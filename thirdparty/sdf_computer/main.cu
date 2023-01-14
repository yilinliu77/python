#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/path.h>

using namespace ngp;
using namespace std;
//  using namespace tcnn;
namespace fs = ::filesystem;

using namespace Eigen;
// using namespace tcnn;

__host__ __device__ uint32_t sample_discrete(float uniform_sample, const float* __restrict__ cdf, int length)
{
    return binary_search(uniform_sample, cdf, length);
}

__global__ void sample_uniform_on_triangle_kernel(uint32_t n_elements, const float* __restrict__ cdf, uint32_t length, const Triangle* __restrict__ triangles, Vector3f* __restrict__ sampled_positions)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    Vector3f sample = sampled_positions[i];
    uint32_t tri_idx = sample_discrete(sample.x(), cdf, length);

    sampled_positions[i] = triangles[tri_idx].sample_uniform_position(sample.tail<2>());
}

__global__ void uniform_octree_sample_kernel(
    const uint32_t num_elements,
    default_rng_t rng,
    const TriangleOctreeNode* __restrict__ octree_nodes,
    uint32_t num_nodes,
    uint32_t depth,
    Vector3f* __restrict__ samples)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
        return;

    rng.advance(i * (1 << 8));

    // Samples random nodes until a leaf is picked
    uint32_t node;
    uint32_t child;
    do {
        node = umin((uint32_t)(random_val(rng) * num_nodes), num_nodes - 1);
        child = umin((uint32_t)(random_val(rng) * 8), 8u - 1);
    } while (octree_nodes[node].depth < depth - 2 || octree_nodes[node].children[child] == -1);

    // Here it should be guaranteed that any child of the node is -1
    float size = scalbnf(1.0f, -depth + 1);

    Vector3i16 pos = octree_nodes[node].pos * 2;
    if (child & 1)
        ++pos.x();
    if (child & 2)
        ++pos.y();
    if (child & 4)
        ++pos.z();
    samples[i] = size * (pos.cast<float>() + samples[i]);
}

__global__ void assign_float(uint32_t n_elements, float value, float* __restrict__ out)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    out[i] = value;
}

__global__ void scale_to_aabb_kernel(uint32_t n_elements, BoundingBox aabb, Vector3f* __restrict__ inout)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    inout[i] = aabb.min + inout[i].cwiseProduct(aabb.diag());
}

__global__ void perturb_sdf_samples(uint32_t n_elements, const Vector3f* __restrict__ perturbations, Vector3f* __restrict__ positions, float* __restrict__ distances)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    Vector3f perturbation = perturbations[i];
    positions[i] += perturbation;

    // Small epsilon above 1 to ensure a triangle is always found.
    distances[i] = perturbation.norm() * 1.001f;
}

std::vector<Vector4f> load_mesh(const std::string v_data_path,
    const int v_num_on_face,
    const int v_num_near_face,
    const int v_num_uniform)
{
    std::vector<Vector3f> vertices = load_obj(v_data_path);
    // The expected format is
    // [v1.x][v1.y][v1.z][v2.x]...
    size_t n_vertices = vertices.size();
    size_t n_triangles = n_vertices / 3;

    BoundingBox m_raw_aabb;
    m_raw_aabb.min = Vector3f::Constant(std::numeric_limits<float>::infinity());
    m_raw_aabb.max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < n_vertices; ++i) {
        m_raw_aabb.enlarge(vertices[i]);
    }

    // Inflate AABB by 1% to give the network a little wiggle room.
    const float inflation = 0.005f;
    m_raw_aabb.inflate(m_raw_aabb.diag().norm() * inflation);

    float mesh_scale = m_raw_aabb.diag().maxCoeff();

    // Normalize vertex coordinates to lie within [0,1]^3.
    // This way, none of the constants need to carry around
    // bounding box factors.
    for (size_t i = 0; i < n_vertices; ++i) {
        vertices[i] = (vertices[i] - m_raw_aabb.min - 0.5f * m_raw_aabb.diag()) / mesh_scale + Vector3f::Constant(0.5f);
    }

    BoundingBox m_aabb;
    m_aabb = {};
    for (size_t i = 0; i < n_vertices; ++i) {
        m_aabb.enlarge(vertices[i]);
    }
    m_aabb.inflate(m_aabb.diag().norm() * inflation);
    m_aabb = m_aabb.intersection(BoundingBox { Vector3f::Zero(), Vector3f::Ones() });

    tcnn::GPUMemory<Triangle> triangles_gpu;
    std::vector<Triangle> triangles_cpu;
    std::vector<float> triangle_weights;
    DiscreteDistribution triangle_distribution;
    tcnn::GPUMemory<float> triangle_cdf;
    std::shared_ptr<TriangleBvh> triangle_bvh; // unique_ptr
    std::shared_ptr<TriangleOctree> triangle_octree;

    triangles_cpu.resize(n_triangles);
    for (size_t i = 0; i < n_vertices; i += 3) {
        triangles_cpu[i / 3] = { vertices[i + 0], vertices[i + 1], vertices[i + 2] };
    }

    if (!triangle_bvh)
        triangle_bvh = TriangleBvh::make();
    triangle_bvh->build(triangles_cpu, 8);
    triangles_gpu.resize_and_copy_from_host(triangles_cpu);
    tcnn::StreamAndEvent m_stream;
    triangle_bvh->build_optix(triangles_gpu, m_stream.get());

    triangle_octree.reset(new TriangleOctree {});
    triangle_octree->build(*triangle_bvh, triangles_cpu, 10);

    float m_bounding_radius = 1;
    m_bounding_radius = Vector3f::Constant(0.5f).norm();
    // set_scale(m_bounding_radius * 1.5f);

    // Compute discrete probability distribution for later sampling of the mesh's surface
    triangle_weights.resize(n_triangles);
    for (size_t i = 0; i < n_triangles; ++i) {
        triangle_weights[i] = triangles_cpu[i].surface_area();
    }
    triangle_distribution.build(triangle_weights);

    // Move CDF to gpu
    triangle_cdf.resize_and_copy_from_host(triangle_distribution.cdf);

    // Mesh data
    EMeshSdfMode mesh_sdf_mode = EMeshSdfMode::Raystab;

    bool uses_takikawa_encoding = false;
    bool use_triangle_octree = false;

    float zero_offset = 0;
    float surface_offset_scale = 1.0f;

    tlog::success() << "Loaded mesh: triangles=" << n_triangles << " AABB=" << m_raw_aabb << " after scaling=" << m_aabb;

    const uint32_t n_to_generate = (v_num_on_face + v_num_near_face + v_num_uniform); // Originally it is 4:3:1 -> exact:near:uniform
    tcnn::GPUMemory<Eigen::Vector3f> _positions;
    tcnn::GPUMemory<float> _distances;
    tcnn::GPUMemory<Eigen::Vector3f> _perturbations;
    _positions.enlarge(n_to_generate);
    _distances.enlarge(n_to_generate);

    Vector3f* positions = _positions.data();
    float* distances = _distances.data();
    bool uniform_only = false;

    // uint32_t n_to_generate_base = n_to_generate / 8;
    // const uint32_t n_to_generate_surface_exact = uniform_only ? 0 : n_to_generate_base * 4;
    // const uint32_t n_to_generate_surface_offset = uniform_only ? 0 : n_to_generate_base * 3;
    // const uint32_t n_to_generate_uniform = uniform_only ? n_to_generate : n_to_generate_base * 1;

    const uint32_t n_to_generate_surface_exact = v_num_on_face;
    const uint32_t n_to_generate_surface_offset = v_num_near_face;
    const uint32_t n_to_generate_uniform = v_num_uniform;

    const uint32_t n_to_generate_surface = n_to_generate_surface_exact + n_to_generate_surface_offset;
    cudaStream_t stream = m_stream.get();

    // Generate uniform 3D samples. Some of these will be transformed to cover the surfaces uniformly. Others will be left as-is.
    default_rng_t m_rng;

    // *****************************************************
    // # Generate random sample on surface mesh
    // *****************************************************
    {
        tcnn::generate_random_uniform<float>(stream, m_rng, n_to_generate * 3, (float*)positions);

        tcnn::linear_kernel(sample_uniform_on_triangle_kernel, 0, stream,
            n_to_generate_surface,
            triangle_cdf.data(),
            (uint32_t)triangle_cdf.size(),
            triangles_gpu.data(),
            positions);

        // The distances of points on the mesh are zero. Can immediately set.
        CUDA_CHECK_THROW(cudaMemsetAsync(distances, 0, n_to_generate_surface_exact * sizeof(float), stream));
    }

    // *****************************************************
    // # Generate random sample
    // *****************************************************
    float stddev = m_bounding_radius / 1024.0f * surface_offset_scale;
    {
        // If we have an octree, generate uniform samples within that octree.
        // Otherwise, at least confine uniform samples to the AABB.
        // (For the uniform_only case, we always use the AABB, then the IoU kernel checks against the octree later)
        if (!uniform_only && (uses_takikawa_encoding || use_triangle_octree)) {
            tcnn::linear_kernel(uniform_octree_sample_kernel, 0, stream,
                n_to_generate_uniform,
                m_rng,
                triangle_octree->nodes_gpu(),
                triangle_octree->n_nodes(),
                triangle_octree->depth(),
                positions + n_to_generate_surface);
            m_rng.advance();

            // If we know the finest discretization of the octree, we can concentrate
            // points MUCH closer to the mesh surface
            float leaf_size = scalbnf(1.0f, -triangle_octree->depth() + 1);
            if (leaf_size < stddev) {
                tlog::warning() << "leaf_size < stddev";
                stddev = leaf_size;
            }

            tcnn::linear_kernel(assign_float, 0, stream,
                n_to_generate_uniform,
                Vector3f::Constant(leaf_size).norm() * 1.001f,
                distances + n_to_generate_surface);
        } else {
            BoundingBox sdf_aabb = m_aabb;
            sdf_aabb.inflate(zero_offset);
            tcnn::linear_kernel(scale_to_aabb_kernel, 0, stream,
                n_to_generate_uniform, sdf_aabb,
                positions + n_to_generate_surface);

            tcnn::linear_kernel(assign_float, 0, stream,
                n_to_generate_uniform,
                sdf_aabb.diag().norm() * 1.001f,
                distances + n_to_generate_surface);
        }
    }

    // *****************************************************
    // # Generate samples near the surface
    // *****************************************************
    {
        _perturbations.enlarge(n_to_generate_surface_offset);
        tcnn::generate_random_logistic<float>(stream, m_rng, n_to_generate_surface_offset * 3, (float*)_perturbations.data(), 0.0f, stddev);

        tcnn::linear_kernel(perturb_sdf_samples, 0, stream,
            n_to_generate_surface_offset,
            _perturbations.data(),
            positions + n_to_generate_surface_exact,
            distances + n_to_generate_surface_exact);
    }

    // The following function expects `distances` to contain an upper bound on the
    // true distance. This accelerates lookups.
    triangle_bvh->signed_distance_gpu(
        n_to_generate_uniform + n_to_generate_surface_offset,
        mesh_sdf_mode,
        positions + n_to_generate_surface_exact,
        distances + n_to_generate_surface_exact,
        triangles_gpu.data(),
        true,
        stream);

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    std::vector<Vector4f> results(n_to_generate);
    std::vector<Eigen::Vector3f> results_vertices(n_to_generate);
    std::vector<float> results_distances(n_to_generate);
    _positions.copy_to_host(results_vertices);
    _distances.copy_to_host(results_distances);
    for (int i = 0; i < results_vertices.size();++i)
    {
        results[i][0] = results_vertices[i][0];
        results[i][1] = results_vertices[i][1];
        results[i][2] = results_vertices[i][2];
        results[i][3] = results_distances[i];
    }
    return results;
    
}


int main(int argc, char** argv)
{
    load_mesh(
        // "/root/code/instant-ngp/data/sdf/armadillo.obj",
        "/root/code/nglod/sdf-net/data/detailed_l7_with_ground.obj",
        4000000, 3000000, 1000000
    );
}
