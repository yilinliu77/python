#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
// #include <neural-graphics-primitives/render_buffer.h>
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

class SDF_computer {
public:
    Eigen::Vector3f m_center;
    float m_scale;

    std::vector<Eigen::Vector3f> m_vertices;
    BoundingBox m_aabb = {};

    tcnn::GPUMemory<Triangle> triangles_gpu;
    std::vector<Triangle> triangles_cpu;
    std::vector<float> triangle_weights;
    DiscreteDistribution triangle_distribution;
    tcnn::GPUMemory<float> triangle_cdf;
    std::shared_ptr<TriangleBvh> triangle_bvh; // unique_ptr
    std::shared_ptr<TriangleOctree> triangle_octree;

    tcnn::StreamAndEvent m_stream;

    bool m_is_initialized_bounds;

    SDF_computer(): m_is_initialized_bounds(false) { }
    void setup_bounds(
        const Eigen::Vector3f& v_center,
        const float v_scale)
    {
        m_is_initialized_bounds=true;
        m_center = v_center;
        m_scale = v_scale;
    }

    // v_triangles: (n_triangles, 3, 3)
    void setup_mesh(const std::vector<Eigen::Vector3f>& v_vertices, const bool v_need_normalize)
    {
        m_vertices = v_vertices;
        size_t n_vertices = m_vertices.size();
        size_t n_triangles = n_vertices / 3;

        // Initialized bounds
        if (!m_is_initialized_bounds)
        {
            BoundingBox m_raw_aabb;
            m_raw_aabb.min = Vector3f::Constant(std::numeric_limits<float>::infinity());
            m_raw_aabb.max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < n_vertices; ++i) {
                m_raw_aabb.enlarge(m_vertices[i]);
            }
            // Inflate AABB by 1% to give the network a little wiggle room.
            const float inflation = 0.005f;
            m_raw_aabb.inflate(m_raw_aabb.diag().norm() * inflation);

            m_scale = m_raw_aabb.diag().maxCoeff();
            m_center = m_raw_aabb.min + 0.5f * m_raw_aabb.diag();
        }
        else
        {
        }

        if (v_need_normalize)
        {
            // Normalize vertex coordinates to lie within [0,1]^3.
            // This way, none of the constants need to carry around
            // bounding box factors.
            for (size_t i = 0; i < n_vertices; ++i) {
                m_vertices[i] = (m_vertices[i] - m_center) / m_scale + Vector3f::Constant(0.5f);
            }
        }

        for (size_t i = 0; i < n_vertices; ++i) {
            m_aabb.enlarge(m_vertices[i]);
        }

        triangles_cpu.resize(n_triangles);
        for (size_t i = 0; i < n_vertices; i += 3) {
            triangles_cpu[i / 3] = { m_vertices[i + 0], m_vertices[i + 1], m_vertices[i + 2] };
        }
        if (!triangle_bvh)
            triangle_bvh = TriangleBvh::make();

        triangle_bvh->build(triangles_cpu, 8);
        triangles_gpu.resize_and_copy_from_host(triangles_cpu);
        triangle_bvh->build_optix(triangles_gpu, m_stream.get());

        triangle_octree.reset(new TriangleOctree {});
        triangle_octree->build(*triangle_bvh, triangles_cpu, 10);

        // set_scale(m_bounding_radius * 1.5f);

        // Compute discrete probability distribution for later sampling of the mesh's surface
        triangle_weights.resize(n_triangles);
        for (size_t i = 0; i < n_triangles; ++i) {
            triangle_weights[i] = triangles_cpu[i].surface_area();
        }
        triangle_distribution.build(triangle_weights);

        // Move CDF to gpu
        triangle_cdf.resize_and_copy_from_host(triangle_distribution.cdf);
    }

    std::vector<short> compute_visibility(
        const std::vector<Vector3f>& v_camera_pos,
        const std::vector<Vector3f>& v_point_pos)
    {
        const uint32_t num_camera = v_camera_pos.size();
        const uint32_t num_point = v_point_pos.size();
        tcnn::GPUMemory<Vector3f> point_pos_gpu;
        tcnn::GPUMemory<Vector3f> camera_pos_gpu;
        camera_pos_gpu.resize_and_copy_from_host(v_camera_pos);
        point_pos_gpu.resize_and_copy_from_host(v_point_pos);
        tcnn::GPUMemory<short> visibility;
        visibility.enlarge(num_point * num_camera);

        cudaStream_t stream = m_stream.get();

        triangle_bvh->check_visibility(num_camera * num_point,
            num_camera,
            num_point,
            camera_pos_gpu.data(),
            point_pos_gpu.data(),
            triangles_gpu.data(),
            visibility.data(),
            stream);
        triangle_bvh->cl
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        std::vector<short> results(num_camera * num_point);
        visibility.copy_to_host(results);
        return results;
    }

    std::pair<std::vector<Eigen::Vector3f>, std::vector<float>> compute_sdf(
        const int v_num_on_face,
        const int v_num_near_face,
        const int v_num_uniform,
        bool is_scale_to_aabb)
    {
        const uint32_t n_to_generate = (v_num_on_face + v_num_near_face + v_num_uniform);
        const uint32_t n_to_generate_surface_exact = v_num_on_face;
        const uint32_t n_to_generate_surface_offset = v_num_near_face;
        const uint32_t n_to_generate_uniform = v_num_uniform;

        const uint32_t n_to_generate_surface = n_to_generate_surface_exact + n_to_generate_surface_offset;
        cudaStream_t stream = m_stream.get();

        tcnn::GPUMemory<Eigen::Vector3f> _positions;
        tcnn::GPUMemory<float> _distances;
        tcnn::GPUMemory<Eigen::Vector3f> _perturbations;
        _positions.enlarge(n_to_generate);
        _distances.enlarge(n_to_generate);

        Vector3f* positions = _positions.data();
        float* distances = _distances.data();

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
        float m_bounding_radius = Vector3f::Constant(0.5f).norm();
        float stddev = m_bounding_radius / 1024.0f;
        {
            // If we have an octree, generate uniform samples within that octree.
            // Otherwise, at least confine uniform samples to the AABB.
            // (For the uniform_only case, we always use the AABB, then the IoU kernel checks against the octree later)

            if (is_scale_to_aabb) {
                BoundingBox sdf_aabb = m_aabb;
                sdf_aabb.inflate(0);
                tcnn::linear_kernel(scale_to_aabb_kernel, 0, stream,
                    n_to_generate_uniform, sdf_aabb,
                    positions + n_to_generate_surface);

                tcnn::linear_kernel(assign_float, 0, stream,
                    n_to_generate_uniform,
                    sdf_aabb.diag().norm() * 1.001f,
                    distances + n_to_generate_surface);
            } else {
                // Used for acclerate computation of SDF
                tcnn::linear_kernel(assign_float, 0, stream,
                    n_to_generate_uniform,
                    std::sqrt(1 + 1 + 1) * 1.001f,
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
            EMeshSdfMode::Raystab,
            positions + n_to_generate_surface_exact,
            distances + n_to_generate_surface_exact,
            triangles_gpu.data(),
            true,
            stream);

        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        std::vector<Eigen::Vector3f> results_vertices(n_to_generate);
        std::vector<float> results_distances(n_to_generate);
        _positions.copy_to_host(results_vertices);
        _distances.copy_to_host(results_distances);
        return { results_vertices, results_distances };
    }
};