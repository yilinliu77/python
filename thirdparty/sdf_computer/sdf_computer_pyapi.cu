#include <neural-graphics-primitives/common_device.cuh>
// #include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <json/json.hpp>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

#include "sdf_computer.cuh"

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

// py::array_t<float> compute_sdf(
//     const std::string v_path,
//     const int v_num_on_face,
//     const int v_num_near_face,
//     const int v_num_uniform)
// {
//     std::vector<Eigen::Vector4f> _results = load_mesh(v_path, v_num_on_face, v_num_near_face, v_num_uniform);
//     // py::array_t<float> result;
//     py::array_t<float> result({ _results.size(), (size_t)4 });
//     py::buffer_info buf = result.request();
//     float* data = (float*)buf.ptr;
//     for (int i = 0; i < _results.size(); ++i) {
//         data[i * 4 + 0] = _results[i][0];
//         data[i * 4 + 1] = _results[i][1];
//         data[i * 4 + 2] = _results[i][2];
//         data[i * 4 + 3] = _results[i][3];
//     }
//     return result;
// }

// class SDF_computer {
// public:
//     Eigen::Vector3f m_center;
//     float m_scale;

//     std::vector<Eigen::Vector3f> m_vertices;
//     BoundingBox m_aabb = {};

//     tcnn::GPUMemory<Triangle> triangles_gpu;
//     std::vector<Triangle> triangles_cpu;
//     std::vector<float> triangle_weights;
//     DiscreteDistribution triangle_distribution;
//     tcnn::GPUMemory<float> triangle_cdf;
//     std::shared_ptr<TriangleBvh> triangle_bvh; // unique_ptr
//     std::shared_ptr<TriangleOctree> triangle_octree;

//     tcnn::StreamAndEvent m_stream;

//     // v_triangles: (n_triangles, 3, 3)
//     SDF_computer(const py::array_t<float>& v_triangles)
//     {
//         auto r = v_triangles.unchecked<3>();
//         // m_vertices = v_triangles;
//         m_vertices.resize(v_triangles.shape(0) * 3);
//         for (int i = 0; i < v_triangles.shape(0); ++i)
//             for (int j = 0; j < 3; ++j) {
//                 m_vertices[i * 3 + j] = Eigen::Vector3f(r(i, j, 0), r(i, j, 1), r(i, j, 2));
//             }

//         size_t n_vertices = m_vertices.size();
//         size_t n_triangles = n_vertices / 3;

//         BoundingBox m_raw_aabb;
//         m_raw_aabb.min = Vector3f::Constant(std::numeric_limits<float>::infinity());
//         m_raw_aabb.max = Vector3f::Constant(-std::numeric_limits<float>::infinity());
//         for (size_t i = 0; i < n_vertices; ++i) {
//             m_raw_aabb.enlarge(m_vertices[i]);
//         }

//         // Inflate AABB by 1% to give the network a little wiggle room.
//         const float inflation = 0.005f;
//         m_raw_aabb.inflate(m_raw_aabb.diag().norm() * inflation);

//         float mesh_scale = m_raw_aabb.diag().maxCoeff();

//         // Normalize vertex coordinates to lie within [0,1]^3.
//         // This way, none of the constants need to carry around
//         // bounding box factors.
//         for (size_t i = 0; i < n_vertices; ++i) {
//             m_vertices[i] = (m_vertices[i] - m_raw_aabb.min - 0.5f * m_raw_aabb.diag()) / mesh_scale + Vector3f::Constant(0.5f);
//         }
//         m_center = 0.5f * m_raw_aabb.diag();
//         m_scale = mesh_scale;

//         for (size_t i = 0; i < n_vertices; ++i) {
//             m_aabb.enlarge(m_vertices[i]);
//         }
//         m_aabb.inflate(m_aabb.diag().norm() * inflation);
//         m_aabb = m_aabb.intersection(BoundingBox { Vector3f::Zero(), Vector3f::Ones() });

//         triangles_cpu.resize(n_triangles);
//         for (size_t i = 0; i < n_vertices; i += 3) {
//             triangles_cpu[i / 3] = { m_vertices[i + 0], m_vertices[i + 1], m_vertices[i + 2] };
//         }
//         if (!triangle_bvh)
//             triangle_bvh = TriangleBvh::make();

//         triangle_bvh->build(triangles_cpu, 8);
//         triangles_gpu.resize_and_copy_from_host(triangles_cpu);
//         triangle_bvh->build_optix(triangles_gpu, m_stream.get());

//         triangle_octree.reset(new TriangleOctree {});
//         triangle_octree->build(*triangle_bvh, triangles_cpu, 10);

//         // set_scale(m_bounding_radius * 1.5f);

//         // Compute discrete probability distribution for later sampling of the mesh's surface
//         triangle_weights.resize(n_triangles);
//         for (size_t i = 0; i < n_triangles; ++i) {
//             triangle_weights[i] = triangles_cpu[i].surface_area();
//         }
//         triangle_distribution.build(triangle_weights);

//         // Move CDF to gpu
//         triangle_cdf.resize_and_copy_from_host(triangle_distribution.cdf);
//     }

//     py::array_t<float> compute_sdf(
//         const int v_num_on_face,
//         const int v_num_near_face,
//         const int v_num_uniform,
//         bool is_scale_to_aabb)
//     {
//         const uint32_t n_to_generate = (v_num_on_face + v_num_near_face + v_num_uniform);
//         const uint32_t n_to_generate_surface_exact = v_num_on_face;
//         const uint32_t n_to_generate_surface_offset = v_num_near_face;
//         const uint32_t n_to_generate_uniform = v_num_uniform;

//         const uint32_t n_to_generate_surface = n_to_generate_surface_exact + n_to_generate_surface_offset;
//         cudaStream_t stream = m_stream.get();

//         tcnn::GPUMemory<Eigen::Vector3f> _positions;
//         tcnn::GPUMemory<float> _distances;
//         tcnn::GPUMemory<Eigen::Vector3f> _perturbations;
//         _positions.enlarge(n_to_generate);
//         _distances.enlarge(n_to_generate);

//         Vector3f* positions = _positions.data();
//         float* distances = _distances.data();

//         // Generate uniform 3D samples. Some of these will be transformed to cover the surfaces uniformly. Others will be left as-is.
//         default_rng_t m_rng;

//         // *****************************************************
//         // # Generate random sample on surface mesh
//         // *****************************************************
//         {
//             tcnn::generate_random_uniform<float>(stream, m_rng, n_to_generate * 3, (float*)positions);

//             tcnn::linear_kernel(sample_uniform_on_triangle_kernel, 0, stream,
//                 n_to_generate_surface,
//                 triangle_cdf.data(),
//                 (uint32_t)triangle_cdf.size(),
//                 triangles_gpu.data(),
//                 positions);

//             // The distances of points on the mesh are zero. Can immediately set.
//             CUDA_CHECK_THROW(cudaMemsetAsync(distances, 0, n_to_generate_surface_exact * sizeof(float), stream));
//         }

//         // *****************************************************
//         // # Generate random sample
//         // *****************************************************
//         float m_bounding_radius = Vector3f::Constant(0.5f).norm();
//         float stddev = m_bounding_radius / 1024.0f;
//         {
//             // If we have an octree, generate uniform samples within that octree.
//             // Otherwise, at least confine uniform samples to the AABB.
//             // (For the uniform_only case, we always use the AABB, then the IoU kernel checks against the octree later)

//             if (is_scale_to_aabb) {
//                 BoundingBox sdf_aabb = m_aabb;
//                 sdf_aabb.inflate(0);
//                 tcnn::linear_kernel(scale_to_aabb_kernel, 0, stream,
//                     n_to_generate_uniform, sdf_aabb,
//                     positions + n_to_generate_surface);

//                 tcnn::linear_kernel(assign_float, 0, stream,
//                     n_to_generate_uniform,
//                     sdf_aabb.diag().norm() * 1.001f,
//                     distances + n_to_generate_surface);
//             } else {
//                 // Used for acclerate computation of SDF
//                 tcnn::linear_kernel(assign_float, 0, stream,
//                     n_to_generate_uniform,
//                     std::sqrt(1 + 1 + 1) * 1.001f,
//                     distances + n_to_generate_surface);
//             }
//         }

//         // *****************************************************
//         // # Generate samples near the surface
//         // *****************************************************
//         {
//             _perturbations.enlarge(n_to_generate_surface_offset);
//             tcnn::generate_random_logistic<float>(stream, m_rng, n_to_generate_surface_offset * 3, (float*)_perturbations.data(), 0.0f, stddev);

//             tcnn::linear_kernel(perturb_sdf_samples, 0, stream,
//                 n_to_generate_surface_offset,
//                 _perturbations.data(),
//                 positions + n_to_generate_surface_exact,
//                 distances + n_to_generate_surface_exact);
//         }

//         // The following function expects `distances` to contain an upper bound on the
//         // true distance. This accelerates lookups.
//         triangle_bvh->signed_distance_gpu(
//             n_to_generate_uniform + n_to_generate_surface_offset,
//             EMeshSdfMode::Raystab,
//             positions + n_to_generate_surface_exact,
//             distances + n_to_generate_surface_exact,
//             triangles_gpu.data(),
//             true,
//             stream);

//         CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//         std::vector<Eigen::Vector3f> results_vertices(n_to_generate);
//         std::vector<float> results_distances(n_to_generate);
//         _positions.copy_to_host(results_vertices);
//         _distances.copy_to_host(results_distances);

//         py::array_t<float> results({ results_vertices.size(), (size_t)4 });
//         auto r = results.mutable_unchecked<2>();
//         for (int i = 0; i < results_vertices.size(); ++i) {
//             r(i, 0) = results_vertices[i][0];
//             r(i, 1) = results_vertices[i][1];
//             r(i, 2) = results_vertices[i][2];
//             r(i, 3) = results_distances[i];
//         }
//         return results;
//     }
// };

class PYSDF_computer {
public:
    Eigen::Vector3f m_center;
    float m_scale;
    SDF_computer sdf_computer;

    PYSDF_computer() { }
    void setup_bounds(const py::array_t<float>& v_bounds) 
    {
        auto r = v_bounds.unchecked<1>();

        Eigen::Vector3f center(r(0), r(1), r(2));

        sdf_computer.setup_bounds(center, r(3));
    }

    // v_triangles: (n_triangles, 3, 3)
    void setup_mesh(const py::array_t<float>& v_triangles, const bool v_need_normalize)
    {
        auto r = v_triangles.unchecked<3>();
        std::vector<Eigen::Vector3f> m_vertices;
        m_vertices.resize(v_triangles.shape(0) * 3);
        for (int i = 0; i < v_triangles.shape(0); ++i)
            for (int j = 0; j < 3; ++j) {
                m_vertices[i * 3 + j] = Eigen::Vector3f(r(i, j, 0), r(i, j, 1), r(i, j, 2));
            }

        sdf_computer.setup_mesh(m_vertices, v_need_normalize);
        m_center = sdf_computer.m_center;
        m_scale = sdf_computer.m_scale;
    }

    py::array_t<float> compute_sdf(
        const int v_num_on_face,
        const int v_num_near_face,
        const int v_num_uniform,
        bool is_scale_to_aabb)
    {
        const auto result = sdf_computer.compute_sdf(v_num_on_face, v_num_near_face, v_num_uniform, is_scale_to_aabb);

        py::array_t<float> results({ result.first.size(), (size_t)4 });
        auto r = results.mutable_unchecked<2>();
        for (int i = 0; i < result.first.size(); ++i) {
            r(i, 0) = result.first[i][0];
            r(i, 1) = result.first[i][1];
            r(i, 2) = result.first[i][2];
            r(i, 3) = result.second[i];
        }
        return results;
    }

    py::array_t<short> check_visibility(
        const py::array_t<float>& v_cameras,
        const py::array_t<float>& v_points)
    {
        // std::cout << "Start 1" << std::endl;
        std::vector<Eigen::Vector3f> cameras;
        std::vector<Eigen::Vector3f> points;
        cameras.resize(v_cameras.shape(0));
        points.resize(v_points.shape(0));

        auto r_cameras = v_cameras.unchecked<2>();
        auto r_points = v_points.unchecked<2>();

        for (size_t i = 0; i < cameras.size(); ++i) {
            cameras[i][0] = r_cameras(i, 0);
            cameras[i][1] = r_cameras(i, 1);
            cameras[i][2] = r_cameras(i, 2);
        }

        for (size_t i = 0; i < points.size(); ++i) {
            points[i][0] = r_points(i, 0);
            points[i][1] = r_points(i, 1);
            points[i][2] = r_points(i, 2);
        }

        // std::cout << "Start 2" << std::endl;
        const auto result = sdf_computer.compute_visibility(
            cameras, points);

        // std::cout << "Start 3" << std::endl;
        py::array_t<short> results(result.size());
        auto r = results.mutable_unchecked<1>();
        for (int i = 0; i < result.size(); ++i) {
            r(i) = result[i];
        }
        // std::cout << "Start 4" << std::endl;
        return results;
    }
};

PYBIND11_MODULE(pysdf, m)
{
    m.doc() = "Fast SDF generation";

    // m.def("compute_sdf", &compute_sdf);

    py::class_<PYSDF_computer> sdf_computer(m, "PYSDF_computer");
    sdf_computer
        .def(py::init<>())
        .def("setup_bounds", &PYSDF_computer::setup_bounds)
        .def("compute_sdf", &PYSDF_computer::compute_sdf)
        .def("setup_mesh", &PYSDF_computer::setup_mesh)
        .def("check_visibility", &PYSDF_computer::check_visibility)
        .def_readonly("m_center", &PYSDF_computer::m_center)
        .def_readonly("m_scale", &PYSDF_computer::m_scale);
}
