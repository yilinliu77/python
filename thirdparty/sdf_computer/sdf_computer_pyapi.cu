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
