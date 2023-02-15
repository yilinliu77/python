#include "sdf_computer.cuh"

int main(int argc, char** argv)
{
    // load_mesh(
    //     "/root/code/nglod/sdf-net/data/detailed_l7_with_ground.obj",
    //     4000000, 3000000, 1000000
    // );

    std::vector<Vector3f> cameras;
    std::vector<Vector3f> points;

    cameras.emplace_back(53.011044,38.780777, 29.532574);
    points.emplace_back(53.011044, 38.780777, 9.632574);
    points.emplace_back(53.011044, 38.780777, 19.432574);
    points.emplace_back(53.011044, 38.780777, 19.532574);
    points.emplace_back(53.011044, 38.780777, 19.732574);

    SDF_computer sdf_computer;

    std::vector<Vector3f> vertices = load_obj("C:/DATASET/Test_imgs2_colmap_neural/sparse_align/detailed_l7_with_ground.obj");
    sdf_computer.setup_mesh(vertices, true);

    for (auto& item : cameras)
    {
        item = (item - sdf_computer.m_center) / sdf_computer.m_scale + Vector3f::Constant(0.5);
    }
    for (auto& item : points) {
        item = (item - sdf_computer.m_center) / sdf_computer.m_scale + Vector3f::Constant(0.5);
    }
    auto result = sdf_computer.compute_visibility(cameras, points);
    std::cout << result[0] << std::endl;
    std::cout << result[1] << std::endl;
    std::cout << result[2] << std::endl;
    std::cout << result[3] << std::endl;
    return 0;
}
