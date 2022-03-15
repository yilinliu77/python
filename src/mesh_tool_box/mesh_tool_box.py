import os
import sys
from multiprocessing import Pool

import open3d as o3d
from numba import njit
from pyproj import Transformer, CRS
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numba
import numpy as np
from tqdm.contrib.concurrent import thread_map, process_map

from shared.trajectory import *

# centralized_point = (1.26826e7, 2.57652e6, 0)  # Translate to origin point to prevent overflow
centralized_point = (493260.00, 2492700.00, 0)  # Translate to origin point to prevent overflow

root_file = r"D:\Projects\Photos\2110-OPT-GDSZ-VCC-shendaL7_box3-52.1-0.02-8890-1464-PTBGAJKL\07.重建模型\SD_OPT_box3_obj4547"
output_root = r"D:\Projects\Photos\2110-OPT-GDSZ-VCC-shendaL7_box3-52.1-0.02-8890-1464-PTBGAJKL\merge"
origin_point = (493504.4363, 2492785.125, 131.5)  # Origin point in "metadata.xml"
# Set to -99999 if no requirement
filter_z = 1.5  # L7 CGCS 2000

# L7
l7_boundary_point_wgs84 = [
    # 0th
    [
        [113.9346571, 22.53175914],
        [113.935193, 22.53221303],
        [113.9354288, 22.53270332],
        [113.9354285, 22.53289409],
        [113.9349824, 22.53289853],
        [113.9346911, 22.53248555],
        [113.9342935, 22.5321344]
    ],
    # 1th
    [
        [113.9352184, 22.53295035],
        [113.9364326, 22.53296924],
        [113.9364192, 22.53329267],
        [113.9362734, 22.53342562],
        [113.9354814, 22.53340215],
        [113.9352163, 22.53317406],
    ]
]

# L7
L7_boundary_point_cgcs2000 = [
    [
        [493280.264198, 2492680.963400],
        [493234.691498, 2492729.714399],
        [493281.155197, 2492765.563797],
        [493311.327599, 2492813.793999],
        [493358.606003, 2492814.020699],
        [493359.778595, 2492789.761703],
        [493335.862503, 2492733.190300]
    ],
    [
        [493333.475189, 2492817.767899],
        [493333.395905, 2492842.758804],
        [493360.219696, 2492869.892502],
        [493447.201706, 2492870.637405],
        [493468.051208, 2492850.659195],
        [493467.150208, 2492816.107903]
    ]
]

l7_boundary_point_cgcs2000_huyue = [
        [
            [493280.264198, 2492680.963400],
            [493234.691498, 2492729.714399],
            [493281.155197, 2492765.563797],
            [493311.327599, 2492813.793999],
            [493358.606003, 2492814.020699],
            [493359.778595, 2492789.761703],
            [493335.862503, 2492733.190300]
        ],
        [
            [493333.475189, 2492817.767899],
            [493333.395905, 2492842.758804],
            [493360.219696, 2492869.892502],
            [493447.201706, 2492870.637405],
            [493468.051208, 2492850.659195],
            [493467.150208, 2492816.107903]
        ]
    ]

# Huiwen
huiwen_boundary_point_wgs84 = [
    [
        [113.9305898, 22.53988704],
        [113.9308016, 22.5402481],
        [113.9306809, 22.54033197],
        [113.9304613, 22.54017008],
        [113.9298159, 22.54062068],
        [113.9295466, 22.54061191],
        [113.9295447, 22.54083723],
        [113.9287873, 22.54083075],
        [113.9287853, 22.54061132],
        [113.9286057, 22.54060819],
        [113.9286083, 22.54037422],
        [113.9297707, 22.54039303],
    ],
    [
        [113.9308276, 22.54018904],
        [113.9301828, 22.540681],
        [113.930327, 22.54086299],
        [113.9302004, 22.54095884],
        [113.9301935, 22.5411511],
        [113.9301134, 22.5411469],
        [113.9301188, 22.54096347],
        [113.9292388, 22.54096648],
        [113.9292288, 22.54115068],
        [113.9296099, 22.54115351],
        [113.9296126, 22.54140958],
        [113.9304284, 22.54142874],
        [113.9303327, 22.5410559],
        [113.9310774, 22.54058873],
    ],
    [
        [113.9310857, 22.54061253],
        [113.9306105, 22.54106074],
        [113.9308258, 22.54142147],
        [113.931417, 22.54110089],

    ]
]

used_boundary = l7_boundary_point_cgcs2000_huyue

for i_building, _ in enumerate(used_boundary):
    for i_point, _ in enumerate(used_boundary[i_building]):
        # mercator = lonLat2Mercator(used_boundary[i_building][i_point])
        mercator = used_boundary[i_building][i_point]
        used_boundary[i_building][i_point][0] = mercator[0] - centralized_point[0]
        used_boundary[i_building][i_point][1] = mercator[1] - centralized_point[1]

poly = [Polygon(item) for item in used_boundary]


def back_up_cgcs2000():
    # centralized_point=(-492700,-2493600,-0)

    # root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\qingxie-obj\Data"
    # origin_point = (12682696, 2576546, 0)

    # root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\obj-vertical_0.8_explor\obj-v08\Data"
    # origin_point = (12682641, 2576576, 0)

    # root_file = r"D:\Projects\SEAR\Real\Huiyuan\0512\qinxie-obj-cgcs2000\Data"
    # origin_point = (492854.0403, 2493646.463, 49.5)

    # root_file=r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data"
    # origin_point = (492796,2493689,0)
    # root_file=r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan_plusplus\obj\Data"
    # origin_point = (492703,2493728,0)

    # root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\cgcs_2000_0.8\0.8-obj\Data"
    # origin_point = (492795.0207,[2493679.919,67.45)]

    boundary_point_wgs84 = [
        [12682473.000000, 2576538.250000],
        [12682539.000000, 2576534.250000],
        [12682540.000000, 2576504.000000],
        [12682597.000000, 2576504.250000],
        [12682599.000000, 2576537.000000],
        [12682689.000000, 2576470.750000],
        [12682745.000000, 2576558.500000],
        [12682670.000000, 2576612.500000],
        [12682669.000000, 2576656.500000],
        [12682572.000000, 2576652.500000],
        [12682470.000000, 2576564.500000],
    ]
    # boundary_point_cgcs2000 = [
    #     [
    #         [492650.281250, 2493639.000000],
    #         [492714.218750, 2493643.750000],
    #         [492712.531250, 2493611.250000],
    #         [492769.437500, 2493613.000000],
    #         [492774.375000, 2493643.000000],
    #         [492861.781250, 2493582.250000],
    #         [492907.406250, 2493667.750000],
    #         [492833.656250, 2493722.500000],
    #         [492838.218750, 2493760.000000],
    #         [492754.843750, 2493756.500000],
    #         [492752.968750, 2493731.250000],
    #         [492713.843750, 2493727.000000],
    #         [492713.968750, 2493698.500000],
    #         [492652.906250, 2493682.500000],
    #     ]
    # ]

    boundary_point_cgcs2000 = [
        [
            [492656.1308, 2493666.683],
            [492655.6239, 2493648.669],
            [492775.662, 2493650.52],
            [492850.6935, 2493595.252],
            [492873.4137, 2493625.907],
            [492860.1616, 2493635.193],
            [492847.5828, 2493618.194],
            [492781.7257, 2493666.577],
            [492753.6768, 2493667.39],
            [492752.8509, 2493691.605],
            [492673.7985, 2493691.177],
            [492673.8727, 2493666.413],

        ],
        [
            [492720.8676, 2493705.64],
            [492802.453, 2493707.444],
            [492802.2385, 2493734.866],
            [492820.7304, 2493735.445],
            [492820.071, 2493703.228],
            [492832.5625, 2493693.471],
            [492817.7006, 2493672.885],
            [492877.2412, 2493630.432],
            [492901.5286, 2493663.045],
            [492835.874, 2493714.689],
            [492834.2308, 2493756.481],
            [492759.2705, 2493754.557],
            [492759.3394, 2493726.695],
            [492721.4676, 2493725.875],
        ],
    ]
    filter_z = 20

    for i_building, _ in enumerate(boundary_point_cgcs2000):
        for i_point, _ in enumerate(boundary_point_cgcs2000[i_building]):
            boundary_point_cgcs2000[i_building][i_point][0] += centralized_point[0]
            boundary_point_cgcs2000[i_building][i_point][1] += centralized_point[1]

    poly = [Polygon(item) for item in boundary_point_cgcs2000]


def translate_gt_point():
    total_points = o3d.io.read_point_cloud(r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\huiwen_0_05_lidar.ply")

    pools = Pool(16)
    mesh_point = np.asarray(total_points.points)
    mask = pools.map(f, mesh_point)
    mask = np.asarray(mask, np.int16)
    total_points.points = o3d.utility.Vector3dVector(mesh_point[np.logical_not(mask)])
    o3d.io.write_point_cloud(r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\gt.ply", total_points)
    pass


def test():
    files = [os.path.join(r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data",
                          folder, folder + ".obj") for folder in os.listdir(
        r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data")]
    total_mesh = o3d.geometry.TriangleMesh()
    for item in tqdm(files):
        mesh = o3d.io.read_triangle_mesh(item)
        mesh = mesh.translate((492796, 2493689, 0))
        total_mesh += mesh

    pcl_1e8 = total_mesh.sample_points_uniformly(number_of_points=int(1e7))
    pcl_1e3 = total_mesh.sample_points_uniformly(number_of_points=int(1e3))
    o3d.io.write_point_cloud("drone_scan_1e8.ply", pcl_1e8)
    o3d.io.write_point_cloud("drone_scan_1e3.ply", pcl_1e3)
    pass


def test2():
    txt_array = [item.strip()[:-1].replace(" ", ",") for item in
                 open(r"C:\repo\python\src\test\drone_scan_1e8.xyz").readlines()]
    with open(r"C:\Users\yilin\Desktop\test\111.csv", "w") as f:
        for line in txt_array:
            f.write(line + "\n")


def filter_mesh_according_to_boundary_and_sample_points(v_input_dir,v_input_name):
    v_output_folder = v_input_dir
    mesh = o3d.io.read_triangle_mesh(os.path.join(v_output_folder, v_input_name))
    mesh_point = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    # remove_flag = thread_map(f, mesh_point[mesh_faces])
    remove_flag = process_map(f,mesh_point[mesh_faces],chunksize=1000)
    remove_flag = np.asarray(remove_flag, np.int16)
    mesh.remove_triangles_by_mask(remove_flag)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(
        os.path.join(v_output_folder, "mesh_centralized.ply"), mesh)
    pcl_4 = mesh.sample_points_poisson_disk(number_of_points=int(1e4))
    o3d.io.write_point_cloud(os.path.join(v_output_folder, "1e4.ply"), pcl_4)


def f(v_args):
    item = v_args
    p1 = Point(v_args[0, :2])
    p2 = Point(v_args[1, :2])
    p3 = Point(v_args[2, :2])
    is_in_boundary1 = np.array([p1.within(item) for item in poly])
    is_in_boundary2 = np.array([p2.within(item) for item in poly])
    is_in_boundary3 = np.array([p3.within(item) for item in poly])
    # is_in_boundary = [True]
    return not (np.logical_and(is_in_boundary1, is_in_boundary2, is_in_boundary3).max() and (
            v_args[:, 2] > filter_z).max())

def f_point(v_args):
    p1 = Point(v_args[:2])
    is_in_boundary1 = np.array([p1.within(item) for item in poly])
    return not (is_in_boundary1.max() and (
            v_args[2] > filter_z))


def filter_points_according_to_boundary():
    v_output_folder = r"D:\Projects\Building_data"
    total_points = o3d.io.read_point_cloud(os.path.join(
        r"D:\Projects\Building_data\2110-las-sz-vcc-hwl+L7S-0.04+0.03-las", "L7S_0 - Cloud.ply"))

    mesh_point = np.asarray(total_points.points)
    remove_flag = process_map(f_point,mesh_point,chunksize=1000)
    remove_flag = np.asarray(remove_flag, np.int16)
    total_points.points = o3d.utility.Vector3dVector(mesh_point[np.logical_not(remove_flag)])
    o3d.io.write_point_cloud(os.path.join(v_output_folder,"gt_points.ply"), total_points)
    pass

def merge_mesh_and_filter_the_points_outside_boundary(v_root_file, v_output_folder):
    if not os.path.exists(v_output_folder):
        os.mkdir(v_output_folder)

    files = [os.path.join(v_root_file, folder, folder + ".obj") for folder in os.listdir(v_root_file)]

    total_mesh = o3d.geometry.TriangleMesh()

    pools = Pool(16)
    for item in tqdm(files):
        mesh = o3d.io.read_triangle_mesh(item)
        mesh = mesh.translate(origin_point)
        mesh = mesh.translate(-np.array(centralized_point))
        mesh_point = np.asarray(mesh.vertices)
        mesh_faces = np.asarray(mesh.triangles)
        remove_flag = pools.map(f, mesh_point[mesh_faces])
        remove_flag = np.asarray(remove_flag, np.int16)
        if (remove_flag > 0).sum() - mesh_point.shape[0] == 0:
            continue
        mesh.remove_triangles_by_mask(remove_flag)
        mesh.remove_unreferenced_vertices()
        total_mesh += mesh

    o3d.io.write_triangle_mesh(os.path.join(v_output_folder, "mesh_centralized.ply"), total_mesh)
    pcl_5 = total_mesh.sample_points_uniformly(number_of_points=int(1e5))
    pcl_7 = total_mesh.sample_points_uniformly(number_of_points=int(1e7))
    o3d.io.write_point_cloud(os.path.join(v_output_folder, "1e5.ply"), pcl_5)
    o3d.io.write_point_cloud(os.path.join(v_output_folder, "1e7.ply"), pcl_7)


def convert_coordinate(v_source_coor: int, v_source_coor_shift: np.ndarray, v_target_coor: int, v_input_mesh: str,
                       v_output_mesh: str):
    mesh = o3d.io.read_triangle_mesh(v_input_mesh)
    mesh = mesh.translate(v_source_coor_shift)
    vertices = np.asarray(mesh.vertices)

    source_coor: int = v_source_coor  # First x (12683207), then y (2575391)
    # source_coor: int = 4326 # First latitude (22), then longitude (113)
    target_coor: int = v_target_coor  # First x (2492686), then y (493332)

    transformer = Transformer.from_crs(CRS.from_epsg(source_coor), CRS.from_epsg(target_coor))
    if source_coor == 3857:
        target_vertices = transformer.transform(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        mesh.vertices = o3d.utility.Vector3dVector(np.stack([target_vertices[1],target_vertices[0],target_vertices[2]], axis=1))
    else:
        raise "Unsupported"

    o3d.io.write_triangle_mesh(v_output_mesh, mesh)


if __name__ == '__main__':
    selected_tool: int = 1
    if len(sys.argv) > 1:
        selected_tool = int(sys.argv[1])
    if selected_tool == 1:
        filter_mesh_according_to_boundary_and_sample_points(sys.argv[2],sys.argv[3])
    if selected_tool == 2:
        filter_points_according_to_boundary()
    elif selected_tool == 3:  # Convert coordinate
        source_coor: int = int(sys.argv[2])
        source_coor_shift: np.ndarray = np.array([1.26826e7,2.57652e6,0])
        target_coor: int = int(sys.argv[4])
        input_mesh: str = str(sys.argv[5])
        output_mesh: str = str(sys.argv[6])
        convert_coordinate(source_coor, source_coor_shift, target_coor, input_mesh, output_mesh)
    elif selected_tool == 24:  # Merge mesh
        merge_mesh_and_filter_the_points_outside_boundary(root_file,
                                                          # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\Ours"
                                                          # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan"
                                                          # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan_plusplus"
                                                          # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\oblique"
                                                          output_root
                                                          # r"D:\Projects\Building_data\Huiwen\CON_Fine"
                                                          )
