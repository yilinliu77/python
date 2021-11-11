import os
from multiprocessing import Pool

import open3d as o3d
from numba import njit
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numba
import numpy as np
from shared.trajectory import *

centralized_point=(-12683215., -2575657., 0) # Translate to origin point to prevent overflow

root_file = r"C:\Users\yilin\Documents\Tencent Files\1787609016\FileRecv\L7S_10W_youshi_2000\L7S_10W_youshi_2000"
origin_point = (38493481.32,2492744.246,144.5) # Origin point in "metadata.xml"
filter_z=-99999 # Set to -99999 if no requirement

# root_file = r"D:\Projects\Building_data\2110-CON-SZ-VCC-huiwenlou+Fine1-51-0.03-8853-1319-PTBGAJKL\重建模型\obj"
# origin_point = (12682233.3,2576043.55,128) # Origin point in "metadata.xml"
# filter_z=-99999 # Set to -99999 if no requirement

# L7
boundary_point_wgs84 = [
    # 0th
    [
        [113.9346571,22.53175914],
        [113.935193,22.53221303],
        [113.9354288,22.53270332],
        [113.9354285,22.53289409],
        [113.9349824,22.53289853],
        [113.9346911,22.53248555],
        [113.9342935,22.5321344]
    ],
    # 1th
    [
        [113.9352184,22.53295035],
        [113.9364326,22.53296924],
        [113.9364192,22.53329267],
        [113.9362734,22.53342562],
        [113.9354814,22.53340215],
        [113.9352163,22.53317406],
    ]
]

# Huiwen
# boundary_point_wgs84 =[
#     [
#         [113.9304898,22.53998704],
#         [113.9307016,22.5402481],
#         [113.9305809,22.54033197],
#         [113.9304613,22.54017008],
#         [113.9298159,22.54062068],
#         [113.9295466,22.54061191],
#         [113.9295347,22.54083723],
#         [113.9287873,22.54083075],
#         [113.9287853,22.54061132],
#         [113.9286057,22.54060819],
#         [113.9286083,22.54047422],
#         [113.9297707,22.54049303],
#     ],
#     [
#         [113.9307276,22.54028904],
#         [113.9301828,22.540681],
#         [113.930327,22.54086299],
#         [113.9302004,22.54095884],
#         [113.9301935,22.5412511],
#         [113.9300134,22.5412469],
#         [113.9300188,22.54099347],
#         [113.9292388,22.54098648],
#         [113.9292288,22.54115068],
#         [113.9296099,22.54115351],
#         [113.9296126,22.54140958],
#         [113.9303284,22.54142874],
#         [113.9303327,22.5410559],
#         [113.9309774,22.54058873],
#     ],
#     [
#         [113.9310857,22.54071253],
#         [113.9306105,22.54106074],
#         [113.9308258,22.54132147],
#         [113.931317,22.54100089],
#
#     ]
# ]


for i_building, _ in enumerate(boundary_point_wgs84):
    for i_point, _ in enumerate(boundary_point_wgs84[i_building]):
        mercator = lonLat2Mercator(boundary_point_wgs84[i_building][i_point])
        boundary_point_wgs84[i_building][i_point][0] = mercator[0]+centralized_point[0]
        boundary_point_wgs84[i_building][i_point][1] = mercator[1]+centralized_point[1]

poly = [Polygon(item) for item in boundary_point_wgs84]

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


def f(v_args):
    item = v_args
    p = Point(item[:2])
    # is_in_boundary = [p.within(item) for item in poly]
    is_in_boundary = [True]
    return not (max(is_in_boundary) and item[2] > filter_z)

def merge_mesh_and_filter_the_points_outside_boundary(v_root_file, v_output_folder):
    files = [os.path.join(v_root_file, folder, folder + ".obj") for folder in os.listdir(v_root_file)]

    total_mesh = o3d.geometry.TriangleMesh()

    pools = Pool(16)
    for item in tqdm(files):
        mesh = o3d.io.read_triangle_mesh(item)
        mesh = mesh.translate(origin_point)
        mesh = mesh.translate(centralized_point)
        mesh_point = np.asarray(mesh.vertices)
        remove_flag = pools.map(f, mesh_point)
        remove_flag = np.asarray(remove_flag, np.int16)
        if (remove_flag > 0).sum() - mesh_point.shape[0] == 0:
            continue
        mesh.remove_vertices_by_mask(remove_flag)
        total_mesh += mesh

    o3d.io.write_triangle_mesh(os.path.join(v_output_folder, "mesh_centralized.ply"), total_mesh)
    pcl_5 = total_mesh.sample_points_uniformly(number_of_points=int(1e5))
    pcl_7 = total_mesh.sample_points_uniformly(number_of_points=int(1e7))
    o3d.io.write_point_cloud(os.path.join(v_output_folder, "1e5.ply"), pcl_5)
    o3d.io.write_point_cloud(os.path.join(v_output_folder, "1e7.ply"), pcl_7)


if __name__ == '__main__':
    # test3()
    # test2()
    # test()
    merge_mesh_and_filter_the_points_outside_boundary(root_file,
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\Ours"
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan"
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan_plusplus"
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\oblique"
                                                      r"D:\Projects\Building_data\L7\test"
                                                      # r"D:\Projects\Building_data\Huiwen\CON_Fine"
                                                      )
