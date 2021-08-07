import os
from multiprocessing import Pool

import open3d as o3d
from numba import njit
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numba
import numpy as np

boundary_point_wgs84 = [
        [12682473.000000,2576538.250000],
        [12682539.000000,2576534.250000],
        [12682540.000000,2576504.000000],
        [12682597.000000,2576504.250000],
        [12682599.000000,2576537.000000],
        [12682689.000000,2576470.750000],
        [12682745.000000,2576558.500000],
        [12682670.000000,2576612.500000],
        [12682669.000000,2576656.500000],
        [12682572.000000,2576652.500000],
        [12682470.000000,2576564.500000],
    ]
boundary_point_cgcs2000 = [
    [492650.281250, 2493639.000000],
    [492714.218750, 2493643.750000],
    [492712.531250, 2493611.250000],
    [492769.437500, 2493613.000000],
    [492774.375000, 2493643.000000],
    [492861.781250, 2493582.250000],
    [492907.406250, 2493667.750000],
    [492833.656250, 2493722.500000],
    [492838.218750, 2493760.000000],
    [492754.843750, 2493756.500000],
    [492752.968750, 2493731.250000],
    [492713.843750, 2493727.000000],
    [492713.968750, 2493698.500000],
    [492652.906250, 2493682.500000],
    ]
poly=Polygon(boundary_point_cgcs2000)
# root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\qingxie-obj\Data"
# origin_point = (12682696, 2576546, 0)

# root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\obj-vertical_0.8_explor\obj-v08\Data"
# origin_point = (12682641, 2576576, 0)

# root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\qinxie-obj-cgcs2000\Data"
# origin_point = (492854.0403, 2493646.463, 49.5)

# root_file=r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data"
# origin_point = (492796,2493689,0)
root_file=r"D:\Projects\SEAR\Real\Huiyuan\0512\cgcs_2000_0.8\0.8-obj\Data"
origin_point = (492795.0207,2493679.919,67.45)

def test3():
    total_points = o3d.io.read_point_cloud(r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\huiwen_0_05_lidar.ply")

    pools = Pool(16)
    mesh_point = np.asarray(total_points.points)
    mask = pools.map(f, mesh_point)
    mask = np.asarray(mask,np.int16)
    total_points.points=o3d.utility.Vector3dVector(mesh_point[np.logical_not(mask)])
    o3d.io.write_point_cloud(r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\gt.ply", total_points)
    pass

def test():
    files = [os.path.join(r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data",
                          folder, folder + ".obj") for folder in os.listdir(
        r"D:\Projects\SEAR\Real\Huiyuan\0427_dronescan\model\obj-jingzhun0427\obj-jingzhun0427\Data")]
    total_mesh = o3d.geometry.TriangleMesh()
    for item in tqdm(files):
        mesh = o3d.io.read_triangle_mesh(item)
        mesh = mesh.translate((492796,2493689,0))
        total_mesh += mesh

    pcl_1e8 = total_mesh.sample_points_uniformly(number_of_points=int(1e8))
    pcl_1e3 = total_mesh.sample_points_uniformly(number_of_points=int(1e3))
    o3d.io.write_point_cloud("drone_scan_1e8.ply",pcl_1e8)
    o3d.io.write_point_cloud("drone_scan_1e3.ply",pcl_1e3)
    pass

def test2():
    txt_array = [item.strip()[:-1].replace(" ",",") for item in open(r"C:\repo\python\src\test\drone_scan_1e8.xyz").readlines()]
    with open(r"C:\Users\yilin\Desktop\test\111.csv","w") as f:
        for line in txt_array:
            f.write(line+"\n")

def f(v_args):
    item = v_args
    p = Point(item[:2])
    return not p.within(poly)

def merge_mesh_and_filter_the_points_outside_boundary(v_root_file,v_output_folder):
    files = [os.path.join(v_root_file, folder, folder + ".obj") for folder in os.listdir(v_root_file)]

    total_mesh = o3d.geometry.TriangleMesh()

    pools = Pool(16)
    for item in tqdm(files):
        mesh = o3d.io.read_triangle_mesh(item)
        mesh = mesh.translate(origin_point)
        mesh_point = np.asarray(mesh.vertices)
        mask = pools.map(f, mesh_point)
        mask = np.asarray(mask, np.int16)
        if (mask > 0).sum() - mesh_point.shape[0] == 0:
            continue
        mesh.remove_vertices_by_mask(mask)
        total_mesh += mesh
    pcl_5 = total_mesh.sample_points_uniformly(number_of_points=int(1e5))
    pcl_7 = total_mesh.sample_points_uniformly(number_of_points=int(1e7))
    o3d.io.write_point_cloud(os.path.join(v_output_folder,"1e5.ply"), pcl_5)
    o3d.io.write_point_cloud(os.path.join(v_output_folder,"1e7.ply"), pcl_7)

if __name__ == '__main__':
    # test3()
    # test2()
    # test()
    merge_mesh_and_filter_the_points_outside_boundary(root_file,
                                                      r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\Ours"
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\dronescan"
                                                      # r"D:\Projects\SEAR\Real\Huiyuan\COMPARISON\oblique"
                                                      )
