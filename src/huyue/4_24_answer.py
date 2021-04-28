import io
import os

import math
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import open3d as o3d
from scipy.spatial.transform import Rotation as R

"""
Learn to load and transform point clouds, meshes
"""

"""
Given a point cloud file, try to load and visualize it using open3d. After that, try to manually write it as xyz format 
params:
    - v_point_clout_path (str): The path of the point cloud 
    - v_output_path (str): The path of the desired output point cloud 
Note:
    - Get familiar with the data structure to store the points. e.g. try to convert the loaded point cloud as numpy's array
    - Links might be help:
        - http://www.open3d.org/docs/release/getting_started.html
"""
def draw_point_cloud(v_point_clout_path,v_output_path):
    pcd = o3d.io.read_point_cloud(v_point_clout_path)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    points = np.asarray(pcd.points)
    output = io.open(v_output_path, "w")
    for i in range(0, int(points.size/3)):
        output.write(f"{points[i][0]:.12f} {points[i][1]:.12f} {points[i][2]:.12f}\n")
    output.close()
    pass

"""
Given a mesh file, try to load and visualize it using open3d. Try to manually write it as obj format 
params:
    - v_mesh_path (str): The path of the mesh 
    - v_output_path (str): The path of the desired output mesh 
    
Note:
    - Links might be help:
        - http://www.open3d.org/docs/release/getting_started.html
"""
def draw_meshes(v_mesh_path,v_output_path):

    mesh = o3d.io.read_triangle_mesh(v_mesh_path)
    o3d.visualization.draw_geometries([mesh])


    vex_arr = np.asarray(mesh.vertices, dtype=np.float64)
    vex_color_arr = np.asarray(mesh.vertex_colors)
    triangles_arr = np.asarray(mesh.triangles)

    output = io.open(v_output_path, "w")
    for i in range(0, int(vex_arr.size / 3)):
        output.write(f"v {vex_arr[i][0]:.17e} {vex_arr[i][1]:.17e} {vex_arr[i][2]:.17e} {vex_color_arr[i][0]:.6f} "
                     f"{vex_color_arr[i][1]:.6f} {vex_color_arr[i][2]:.6f}\n")
    for i in range(0,int(triangles_arr.size/3)):
        output.write(f"f {triangles_arr[i][0]+1} {triangles_arr[i][1]+1} {triangles_arr[i][2]+1}\n")
    output.close()
    pass


"""
Given a mesh file, try to load it and rotate 90 degree along the X axis 
params:
    - v_mesh_path (str): The path of the mesh 
    - v_output_path (str): The path of the desired output mesh 

Note:
    - Use two ways to accomplish that (Use Open3d or manually do rotation). 
    - Links might be help:
        - http://www.open3d.org/docs/release/getting_started.html
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
"""


def rotate_meshes(v_mesh_path, v_output_path):
    '''
    #manually do rotation
    mesh = o3d.io.read_triangle_mesh(v_mesh_path)
    vers = np.asarray(mesh.vertices)
    for v in vers:
        y=v[1]
        z=v[2]
        v[1]=-z
        v[2]=y
    o3d.io.write_triangle_mesh(v_output_path, mesh)
    '''
    mesh = o3d.io.read_triangle_mesh(v_mesh_path)
    mesh.rotate(R.from_euler('x', 90, degrees=True).as_matrix(), center=(0,0,0))
    o3d.io.write_triangle_mesh(v_output_path, mesh)

    pass


"""
Try to beat the BIG BOSS:

Given a point cloud file, try to learn RANSAC algorithm and use it to detect several planes
params:
    - v_mesh_path (str): The path of the mesh 
    - v_output_path (str): The path of the desired output planes.

Note:
    - Use Ransac provided by Open3D to verify your result
    - Links might be help:
        - http://www.open3d.org/docs/release/getting_started.html
        - Try to search this algorithm yourself. e.g. keywords: RANSAC, Plane fitting
"""

'''
Noise handling must be done in RANSAC, otherwise data sets containing fullful noise are
more likely to be returned after several iterations.
'''
def plane_segment_by_three_points(pcd,a,b,c):
    points = np.asarray(pcd.points)
    fin = []
    d1 = points[a] - points[b]
    d2 = points[a] - points[c]
    ABC = np.cross(d1, d2)
    ABC /= np.linalg.norm(ABC)
    D = -(ABC[0] * points[a][0] + ABC[1] * points[a][1] + ABC[2] * points[a][2])
    [a1, b1, c1] = ABC
    pn = np.array([a1, b1, c1]) / np.linalg.norm(np.array([a1, b1, c1]))
    for i in range(0, int(points.size / 3)):
        d = ABC[0] * points[i][0] + ABC[1] * points[i][1] + ABC[2] * points[i][2] + D
        dot = pn[0] * pcd.normals[i][0] + pn[1] * pcd.normals[i][1] + pn[2] * pcd.normals[i][2]
        if d < 2 and d > -2 and (dot > 0.70 or dot < -0.70):
            fin.append(i)
    return [ABC[0], ABC[1], ABC[2], D], fin
    pass

def ransac_plane_segment(pcd):
    points = np.asarray(pcd.points)
    fin = []
    plane_model = []
    for i in range(0, 400):
        print(f"RANSAC:{i}/400")
        a = 0
        b = 0
        c = 0
        #Make sure the three variables are different
        while a==b or b==c or c==a:
            a = random.randint(0, points.size/3-1)
            b = random.randint(0, points.size/3-1)
            c = random.randint(0, points.size/3-1)
        temp_N, temp = plane_segment_by_three_points(pcd, a, b, c)
        if temp.__sizeof__() > fin.__sizeof__():
            fin = temp
            plane_model = temp_N
    return plane_model, fin
    pass

def k_means(points,k):
    fin_center_points = np.empty([k, 3], dtype=np.float64)
    fin_points_index = np.empty([int(points.size/3)], dtype=np.int16)
    for i in range(0, k):
        fin_center_points[i][0] = points[i][0]
        fin_center_points[i][1] = points[i][1]
        fin_center_points[i][2] = points[i][2]
    for count in range(0, 100):
        print(f"k_means:{count}/100")
        for i in range(0, int(points.size/3)):
            near_index = 0
            d2 = math.pow(points[i][0] - fin_center_points[0][0], 2) + \
                 math.pow(points[i][1] - fin_center_points[0][1], 2) + \
                 math.pow(points[i][2] - fin_center_points[0][2], 2)
            for i_to_center in range(1, k):
                d2new = math.pow(points[i][0] - fin_center_points[i_to_center][0], 2) + \
                    math.pow(points[i][1] - fin_center_points[i_to_center][1], 2) + \
                    math.pow(points[i][2] - fin_center_points[i_to_center][2], 2)
                if d2new < d2:
                    d2 = d2new
                    near_index = i_to_center
            fin_points_index[i] = near_index
        for i_to_center in range(0, k):
            sum_x = 0
            sum_y = 0
            sum_z = 0
            sum_count = 0
            for i_to_points in range(0, int(points.size/3)):
                if fin_points_index[i_to_points] == i_to_center:
                    sum_x += points[i_to_points][0]
                    sum_y += points[i_to_points][1]
                    sum_z += points[i_to_points][2]
                    sum_count += 1
            fin_center_points[i_to_center][0] = sum_x / sum_count
            fin_center_points[i_to_center][1] = sum_y / sum_count
            fin_center_points[i_to_center][2] = sum_z / sum_count
    return fin_points_index
    pass

def distence_two_cloud(v_cloud_points1, v_cloud_points2):
    points1 = np.asarray(v_cloud_points1.points)
    points2 = np.asarray(v_cloud_points2.points)
    fin_d = np.linalg.norm(points1[0] - v_cloud_points2.get_center().T)
    min_i = 0
    for i in range(1, int(points1.size / 3)):
        d = np.linalg.norm(points1[i] - v_cloud_points2.get_center().T)
        if d < fin_d:
            fin_d = d
            min_i = i
    for j in range(0, int(points2.size/3)):
        d = np.linalg.norm(points1[min_i] - points2[j])
        if d < fin_d:
            fin_d = d
    return fin_d

def find_root(v_ufs,v_i):
    fin = v_i
    while fin > 0:
        fin = v_ufs[fin]
    return fin
    pass

def k_means_aggregate(v_cloud_points, v_k):
    points = np.asarray(v_cloud_points.points)
    points_index = k_means(points, v_k)
    fin_array = []
    for i in range(0, v_k):
        points_index_array = []
        for i_to_points in range(0, points_index.size):
            if i == points_index[i_to_points]:
                points_index_array.append(i_to_points)
        pcd1 = v_cloud_points.select_by_index(points_index_array)
        fin_array.append(pcd1)

    print(fin_array)
    UFS = np.empty([v_k], dtype=np.int16)
    for i in range(0, v_k):
        UFS[i] = -1
    for i in range(0, v_k):
        for test_i in range(i + 1, v_k):
            d = distence_two_cloud(fin_array[i], fin_array[test_i])
            print(f"distence({i * v_k + test_i}/{v_k*v_k}):{d}")
            if d < 10:
                UFS[i] = test_i
    fin = []
    print("handle UFS")
    for i in range(0, v_k):
        root = find_root(UFS, i)
        if root != i:
            fin_array[root] = fin_array[root] + fin_array[i]
    print("aggregate points cloud")
    for i in range(0, v_k):
        if UFS[i] < 0:
            fin.append(fin_array[i])
    return fin
    pass

def detect_planes(v_point_clout_path, v_output_path):
    pcd = o3d.io.read_point_cloud(v_point_clout_path)
    pcd.paint_uniform_color((0.5, 0.5, 0.5))
    pcd.estimate_normals()
    pcd.normalize_normals()
    points_cloud_array = []
    pcd1 = pcd
    while(np.asarray(pcd1.points).size / 3 > 15000):
        print(f"last points:{np.asarray(pcd1.points).size / 3}")
        points = np.asarray(pcd1.points)
        plane_model, inliers = ransac_plane_segment(pcd1)
        inlier_cloud = pcd1.select_by_index(inliers)
        inlier_cloud.paint_uniform_color((random.random(), random.random(), random.random(),))
        points_cloud_array.append(inlier_cloud)
        pcd1 = pcd1.select_by_index(inliers, invert=True)

    print(points_cloud_array)
    for i in points_cloud_array:
        pcd1 = pcd1 + i

    o3d.visualization.draw_geometries([pcd1],
                                      zoom=0.8,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])


    o3d.io.write_point_cloud(v_output_path,pcd1)
    pass

if __name__ == '__main__':
    draw_point_cloud("szu_north_0422_points.ply","my_point_cloud.xyz")
    draw_meshes("szu_north_0422.obj","my_mesh.obj")
    rotate_meshes("szu_north_0422.obj","my_mesh_rotated.obj")
    detect_planes("szu_north_0422_points.ply","my_planes.ply")
