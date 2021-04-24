import numpy as np
from matplotlib import pyplot as plt
import cv2
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
    pass


"""
Try to beat the BIG BOSS:

Given a point cloud file, try to learn RANSAC algorithm and use it to detect several planes
params:
    - v_mesh_path (str): The path of the mesh 
    - v_output_path (str): The path of the desired output planes. Each plane has two triangles 

Note:
    - Use Ransac provided by Open3D to verify your result
    - Links might be help:
        - http://www.open3d.org/docs/release/getting_started.html
        - Try to search this algorithm yourself. e.g. keywords: RANSAC, Plane fitting
"""

def detect_planes(v_mesh_path, v_output_path):
    pass

if __name__ == '__main__':
    draw_point_cloud("szu_north_0422_points.ply","my_point_cloud.xyz")
    draw_meshes("szu_north_0422.obj","my_mesh.obj")
    rotate_meshes("szu_north_0422.obj","my_mesh_rotated.obj")
    detect_planes("szu_north_0422.obj","my_planes.obj")
