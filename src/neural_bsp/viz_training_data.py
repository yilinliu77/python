import math
import shutil

import h5py
import open3d as o3d
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pyrender
from PIL import Image

from src.neural_bsp.abc_hdf5_dataset import normalize_points
from shared.common_utils import *

def viz_training_data():
    root_file = r"C:\repo\python\output\neural_bsp\training.h5"
    # root_file = r"C:\repo\python\output\neural_bsp\training.h5"

    with h5py.File(root_file, 'r') as f:
        num_items = f["points"].shape[0]
        num_patches = f["points"].shape[1]
        while True:
            id1 = np.random.randint(0, num_items - 1)
            id2 = np.random.randint(0, num_patches - 1)

            # id1 = 876
            # id2 = 724

            if f["names"][id1] >= 10000:
                continue
            print("{}_{}:{}_{}".format(f["names"][id1], f["ids"][id1], id1,id2))
            points = f["points"][id1, id2] / 32767
            point_flags = f["point_flags"][id1, id2]

            shifts = np.arange(8)
            point_flags = ((point_flags[:,None] & (1 << shifts)[None, :]) > 0).reshape(-1)

            voronoi_flags = f["voronoi_flags"][id1, id2]
            voronoi_flags = ((voronoi_flags[:,None] & (1 << shifts)[None, :]) > 0).reshape(-1)
            voronoi_flags = voronoi_flags.reshape(32,32,32)

            query = np.stack(np.meshgrid(
                np.arange(32), np.arange(32), np.arange(32), indexing='ij'), axis=-1) / 31 * 2 - 1

            query -= 1/31

            pc1 = o3d.geometry.PointCloud()
            pc1.points=o3d.utility.Vector3dVector(points[point_flags])
            colors = np.zeros_like(points[point_flags])
            colors[:,0] = 1
            pc1.colors=o3d.utility.Vector3dVector(colors)
            pc2 = o3d.geometry.PointCloud()
            pc2.points = o3d.utility.Vector3dVector(query[voronoi_flags])
            colors = np.zeros_like(query[voronoi_flags])
            colors[:, 1] = 1
            pc2.colors=o3d.utility.Vector3dVector(colors)

            coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([pc1, pc2, coor])

            # export_point_cloud("voronoi.ply", query[voronoi_flags])
            # export_point_cloud("points.ply", points[point_flags])
            pass

if __name__ == '__main__':
    viz_training_data()
