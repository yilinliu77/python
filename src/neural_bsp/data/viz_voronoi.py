import math
import shutil

import open3d as o3d
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pyrender
from PIL import Image

opj = os.path.join

root = r"g:\Dataset\GSP\GSP_debug"

if __name__ == '__main__':
    files = sorted([file for file in os.listdir(opj(root, "debug"))])

    queries = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij')
    queries = np.stack(queries, axis=-1) / 255 * 2 - 1

    for file in tqdm(files):
        prefix = file[:10]
        content = open(opj(root, "debug", prefix + ".bin"), "rb").read()
        content = np.frombuffer(content, dtype=np.uint8)
        shifts = np.arange(8)
        voronoi_flags = ((content[:, None] & (1 << shifts)[None, :]) > 0).reshape(-1)
        voronoi_flags = voronoi_flags.reshape(256, 256, 256)

        voronoi_edges = queries[voronoi_flags]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voronoi_edges.reshape(-1, 3)))
        o3d.io.write_point_cloud(opj(root, prefix + ".ply"), pcd)

        continue
