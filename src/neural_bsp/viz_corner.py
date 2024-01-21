import itertools
import os
from sys import argv

import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm
import json

from src.neural_bsp.viz_corner_and_curves import create_sphere

if __name__ == "__main__":
    input_file = argv[1]
    output_file = argv[2]

    pcd = trimesh.load(input_file)
    if False:
        r = np.asarray([item[7] for item in pcd.metadata["_ply_raw"]["vertex"]["data"]], dtype=np.int32)
        g = np.asarray([item[8] for item in pcd.metadata["_ply_raw"]["vertex"]["data"]], dtype=np.int32)
        b = np.asarray([item[9] for item in pcd.metadata["_ply_raw"]["vertex"]["data"]], dtype=np.int32)
        rgb = np.stack([r, g, b], axis=1)
        rgb[(rgb==np.array([(204,255,153),])).all(axis=1)] = np.array([129,72,105])
    points = np.asarray(pcd.vertices)

    mesh = o3d.geometry.TriangleMesh()
    colors = []
    for i, point in enumerate(points):
        local_mesh = create_sphere(point, radius=0.005, resolution=3)
        # colors.append(np.tile(rgb[i][None,:], (np.asarray(local_mesh.triangles).shape[0], 1)))
        mesh+=local_mesh

    trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        # face_colors=np.concatenate(colors,axis=0)
    ).export(output_file)
