import os
from pathlib import Path
import h5py
import numpy as np
import open3d as o3d
import trimesh
from OCC.Extend.DataExchange import read_step_file
from open3d.examples.open3d_example import draw_registration_result
from tqdm import tqdm

from shared.occ_utils import normalize_shape, get_triangulations

abc_root = Path("D:/Datasets/data_step")
root = Path("D:/brepnet/real_data/points_align4mm_partmesh_whole/test")

files = os.listdir(root)
for file in tqdm(files):
    with h5py.File(root/file) as f:
        name = f["item_id"][0].decode("utf-8")
        prefix = name[0:8]

        step_name = name.split("_")
        step_name.insert(-1, "step")
        step_name = "_".join(step_name)
        shape = read_step_file(str(abc_root/prefix/(step_name+".step")), True)

        shape = normalize_shape(shape, 0.9)
        # write_step_file(shape, str(output_root / v_folder / "normalized_shape.step"))

        v, fa = get_triangulations(shape, 0.001)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(fa)
        o3d.io.write_triangle_mesh(str(root/f"../pc/{prefix}.ply"), mesh)
        tgt_pcd = mesh.sample_points_poisson_disk(10000)

        num_patches = f["points"].shape[0]
        points = np.stack(f["points"][:]).reshape(num_patches, -1, 3).reshape(-1, 3)

        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2
        points -= center
        scale = np.max(bbox_max - bbox_min)
        points /= scale
        points *= 0.9 * 2

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(root/f"../pc/{prefix}_pc.ply"), pcd)
