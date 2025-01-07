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

root = Path("D:/brepnet/real_data/points_align4mm_partmesh_whole/pc")

files = os.listdir(root)
for file in tqdm(files):
    if len(file)!=12:
        continue
    prefix = file[:8]

    mesh = o3d.io.read_triangle_mesh(str(root/f"{prefix}.ply"))
    gt_pcd = mesh.sample_points_poisson_disk(100000, use_triangle_normal=True)

    pcd = o3d.io.read_point_cloud(str(root/f"{prefix}_pc.ply"))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, gt_pcd, 0.075, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1000))

    # draw_registration_result(pcd, gt_pcd, reg_p2p.transformation)
    pcd = pcd.transform(reg_p2p.transformation)
    pcd_reg = o3d.geometry.PointCloud()
    pcd_reg.points = pcd.points
    o3d.io.write_point_cloud(str(root/f"{prefix}_pc_reg.ply"), pcd_reg)