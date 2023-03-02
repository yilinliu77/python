import sys
from functools import partial
from multiprocessing import Pool

from shared.common_utils import to_homogeneous
from src.neural_recon.colmap_io import check_visibility

sys.path.append("thirdparty/sdf_computer/build/")
import numpy as np

import json
import open3d as o3d
import pysdf


def normalize_extrinsic(v_bounds_center, v_bounds_size, v_intrinsic, v_extrinsic):
    model_matrix = np.zeros((4, 4), dtype=np.float32)
    model_matrix[0, 0] = v_bounds_size
    model_matrix[1, 1] = v_bounds_size
    model_matrix[2, 2] = v_bounds_size
    model_matrix[0, 3] = v_bounds_center[0] - v_bounds_size / 2
    model_matrix[1, 3] = v_bounds_center[1] - v_bounds_size / 2
    model_matrix[2, 3] = v_bounds_center[2] - v_bounds_size / 2
    model_matrix[3, 3] = 1

    projections = []
    for item in v_extrinsic:
        camera_coor = np.matmul(item, model_matrix)
        camera_coor2 = np.matmul(to_homogeneous(np.asarray(
                ((1, 0, 0),
                 (0, -1, 0),
                 (0, 0, -1)))), camera_coor)
        projection = np.matmul(to_homogeneous(v_intrinsic), camera_coor2)
        projections.append(projection)

    return projections


if __name__ == '__main__':
    pose = json.loads(open(r"C:\Users\whats\Downloads\wo_data\wo_data\end\camera_train.json", "r").read())
    K = np.asarray(pose["K"])

    K[0, 0] /= 960
    K[1, 1] /= 720
    K[0, 2] /= 960
    K[1, 2] /= 720

    bounds_center = np.array([0., 0., 0.], dtype=np.float32)
    bounds_size = 2.
    extrinsics = np.asarray([np.asarray(pose[item]) for item in pose if item != "K"], dtype=np.float32)
    poses = [(item[:3,3] - bounds_center) / bounds_size + 0.5 for item in extrinsics]
    extrinsics = np.asarray([np.linalg.inv(item) for item in extrinsics], dtype=np.float32)
    projections = np.asarray(normalize_extrinsic(bounds_center, bounds_size, K, extrinsics), dtype=np.float32)

    mesh = o3d.io.read_triangle_mesh(r"C:\Users\whats\Downloads\wo_data\wo_data\end\end_rotate.ply")
    gt_mesh_vertices = np.asarray(mesh.vertices)
    gt_mesh_vertices = (gt_mesh_vertices - bounds_center) / bounds_size + 0.5

    gt_mesh_faces = np.asarray(mesh.triangles)
    sdf_computer = pysdf.PYSDF_computer()

    sdf_computer.setup_bounds(
        np.append(bounds_center, bounds_size)
    )
    sdf_computer.setup_mesh(gt_mesh_vertices[gt_mesh_faces],
                            False)  # Do not automatically compute the bounds
    sdf = sdf_computer.compute_sdf(int(1e5), int(1e5), 0, False)
    sample_points = sdf[:, :3]
    sample_distances = sdf[:, 3:]

    print("Start to check visibility")
    # Start to check visibility
    pool = Pool(16)
    check_visibility(np.zeros((4, 4), dtype=np.float32),
                     np.zeros((2, 3), np.float32))  # Dummy, just for compiling the function
    visibility_inside_frustum = pool.map(
        partial(check_visibility, v_points=sample_points),
        projections, chunksize=10)
    visibility_inside_frustum = np.stack(visibility_inside_frustum, axis=0)

    visibility_intersection_free = sdf_computer.check_visibility(
        poses,
        sample_points
    )
    visibility_intersection_free = visibility_intersection_free.reshape([len(poses), -1]).astype(bool)
    final_visibility = np.logical_and(visibility_inside_frustum, visibility_intersection_free)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(
        (sample_points[np.max(final_visibility, axis=0)] - 0.5) * bounds_size + bounds_center)
    o3d.io.write_point_cloud("output/aaa.ply", pc)

    id_test_img = 1
    tr = o3d.geometry.TriangleMesh()
    pc = o3d.geometry.PointCloud()
    # Store normalized model
    tr.vertices = o3d.utility.Vector3dVector(gt_mesh_vertices)
    tr.triangles = o3d.utility.Vector3iVector(gt_mesh_faces)
    o3d.io.write_triangle_mesh("output/model.ply", tr)
    # Store normalized cameras
    pc.points = o3d.utility.Vector3dVector(poses)
    o3d.io.write_point_cloud("output/cameras.ply", pc)
    # Store normalized sample points
    pc.points = o3d.utility.Vector3dVector(sample_points)
    o3d.io.write_point_cloud("output/1.ply", pc)
    # Store points inside frustum
    pc.points = o3d.utility.Vector3dVector(sample_points[visibility_inside_frustum[id_test_img]])
    o3d.io.write_point_cloud("output/2.ply", pc)
    # Store points that are collision-free
    pc.points = o3d.utility.Vector3dVector(
        sample_points[visibility_intersection_free[id_test_img] == 1])
    o3d.io.write_point_cloud("output/3.ply", pc)
    # Store points that are visible to both
    pc.points = o3d.utility.Vector3dVector(sample_points[final_visibility[id_test_img]])
    o3d.io.write_point_cloud("output/4.ply", pc)
    pc.clear()
