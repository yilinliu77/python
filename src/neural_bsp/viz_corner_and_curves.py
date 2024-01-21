import itertools
import os
from sys import argv

import open3d as o3d
import numpy as np
import trimesh
from tqdm import tqdm
import json

root = r"G:/Dataset/GSP/test_data_final/gt/curves"
type = "gt"

# root = r"G:/Dataset/GSP/Results/Baselines/SED_mesh_1228"
# type = "point2cad"

# root = r"G:/Dataset/GSP/Results/Ours/0109_viz_small_mesh"
# type = "ours"


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-6))
    return rotation_matrix


def create_pipe(start_point, end_point, radius=0.005):
    cylinder_vector = np.array([0, 0, 1], dtype=np.float64)
    target_vector = np.array(end_point) - np.array(start_point) + 1e-6

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(target_vector))

    transform_mat = np.eye(4)

    transform_mat[:3, :3] = rotation_matrix_from_vectors(cylinder_vector, target_vector)
    transform_mat[:3, 3] = (np.array(start_point) + np.array(end_point)) / 2

    cylinder.transform(transform_mat)
    return cylinder


def create_sphere(center, radius=0.02, resolution=10):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)

    transform_mat = np.eye(4)
    transform_mat[:3, 3] = center
    sphere.transform(transform_mat)
    return sphere


if __name__ == "__main__":
    # list_file = r"G:/Dataset/GSP/List/vis31.txt"
    list_file = r"G:/Dataset/GSP/List/vis_random.txt"
    root = argv[1]
    type = argv[2]
    prefix = []
    if len(argv) > 3:
        tasks = [argv[3]]
    else:
        tasks = [file for file in sorted(os.listdir(root)) if type == "gt" or os.path.isdir(os.path.join(root, file))]

    viz_ids = [item.strip() for item in open(list_file).readlines()]

    # tasks = ["00990023.ply", "00990042.ply", "00990383.ply", "00990602.ply", "00990610.ply", "00990738.ply"]
    for task in tqdm(tasks):
        if task[:8] not in viz_ids:
            continue
        curves_mesh = o3d.geometry.TriangleMesh()
        vertices_mesh = o3d.geometry.TriangleMesh()

        if type == "ours":
            data = os.listdir(os.path.join(root, task, "viz_curve_and_vertex"))
            curves = [item for item in data if item.endswith("curve.ply")]
            for curve in curves:
                lines = np.asarray(o3d.io.read_point_cloud(
                    os.path.join(root, task, "viz_curve_and_vertex", curve)).points)
                idx = np.asarray(range(lines.shape[0]-1))
                idy = np.asarray(range(1,lines.shape[0]))
                id = np.stack([idx, idy], axis=1)
                lines = lines[id]
                for line in lines:
                    if lines.shape[0] != 1 and np.linalg.norm(line[0] - line[1]) > 1e-2:
                        continue
                    curves_mesh += create_pipe(line[0], line[1])

            corners = [item for item in data if item.endswith("vertex.ply")]
            for corner in corners:
                vertex = np.asarray(o3d.io.read_point_cloud(os.path.join(root, task, "viz_curve_and_vertex", corner)).points)
                vertices_mesh += create_sphere(vertex)

            o3d.io.write_triangle_mesh(os.path.join(root, task, "viz_curve_and_vertex/viz_curves.obj"), curves_mesh)
            o3d.io.write_triangle_mesh(os.path.join(root, task, "viz_curve_and_vertex/viz_corners.obj"), vertices_mesh)

        elif type == "point2cad":
            tasks = [file for file in sorted(os.listdir(os.path.join(root, task)))]
            curves = o3d.geometry.TriangleMesh()
            vertices = o3d.geometry.TriangleMesh()
            with open(os.path.join(root, task, "topo/topo_transformed.json"), "r") as f:
                data = json.loads(f.read())

            for curve in data["curves"]:
                points = np.array(curve["pv_points"])
                indices = np.array(curve["pv_lines"])
                lines = points[indices]
                for line in lines:
                    if lines.shape[0] != 2 and np.linalg.norm(line[0] - line[1]) > 1e-1:
                        continue
                    curves += create_pipe(line[0], line[1])

            for corner in data["corners"]:
                vertices += create_sphere(np.array(corner["corner"]))

            o3d.io.write_triangle_mesh(os.path.join(root, task, "clipped/viz_curves.obj"), curves)
            o3d.io.write_triangle_mesh(os.path.join(root, task, "clipped/viz_corners.obj"), vertices)

        elif type == "gt":
            curves = o3d.geometry.TriangleMesh()
            vertices = o3d.geometry.TriangleMesh()
            pcd = trimesh.load(os.path.join(root, task))
            if not pcd.is_empty:
                points = np.array(pcd.vertices)
                primitive_index = np.asarray(
                    [item[3] for item in pcd.metadata["_ply_raw"]["vertex"]["data"]], dtype=np.int32)

                for i_curve in range(primitive_index.max() + 1):
                    curve = points[primitive_index == i_curve]

                    if curve.shape[0] <= 10:
                        idx = np.asarray(range(curve.shape[0] - 1))
                        idy = np.asarray(range(1, curve.shape[0]))
                        id = np.stack([idx, idy], axis=1)
                    else:
                        idx = np.asarray(range(0, curve.shape[0] - 10, 10))
                        idy = np.asarray(range(10, curve.shape[0], 10))
                        idx = np.insert(idx, idx.shape[0], idy[-1])
                        idy = np.insert(idy, idy.shape[0], curve.shape[0] - 1)
                        id = np.stack([idx, idy], axis=1)

                    lines = curve[id]
                    for line in lines:
                        if lines.shape[0] != 2 and np.linalg.norm(line[0] - line[1]) > 1e-1:
                            continue
                        curves += create_pipe(line[0], line[1])
                    pass

            pcd = trimesh.load(os.path.join(root, "../vertices/", task))
            if not pcd.is_empty:
                corners = np.asarray(pcd.vertices)
                for corner in corners:
                    vertices += create_sphere(corner)

            output_root = os.path.join(root, "../viz_curve_and_vertex")
            os.makedirs(output_root, exist_ok=True)
            o3d.io.write_triangle_mesh(
                os.path.join(root, "../viz_curve_and_vertex/{}_curve.obj".format(task[:8])), curves)
            o3d.io.write_triangle_mesh(
                os.path.join(root, "../viz_curve_and_vertex/{}_corner.obj".format(task[:8])), vertices)

        elif type=="complex":
            curves = o3d.geometry.TriangleMesh()
            vertices = o3d.geometry.TriangleMesh()
            tasks = os.listdir(os.path.join(root, task))
            for item in tasks:
                if item.endswith("curve.ply"):
                    pcd = o3d.io.read_point_cloud(os.path.join(root, task, item))
                    if len(pcd.points) == 2:
                        curves += create_pipe(pcd.points[0], pcd.points[1])
                    else:
                        for i in range(len(pcd.points) - 1):
                            curves += create_pipe(pcd.points[i], pcd.points[i + 1])
                if item.endswith("vertex.ply"):
                    pcd = o3d.io.read_point_cloud(os.path.join(root, task, item))
                    assert len(pcd.points) == 1
                    vertices += create_sphere(pcd.points[0])
            os.makedirs(os.path.join(root, "../viz_corner_curve", task), exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(root, "../viz_corner_curve", task, "curves.obj"), curves)
            o3d.io.write_triangle_mesh(os.path.join(root, "../viz_corner_curve", task, "vertices.obj"), vertices)

        # break
