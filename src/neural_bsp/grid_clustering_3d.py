import os.path
import shutil
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import ray

import yaml

import open3d as o3d
import torch

import bvh_distance_queries
from shared.common_utils import check_dir


def normalize(v, v_axis=-1):
    norm = np.linalg.norm(v, axis=v_axis)
    new_v = np.copy(v)
    new_v[norm > 0] = new_v[norm > 0] / norm[norm > 0][:, None]
    return new_v


def calculate_distance(source_coords_ref, obj_file, use_cpu=True):
    with open(obj_file) as f:
        vertices = []
        faces = []
        for line in f.readlines():
            if line[:2] == "v ":
                data = line[2:].strip().split(" ")
                vertices.append(np.asarray((data[0], data[1], data[2])))
            elif line[0] == "f":
                data = line[2:].strip().split(" ")
                data = [item.split("//")[0] for item in data]
                faces.append(np.asarray((data[0], data[1], data[2])))
    vertices = np.stack(vertices).astype(np.float32)
    faces = np.stack(faces).astype(np.int64) - 1

    # Calculate the bounds
    vmin = np.min(vertices, axis=0)
    vmax = np.max(vertices, axis=0)
    vcenter = (vmin + vmax) / 2
    diag = np.sqrt(np.sum((vmax - vmin) ** 2))

    # Normalize the vertices
    vertices = ((vertices - vcenter) / diag) * 2

    query_points = source_coords_ref.astype(np.float32) / (resolution - 1) * 2 - 1

    if use_cpu:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        results = scene.compute_closest_points(
            o3d.core.Tensor.from_numpy(query_points)
        )

        closest_points = results["points"].numpy()
        distances = np.linalg.norm(closest_points - query_points, axis=-1)
        closest_faces = results["primitive_ids"].numpy()

        # test_results = scene.compute_closest_points(
        #     o3d.core.Tensor.from_numpy(np.array([
        #         [1, -0.129412, 0.639216],
        #         [0.615686, -0.00392157, 0.670588],
        #     ], dtype=np.float32))
        # )
        # print(test_results["primitive_ids"])

    else:
        with torch.no_grad():
            device = torch.device('cuda')

            m = bvh_distance_queries.BVH()

            distances, closest_points, closest_faces, closest_bcs = m(
                torch.from_numpy(vertices).to(device)[faces].unsqueeze(0),
                torch.from_numpy(query_points).to(device).unsqueeze(0))
            # torch.cuda.synchronize()
            distances = torch.sqrt(distances).detach().cpu().numpy()[0]
            # distances = distances.detach().cpu().numpy()[0]
            closest_points = closest_points.detach().cpu().numpy()[0]
            closest_faces = closest_faces[0].cpu().numpy()

    if False:
        for test_distance in [0.01, 0.1, 0.5]:
            selected_points = query_points[distances < test_distance]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(selected_points)
            o3d.io.write_point_cloud("output/{}.ply".format(test_distance), pc)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    return np.concatenate((query_points, distances[:, None]), axis=1), closest_faces, mesh




# 1. Merge the same curves
def filter_primitives(primitive_dict, v_faces):
    curves = []
    for primitive in primitive_dict["curves"]:
        # It will cause problems, do not open it
        # if not primitive["sharp"]:
        #     continue
        if primitive["type"] == "Circle":
            is_new_primitive = True
            for id_existing_curve in range(len(curves)):
                if curves[id_existing_curve]["type"] == primitive["type"] and \
                        curves[id_existing_curve]["location"] == primitive["location"] and \
                        curves[id_existing_curve]["radius"] == primitive["radius"]:
                    curves[id_existing_curve]["vert_indices"].update(primitive["vert_indices"])
                    is_new_primitive = False
                    break
            if is_new_primitive:
                curves.append({
                    "type": primitive["type"],
                    "location": primitive["location"],
                    "radius": primitive["radius"],
                    "vert_indices": set(primitive["vert_indices"])
                })
        elif primitive["type"] == "BSpline":
            is_new_primitive = True
            for id_existing_curve in range(len(curves)):
                if curves[id_existing_curve]["type"] == primitive["type"] and \
                        curves[id_existing_curve]["knots"] == primitive["knots"] and \
                        curves[id_existing_curve]["degree"] == primitive["degree"] and \
                        curves[id_existing_curve]["poles"] == primitive["poles"]:
                    curves[id_existing_curve]["vert_indices"].update(primitive["vert_indices"])
                    is_new_primitive = False
                    break
            if is_new_primitive:
                curves.append({
                    "type": primitive["type"],
                    "knots": primitive["knots"],
                    "degree": primitive["degree"],
                    "poles": primitive["poles"],
                    "vert_indices": set(primitive["vert_indices"])
                })
        elif primitive["type"] == "Line":
            is_new_primitive = True
            for id_existing_curve in range(len(curves)):
                if curves[id_existing_curve]["type"] == primitive["type"] and \
                        curves[id_existing_curve]["direction"] == primitive["direction"] and \
                        curves[id_existing_curve]["location"] == primitive["location"]:
                    curves[id_existing_curve]["vert_indices"].update(primitive["vert_indices"])
                    is_new_primitive = False
                    break
            if is_new_primitive:
                curves.append({
                    "type": primitive["type"],
                    "direction": primitive["direction"],
                    "location": primitive["location"],
                    "vert_indices": set(primitive["vert_indices"])
                })
        elif primitive["type"] == "Ellipse":
            is_new_primitive = True
            for id_existing_curve in range(len(curves)):
                if curves[id_existing_curve]["type"] == primitive["type"] and \
                        curves[id_existing_curve]["focus1"] == primitive["focus1"] and \
                        curves[id_existing_curve]["focus2"] == primitive["focus2"] and \
                        curves[id_existing_curve]["maj_radius"] == primitive["maj_radius"] and \
                        curves[id_existing_curve]["min_radius"] == primitive["min_radius"]:
                    curves[id_existing_curve]["vert_indices"].update(primitive["vert_indices"])
                    is_new_primitive = False
                    break
            if is_new_primitive:
                curves.append({
                    "type": primitive["type"],
                    "focus1": primitive["focus1"],
                    "focus2": primitive["focus2"],
                    "maj_radius": primitive["maj_radius"],
                    "min_radius": primitive["min_radius"],
                    "vert_indices": set(primitive["vert_indices"])
                })
        else:
            raise

    # Detect sharp edges
    corner_points = []

    id_primitive_per_vertex = {}
    for id_curve, curve in enumerate(curves):
        for vert_id in curve["vert_indices"]:
            if vert_id in corner_points:
                continue
            if vert_id in id_primitive_per_vertex:
                id_primitive_per_vertex.pop(vert_id)
                corner_points.append(vert_id)
            else:
                id_primitive_per_vertex[vert_id] = set()

    for id_surface, surface in enumerate(primitive_dict["surfaces"]):
        for id_face in surface["face_indices"]:
            vert_ids = v_faces[id_face]
            for vert_id in vert_ids:
                if vert_id in id_primitive_per_vertex:
                    id_primitive_per_vertex[vert_id].add(id_surface)

    filtered_curves = []
    for id_curve, curve in enumerate(curves):
        neighbour_primitives = set()
        for vert_id in curve["vert_indices"]:
            if vert_id not in id_primitive_per_vertex:
                continue
            for item in id_primitive_per_vertex[vert_id]:
                neighbour_primitives.add(item)
        if len(neighbour_primitives) > 1:
            filtered_curves.append(curve)

    return filtered_curves, primitive_dict["surfaces"]


# 2. Calculate the index of primitives for each vertex and face
def calculate_indices(curves, surfaces, mesh, v_is_log=""):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]
    num_curves = len(curves)
    num_surfaces = len(surfaces)
    num_primitives = num_curves + num_surfaces
    vert_id_to_primitives = np.zeros(num_vertices, dtype=np.int64) - 1
    surface_id_to_primitives = np.zeros(num_faces, dtype=np.int64)

    id_corner_points = OrderedDict()
    for id_curve, curve in enumerate(curves):
        for id_vert in curve["vert_indices"]:
            if vert_id_to_primitives[id_vert] != -1:
                if id_vert not in id_corner_points:
                    vert_id_to_primitives[id_vert] = num_primitives + len(id_corner_points)
                    id_corner_points[id_vert] = []
                else:
                    vert_id_to_primitives[id_vert] = num_primitives + list(id_corner_points.keys()).index(id_vert)
                id_corner_points[id_vert].append(id_curve)
                id_corner_points[id_vert].append(vert_id_to_primitives[id_vert])
            else:
                vert_id_to_primitives[id_vert] = id_curve
        pass
    num_corner_points = len(id_corner_points)
    num_primitives = num_primitives + num_corner_points

    if v_is_log!="":
        export_point_cloud("{}/corner_points.ply".format(v_is_log), vertices[list(id_corner_points.keys())])

    for id_surface, surface in enumerate(surfaces):
        for id_face in surface["face_indices"]:
            primitive_ids = vert_id_to_primitives[faces[id_face]]
            # This is a triangle related to a corner point
            if (primitive_ids >= num_curves + num_surfaces).any():
                # Index of the corner point, guarantee to be larger than the curves and the surfaces
                surface_id_to_primitives[id_face] = np.max(primitive_ids)
            # This is a triangle related to a feature curve
            elif (primitive_ids > -1).any():
                surface_id_to_primitives[id_face] = np.max(primitive_ids)
            # This is a triangle related to a surface
            else:
                surface_id_to_primitives[id_face] = id_surface + num_curves
            pass

    return surface_id_to_primitives, id_corner_points


@ray.remote
def process_item(v_root, v_output_root, v_files,
                 source_coords_ref, target_coords_ref, valid_flag_ref, v_resolution, v_is_log=False):
    for obj_file in v_files:
        prefix = "_".join(obj_file.split("_")[:2])
        # 1. Setup file path
        prefix_path = os.path.join(v_output_root, prefix)
        os.makedirs(os.path.join(v_output_root, prefix), exist_ok=True)
        feature_file = obj_file.replace("trimesh", "features")
        feature_file = feature_file.replace("obj", "yml")
        feature_file = os.path.join(v_root, "feat", feature_file)
        obj_file = os.path.join(v_root, "obj", obj_file)

        # 2. Calculate closest distance
        udf, closest_faces, mesh = calculate_distance(source_coords_ref, obj_file, use_cpu=False)
        o3d.io.write_triangle_mesh(os.path.join(prefix_path, "normalized_mesh.ply"), mesh)

        # 3. Read primitives
        with open(feature_file) as f:
            primitive_dict = yaml.load(f, yaml.CLoader)

        # 4. Extract and merge the same curves
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_faces = np.asarray(mesh.triangles)
        curves, surfaces = filter_primitives(primitive_dict, mesh_faces)

        num_curves = len(curves)
        num_surfaces = len(surfaces)
        num_primitives = num_curves + num_surfaces

        # 5. Calculate the primitive id for each surface
        surface_id_to_primitives, id_corner_points = calculate_indices(curves, surfaces, mesh, prefix_path)

        num_corner_points = len(id_corner_points)
        num_primitives = num_primitives + num_corner_points

        # 6. Calculate the input features: gradient and udf
        gx, gy, gz = np.gradient(udf[:, 3].reshape(v_resolution, v_resolution, v_resolution))
        g = -np.stack((gx, gy, gz), axis=-1)
        g = normalize(g, v_axis=-1)

        input_features = np.concatenate((
            udf[:, 3:4].reshape(v_resolution, v_resolution, v_resolution, 1),
            g), axis=-1)

        # 7. Calculate the consistency
        consistent_flag = np.zeros((resolution, resolution, resolution, 26), dtype=bool)
        valid_flag_3 = valid_flag_ref.reshape(resolution, resolution, resolution, 26)
        consistent_flag[valid_flag_3] = \
            np.tile(surface_id_to_primitives[closest_faces][:, None], (1, 26))[valid_flag_ref] != \
            surface_id_to_primitives[closest_faces[target_coords_ref[valid_flag_ref]]]

        query_points = udf[:, :3].reshape(resolution, resolution, resolution, 3)

        # Visualization
        if v_is_log:
            # 1. Visualize udf
            test_distance = 0.05
            selected_points = udf[:, :3][udf[:, 3] < test_distance]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(selected_points)
            o3d.io.write_point_cloud(os.path.join(prefix_path,"levelset_005.ply"), pc)

            # 2. Visualize the nearest point for each primitive
            is_viz_details = False
            if is_viz_details:
                for i in range(num_primitives):
                    out_mesh = o3d.geometry.TriangleMesh()
                    out_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                    out_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces[surface_id_to_primitives == i])
                    o3d.io.write_triangle_mesh("{}/{}.ply".format(v_output_root, i), out_mesh)

                    mask = surface_id_to_primitives[closest_faces] == i
                    export_point_cloud("{}/p_{}.ply".format(v_output_root, i), udf[mask, :3])

            # 3. Visualize the boundary points
            consistent_flag_ = consistent_flag.reshape(-1, 26)
            s_p = udf[:, :3][np.any(consistent_flag_, axis=1)]
            export_point_cloud(os.path.join(prefix_path,"boundary.ply"), s_p)

            # 4. Visualize the edges
            edge_mesh = o3d.geometry.TriangleMesh()
            for i in range(num_curves):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(mesh_faces[surface_id_to_primitives == i])
                mesh.remove_unreferenced_vertices()
                edge_mesh+=mesh.remove_unreferenced_vertices()
            o3d.io.write_triangle_mesh(os.path.join(prefix_path,"edges.ply"), edge_mesh)


        np.save(
            os.path.join(
                v_output_root,
                prefix,
                "data"),
            {
                "resolution": resolution,
                "input_features": input_features.astype(np.float16),
                # "valid_flags": valid_flag_3,
                "consistent_flags": np.packbits(consistent_flag, axis=None),
                # "query_points": query_points,
            })
    pass


# Construct the edge graph
def construct_graph_3d(v_resolution):
    source_coords = np.stack(np.meshgrid(
        np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution), indexing="ij"),
        axis=3).reshape(-1, 3)
    target_coords_list = []
    for delta_x in [-1, 0, 1]:
        for delta_y in [-1, 0, 1]:
            for delta_z in [-1, 0, 1]:
                if delta_x == delta_y == delta_z == 0:
                    continue
                target_coords_list.append(source_coords + np.array([delta_x, delta_y, delta_z])[None, :])
    target_coords = np.stack(target_coords_list, axis=1)
    valid_flag = np.logical_and(target_coords >= 0, target_coords < v_resolution).all(axis=2)

    target_coords = target_coords[:, :, 0] * v_resolution * v_resolution + \
                    target_coords[:, :, 1] * v_resolution + target_coords[:, :, 2]

    return source_coords, target_coords, valid_flag


if __name__ == '__main__':
    print("Start to construct dataset")
    np.random.seed(0)

    log_root = "output/grid_clustering_3d"
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    resolution = 128
    data_root = r"E:\DATASET\SIGA2023\Mechanism\ABC_NEF_obj"
    output_root = r"G:\Dataset\GSP"

    check_dir(output_root)

    source_coords, target_coords, valid_flag = construct_graph_3d(resolution)

    files = [item for item in os.listdir(os.path.join(data_root, "obj"))]

    num_cores = 8
    num_task_per_core = len(files) // num_cores + 1

    ray.init(
        # local_mode=True,
        # num_cpus=0,
        num_cpus=num_cores,
        num_gpus=1
    )
    tasks = []
    source_coords_ref = ray.put(source_coords)
    target_coords_ref = ray.put(target_coords)
    valid_flag_ref = ray.put(valid_flag)
    for i in range(0, num_cores):
        i_start = i * num_task_per_core
        i_end = min(len(files), (i + 1) * num_task_per_core)
        tasks.append(process_item.remote(data_root, output_root,
                                         files[i_start:i_end],
                                         # files[2:3],
                                         # files[240:241],
                                         source_coords_ref, target_coords_ref, valid_flag_ref,
                                         resolution, True))
    results = ray.get(tasks)
    ray.shutdown()

    exit(0)