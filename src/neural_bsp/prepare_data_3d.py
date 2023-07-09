import os.path
import sys
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
from shared.common_utils import check_dir, export_point_cloud


def normalize(v, v_axis=-1):
    norm = np.linalg.norm(v, axis=v_axis)
    new_v = np.copy(v)
    new_v[norm > 0] = new_v[norm > 0] / norm[norm > 0][:, None]
    return new_v


# @ray.remote(num_gpus=0.5)
def calculate_distance_gpu(vertices, faces, query_points):
    device = torch.device('cuda')

    with torch.no_grad():
        m = bvh_distance_queries.BVH()

        distances, closest_points, closest_faces, closest_bcs = m(
            torch.from_numpy(vertices[faces].copy()).to(device).unsqueeze(0),
            torch.from_numpy(query_points.copy()).to(device).unsqueeze(0))
        # torch.cuda.synchronize()
        distances = torch.sqrt(distances).detach().cpu().numpy()[0]
        # distances = distances.detach().cpu().numpy()[0]
        closest_points = closest_points.detach().cpu().numpy()[0]
        closest_faces = closest_faces[0].cpu().numpy()
        closest_bcs = closest_bcs[0].cpu().numpy()

    # print(distances.shape[0])
    # print(closest_faces.shape[0])
    # print(closest_bcs.shape[0])
    return distances, closest_faces, closest_bcs


def calculate_distance_cpu(vertices, faces, query_points):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    results = scene.compute_closest_points(
        o3d.core.Tensor.from_numpy(query_points)
    )
    uvs = results["primitive_uvs"].numpy()
    closest_points = results["points"].numpy()
    distances = np.linalg.norm(closest_points - query_points, axis=-1)
    closest_faces = results["primitive_ids"].numpy()

    uvs = np.concatenate((1 - uvs[:, 0:1] - uvs[:, 1:2], uvs), axis=1)

    return distances, closest_faces, uvs


def calculate_distance(query_points, vertices, faces,
                       surface_id_to_primitives, face_edge_indicator,
                       num_curves,
                       use_cpu=True):
    if use_cpu:
        distances, closest_faces, closest_bcs = calculate_distance_cpu(
            vertices, faces, query_points)
    else:
        distances, closest_faces, closest_bcs = calculate_distance_gpu(
            vertices, faces, query_points)
    #
    closest_primitive = surface_id_to_primitives[closest_faces]
    face_edge_indicator_expanded = face_edge_indicator[closest_faces]

    all_mask = np.any(face_edge_indicator_expanded >= 0, axis=-1)
    all_mask = np.logical_and(all_mask, np.any(closest_bcs < 1e-2, axis=-1))

    corner_points_mask = np.any(
        np.logical_and(face_edge_indicator_expanded > num_curves, closest_bcs > 1 - 1e-2),
        axis=-1
    )
    closest_primitive[corner_points_mask] = np.max(face_edge_indicator_expanded[corner_points_mask], axis=-1)

    edge_mask = np.logical_and(all_mask, np.all(face_edge_indicator_expanded < num_curves, axis=-1))
    closest_primitive[edge_mask] = np.max(face_edge_indicator_expanded[edge_mask], axis=-1)

    pass

    if False:
        for test_distance in [0.01, 0.1, 0.5]:
            selected_points = query_points[distances < test_distance]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(selected_points)
            o3d.io.write_point_cloud("output/{}.ply".format(test_distance), pc)

    return distances, closest_primitive


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
            return None, None

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
def calculate_indices(curves, surfaces, vertices, faces, v_is_log=""):
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
    # curves: [0, num_curves]
    # surfaces: [num_curves, num_curves+num_surfaces]
    # corner points: [num_curves+num_surfaces, num_primitives]
    if v_is_log != "":
        if len(list(id_corner_points.keys())) == 0:
            # print("{} don't have corner points".format(v_is_log))
            pass
        else:
            export_point_cloud("{}/corner_points.ply".format(v_is_log), vertices[list(id_corner_points.keys())])

    # Each face has three flags for each vertex. 0 is normal face, negative is edges, positive is corner points
    face_edge_indicator = -np.ones((faces.shape[0], 3), np.int64)
    for id_surface, surface in enumerate(surfaces):
        for id_face in surface["face_indices"]:
            primitive_ids = vert_id_to_primitives[faces[id_face]]

            for idx, primitive_id in enumerate(primitive_ids):
                if primitive_id >= num_curves + num_surfaces:
                    face_edge_indicator[id_face, idx] = primitive_id
                elif primitive_id > -1:
                    face_edge_indicator[id_face, idx] = primitive_id

            surface_id_to_primitives[id_face] = id_surface + num_curves

    return surface_id_to_primitives, face_edge_indicator, id_corner_points


@ray.remote(num_cpus=1)
def process_item(v_root, v_output_root, v_files,
                 source_coords_ref, target_coords_ref, valid_flag_ref, v_resolution, v_is_log=False):
    for prefix in v_files:
        times = [0] * 10
        cur_time = time.time()
        # prefix = "_".join(obj_file.split("_")[:2])
        # 1. Setup file path
        prefix_path = os.path.join(v_output_root, prefix)
        os.makedirs(os.path.join(v_output_root, prefix), exist_ok=True)
        feature_file = obj_file = None
        for file in os.listdir(os.path.join(v_root, prefix)):
            if "features" in file:
                feature_file = os.path.join(v_root, prefix, file)
            elif "trimesh" in file:
                obj_file = os.path.join(v_root, prefix, file)
        if feature_file is None or obj_file is None:
            continue
        # 2. Read obj
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
        mesh_vertices = np.stack(vertices).astype(np.float32)
        mesh_faces = np.stack(faces).astype(np.int64) - 1

        # Calculate the bounds
        vmin = np.min(mesh_vertices, axis=0)
        vmax = np.max(mesh_vertices, axis=0)
        vcenter = (vmin + vmax) / 2
        diag = np.sqrt(np.sum((vmax - vmin) ** 2))
        # Normalize the vertices
        mesh_vertices = ((mesh_vertices - vcenter) / diag) * 2
        query_points = source_coords_ref.astype(np.float32) / (resolution - 1) * 2 - 1
        if v_is_log:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh_vertices),
                o3d.utility.Vector3iVector(mesh_faces),
            )
            o3d.io.write_triangle_mesh(os.path.join(prefix_path, "normalized_mesh.ply"), mesh)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        # 3. Read primitives
        with open(feature_file) as f:
            # primitive_dict = yaml.load(f, yaml.CSafeLoader)
            primitive_dict = yaml.load(f, yaml.CLoader)
        times[1] += time.time() - cur_time
        cur_time = time.time()
        # 4. Extract and merge the same curves
        curves, surfaces = filter_primitives(primitive_dict, mesh_faces)

        if curves is None:
            continue
        times[2] += time.time() - cur_time
        cur_time = time.time()

        num_curves = len(curves)
        num_surfaces = len(surfaces)
        num_primitives = num_curves + num_surfaces

        # 5. Calculate the primitive id for each surface
        surface_id_to_primitives, face_edge_indicator, id_corner_points = calculate_indices(
            curves, surfaces, mesh_vertices, mesh_faces, prefix_path if v_is_log else "")
        num_corner_points = len(id_corner_points)
        num_primitives = num_primitives + num_corner_points
        times[3] += time.time() - cur_time
        cur_time = time.time()
        udf, closest_primitives = calculate_distance(query_points, mesh_vertices, mesh_faces,
                                                     surface_id_to_primitives, face_edge_indicator,
                                                     num_curves,
                                                     use_cpu=True)
        times[4] += time.time() - cur_time
        cur_time = time.time()
        # 6. Calculate the input features: gradient and udf
        gx, gy, gz = np.gradient(udf.reshape((v_resolution, v_resolution, v_resolution)))
        g = -np.stack((gx, gy, gz), axis=-1)
        g = normalize(g, v_axis=-1)

        input_features = np.concatenate((
            udf.reshape((v_resolution, v_resolution, v_resolution, 1)),
            g), axis=-1)
        times[5] += time.time() - cur_time
        cur_time = time.time()
        # 7. Calculate the consistency
        consistent_flag = np.zeros((resolution, resolution, resolution, 26), dtype=bool)
        valid_flag_3 = valid_flag_ref.reshape((resolution, resolution, resolution, 26))
        consistent_flag[valid_flag_3] = \
            np.tile(closest_primitives[:, None], (1, 26))[valid_flag_ref] != \
            closest_primitives[target_coords_ref[valid_flag_ref]]

        consistent_flag = np.any(consistent_flag, axis=3)
        times[6] += time.time() - cur_time
        cur_time = time.time()
        # Visualization
        if v_is_log:
            # 1. Visualize udf
            test_distance = 0.05
            selected_points = query_points[udf < test_distance]

            if selected_points.shape[0] == 0:
                print("{} don't have levelset_005 points".format(prefix))
            else:
                export_point_cloud(os.path.join(prefix_path, "levelset_005.ply"), selected_points)

            # 2. Visualize the nearest point for each primitive
            is_viz_details = False
            if is_viz_details:
                for i in range(num_primitives):
                    # out_mesh = o3d.geometry.TriangleMesh()
                    # out_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                    # out_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces[surface_id_to_primitives == i])
                    # o3d.io.write_triangle_mesh("{}/{}.ply".format(v_output_root, i), out_mesh)

                    mask = closest_primitives == i
                    export_point_cloud("{}/p_{}.ply".format(v_output_root, i), query_points[mask])

            # 3. Visualize the boundary points
            consistent_flag_ = consistent_flag.reshape(-1)
            s_p = query_points[consistent_flag_]
            if s_p.shape[0] == 0:
                print("{} don't have boundary points".format(prefix))
            else:
                export_point_cloud(os.path.join(prefix_path, "boundary.ply"), s_p)

            # 4. Visualize the edges
            edge_mesh = o3d.geometry.TriangleMesh()
            for i in range(num_curves):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(mesh_faces[np.any(face_edge_indicator == i, axis=-1)])
                mesh.remove_unreferenced_vertices()
                edge_mesh += mesh.remove_unreferenced_vertices()
            if np.asarray(edge_mesh.triangles).shape[0] == 0:
                print("{} don't have curves".format(prefix))
            else:
                o3d.io.write_triangle_mesh(os.path.join(prefix_path, "edges.ply"), edge_mesh)

        np.save(
            os.path.join(
                v_output_root,
                "training",
                "{}".format(prefix)),
            {
                "resolution": resolution,
                "input_features": input_features.astype(np.float16),
                # "valid_flags": valid_flag_3,
                "consistent_flags": np.packbits(consistent_flag, axis=None),
                # "query_points": query_points,
            })
        times[7] += time.time() - cur_time
        cur_time = time.time()
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

    resolution = 256
    # data_root = r"E:\DATASET\SIGA2023\Mechanism\ABC_NEF_obj"
    # output_root = r"G:\Dataset\GSP"

    assert len(sys.argv) == 5 or len(sys.argv) == 7
    data_root = sys.argv[1]
    output_root = sys.argv[2]

    # if os.path.exists(output_root):
    #     print("Output directory already exists")
    #     exit()
    check_dir(output_root)
    check_dir(os.path.join(output_root, "training"))

    source_coords, target_coords, valid_flag = construct_graph_3d(resolution)

    if len(sys.argv) > 5:
        id_start = int(sys.argv[5])
        id_end = int(sys.argv[6])
    else:
        id_start = 0
        id_end = 1000000
    assert id_end>=id_start

    files = [item for item in os.listdir(data_root) if int(item)>id_start and int(item) < id_end]

    num_cores = int(sys.argv[3])
    num_gpus = int(sys.argv[4])
    num_task_per_core = len(files) // max(1, num_cores) + 1

    ray.init(
        # local_mode=True,
        # num_cpus=1,
        num_cpus=num_cores,
        num_gpus=num_gpus
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
                                         resolution, False))
    results = ray.get(tasks)
    ray.shutdown()

    exit(0)
