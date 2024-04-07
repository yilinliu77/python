import os
import queue
from pathlib import Path

import ray, trimesh
import open3d as o3d
import numpy as np
import yaml
from tqdm import tqdm
import networkx as nx
from shared.common_utils import *

import igraph as ig

import traceback, sys

from src.img2brep.sample_points import sample_points_on_line, sample_points_on_plane, get_plane_points, \
    sample_points_on_circle, sample_points_on_bspline

from scipy.interpolate import BSpline

data_root = Path(r"G:/Dataset/ABC/raw_data/abc_0000_obj_v00")
# data_split = r"valid_planar_shapes_except_cube.txt"
data_split = r"C:/repo/python/src/img2brep/data/train_split_deepcad_10000.txt"
output_root = Path(r"G:/Dataset/img2brep/deepcad_10000/main_data")


# data_root = Path(r"H:\Data\SIGA23\Baseline\data\abc_0000_obj_v00")
# data_split = r"valid_planar_shapes_except_cube.txt"
# output_root = Path(r"H:\Data\SIGA23\Baseline\data\0planar_shapes_test")


def pad_to_numpy1(array_of_lists, pad_value=-1):
    max_length = max(len(lst) for lst in array_of_lists)
    padded_lists = [lst + [pad_value] * (max_length - len(lst)) for lst in array_of_lists]
    return np.array(padded_lists)


def pad_to_numpy2(arrays, pad_value=-1):
    # Determine the maximum length of arrays in the list
    MAX_N = max(array.shape[0] for array in arrays)

    # Initialize a new 3D array with shape M*MAX_N*2 filled with the pad_value
    M = len(arrays)
    padded_arrays = np.full((M, MAX_N, 2), pad_value)

    # Copy the contents of each array into the corresponding slice of the new array
    for i, array in enumerate(arrays):
        padded_arrays[i, :array.shape[0], :] = array

    return padded_arrays


# @ray.remote(num_cpus=1)
def get_brep(v_root, output_root, v_folders):
    # v_folders = ["00000325"]
    single_loop_folder = []

    sample_points_lines_list = []
    sample_points_faces_list = []

    for idx, v_folder in enumerate(v_folders):
        # for idx, v_folder in enumerate(tqdm(v_folders)):
        safe_check_dir(output_root / v_folder)

        try:
            # Load mesh and yml files
            all_files = os.listdir(v_root / v_folder)
            obj_file = [ff for ff in all_files if ff.endswith(".obj")][0]
            yml_file = [ff for ff in all_files if ff.endswith(".yml") and "features" in ff][0]

            mesh = trimesh.load_mesh(v_root / v_folder / obj_file, process=False, maintain_order=True)

            # Normalize with bounding box
            extent = mesh.bounding_box.extents
            diag = np.linalg.norm(extent)
            centroid = mesh.bounding_box.centroid

            mesh.vertices -= centroid
            mesh.vertices /= diag

            transform = lambda x: (x - centroid) / diag

            mesh.export(output_root / v_folder / "mesh.ply")

            # Start to extract BREP
            files = yaml.load(open(v_root / v_folder / yml_file, "r").read(), yaml.CLoader)
            curves = files["curves"]
            surfaces = files["surfaces"]

            planes = []
            lines = []

            sample_points_lines = []
            sample_points_faces = []

            # Read all the curves and plane
            edge_idx_to_line_idx = {}
            line_idx = 0
            for line in curves:
                line_type = line["type"]
                if line_type == "Line":
                    coords = np.asarray(line["location"])
                    direction = np.asarray(line["direction"])
                    vertex_index = line["vert_indices"]
                    vert_parameters = np.asarray(line["vert_parameters"])

                    sample_points = sample_points_on_line(coords, direction, vert_parameters, num_samples=20)

                elif line_type == "Circle":
                    coords = np.asarray(line["location"])
                    vertex_index = line["vert_indices"]
                    vert_parameters = np.asarray(line["vert_parameters"])
                    radius = line["radius"]
                    xyz_axis = np.stack([np.asarray(line["x_axis"]),
                                         np.asarray(line["y_axis"]),
                                         np.asarray(line["z_axis"])])
                    sample_points = sample_points_on_circle(coords, xyz_axis, radius, num_samples=20)

                elif line_type == "BSpline":
                    knots = np.asarray(line["knots"])
                    poles = np.asarray(line["poles"])
                    degree = int(line["degree"])
                    rational = bool(line["rational"])
                    weights = np.asarray(line["weights"])
                    vert_parameters = np.asarray(line["vert_parameters"])
                    sample_points = sample_points_on_bspline(knots, poles, degree, rational, weights, vert_parameters)

                else:
                    raise ValueError("Unknown line type")

                sample_points_lines.append(transform(sample_points))

                lines.append({"type"        : line_type,
                              "vertex_index": vertex_index, "sample_points": sample_points})

                edge_idx_to_line_idx[(vertex_index[0], vertex_index[-1])] = line_idx
                edge_idx_to_line_idx[(vertex_index[-1], vertex_index[0])] = line_idx
                line_idx += 1

            assert len(edge_idx_to_line_idx) == len(curves) * 2

            # safe_check_dir(output_root / v_folder / "debug")
            count = 0
            for plane in surfaces:
                faces_type = plane["type"]

                if faces_type == "Plane":
                    params = np.asarray(plane["coefficients"])
                    vertex_index = plane["vert_indices"]

                    abcd = np.array(plane["coefficients"])
                    coords = np.asarray(plane["location"])
                    vert_parameters = np.asarray(plane["vert_parameters"])
                    xyz_axis = np.stack([np.asarray(plane["x_axis"]),
                                         np.asarray(plane["y_axis"]),
                                         np.asarray(plane["z_axis"])])

                    sample_points = sample_points_on_plane(coords, abcd, xyz_axis, vert_parameters, num_samples=20)

                else:
                    raise ValueError("Unknown faces type")

                sample_points_faces.append(transform(sample_points))

                planes.append({"params": params, "vertex_index": vertex_index, "sample_points": sample_points})

                points_in_plane_c = get_plane_points(coords, abcd, xyz_axis, vert_parameters)
                points_in_plane_c = transform(points_in_plane_c)

                # point_cloud = o3d.geometry.PointCloud()
                # point_cloud.points = o3d.utility.Vector3dVector(points_in_plane_c)
                # o3d.io.write_point_cloud(str(output_root / v_folder / "debug" / (str(count) + ".ply")), point_cloud)
                # count += 1

            # vis the sample points
            sample_points_lines_cat = np.concatenate(sample_points_lines, axis=0)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(sample_points_lines_cat)
            o3d.io.write_point_cloud(str(output_root / v_folder / "sample_points_lines.ply"), point_cloud)

            sample_points_faces_cat = np.concatenate(sample_points_faces, axis=0)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(sample_points_faces_cat)
            o3d.io.write_point_cloud(str(output_root / v_folder / "sample_points_faces.ply"), point_cloud)

            # save to global list
            # np.save(output_root / v_folder / "sample_points_lines.npy", np.stack(sample_points_lines))
            # np.save(output_root / v_folder / "sample_points_faces.npy", np.stack(sample_points_faces))

            sample_points_lines_list.append(np.stack(sample_points_lines))
            sample_points_faces_list.append(np.stack(sample_points_faces))

            if False:
                sample_points_vis = np.concatenate((sample_points_vis_lines, sample_points_vis_faces), axis=0)
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(sample_points_vis)
                o3d.visualization.draw_geometries([point_cloud], window_name='Open3D Point Cloud', width=800,
                                                  height=600)

            # Plane-to-line connectivity
            vv_adj = np.zeros((mesh.vertices.shape[0], mesh.vertices.shape[0]), dtype=bool)
            vv_adj[mesh.edges_unique[:, 0], mesh.edges_unique[:, 1]] = True
            vv_adj[mesh.edges_unique[:, 1], mesh.edges_unique[:, 0]] = True

            # Loop for each plane
            plane_to_line = [[] for _ in range(len(planes))]
            for id_plane, plane in enumerate(planes):
                id_lines = []
                id_vertices = []
                id_common_points = []
                for id_line, line in enumerate(lines):
                    set_2 = frozenset(plane["vertex_index"])
                    id_common_point = [x for x in line["vertex_index"] if x in set_2]
                    if len(id_common_point) < 2:
                        continue
                    id_lines.append(id_line)
                    id_vertices.append(id_common_point)
                    id_common_points += id_common_point

                corner_points = set()
                for id_line1, line1 in enumerate(id_vertices):
                    for id_line2, line2 in enumerate(id_vertices):
                        if id_line1 == id_line2:
                            continue
                        corner_points |= (set(line1) & set(line2))

                graph = nx.Graph()
                for id_vertex in corner_points:
                    graph.add_node(id_vertex)
                for id_line1, line1 in enumerate(id_vertices):
                    idx_last = -1
                    id_last = -1
                    for idx, id_vert in enumerate(line1):
                        if id_vert in corner_points:
                            if idx_last == -1:
                                idx_last = idx
                                id_last = id_vert
                            # elif idx_last != idx-1:
                            else:
                                graph.add_edge(id_last, id_vert)
                                id_last = id_vert
                                idx_last = idx
                            continue

                # Graph for all the line points on this plane
                # graph = nx.Graph()
                # for id_vertex in id_common_points:
                #     graph.add_node(id_vertex)
                # edges = [item for item in mesh.edges_unique if item[0] in id_common_points and item[1] in id_common_points]
                # graph.add_edges_from(edges)
                #
                # for vertex in graph.nodes:
                #     # Degree of the vertex is 2
                #     if graph.degree[vertex] == 3 and vertex not in corner_points:
                #         edges = list(graph.edges(vertex))
                #         for edge in edges:
                #             if graph.degree[edge[0]] != 2 and graph.degree[edge[1]] != 2:
                #                 graph.remove_edge(*edge)
                #                 break

                # raw_sequences = nx.minimum_cycle_basis(graph)
                # raw_sequences = [[item[0] for item in nx.find_cycle(graph, loop)] for loop in raw_sequences]
                igraph = ig.Graph.from_networkx(graph)
                raw_sequences_edges = igraph.minimum_cycle_basis(use_cycle_order=True)
                raw_sequences = []
                for edges in raw_sequences_edges:
                    # form a loop
                    loop = []
                    # First edge
                    if igraph.es[edges[0]].source == igraph.es[edges[1]].source or \
                            igraph.es[edges[0]].source == igraph.es[edges[1]].target:
                        loop.append(igraph.es[edges[0]].target_vertex["_nx_name"])
                        loop.append(igraph.es[edges[0]].source_vertex["_nx_name"])
                    else:
                        loop.append(igraph.es[edges[0]].source_vertex["_nx_name"])
                        loop.append(igraph.es[edges[0]].target_vertex["_nx_name"])

                    for edge in edges[1:-1]:
                        id_candidate1 = igraph.es[edge].source_vertex["_nx_name"]
                        id_candidate2 = igraph.es[edge].target_vertex["_nx_name"]
                        loop.append(id_candidate1 if loop[-1] == id_candidate2 else id_candidate2)
                    raw_sequences.append(loop)

                # Travel
                corner_sequences = []
                for loop in raw_sequences:
                    corner_sequences.append([])
                    for id in loop:
                        if id in corner_points:
                            corner_sequences[-1].append(id)

                plane_to_line[id_plane] = corner_sequences

            all_face_single_loop = True
            with open(output_root / v_folder / "plane_to_line.txt", "w") as f:
                for plane in plane_to_line:
                    f.write("p\n")

                    for sequences in plane:
                        f.write(",".join([str(item) for item in sequences]) + "\n")

                    if len(plane) > 1:
                        all_face_single_loop = False

            if all_face_single_loop:
                single_loop_folder.append(v_folder)

            face_edge_idx = []
            edge_adj = []
            for face_idx in range(len(plane_to_line)):
                # each begins with face_idx
                face_egde_idx_c = [face_idx]
                plane = plane_to_line[face_idx]

                # transform the vertex index to line index
                edge_adj_c_list = []
                for sequence in plane:
                    # each loop begins with -1
                    face_egde_idx_c.append(-1)
                    face_edge_idx_list_c = []

                    for edge in zip(sequence, sequence[1:] + sequence[0:1]):
                        if edge not in edge_idx_to_line_idx:
                            raise "Error"
                        line_idx = edge_idx_to_line_idx[edge]
                        face_egde_idx_c.append(line_idx)
                        face_edge_idx_list_c.append(line_idx)

                    # adj line idx mean adj
                    edge_adj_c = np.column_stack((face_edge_idx_list_c, np.roll(face_edge_idx_list_c, -1)))  # N * 2
                    edge_adj_c_list.append(edge_adj_c)

                edge_adj.append(np.concatenate(edge_adj_c_list, axis=0))

                face_edge_idx.append(face_egde_idx_c)

            face_edge_idx = pad_to_numpy1(face_edge_idx, pad_value=-1)
            edge_adj = pad_to_numpy2(edge_adj, pad_value=-1)

            # np.save(output_root / v_folder / "face_edge_idx.npy", face_edge_idx)

            data_dict = {
                'face_edge_idx'      : face_edge_idx,
                'sample_points_lines': np.stack(sample_points_lines),
                'sample_points_faces': np.stack(sample_points_faces),
                'edge_adj'           : edge_adj
                }

            np.savez_compressed(output_root / v_folder / "data.npz", **data_dict)

            # Check
            line_mesh = o3d.geometry.LineSet()
            points = []
            indices = []
            existing_vertices_id = []
            for plane in plane_to_line:
                # points = []
                for sequence in plane:
                    lines = zip(sequence, sequence[1:] + sequence[0:1])
                    for seg in lines:
                        indices.append([])
                        if seg[0] not in existing_vertices_id:
                            points.append(mesh.vertices[seg[0]])
                            existing_vertices_id.append(seg[0])
                        indices[-1].append(existing_vertices_id.index(seg[0]))

                        if seg[1] not in existing_vertices_id:
                            points.append(mesh.vertices[seg[1]])
                            existing_vertices_id.append(seg[1])
                        indices[-1].append(existing_vertices_id.index(seg[1]))

                        # points.append(mesh.vertices[seg[0]])
                        # points.append(mesh.vertices[seg[1]])
                # line_mesh.points = o3d.utility.Vector3dVector(points)
                # line_mesh.lines = o3d.utility.Vector2iVector(np.array([i for i in range(len(points))]).reshape(-1, 2))
                # o3d.io.write_line_set("{}_loops.ply".format(v_folder), line_mesh)
                continue

            line_mesh.points = o3d.utility.Vector3dVector(points)
            line_mesh.lines = o3d.utility.Vector2iVector(indices)
            o3d.io.write_line_set(str(output_root / v_folder / "wireframe.ply"), line_mesh)

            plane_adj = np.zeros((len(planes), len(planes)), dtype=np.int32)
            for id_plane1, plane1 in enumerate(planes):
                for id_plane2, plane2 in enumerate(planes):
                    if id_plane1 >= id_plane2:
                        continue
                    if len(set(plane1["vertex_index"]) & set(plane2["vertex_index"])) != 0:
                        plane_adj[id_plane1, id_plane2] = 1
                        plane_adj[id_plane2, id_plane1] = 1

            with open(output_root / v_folder / "plane_adj.txt", "w") as f:
                for row in plane_adj:
                    f.write(",".join([str(item) for item in row]) + "\n")
        except Exception as e:
            with open(output_root / "error.txt", "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(v_folder + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(e)

    for folder in single_loop_folder:
        with open(output_root / "single_loop.txt", "a") as f:
            f.write(folder + "\n")

    # max_len = max([item.shape[0] for item in sample_points_lines_list])
    # for idx, item in enumerate(sample_points_lines_list):
    #     if item.shape[0] < max_len:
    #         sample_points_lines_list[idx] = np.concatenate((item, np.zeros((max_len - item.shape[0], 20, 3))), axis=0)
    #
    # max_len = max([item.shape[0] for item in sample_points_faces_list])
    # for idx, item in enumerate(sample_points_faces_list):
    #     if item.shape[0] < max_len:
    #         sample_points_faces_list[idx] = np.concatenate((item, np.zeros((max_len - item.shape[0], 20, 3))), axis=0)

    # sample_points_lines_list = np.concatenate(sample_points_lines_list, axis=0)
    # sample_points_faces_list = np.concatenate(sample_points_faces_list, axis=0)
    return


get_brep_ray = ray.remote(get_brep)

if __name__ == '__main__':
    total_ids = [item.strip() for item in open(data_split, "r").readlines()]

    safe_check_dir(output_root)

    # single process
    if False:
        get_brep(data_root, output_root, total_ids)
    else:
        ray.init(
                num_cpus=1,
                local_mode=True
                )
        num_batches = 40
        batch_size = len(total_ids) // num_batches + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(
                    get_brep_ray.remote(data_root,
                                        output_root,
                                        total_ids[i * batch_size:min(len(total_ids), (i + 1) * batch_size)]))
        ray.get(tasks)
        print("Done")
