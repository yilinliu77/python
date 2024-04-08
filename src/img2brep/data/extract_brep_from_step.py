import os
import queue
from pathlib import Path

import ray, trimesh
import open3d as o3d
import numpy as np
import yaml
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.NCollection import NCollection_Map
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.GeomAbs import (GeomAbs_Circle, GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_Ellipse,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Extend.DataExchange import read_step_file
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
data_split = r"C:/repo/python/src/img2brep/data/deepcad_train_10000.txt"
exception_files = [
    r"C:/repo/python/src/img2brep/data/abc_multiple_component_or_few_faces_ids_.txt",
    r"C:/repo/python/src/img2brep/data/abc_cube_ids.txt",
    r"C:/repo/python/src/img2brep/data/abc_with_others_ids.txt",
]
output_root = Path(r"G:/Dataset/img2brep/deepcad_10000/main_data")

num_max_primitives = 100000

# @ray.remote(num_cpus=1)
def get_brep(v_root, output_root, v_folders):
    # v_folders = ["00001000"]
    single_loop_folder = []

    for idx, v_folder in enumerate(v_folders):
        # for idx, v_folder in enumerate(tqdm(v_folders)):
        safe_check_dir(output_root / v_folder)

        try:
            # Load mesh and yml files
            all_files = os.listdir(v_root / v_folder)
            obj_file = [ff for ff in all_files if ff.endswith(".obj")][0]
            step_file = [ff for ff in all_files if ff.endswith(".step") and "step" in ff][0]

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
            shape = read_step_file(str(v_root / v_folder / step_file), verbosity=False)
            if shape.NbChildren() != 1:
                print("Multiple components: {}; Jump over".format(v_folder))
                shutil.rmtree(output_root / v_folder)
                continue

            # Function to explore and print the elements of a shape
            def explore_shape(shape, shape_type):
                explorer = TopExp_Explorer(shape, shape_type)
                while explorer.More():
                    yield explorer.Current()
                    explorer.Next()

            # Explore and list faces, edges, and vertices
            face_dict = {}
            for face in explore_shape(shape, TopAbs_FACE):
                if face not in face_dict and face.Reversed() not in face_dict:
                    face_dict[face] = len(face_dict)
            num_faces = len(face_dict)
            edge_dict = {}
            for edge in explore_shape(shape, TopAbs_EDGE):
                if edge not in edge_dict and edge.Reversed() not in edge_dict:
                    edge_dict[edge] = len(edge_dict)
            num_edges = len(edge_dict)
            all_vertices = list(explore_shape(shape, TopAbs_VERTEX))

            # Sample points in face
            face_sample_points = []
            for face in face_dict:
                surface = BRepAdaptor_Surface(face)

                if surface.GetType() not in [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                                             GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface]:
                    raise ValueError("Unsupported surface type: {}".format(surface.GetType()))

                first_u = surface.FirstUParameter()
                last_u = surface.LastUParameter()
                first_v = surface.FirstVParameter()
                last_v = surface.LastVParameter()
                u = np.linspace(first_u, last_u, num=20)
                v = np.linspace(first_v, last_v, num=20)
                u, v = np.meshgrid(u, v)
                points = []
                for i in range(u.shape[0]):
                    for j in range(u.shape[1]):
                        pnt = surface.Value(u[i, j], v[i, j])
                        points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
                face_sample_points.append(np.stack(points, axis=0).reshape(20, 20, 3))
            face_sample_points = np.stack(face_sample_points, axis=0)
            face_sample_points = transform(face_sample_points)
            assert len(face_dict) == num_faces == face_sample_points.shape[0]

            # Sample points in edges
            edge_sample_points = []
            for edge in edge_dict:
                curve = BRepAdaptor_Curve(edge)
                if curve.GetType() not in [GeomAbs_Circle, GeomAbs_Line, GeomAbs_Ellipse, GeomAbs_BSplineCurve]:
                    raise ValueError("Unsupported curve type: {}".format(curve.GetType()))
                # Sample 20 points along it
                range_start = curve.FirstParameter()
                range_end = curve.LastParameter()
                sample_u = np.linspace(range_start, range_end, num=20)
                sample_points = []
                for u in sample_u:
                    pnt = curve.Value(u)
                    sample_points.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32))
                edge_sample_points.append(np.stack(sample_points, axis=0))
            edge_sample_points = np.stack(edge_sample_points, axis=0)
            edge_sample_points = transform(edge_sample_points)
            assert len(edge_dict) == num_edges == edge_sample_points.shape[0]

            # (N, X) shows every loop with each face, -2 denotes a start token and pad with -1
            face_edge_connectivity = []
            for face in face_dict:
                loops = []

                for wire in explore_shape(face, TopAbs_WIRE):
                    loops.append(-2)
                    for edge in explore_shape(wire, TopAbs_EDGE):
                        if edge in edge_dict:
                            loops.append(edge_dict[edge])
                        elif edge.Reversed() in edge_dict:
                            loops.append(edge_dict[edge.Reversed()])
                        else:
                            raise ValueError("Edge not in edge_dict")
                face_edge_connectivity.append(loops)

            # (M, 2) shows the indexes of all the connected edges
            edge_connectivity = []
            edge_vertex_map = {}
            for edge in edge_dict:
                # Get vertices of the current edge
                for vertex in explore_shape(edge, TopAbs_VERTEX):
                    if vertex not in edge_vertex_map:
                        edge_vertex_map[vertex] = [edge]
                    else:
                        edge_vertex_map[vertex].append(edge)
            for vertex, edges in edge_vertex_map.items():
                for i in range(len(edges)):
                    id1 = edge_dict[edges[i]] if edges[i] in edge_dict else edge_dict[edges[i].Reversed()]
                    edge2 = edges[(i + 1) % len(edges)]
                    id2 = edge_dict[edge2] if edge2 in edge_dict else edge_dict[edge2.Reversed()]
                    if id1 < id2:
                        edge_connectivity.append([id1, id2])
                    else:
                        edge_connectivity.append([id2, id1])

            # (N, N) shows the adjacency of faces
            face_connectivity = np.zeros((num_faces, num_faces), dtype=np.int8)
            face_edge_map = {}
            # (M, 2) shows the index of faces to intersect with an edge
            edge_face_idx = np.zeros((len(edge_dict), 2), dtype=np.int32)
            for face in face_dict:
                # Get edges of the current face
                for edge in explore_shape(face, TopAbs_EDGE):
                    if edge in face_edge_map:
                        face_edge_map[edge].append(face)
                    elif edge.Reversed() in face_edge_map:
                        face_edge_map[edge.Reversed()].append(face)
                    else:
                        face_edge_map[edge] = [face]
            for item in face_edge_map:
                if len(face_edge_map[item]) != 2:
                    raise ValueError("Edge results by more than 2 faces.")
            for edge, faces in face_edge_map.items():
                id1 = face_dict[faces[0]] if faces[0] in face_dict else face_dict[faces[0].Reversed()]
                id2 = face_dict[faces[1]] if faces[1] in face_dict else face_dict[faces[1].Reversed()]
                face_connectivity[id1, id2] = 1
                face_connectivity[id2, id1] = 1
                if edge in edge_dict:
                    edge_face_idx[edge_dict[edge], 0] = id1
                    edge_face_idx[edge_dict[edge], 1] = id2
                else:
                    edge_face_idx[edge_dict[edge.Reversed()], 0] = id1
                    edge_face_idx[edge_dict[edge.Reversed()], 1] = id2

            max_length = max(len(lst) for lst in face_edge_connectivity)
            face_edge_connectivity = np.array(
                [i + [-1]*(max_length-len(i)) for i in face_edge_connectivity], dtype=np.int32)

            edge_connectivity = np.asarray(edge_connectivity, dtype=np.int32)

            data_dict = {
                'face_edge_idx'      : face_edge_connectivity,
                'sample_points_lines': edge_sample_points,
                'sample_points_faces': face_sample_points,
                'edge_adj'           : edge_connectivity,
                'face_adj'           : face_connectivity,
                'edge_face_idx'           : edge_face_idx,
                }

            np.savez_compressed(output_root / v_folder / "data.npz", **data_dict)
            # continue

            # Check
            # Write face
            pc_model = o3d.geometry.PointCloud()
            pc_model.points = o3d.utility.Vector3dVector(face_sample_points.reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "face_sample_points.ply"), pc_model)

            pc_model.points = o3d.utility.Vector3dVector(edge_sample_points.reshape(-1, 3))
            o3d.io.write_point_cloud(str(output_root / v_folder / "edge_sample_points.ply"), pc_model)

            check_dir(output_root / v_folder / "debug_topology")
            for i in range(face_edge_connectivity.shape[0]):
                points = []
                for item in face_edge_connectivity[i]:
                    if item == -1 or item == -2:
                        continue
                    points.append(edge_sample_points[item])
                pc_model.points = o3d.utility.Vector3dVector(np.stack(points, axis=0).reshape(-1, 3))
                o3d.io.write_point_cloud(str(output_root / v_folder / "debug_topology" / "{}.ply".format(i)), pc_model)
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

    return


get_brep_ray = ray.remote(get_brep)

if __name__ == '__main__':
    total_ids = [item.strip() for item in open(data_split, "r").readlines()]

    exception_ids = []
    for file in exception_files:
        exception_ids += [item.strip() for item in open(file, "r").readlines()]
    exception_ids = list(set(exception_ids))

    num_original = len(total_ids)
    total_ids = list(set(total_ids) - set(exception_ids))
    total_ids.sort()
    print("Total ids: {} -> {}".format(num_original, len(total_ids)))
    check_dir(output_root)

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
