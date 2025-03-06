import copy
import itertools
import math
import os, sys, shutil, traceback
from pathlib import Path

import numpy as np
import torch
from OCC.Core import Message
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.IFSelect import IFSelect_ReturnStatus
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.Interface import Interface_Static
from OCC.Core.Message import Message_PrinterOStream, Message_Alarm
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_ManifoldSolidBrep, \
    STEPControl_FacetedBrep, STEPControl_ShellBasedSurfaceModel
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopAbs import TopAbs_SHAPE
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Extend.DataExchange import read_step_file, write_stl_file

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud
from shared.occ_utils import get_primitives, disable_occ_log
from src.brepnet.eval.check_valid import check_step_valid_soild, save_step_file

from src.brepnet.post.utils import *
from src.brepnet.post.geom_optimization import optimize_geom, test_optimize_geom

import ray
import argparse
import trimesh

import time


def get_data(v_filename):
    # specify the key to get the face points, edge points and edge_face_connectivity in data.npz
    # data_npz = np.load(os.path.join(data_root, folder_name, 'data.npz'), allow_pickle=True)['arr_0'].item()
    data_npz = np.load(v_filename, allow_pickle=True)
    if 'sample_points_faces' in data_npz and 'edge_face_connectivity' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
        edge_points = data_npz['sample_points_lines']  # Edge sample points (num_lines*20*3)
        edge_face_connectivity = data_npz['edge_face_connectivity']  # (num_intersection, (id_edge, id_face1, id_face2))
    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
        edge_points = data_npz['pred_edge']
        edge_face_connectivity = data_npz['pred_edge_face_connectivity']
    elif 'pred_face' in data_npz and 'face_edge_adj' in data_npz:
        face_points = data_npz['pred_face'].astype(np.float32)
        edge_points = data_npz['pred_edge'].astype(np.float32)
        face_edge_adj = data_npz['face_edge_adj']
        edge_face_connectivity = []
        N = face_points.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                intersection = list(set(face_edge_adj[i]).intersection(set(face_edge_adj[j])))
                if len(intersection) > 0:
                    edge_face_connectivity.append([intersection[0], i, j])
        edge_face_connectivity = np.array(edge_face_connectivity)

    else:
        raise ValueError(f"Unknown data npz format {v_filename}")

    face_points = face_points[..., :3]
    edge_points = edge_points[..., :3]
    shape = Shape(face_points, edge_points, edge_face_connectivity, False)
    return shape


def get_candidate_shapes(num_drop, v_faces, v_curves, v_conn):
    if num_drop == 0:
        new_faces = [item for item in v_faces]
        new_curves = [item for item in v_curves]
        new_edge_face_connectivity = [item for item in v_conn]
        return [(new_faces, new_curves, new_edge_face_connectivity)]
    num_faces = len(v_faces)
    candidate_shapes = []
    drop_ids = list(itertools.combinations(range(num_faces), num_drop))

    for drop_id in drop_ids:
        preserved_ids = np.array(list(set(range(num_faces)) - set(drop_id)))
        prev_id_to_new_id = {prev_id: new_id for new_id, prev_id in enumerate(preserved_ids)}
        new_faces = [v_faces[idx] for idx in preserved_ids]
        new_curves = [item for item in v_curves]
        new_edge_face_connectivity = []
        for edge_id, face_id1, face_id2 in v_conn:
            if face_id1 in preserved_ids and face_id2 in preserved_ids:
                new_edge_face_connectivity.append([edge_id, prev_id_to_new_id[face_id1], prev_id_to_new_id[face_id2]])
        candidate_shapes.append((new_faces, new_curves, new_edge_face_connectivity))
    return candidate_shapes


def construct_brep_from_datanpz(data_root, out_root, folder_name, v_drop_num=0,
                                is_ray=False, is_log=True,
                                is_optimize_geom=True, v_max_optimize_iter=200,
                                isdebug=False, use_cuda=False, from_scratch=True,
                                is_save_data=False):
    disable_occ_log()
    # is_log = False
    # isdebug = False
    time_records = [0, 0, 0, 0, 0, 0]
    timer = time.time()
    data_root = Path(data_root)
    out_root = Path(out_root)
    if from_scratch:
        check_dir(out_root / folder_name)

    # Check if it is already processed
    if (out_root / folder_name / "success.txt").exists():
        return time_records
    safe_check_dir(out_root / folder_name)

    debug_face_save_path = out_root / folder_name / "debug_face_loop"
    if is_save_data:
        safe_check_dir(debug_face_save_path)

    if is_log:
        print(
                f"{Colors.GREEN}############################# Processing {folder_name} #############################{Colors.RESET}")

    # Prepare the data
    shape = get_data(os.path.join(data_root, folder_name, 'data.npz'))
    if isdebug:
        export_edges(shape.recon_edge_points, debug_face_save_path / 'edge_ori.obj')
    shape.remove_half_edges()
    shape.check_openness()
    shape.build_fe()
    shape.build_vertices(0.2)

    if isdebug:
        print(
                f"{Colors.GREEN}Remove {len(shape.remove_edge_idx_src) + len(shape.remove_edge_idx_new)} edges{Colors.RESET}")

    if is_save_data:
        export_point_cloud(os.path.join(debug_face_save_path, 'face.ply'), shape.recon_face_points.reshape(-1, 3))
        updated_edge_points = np.delete(shape.recon_edge_points, shape.remove_edge_idx_new, axis=0)
        export_edges(updated_edge_points, os.path.join(debug_face_save_path, 'edge.obj'))
        for face_idx in range(len(shape.face_edge_adj)):
            export_point_cloud(os.path.join(debug_face_save_path, f"face{face_idx}.ply"),
                               shape.recon_face_points[face_idx].reshape(-1, 3))
            for edge_idx in shape.face_edge_adj[face_idx]:
                idx = np.where(shape.edge_face_connectivity[:, 0] == edge_idx)[0][0]
                adj_face = shape.edge_face_connectivity[idx][1:]
                export_point_cloud(
                        os.path.join(debug_face_save_path, f"face{face_idx}_edge_idx{edge_idx}_face{adj_face}.ply"),
                        shape.recon_edge_points[edge_idx].reshape(-1, 3),
                        np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))
        for edge_idx in range(len(shape.recon_edge_points)):
            if edge_idx in shape.remove_edge_idx_new:
                continue
            export_point_cloud(os.path.join(
                    debug_face_save_path, f'edge{edge_idx}.ply'),
                    shape.recon_edge_points[edge_idx].reshape(-1, 3),
                    np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))

    # Optimize data
    if is_optimize_geom:
        interpolation_face = []
        for item in shape.interpolation_face:
            interpolation_face.append(item)

        if not is_ray:
            shape.recon_face_points, shape.recon_edge_points = optimize(
                    interpolation_face, shape.recon_edge_points, shape.recon_face_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_islog=isdebug, v_max_iter=v_max_optimize_iter, use_cuda=use_cuda)
        else:
            shape.recon_face_points, shape.recon_edge_points = optimize(
                    shape.interpolation_face, shape.recon_edge_points, shape.recon_face_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_islog=False, v_max_iter=v_max_optimize_iter, use_cuda=use_cuda)

        if is_save_data:
            updated_edge_points = np.delete(shape.recon_edge_points, shape.remove_edge_idx_new, axis=0)
            export_edges(updated_edge_points, os.path.join(debug_face_save_path, 'optimized_edge.obj'))
            for face_idx in range(len(shape.face_edge_adj)):
                for edge_idx in shape.face_edge_adj[face_idx]:
                    idx = np.where(shape.edge_face_connectivity[:, 0] == edge_idx)[0][0]
                    adj_face = shape.edge_face_connectivity[idx][1:]
                    export_point_cloud(
                            os.path.join(debug_face_save_path,
                                         f"face{face_idx}_optim_edge_idx{edge_idx}_face{adj_face}.ply"),
                            shape.recon_edge_points[edge_idx].reshape(-1, 3),
                            np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))
                export_point_cloud(debug_face_save_path / f'optim_face{face_idx}.ply',
                                   shape.recon_face_points[face_idx].reshape(-1, 3))
            for edge_idx in range(len(shape.recon_edge_points)):
                if edge_idx in shape.remove_edge_idx_new:
                    continue
                export_point_cloud(
                        os.path.join(debug_face_save_path, f'optim_edge{edge_idx}.ply'),
                        shape.recon_edge_points[edge_idx].reshape(-1, 3),
                        np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))

    ori_shape = copy.deepcopy(shape)

    recon_geom_faces = [create_surface(points) for points in shape.recon_face_points]
    recon_topo_faces = [
        BRepBuilderAPI_MakeFace(geom_face, TRANSFER_PRECISION).Face() for geom_face in recon_geom_faces]
    recon_geom_curves = [create_edge(points) for points in shape.recon_edge_points]
    recon_topo_curves = [BRepBuilderAPI_MakeEdge(curve).Edge() for curve in recon_geom_curves]

    shape.recon_geom_faces = [item for item in recon_geom_faces]
    shape.recon_topo_faces = [item for item in recon_topo_faces]
    shape.recon_geom_curves = [item for item in recon_geom_curves]
    shape.recon_topo_curves = [item for item in recon_topo_curves]
    shape.build_geom(is_replace_edge=True)
    recon_topo_curves = [item for item in shape.recon_topo_curves]

    # Write separate faces
    v, f = get_separated_surface(shape.recon_topo_faces, v_precision1=0.1, v_precision2=0.2)
    trimesh.Trimesh(vertices=v, faces=f).export(out_root / folder_name / "separate_faces.ply")

    num_max_drop = min(v_drop_num, math.ceil(0.2 * len(ori_shape.recon_face_points)))
    is_success = False

    for num_drop in range(num_max_drop + 1):
        candidate_shapes = get_candidate_shapes(num_drop, recon_geom_faces, recon_topo_curves, ori_shape.edge_face_connectivity)

        for (faces, curves, connectivity) in candidate_shapes:
            if len(faces) == 0:
                if is_log:
                    print(f"{Colors.RED}No data in {folder_name}{Colors.RESET}")
                # shutil.rmtree(os.path.join(out_root, folder_name))
                continue

            num_faces = len(faces)
            face_edge_adj = [[] for _ in range(num_faces)]
            for edge_face1_face2 in connectivity:
                edge, face1, face2 = edge_face1_face2
                if face1 == face2:
                    # raise ValueError("Face1 and Face2 should be different")
                    print("Face1 and Face2 should be different")
                    continue
                assert edge not in face_edge_adj[face1]
                face_edge_adj[face1].append(edge)
                face_edge_adj[face2].append(edge)

            # Construct trimmed surface
            trimmed_faces = []
            for i_face in range(num_faces):
                if len(face_edge_adj[i_face]) == 0:
                    trimmed_faces.append(None)
                    continue
                face_edge_idx = face_edge_adj[i_face]
                geom_face = faces[i_face]
                face_edges = [curves[edge_idx] for edge_idx in face_edge_idx]

                # Build wire
                trimmed_face = None
                for threshold in CONNECT_TOLERANCE:
                    wire_list = create_wire_from_unordered_edges(face_edges, threshold)
                    if wire_list is None:
                        continue

                    trimmed_face = create_trimmed_face_from_wire(geom_face, face_edges, wire_list, threshold)
                    if trimmed_face is not None:
                        break

                trimmed_faces.append(trimmed_face)

            trimmed_faces = [face for face in trimmed_faces if face is not None]
            if len(trimmed_faces) < 0.8 * num_faces:
                continue

            # Try construct solid from trimmed faces only
            solid = None
            if len(trimmed_faces) > 0.8 * num_faces:
                for connected_tolerance in CONNECT_TOLERANCE:
                    if is_log:
                        print(f"Try construct solid with {connected_tolerance}")
                    solid = get_solid(trimmed_faces, connected_tolerance)
                    if solid is not None:
                        break

            # Check solid
            if solid is not None:
                save_step_file(out_root / folder_name / 'recon_brep.step', solid)
                if not check_step_valid_soild(str(out_root / folder_name / 'recon_brep.step')):
                    # print("Inconsistent solid check in {}".format(folder_name))
                    os.remove(out_root / folder_name / 'recon_brep.step')
                else:
                    write_stl_file(solid, str(out_root / folder_name / "recon_brep.stl"),
                                   linear_deflection=0.1, angular_deflection=0.2)
                    open(out_root / folder_name / "success.txt", 'w').close()
                    is_success = True
                    break
        if is_success:
            break

    # If solid is None, then try to obtain step file with all faces
    if not is_success:
        # Construct trimmed surface
        num_faces = len(recon_topo_faces)
        face_edge_adj = [[] for _ in range(num_faces)]
        for edge_face1_face2 in ori_shape.edge_face_connectivity:
            edge, face1, face2 = edge_face1_face2
            if face1 == face2:
                # raise ValueError("Face1 and Face2 should be different")
                print("Face1 and Face2 should be different")
                continue
            assert edge not in face_edge_adj[face1]
            face_edge_adj[face1].append(edge)
            face_edge_adj[face2].append(edge)

        trimmed_faces = []
        for i_face in range(num_faces):
            if len(face_edge_adj[i_face]) == 0:
                trimmed_faces.append(None)
                continue
            face_edge_idx = face_edge_adj[i_face]
            geom_face = recon_geom_faces[i_face]
            face_edges = [recon_topo_curves[edge_idx] for edge_idx in face_edge_idx]

            # Build wire
            trimmed_face = None
            for threshold in CONNECT_TOLERANCE:
                wire_list = create_wire_from_unordered_edges(face_edges, threshold)
                if wire_list is None:
                    continue

                trimmed_face = create_trimmed_face_from_wire(geom_face, face_edges, wire_list, threshold)
                if trimmed_face is not None:
                    break

            trimmed_faces.append(trimmed_face)

        mixed_faces = []
        for i_face in range(num_faces):
            if trimmed_faces[i_face] is None:
                face = BRepBuilderAPI_MakeFace(recon_geom_faces[i_face], TRANSFER_PRECISION).Face()
                mixed_faces.append(face)
            else:
                mixed_faces.append(trimmed_faces[i_face])

        # trimmed_faces = [face for face in trimmed_faces if face is not None]
        # if len(trimmed_faces) < 0.8 * num_faces:
        #     continue

        compound = None
        for connected_tolerance in CONNECT_TOLERANCE:
            compound = get_compound(mixed_faces, connected_tolerance)
            if compound is not None:
                break

        if compound is not None:
            save_step_file(out_root / folder_name / 'recon_brep.step', compound)
        else:
            print(f"Failed to construct solid in {folder_name}")
    return time_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test")
    parser.add_argument('--list', type=str, default="")
    parser.add_argument('--out_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test_out")
    parser.add_argument('--num_cpus', type=int, default=12)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--max_optimize_iter', type=int, default=200)
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--drop_num', type=int, default=0)
    args = parser.parse_args()
    v_data_root = args.data_root
    v_out_root = args.out_root
    filter_list = args.list
    is_use_ray = args.use_ray
    num_cpus = args.num_cpus
    use_cuda = args.use_cuda
    max_optimize_iter = int(args.max_optimize_iter)
    from_scratch = args.from_scratch
    drop_num = args.drop_num
    safe_check_dir(v_out_root)
    if not os.path.exists(v_data_root):
        raise ValueError(f"Data root path {v_data_root} does not exist.")

    if args.prefix != "":
        construct_brep_from_datanpz(v_data_root, v_out_root, args.prefix,
                                    v_drop_num=drop_num,
                                    use_cuda=use_cuda,
                                    is_optimize_geom=True,
                                    v_max_optimize_iter=max_optimize_iter,
                                    isdebug=True, is_save_data=True, )
        exit()
    all_folders = [folder for folder in os.listdir(v_data_root) if os.path.isdir(os.path.join(v_data_root, folder))]
    if filter_list != "":
        print(f"Use filter_list {filter_list}")
        if not os.path.exists(filter_list):
            raise ValueError(f"List {filter_list} does not exist.")
        if os.path.isdir(filter_list):
            valid_prefies = [f for f in os.listdir(filter_list) if os.path.isdir(os.path.join(filter_list, f))]
        elif filter_list.endswith(".txt"):
            valid_prefies = [item.strip() for item in open(filter_list).readlines()]
        else:
            raise ValueError(f"Invalid list {filter_list}")
        all_folders = list(set(all_folders) & set(valid_prefies))

    all_folders.sort()
    all_folders = all_folders

    print(f"Total {len(all_folders)} folders")

    if not is_use_ray:
        # random.shuffle(all_folders)
        for i in tqdm(range(len(all_folders))):
            construct_brep_from_datanpz(v_data_root, v_out_root, all_folders[i],
                                        v_drop_num=drop_num,
                                        use_cuda=use_cuda, from_scratch=from_scratch,
                                        is_save_data=False, is_log=False,
                                        is_optimize_geom=True, v_max_optimize_iter=max_optimize_iter,
                                        is_ray=False, )
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8080,
                num_cpus=num_cpus,
                # num_gpus=num_gpus,
                # local_mode=True
        )
        construct_brep_from_datanpz_ray = ray.remote(num_gpus=0.1 if use_cuda else 0, max_retries=0)(
                construct_brep_from_datanpz)

        tasks = []
        timeout_cancel_list = []
        for i in range(len(all_folders)):
            tasks.append(construct_brep_from_datanpz_ray.remote(
                    v_data_root, v_out_root,
                    all_folders[i],
                    v_drop_num=drop_num,
                    use_cuda=use_cuda, from_scratch=from_scratch,
                    is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
            ))
        results = []
        for i in tqdm(range(len(all_folders))):
            try:
                results.append(ray.get(tasks[i], timeout=30))
            except ray.exceptions.GetTimeoutError:
                results.append(None)
                timeout_cancel_list.append(all_folders[i])
                ray.cancel(tasks[i])
                print(f"Cancel {all_folders[i]} for timeout")
            except:
                results.append(None)
        results = [item for item in results if item is not None]
        print(f"Cancel for timeout: {timeout_cancel_list}")
        print(len(results))
        results = np.array(results)
        print(results.mean(axis=0))
    print("Done")
