import copy
import os, sys, shutil, traceback

import numpy as np
import torch
from OCC.Core import Message
from OCC.Core.IFSelect import IFSelect_ReturnStatus
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.Interface import Interface_Static
from OCC.Core.Message import Message_PrinterOStream, Message_Alarm
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_ManifoldSolidBrep, \
    STEPControl_FacetedBrep, STEPControl_ShellBasedSurfaceModel
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopAbs import TopAbs_SHAPE
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Extend.DataExchange import read_step_file

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud
from shared.occ_utils import get_primitives
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

    shape = Shape(face_points, edge_points, edge_face_connectivity, False)
    return shape


def construct_brep_from_datanpz(data_root, out_root, folder_name,
                                is_ray=False, is_log=True,
                                is_optimize_geom=True, isdebug=False, use_cuda=False, from_scratch=True, is_save_data=False):
    time_records = [0, 0, 0, 0, 0, 0]
    timer = time.time()
    if not from_scratch and os.path.exists(os.path.join(out_root, folder_name + "/success.txt")):
        return time_records
    # print(folder_name)

    if os.path.exists(os.path.join(out_root, folder_name + "/success.txt")):
        os.remove(os.path.join(out_root, folder_name + "/success.txt"))

    printers = Message.message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)
    if is_log:
        print(f"{Colors.GREEN}############################# "
              f"Processing {folder_name} #############################{Colors.RESET}")

    debug_face_save_path = None
    if is_save_data:
        debug_face_save_path = str(os.path.join(out_root, folder_name, "debug_face_loop"))
        safe_check_dir(debug_face_save_path)

    if from_scratch:
        safe_check_dir(os.path.join(out_root, folder_name))
    else:
        safe_check_dir(os.path.join(out_root, folder_name))

    shape = get_data(os.path.join(data_root, folder_name, 'data.npz'))

    # if shape.recon_face_points.shape[0] < 10:
    #     shutil.rmtree(os.path.join(out_root, folder_name))
    #     return

    if isdebug:
        export_edges(shape.recon_edge_points, os.path.join(debug_face_save_path, 'edge_ori.obj'))
    shape.remove_half_edges()
    shape.check_openness()
    shape.build_fe()
    shape.build_vertices(0.2)
    # shape.remove_isolated_edges()
    # if isdebug:
    #     export_edges(shape.recon_edge_points, os.path.join(debug_face_save_path, 'edge_before_drop.obj'))
    # shape.drop_edges(max_drop_num=2)
    # if isdebug:
    #     export_edges(shape.recon_edge_points, os.path.join(debug_face_save_path, 'edge_after_drop1.obj'))

    if not shape.have_data:
        if is_log:
            print(f"{Colors.RED}No data in {folder_name}{Colors.RESET}")
        # shutil.rmtree(os.path.join(out_root, folder_name))
        return [0, 0, 0, 0, 0, 0]

    if isdebug:
        print(f"{Colors.GREEN}Remove {len(shape.remove_edge_idx_src) + len(shape.remove_edge_idx_new)} edges{Colors.RESET}")

    time_records[0] = time.time() - timer
    timer = time.time()

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

    if is_optimize_geom:
        interpolation_face = []
        for item in shape.interpolation_face:
            interpolation_face.append(item)

        if not is_ray:
            shape.recon_face_points, shape.recon_edge_points = optimize(
                    interpolation_face, shape.recon_edge_points, shape.recon_face_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_islog=isdebug, v_max_iter=200, use_cuda=use_cuda)
        else:
            shape.recon_face_points, shape.recon_edge_points = optimize(
                    shape.interpolation_face, shape.recon_edge_points, shape.recon_face_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_islog=False, v_max_iter=200, use_cuda=use_cuda)
            # shape.recon_edge_points = ray.get(task)

        if is_save_data:
            updated_edge_points = np.delete(shape.recon_edge_points, shape.remove_edge_idx_new, axis=0)
            export_edges(updated_edge_points, os.path.join(debug_face_save_path, 'optimized_edge.obj'))
            for face_idx in range(len(shape.face_edge_adj)):
                for edge_idx in shape.face_edge_adj[face_idx]:
                    idx = np.where(shape.edge_face_connectivity[:, 0] == edge_idx)[0][0]
                    adj_face = shape.edge_face_connectivity[idx][1:]
                    export_point_cloud(
                            os.path.join(debug_face_save_path, f"face{face_idx}_optim_edge_idx{edge_idx}_face{adj_face}.ply"),
                            shape.recon_edge_points[edge_idx].reshape(-1, 3),
                            np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))
            for edge_idx in range(len(shape.recon_edge_points)):
                if edge_idx in shape.remove_edge_idx_new:
                    continue
                export_point_cloud(
                        os.path.join(debug_face_save_path, f'optim_edge{edge_idx}.ply'),
                        shape.recon_edge_points[edge_idx].reshape(-1, 3),
                        np.linspace([1, 0, 0], [0, 1, 0], shape.recon_edge_points[edge_idx].shape[0]))

    time_records[1] = time.time() - timer
    timer = time.time()

    shape.build_geom(is_replace_edge=True)
    if isdebug:
        print(f"{Colors.GREEN}{len(shape.replace_edge_idx)} edges are replace{Colors.RESET}")

    # Construct Brep from face_points, edge_points, face_edge_adj
    solid = None
    mesh = None
    is_successful = False
    for i in range(len(CONNECT_TOLERANCE)):
        connected_tolerance = CONNECT_TOLERANCE[i]
        if is_log:
            print(f"Try connected_tolerance {connected_tolerance}")
        # solid, faces_result = construct_brep(shape, connected_tolerance,
        #                                      isdebug=isdebug, is_save_face=True,
        #                                      folder_path=os.path.join(out_root, folder_name))
        is_face_success_list, mesh, solid = construct_brep(shape, connected_tolerance, isdebug=isdebug)

        # Check
        if mesh is None or solid is None:
            continue
        if solid.ShapeType() == TopAbs_COMPOUND:
            continue

        shape_tol_setter = ShapeFix_ShapeTolerance()
        shape_tol_setter.SetTolerance(solid, 1e-1)
        analyzer = BRepCheck_Analyzer(solid)
        if not analyzer.IsValid():
            result = analyzer.Result(solid)
            continue

        if solid.ShapeType() == TopAbs_SOLID and analyzer.IsValid():
            is_successful = True
        else:
            is_successful = False

        save_step_file(os.path.join(out_root, folder_name, 'recon_brep.step'), solid)
        if not check_step_valid_soild(os.path.join(out_root, folder_name, 'recon_brep.step')):
            continue
        break

    time_records[2] = time.time() - timer
    timer = time.time()

    if mesh is not None:
        mesh.export(os.path.join(out_root, folder_name, 'recon_brep.ply'))

    if solid is not None:
        try:
            save_step_file(os.path.join(out_root, folder_name, 'recon_brep.step'), solid)
            if check_step_valid_soild(os.path.join(out_root, folder_name, 'recon_brep.step')):
                open(os.path.join(out_root, folder_name, 'success.txt'), 'w').close()
                write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'), linear_deflection=0.01,
                               angular_deflection=0.1)
        except:
            pass

    time_records[3] = time.time() - timer
    timer = time.time()
    return time_records
    # if not is_successful:
    #     shutil.rmtree(os.path.join(out_root, folder_name))


def test_construct_brep(v_data_root, v_out_root, v_prefix, use_cuda):
    # debug_folder = os.listdir(v_out_root)
    debug_folder = [v_prefix]
    for folder in debug_folder:
        construct_brep_from_datanpz(v_data_root, v_out_root, folder,
                                    use_cuda=use_cuda, is_optimize_geom=True, isdebug=True, is_save_data=True, )
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test")
    parser.add_argument('--list', type=str, default="")
    parser.add_argument('--out_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test_out")
    parser.add_argument('--is_cover', type=bool, default=False)
    parser.add_argument('--num_cpus', type=int, default=12)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    args = parser.parse_args()
    v_data_root = args.data_root
    v_out_root = args.out_root
    is_cover = args.is_cover
    filter_list = args.list
    is_use_ray = args.use_ray
    num_cpus = args.num_cpus
    use_cuda = args.use_cuda
    from_scratch = args.from_scratch
    safe_check_dir(v_out_root)
    if not os.path.exists(v_data_root):
        raise ValueError(f"Data root path {v_data_root} does not exist.")

    if args.prefix != "":
        test_construct_brep(v_data_root, v_out_root, args.prefix, use_cuda)
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
    # all_folders = os.listdir(r"E:\data\img2brep\.43\2024_09_22_21_57_44_0921_pure_out3_failed")
    # check_dir(v_out_root)

    # print(f"Total {len(all_folders)} folders")
    # if not is_cover:
    # print(f"Skip existing folders")
    # all_folders = [folder for folder in all_folders if not os.path.exists(os.path.join(v_out_root, folder))]
    # print(f"Total {len(all_folders)} folders to process")

    all_folders.sort()
    # all_folders = all_folders[:100]

    print(f"Total {len(all_folders)} folders")

    if not is_use_ray:
        # random.shuffle(all_folders)
        for i in tqdm(range(len(all_folders))):
            construct_brep_from_datanpz(v_data_root, v_out_root, all_folders[i],
                                        use_cuda=use_cuda, from_scratch=from_scratch,
                                        is_save_data=False, is_log=False, is_optimize_geom=True, is_ray=False, )
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8080,
                num_cpus=num_cpus,
                # num_gpus=num_gpus,
                # local_mode=True
        )
        construct_brep_from_datanpz_ray = ray.remote(num_gpus=0.1 if use_cuda else 0, max_retries=0)(construct_brep_from_datanpz)

        tasks = []
        for i in range(len(all_folders)):
            tasks.append(construct_brep_from_datanpz_ray.remote(
                    v_data_root, v_out_root,
                    all_folders[i],
                    use_cuda=use_cuda, from_scratch=from_scratch,
                    is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
            ))
        results = []
        for i in tqdm(range(len(all_folders))):
            results.append(ray.get(tasks[i]))
        results = [item for item in results if item is not None]
        print(len(results))
        results = np.array(results)
        print(results.mean(axis=0))
    print("Done")
