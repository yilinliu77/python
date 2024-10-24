import copy
import os, sys, shutil, traceback

import numpy as np
import torch
from OCC.Core import Message
from OCC.Core.IFSelect import IFSelect_ReturnStatus
from OCC.Core.Message import Message_PrinterOStream, Message_Alarm
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Extend.DataExchange import read_step_file

from shared.common_utils import safe_check_dir, check_dir
from shared.common_utils import export_point_cloud
from shared.occ_utils import get_primitives

from src.brepnet.post.utils import *
from src.brepnet.post.geom_optimization import optimize_geom, test_optimize_geom

import ray
import argparse
import trimesh

import time


@ray.remote(num_gpus=0.05)
def optimize_geom_ray(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, is_use_cuda, is_log=False,
                      max_iter=100):
    return optimize_geom(recon_face, recon_edge, edge_face_connectivity, face_edge_adj, is_use_cuda, max_iter,
                         is_log=is_log)


def construct_brep_item(face_points, edge_points, edge_face_connectivity, ):
    # face_edge_adj store the edge idx list of each face
    face_edge_adj = [[] for _ in range(face_points.shape[0])]
    for edge_face1_face2 in edge_face_connectivity:
        edge, face1, face2 = edge_face1_face2
        assert edge not in face_edge_adj[face1]
        face_edge_adj[face1].append(edge)

    if False:
        with torch.set_grad_enabled(True):
            face_points, edge_points, edge_face_connectivity, face_edge_adj, remove_edge_idx = optimize_geom(
                    face_points, edge_points,
                    edge_face_connectivity,
                    face_edge_adj,
                    is_use_cuda=True,
                    max_iter=150)

    connected_tolerances = copy.deepcopy(CONNECT_TOLERANCE)
    solid = None
    printers = Message.message.DefaultMessenger().Printers()
    for idx in range(printers.Length()):
        printers.Value(idx + 1).SetTraceLevel(Message_Alarm)
    while len(connected_tolerances) > 0:
        connected_tolerance = connected_tolerances.pop()
        solid, faces_result = construct_brep(face_points, edge_points, face_edge_adj, connected_tolerance,
                                             isdebug=False, is_save_face=False,
                                             folder_path="1")
        if solid is None:
            continue

        if solid.ShapeType() == TopAbs_COMPOUND:
            continue

        analyzer = BRepCheck_Analyzer(solid)
        if not analyzer.IsValid():
            continue

        # Valid Solid
        write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
        write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'))
        break

    if solid is None:
        print(f"solid is None {folder_name}")
        return

    if solid.ShapeType() == TopAbs_COMPOUND:
        print(f"solid is TopAbs_COMPOUND {folder_name}")
        # write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_compound.stl'))
        # recon_face_dir = os.path.join(out_root, folder_name, 'recon_face')
        # gen_mesh = trimesh.util.concatenate(
        #         [trimesh.load(os.path.join(recon_face_dir, f)) for f in os.listdir(recon_face_dir) if f.endswith('.stl')])
        # gen_mesh.export(os.path.join(out_root, folder_name, 'recon_brep_compound.stl'))
        return

    analyzer = BRepCheck_Analyzer(solid)
    if not analyzer.IsValid():
        print(f"solid is invalid {folder_name}")
        # write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep_invalid.stl'))
        write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep_invalid.step'))
        # recon_face_dir = os.path.join(out_root, folder_name, 'recon_face')
        # gen_mesh = trimesh.util.concatenate(
        #         [trimesh.load(os.path.join(recon_face_dir, f)) for f in os.listdir(recon_face_dir) if f.endswith('.stl')])
        # gen_mesh.export(os.path.join(out_root, folder_name, 'recon_brep_invalid.stl'))
        return

    # Valid Solid
    write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
    write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'))


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
                                is_optimize_geom=True, isdebug=False, use_cuda=False, from_scratch=False, is_save_data=False):
    if not from_scratch and os.path.exists(os.path.join(out_root, folder_name + "/success.txt")):
        return

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
        return

    if isdebug:
        print(f"{Colors.GREEN}Remove {len(shape.remove_edge_idx_src) + len(shape.remove_edge_idx_new)} edges{Colors.RESET}")

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
            interpolation_face.append(item.cpu().numpy())

        if not is_ray:
            shape.recon_edge_points = optimize(
                    interpolation_face, shape.recon_edge_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_islog=isdebug, v_max_iter=200)
        else:
            task = optimize_ray.remote(
                    interpolation_face, shape.recon_edge_points,
                    shape.edge_face_connectivity, shape.is_end_point, shape.pair1,
                    shape.face_edge_adj, v_max_iter=200)
            shape.recon_edge_points = ray.get(task)

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

    # Construct Brep from face_points, edge_points, face_edge_adj
    solid = None
    mesh = None
    is_successful = False
    for i in range(len(CONNECT_TOLERANCE)):
        connected_tolerance = CONNECT_TOLERANCE[i]
        # solid, faces_result = construct_brep(shape, connected_tolerance,
        #                                      isdebug=isdebug, is_save_face=True,
        #                                      folder_path=os.path.join(out_root, folder_name))
        is_face_success_list, mesh, solid = construct_brep(shape, connected_tolerance, isdebug=isdebug)

        # Check
        if mesh is None or solid is None:
            continue
        if solid.ShapeType() == TopAbs_COMPOUND:
            continue

        analyzer = BRepCheck_Analyzer(solid)
        # analyzer.SetExactMethod(True)
        if not analyzer.IsValid():
            result = analyzer.Result(solid)
            continue

        if solid.ShapeType() == TopAbs_SOLID and analyzer.IsValid():
            is_successful = True
        else:
            is_successful = False
        break

    if solid is not None:
        try:
            write_step_file(solid, os.path.join(out_root, folder_name, 'recon_brep.step'))
        except:
            pass
    if mesh is not None:
        mesh.export(os.path.join(out_root, folder_name, 'recon_brep.ply'))
    if is_successful:
        open(os.path.join(out_root, folder_name, 'success.txt'), 'w').close()
        try:
            write_stl_file(solid, os.path.join(out_root, folder_name, 'recon_brep.stl'), linear_deflection=0.01, angular_deflection=0.5)
        except:
            pass
    # if not is_successful:
    #     shutil.rmtree(os.path.join(out_root, folder_name))


def construct_brep_from_datanpz_batch(data_root, out_root, folder_name_list,
                                      use_cuda=False,
                                      is_optimize_geom=True,
                                      from_scratch=False,
                                      ):
    for folder_name in folder_name_list:
        try:
            construct_brep_from_datanpz(data_root, out_root, folder_name,
                                        is_log=False,
                                        is_ray=True, is_optimize_geom=is_optimize_geom,
                                        isdebug=False, use_cuda=use_cuda, from_scratch=from_scratch)
        except Exception as e:
            with open(os.path.join(out_root, "error.txt"), "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(folder_name + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(folder_name + ": " + str(e))
                print(e)
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                shutil.rmtree(os.path.join(out_root, folder_name))


construct_brep_from_datanpz_batch_ray = ray.remote(max_retries=2)(construct_brep_from_datanpz_batch)


def test_construct_brep(v_data_root, v_out_root, v_prefix, use_cuda, from_scratch):
    # debug_folder = os.listdir(v_out_root)
    debug_folder = [v_prefix]
    for folder in debug_folder:
        construct_brep_from_datanpz(v_data_root, v_out_root, folder,
                                    use_cuda=use_cuda, is_optimize_geom=True, isdebug=True, is_save_data=True,
                                    from_scratch=from_scratch)
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--data_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test")
    parser.add_argument('--list_file', type=str, default="")
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
    list_file = args.list_file
    is_use_ray = args.use_ray
    num_cpus = args.num_cpus
    use_cuda = args.use_cuda
    from_scratch = args.from_scratch
    safe_check_dir(v_out_root)
    if not os.path.exists(v_data_root):
        raise ValueError(f"Data root path {v_data_root} does not exist.")

    if args.prefix != "":
        test_construct_brep(v_data_root, v_out_root, args.prefix, use_cuda, from_scratch=from_scratch)
    all_folders = [folder for folder in os.listdir(v_data_root) if os.path.isdir(os.path.join(v_data_root, folder))]
    if list_file != "":
        valid_prefies = [item.strip() for item in open(list_file).readlines()]
        all_folders = list(set(all_folders) & set(valid_prefies))
    # all_folders = os.listdir(r"E:\data\img2brep\.43\2024_09_22_21_57_44_0921_pure_out3_failed")
    # check_dir(v_out_root)

    # print(f"Total {len(all_folders)} folders")
    # if not is_cover:
    # print(f"Skip existing folders")
    # all_folders = [folder for folder in all_folders if not os.path.exists(os.path.join(v_out_root, folder))]
    # print(f"Total {len(all_folders)} folders to process")

    all_folders.sort()
    # all_folders = all_folders[:50]

    if not is_use_ray:
        # random.shuffle(all_folders)
        for i in tqdm(range(len(all_folders))):
            construct_brep_from_datanpz(v_data_root, v_out_root, all_folders[i], use_cuda=use_cuda, from_scratch=from_scratch)
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8080,
                # num_cpus=1,
                num_cpus=num_cpus,
                # local_mode=True
        )
        batch_size = 1
        num_batches = len(all_folders) // batch_size + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(construct_brep_from_datanpz_batch_ray.remote(
                    v_data_root, v_out_root,
                    all_folders[i * batch_size:(i + 1) * batch_size],
                    use_cuda=use_cuda, from_scratch=from_scratch))
        ray.get(tasks)
    print("Done")
