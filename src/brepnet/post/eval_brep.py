import time, os, random, traceback, sys
import torch
import numpy as np

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from tqdm import tqdm
import trimesh
import argparse

# import pandas as pd
from chamferdist import ChamferDistance

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.DataExchange import read_step_file, write_step_file, write_stl_file
from OCC.Core.BRepCheck import BRepCheck_Analyzer

import ray
import shutil

from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Shell
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL, TopAbs_SOLID

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

shape_type = ["valid_solid", "invalid_solid", "shell", "compound", "else"]


def is_vertex_close(p1, p2, tol):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def get_edge_length(edge, NUM_SEGMENTS=100):
    curve_data = BRep_Tool.Curve(edge)
    if curve_data and len(curve_data) == 3:
        curve_handle, first, last = curve_data
        segment_length = (last - first) / NUM_SEGMENTS
        edge_length = 0
        for i in range(NUM_SEGMENTS):
            u1 = first + segment_length * i
            u2 = first + segment_length * (i + 1)
            edge_length += np.linalg.norm(np.array(curve_handle.Value(u1).Coord()) - np.array(curve_handle.Value(u2).Coord()))
        return edge_length


def read_step_and_get_data(step_file_path, NUM_SAMPLE_EDGE_UNIT=100):
    if not os.path.exists(step_file_path):
        return None, None

    shape = read_step_file(step_file_path, verbosity=False)

    # face
    # face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # faces = []
    # while face_explorer.More():
    #     face = face_explorer.Current()
    #     faces.append(face)
    #     face_explorer.Next()
    # gen_faces = np.stack(faces)

    # vertex
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertexes = []
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        point = BRep_Tool.Pnt(vertex)
        # check if the point is close to the previous point
        is_close = False
        for v in vertexes:
            if is_vertex_close(v, (point.X(), point.Y(), point.Z()), 1e-4):
                is_close = True
                break
        if not is_close:
            vertexes.append((point.X(), point.Y(), point.Z()))
        vertex_explorer.Next()

    # edge
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    topods_edge_list = []
    edge_points = []
    while edge_explorer.More():
        edge = edge_explorer.Current()
        is_saved = False
        for each in topods_edge_list:
            if each.IsEqual(edge) or each.IsEqual(edge.Reversed()):
                is_saved = True
                break
        if not is_saved:
            topods_edge_list.append(edge)
            curve_data = BRep_Tool.Curve(edge)
            if curve_data and len(curve_data) == 3:
                curve_handle, first, last = curve_data
            else:
                continue

            NUM_SAMPLE_EDGE = int(get_edge_length(edge, NUM_SEGMENTS=32) * NUM_SAMPLE_EDGE_UNIT)
            u_values = np.linspace(first, last, NUM_SAMPLE_EDGE)
            points_on_edge = []
            for u in u_values:
                point = curve_handle.Value(u)
                points_on_edge.append((point.X(), point.Y(), point.Z()))
            if len(points_on_edge) != 0:
                points_on_edge = np.array(points_on_edge)
                points_on_edge.reshape(-1, 3)
                edge_points.append(points_on_edge)
        edge_explorer.Next()
    return vertexes, edge_points


def eval_one(eval_root, gt_root, folder_name, SAMPLE_NUM=100000):
    if os.path.exists(os.path.join(eval_root, folder_name, 'error.txt')):
        os.remove(os.path.join(eval_root, folder_name, 'error.txt'))
    if os.path.exists(os.path.join(eval_root, folder_name, 'eval.npz')):
        os.remove(os.path.join(eval_root, folder_name, 'eval.npz'))

    result = {
        'num_recon_face'  : 0,
        'num_gt_face'     : 0,
        'face_acc_cd'     : [],
        'face_com_cd'     : [],
        'face_cd'         : [],

        'num_recon_edge'  : 0,
        'num_gt_edge'     : 0,
        'edge_acc_cd'     : [],
        'edge_com_cd'     : [],
        'edge_cd'         : [],

        'num_recon_vertex': 0,
        'num_gt_vertex'   : 0,
        'vertex_acc_cd'   : [],
        'vertex_com_cd'   : [],
        'vertex_cd'       : [],

        'stl_type'        : -1,
        'stl_acc_cd'      : 0,
        'stl_com_cd'      : 0,
        'stl_cd'          : 0,
    }
    device = torch.device('cpu')
    chamfer_distance = ChamferDistance()

    # gt info
    gt_mesh_path = os.path.join(gt_root, folder_name, 'mesh.ply')
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_pc, _ = trimesh.sample.sample_surface(gt_mesh, SAMPLE_NUM)
    gt_pc_tensor = torch.from_numpy(gt_pc).float().to(device)

    data_npz = np.load(os.path.join(gt_root, folder_name, 'data.npz'))
    result['num_gt_face'] = data_npz['sample_points_faces'].shape[0]
    # result['num_gt_edge'] = data_npz['sample_points_lines'].shape[0]
    # result['num_gt_vertex'] = data_npz['sample_points_vertices'].shape[0]

    gt_step_path = os.path.join(gt_root, folder_name, 'normalized_shape.step')
    gt_vertexes, gt_edge_points = read_step_and_get_data(gt_step_path)
    result['num_gt_edge'] = len(gt_edge_points)
    result['num_gt_vertex'] = len(gt_vertexes)
    gt_vertexes = np.stack(gt_vertexes, axis=0)
    gt_edge_points = np.concatenate(gt_edge_points, axis=0)
    gt_vertex_tensor = torch.from_numpy(gt_vertexes).float().to(device)
    gt_edge_tensor = torch.from_numpy(gt_edge_points).float().to(device)

    gen_step_name = [f for f in os.listdir(os.path.join(eval_root, folder_name)) if f.endswith('.step')]
    gen_stl_name = [f for f in os.listdir(os.path.join(eval_root, folder_name)) if f.endswith('.stl')]
    # if len(gen_step_name) == 0 or len(gen_stl_name) == 0:
    #     raise ValueError(f"Cannot find the step or stl file in {folder_name}")
    # if len(gen_step_name) > 1 or len(gen_stl_name) > 1:
    #     raise ValueError(f"More than one step or stl file in {folder_name}")

    # face
    recon_face_dir = os.path.join(eval_root, folder_name, 'recon_face')
    recon_face_stl_name = [f for f in os.listdir(recon_face_dir) if f.endswith('.stl')]
    if os.path.exists(recon_face_dir) and len(os.listdir(recon_face_dir)) != 0:
        result['num_recon_face'] = len(recon_face_stl_name)
        recon_face_stl_mesh = trimesh.util.concatenate([trimesh.load(os.path.join(recon_face_dir, f)) for f in recon_face_stl_name])
        recon_face_pc, _ = trimesh.sample.sample_surface(recon_face_stl_mesh, SAMPLE_NUM)
        recon_face_pc_tensor = torch.from_numpy(recon_face_pc).float().to(device)
        acc_cd = chamfer_distance(recon_face_pc_tensor.unsqueeze(0), gt_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        com_cd = chamfer_distance(gt_pc_tensor.unsqueeze(0), recon_face_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        cd = (acc_cd + com_cd) / 2
        result['face_acc_cd'].append(acc_cd)
        result['face_com_cd'].append(com_cd)
        result['face_cd'].append(cd)

    # read the edge and vertex from gen step file
    if os.path.exists(os.path.join(eval_root, folder_name, 'recon_brep.step')):
        step_file_path = os.path.join(eval_root, folder_name, 'recon_brep.step')
        gen_vertexes, gen_edge_points = read_step_and_get_data(step_file_path)
    else:
        gen_vertexes, gen_edge_points = None, None

    # edge
    if gen_edge_points is not None:
        result['num_recon_edge'] = len(gen_edge_points)
        gen_edge_points = np.concatenate(gen_edge_points, axis=0)
        recon_edge_pc_tensor = torch.from_numpy(gen_edge_points).float().to(device)
        acc_cd = chamfer_distance(recon_edge_pc_tensor.unsqueeze(0), gt_edge_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        com_cd = chamfer_distance(gt_edge_tensor.unsqueeze(0), recon_edge_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        cd = (acc_cd + com_cd) / 2
        result['edge_acc_cd'].append(acc_cd)
        result['edge_com_cd'].append(com_cd)
        result['edge_cd'].append(cd)

    # vertex
    if gen_vertexes is not None:
        result['num_recon_vertex'] = len(gen_vertexes)
        gen_vertexes = np.stack(gen_vertexes, axis=0)
        recon_vertex_pc_tensor = torch.from_numpy(gen_vertexes).float().to(device)
        acc_cd = chamfer_distance(recon_vertex_pc_tensor.unsqueeze(0), gt_vertex_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        com_cd = chamfer_distance(gt_vertex_tensor.unsqueeze(0), recon_vertex_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        cd = (acc_cd + com_cd) / 2
        result['vertex_acc_cd'].append(acc_cd)
        result['vertex_com_cd'].append(com_cd)
        result['vertex_cd'].append(cd)

    # check the vaildity of the step file
    if len(gen_step_name) != 0:
        try:
            gen_shape = read_step_file(os.path.join(eval_root, folder_name, gen_step_name[0]), as_compound=False, verbosity=False)
            shape_analyzer = BRepCheck_Analyzer(gen_shape)
            if gen_shape.ShapeType() == TopAbs_SOLID:
                if shape_analyzer.IsValid():
                    result['stl_type'] = shape_type[0]
                else:
                    result['stl_type'] = shape_type[1]
            elif gen_shape.ShapeType() == TopoDS_Shell:
                result['stl_type'] = shape_type[2]
            elif gen_shape.ShapeType() == TopAbs_COMPOUND:
                result['stl_type'] = shape_type[3]
            else:
                result['stl_type'] = shape_type[4]
        except Exception as e:
            result['stl_type'] = shape_type[4]
    else:
        result['stl_type'] = shape_type[4]

    # compute the chamfer distance between the stl and gt mesh
    if len(gen_stl_name) != 0:
        gen_mesh = trimesh.load(str(os.path.join(eval_root, folder_name, gen_stl_name[0])))
    else:
        recon_face_dir = os.path.join(eval_root, folder_name, 'recon_face')
        if os.path.exists(recon_face_dir) and len(os.listdir(recon_face_dir)) != 0:
            recon_face_stl_name = [f for f in os.listdir(recon_face_dir) if f.endswith('.stl')]
            gen_mesh = trimesh.util.concatenate([trimesh.load(os.path.join(recon_face_dir, f)) for f in recon_face_stl_name])
        else:
            gen_mesh = None

    if gen_mesh is not None:
        gen_pc, _ = trimesh.sample.sample_surface(gen_mesh, SAMPLE_NUM)
        gen_pc_tensor = torch.from_numpy(gen_pc).float().to(device)
        acc_cd = chamfer_distance(gen_pc_tensor.unsqueeze(0), gt_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        com_cd = chamfer_distance(gt_pc_tensor.unsqueeze(0), gen_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        result['stl_acc_cd'] = acc_cd
        result['stl_com_cd'] = com_cd
        result['stl_cd'] = (acc_cd + com_cd) / 2

    np.savez_compressed(os.path.join(eval_root, folder_name, 'eval.npz'), result=result, allow_pickle=True)


def eval_batch(eval_root, gt_root, folder_name_list, SAMPLE_NUM=100000):
    for folder_name in folder_name_list:
        try:
            eval_one(eval_root, gt_root, folder_name, SAMPLE_NUM)
        except Exception as e:
            with open(os.path.join(eval_root, "error.txt"), "a") as f:
                tb_list = traceback.extract_tb(sys.exc_info()[2])
                last_traceback = tb_list[-1]
                f.write(folder_name + ": " + str(e) + "\n")
                f.write(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")
                print(folder_name + ": " + str(e))
                print(e)
                print(f"An error occurred on line {last_traceback.lineno} in {last_traceback.name}\n\n")


eval_batch_remote = ray.remote(max_retries=3)(eval_batch)


def compute_statistics(eval_root):
    sum_recon_face, sum_gt_face = 0, 0
    sum_recon_edge, sum_gt_edge = 0, 0
    sum_recon_vertex, sum_gt_vertex = 0, 0

    all_face_acc_cd, all_face_com_cd, all_face_cd = [], [], []
    all_edge_acc_cd, all_edge_com_cd, all_edge_cd = [], [], []
    all_vertex_acc_cd, all_vertex_com_cd, all_vertex_cd = [], [], []

    all_stl_acc_cd, all_stl_com_cd, all_stl_cd = [], [], []
    num_valid_solid, num_invalid_solid, num_shell, num_compound, num_non_solid = 0, 0, 0, 0, 0

    valid_solid_acc_cd, valid_solid_com_cd, valid_solid_cd = [], [], []
    invalid_solid_acc_cd, invalid_solid_com_cd, invalid_solid_cd = [], [], []
    shell_acc_cd, shell_com_cd, shell_cd = [], [], []
    compound_acc_cd, compound_com_cd, compound_cd = [], [], []
    non_solid_acc_cd, non_solid_com_cd, non_solid_cd = [], [], []

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, folder))]
    exception_folders = []
    for folder_name in tqdm(all_folders):
        if not os.path.exists(os.path.join(eval_root, folder_name, 'eval.npz')):
            exception_folders.append(folder_name)
            continue

        result = np.load(os.path.join(eval_root, folder_name, 'eval.npz'), allow_pickle=True)['result'].item()
        if result['num_recon_face'] == 0:
            exception_folders.append(folder_name)
            continue

        # record the data
        sum_recon_face += int(result['num_recon_face'])
        sum_gt_face += int(result['num_gt_face'])
        sum_recon_edge += int(result['num_recon_edge'])
        sum_gt_edge += int(result['num_gt_edge'])
        sum_recon_vertex += int(result['num_recon_vertex'])
        sum_gt_vertex += int(result['num_gt_vertex'])

        all_face_acc_cd.extend(result['face_acc_cd'])
        all_face_com_cd.extend(result['face_com_cd'])
        all_face_cd.extend(result['face_cd'])

        all_edge_acc_cd.extend(result['edge_acc_cd'])
        all_edge_com_cd.extend(result['edge_com_cd'])
        all_edge_cd.extend(result['edge_cd'])

        all_vertex_acc_cd.extend(result['vertex_acc_cd'])
        all_vertex_com_cd.extend(result['vertex_com_cd'])
        all_vertex_cd.extend(result['vertex_cd'])

        all_stl_acc_cd.append(result['stl_acc_cd'])
        all_stl_com_cd.append(result['stl_com_cd'])
        all_stl_cd.append(result['stl_cd'])

        if result['stl_type'] == shape_type[0]:
            num_valid_solid += 1
            valid_solid_acc_cd.append(result['stl_acc_cd'])
            valid_solid_com_cd.append(result['stl_com_cd'])
            valid_solid_cd.append(result['stl_cd'])
            if result['stl_acc_cd'] > 0.001:
                print(f"Valid solid {folder_name} has a large CD: {result['stl_cd']}")

        elif result['stl_type'] == shape_type[1]:
            num_invalid_solid += 1
            invalid_solid_acc_cd.append(result['stl_acc_cd'])
            invalid_solid_com_cd.append(result['stl_com_cd'])
            invalid_solid_cd.append(result['stl_cd'])
        else:
            num_non_solid += 1
            non_solid_acc_cd.append(result['stl_acc_cd'])
            non_solid_com_cd.append(result['stl_com_cd'])
            non_solid_cd.append(result['stl_cd'])

        if result['stl_type'] == shape_type[2]:
            num_shell += 1
            shell_acc_cd.append(result['stl_acc_cd'])
            shell_com_cd.append(result['stl_com_cd'])
            shell_cd.append(result['stl_cd'])
        elif result['stl_type'] == shape_type[3]:
            num_compound += 1
            compound_acc_cd.append(result['stl_acc_cd'])
            compound_com_cd.append(result['stl_com_cd'])
            compound_cd.append(result['stl_cd'])

    if len(exception_folders) != 0:
        print(f"Found exception folders: {exception_folders}")

    # print the statistics
    print("\nFace")
    print("Recon: ", sum_recon_face)
    print("GT: ", sum_gt_face)
    print("Average ACC CD: ", np.mean(all_face_acc_cd))
    print("Average COM CD: ", np.mean(all_face_com_cd))
    print("Average CD: ", np.mean(all_face_cd))

    print("\nEdge")
    print("Recon: ", sum_recon_edge)
    print("GT: ", sum_gt_edge)
    print("Average ACC CD: ", np.mean(all_edge_acc_cd))
    print("Average COM CD: ", np.mean(all_edge_com_cd))
    print("Average CD: ", np.mean(all_edge_cd))

    print("\nVertex")
    print("Recon: ", sum_recon_vertex)
    print("GT: ", sum_gt_vertex)
    print("Average ACC CD: ", np.mean(all_vertex_acc_cd))
    print("Average COM CD: ", np.mean(all_vertex_com_cd))
    print("Average CD: ", np.mean(all_vertex_cd))

    print("\nValid Solid: ", num_valid_solid)
    if num_valid_solid != 0:
        print("Average Acc CD: ", np.mean(valid_solid_acc_cd))
        print("Average Com CD: ", np.mean(valid_solid_com_cd))
        print("Average CD: ", np.mean(valid_solid_cd))

    print("\nInvalid Solid: ", num_invalid_solid)
    if num_invalid_solid != 0:
        print("Average Acc CD: ", np.mean(invalid_solid_acc_cd))
        print("Average Com CD: ", np.mean(invalid_solid_com_cd))
        print("Average CD: ", np.mean(invalid_solid_cd))

    print("\nNon Solid: ", num_non_solid)
    if num_non_solid != 0:
        print("Average Acc CD: ", np.mean(non_solid_acc_cd))
        print("Average Com CD: ", np.mean(non_solid_com_cd))
        print("Average CD: ", np.mean(non_solid_cd))

    print("\nShell: ", num_shell)
    if num_shell != 0:
        print("Average Acc CD: ", np.mean(shell_acc_cd))
        print("Average Com CD: ", np.mean(shell_com_cd))
        print("Average CD: ", np.mean(shell_cd))

    print("\nCompound: ", num_compound)
    if num_compound != 0:
        print("Average Acc CD: ", np.mean(compound_acc_cd))
        print("Average Com CD: ", np.mean(compound_com_cd))
        print("Average CD: ", np.mean(compound_cd))

    # data = pd.DataFrame(all_stl_cd, columns=['all_stl_cd'])
    # print(data.info())
    # print(data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print(f"Exception folders: {len(exception_folders)}")


def seg_by_face_num(eval_root):
    seg_root = eval_root.rstrip('\\') + '_seg_by_face_num'
    if os.path.exists(seg_root):
        shutil.rmtree(seg_root)
    os.makedirs(seg_root, exist_ok=False)
    seg_save_root = [os.path.join(seg_root, 'face_30'),
                     os.path.join(seg_root, 'face_20'),
                     os.path.join(seg_root, 'face_10'),
                     os.path.join(seg_root, 'face_0'),
                     os.path.join(seg_root, 'else')]
    for each in seg_save_root:
        os.makedirs(each, exist_ok=True)

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, folder))]
    exception_folders = []
    for folder_name in tqdm(all_folders):
        if not os.path.exists(os.path.join(eval_root, folder_name, 'eval.npz')):
            exception_folders.append(folder_name)
            continue

        result = np.load(os.path.join(eval_root, folder_name, 'eval.npz'), allow_pickle=True)['result'].item()
        if result['num_recon_face'] == 0:
            exception_folders.append(folder_name)
            continue

        if result['stl_type'] == shape_type[0]:
            face_num = int(result['num_recon_face'])
            if face_num > 30:
                shutil.copytree(os.path.join(eval_root, folder_name), os.path.join(seg_save_root[0], folder_name))
            elif face_num > 20:
                shutil.copytree(os.path.join(eval_root, folder_name), os.path.join(seg_save_root[1], folder_name))
            elif face_num > 10:
                shutil.copytree(os.path.join(eval_root, folder_name), os.path.join(seg_save_root[2], folder_name))
            else:
                shutil.copytree(os.path.join(eval_root, folder_name), os.path.join(seg_save_root[3], folder_name))

    print(f"Seg by face num, saved in : {seg_root}")


def test_eval_one(eval_root, gt_root):
    folder_name = "00009559"
    eval_one(eval_root, gt_root, folder_name)
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate The Generated Brep')
    parser.add_argument('--eval_root', type=str, default=r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test_out")
    parser.add_argument('--gt_root', type=str, default=r"E:\data\img2brep\deepcad_whole_v5\deepcad_whole_test_v5")
    parser.add_argument('--is_use_ray', type=bool, default=True)
    parser.add_argument('--num_cpus', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--is_cover', type=bool, default=False)
    args = parser.parse_args()
    eval_root = args.eval_root
    gt_root = args.gt_root
    is_use_ray = args.is_use_ray
    batch_size = args.batch_size
    num_cpus = args.num_cpus
    is_cover = args.is_cover

    if not os.path.exists(eval_root):
        raise ValueError(f"Data root path {eval_root} does not exist.")
    if not os.path.exists(gt_root):
        raise ValueError(f"Output root path {gt_root} does not exist.")

    # test_eval_one(eval_root, gt_root)

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, folder))]
    print(f"Total {len(all_folders)} folders to evaluate")

    if not is_cover:
        print("Filtering the folders that have eval.npz")
        all_folders = [folder for folder in all_folders if not os.path.exists(os.path.join(eval_root, folder, 'eval.npz'))]
        print(f"Total {len(all_folders)} folders to evaluate")

    if not is_use_ray:
        # random.shuffle(self.folder_names)
        for i in tqdm(range(len(all_folders))):
            eval_one(eval_root, gt_root, all_folders[i])
    else:
        ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8080,
                num_cpus=num_cpus,
                # local_mode=True
        )
        batch_size = 1
        num_batches = len(all_folders) // batch_size + 1
        tasks = []
        for i in range(num_batches):
            tasks.append(eval_batch_remote.remote(eval_root, gt_root, all_folders[i * batch_size:(i + 1) * batch_size]))
        ray.get(tasks)

    print("Computing statistics...")
    compute_statistics(eval_root)
    print("Done")
