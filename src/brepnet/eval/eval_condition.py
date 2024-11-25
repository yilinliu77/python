import time, os, random, traceback, sys
from pathlib import Path

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

from shared.occ_utils import get_primitives, get_triangulations, get_points_along_edge, get_curve_length
from src.brepnet.eval.check_valid import check_step_valid_soild


def is_vertex_close(p1, p2, tol=1e-3):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def compute_statistics(eval_root, txt_path):
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

        if len(result['vertex_cd']) > 0 and np.mean(result['vertex_cd']) > 0.1:
            print(f"Vertex CD > 0.1: {folder_name}: {np.mean(result['vertex_cd'])}")

        all_stl_acc_cd.append(result['stl_acc_cd'])
        all_stl_com_cd.append(result['stl_com_cd'])
        all_stl_cd.append(result['stl_cd'])

        if result['stl_type'] == shape_type[0]:
            num_valid_solid += 1
            valid_solid_acc_cd.append(result['stl_acc_cd'])
            valid_solid_com_cd.append(result['stl_com_cd'])
            valid_solid_cd.append(result['stl_cd'])
            if result['stl_acc_cd'] > 0.1:
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

    # print the statistics and save to result.txt
    fp = open(txt_path, "w")
    print("\nFace", file=fp)
    print("Recon: ", sum_recon_face, file=fp)
    print("GT: ", sum_gt_face, file=fp)
    print("Average ACC CD: ", np.mean(all_face_acc_cd), file=fp)
    print("Average COM CD: ", np.mean(all_face_com_cd), file=fp)
    print("Average CD: ", np.mean(all_face_cd), file=fp)
    print(
        f"{sum_recon_face} {sum_gt_face} {sum_recon_face / sum_gt_face} {np.mean(all_face_acc_cd)} {np.mean(all_face_com_cd)} "
        f"{np.mean(all_face_cd)}", file=fp)

    print("\nEdge", file=fp)
    print("Recon: ", sum_recon_edge, file=fp)
    print("GT: ", sum_gt_edge, file=fp)
    print("Average ACC CD: ", np.mean(all_edge_acc_cd), file=fp)
    print("Average COM CD: ", np.mean(all_edge_com_cd), file=fp)
    print("Average CD: ", np.mean(all_edge_cd), file=fp)
    print(
        f"{sum_recon_edge} {sum_gt_edge} {sum_recon_edge / sum_gt_edge} {np.mean(all_edge_acc_cd)} {np.mean(all_edge_com_cd)} "
        f"{np.mean(all_edge_cd)}", file=fp)

    print("\nVertex", file=fp)
    print("Recon: ", sum_recon_vertex, file=fp)
    print("GT: ", sum_gt_vertex, file=fp)
    print("Average ACC CD: ", np.mean(all_vertex_acc_cd), file=fp)
    print("Average COM CD: ", np.mean(all_vertex_com_cd), file=fp)
    print("Average CD: ", np.mean(all_vertex_cd), file=fp)
    print(f"{sum_recon_vertex} {sum_gt_vertex} {sum_recon_vertex / sum_gt_vertex} {np.mean(all_vertex_acc_cd)} "
          f"{np.mean(all_vertex_com_cd)} {np.mean(all_vertex_cd)}", file=fp)

    print("\nValid Solid: ", num_valid_solid, file=fp)
    if num_valid_solid != 0:
        print("Average Acc CD: ", np.mean(valid_solid_acc_cd), file=fp)
        print("Average Com CD: ", np.mean(valid_solid_com_cd), file=fp)
        print("Average CD: ", np.mean(valid_solid_cd), file=fp)
        print(
            f"{num_valid_solid} {len(all_folders)} {num_valid_solid / len(all_folders)} {np.mean(valid_solid_acc_cd)} "
            f"{np.mean(valid_solid_com_cd)} {np.mean(valid_solid_cd)}", file=fp)

    print("\nInvalid Solid: ", num_invalid_solid, file=fp)
    if num_invalid_solid != 0:
        print("Average Acc CD: ", np.mean(invalid_solid_acc_cd), file=fp)
        print("Average Com CD: ", np.mean(invalid_solid_com_cd), file=fp)
        print("Average CD: ", np.mean(invalid_solid_cd), file=fp)

    print("\nNon Solid: ", num_non_solid, file=fp)
    if num_non_solid != 0:
        print("Average Acc CD: ", np.mean(non_solid_acc_cd), file=fp)
        print("Average Com CD: ", np.mean(non_solid_com_cd), file=fp)
        print("Average CD: ", np.mean(non_solid_cd), file=fp)

    print("\nShell: ", num_shell, file=fp)
    if num_shell != 0:
        print("Average Acc CD: ", np.mean(shell_acc_cd), file=fp)
        print("Average Com CD: ", np.mean(shell_com_cd), file=fp)
        print("Average CD: ", np.mean(shell_cd), file=fp)

    print("\nCompound: ", num_compound, file=fp)
    if num_compound != 0:
        print("Average Acc CD: ", np.mean(compound_acc_cd), file=fp)
        print("Average Com CD: ", np.mean(compound_com_cd), file=fp)
        print("Average CD: ", np.mean(compound_cd), file=fp)

    print(f"Exception folders: {len(exception_folders)}", file=fp)
    print(f"Total folders: {len(all_folders)}", file=fp)

    print(
        f"{sum_recon_face} {sum_gt_face} {sum_recon_face / sum_gt_face} {np.mean(all_face_acc_cd)} {np.mean(all_face_com_cd)} "
        f"{np.mean(all_face_cd)}", file=fp)
    print(
        f"{sum_recon_edge} {sum_gt_edge} {sum_recon_edge / sum_gt_edge} {np.mean(all_edge_acc_cd)} {np.mean(all_edge_com_cd)} "
        f"{np.mean(all_edge_cd)}", file=fp)
    print(f"{sum_recon_vertex} {sum_gt_vertex} {sum_recon_vertex / sum_gt_vertex} {np.mean(all_vertex_acc_cd)} "
          f"{np.mean(all_vertex_com_cd)} {np.mean(all_vertex_cd)}", file=fp)
    print(f"{num_valid_solid} {len(all_folders)} {num_valid_solid / len(all_folders)} {np.mean(valid_solid_acc_cd)} "
          f"{np.mean(valid_solid_com_cd)} {np.mean(valid_solid_cd)}", file=fp)

    fp.close()


def get_data(v_shape, v_num_per_m=100):
    faces, face_points, edges, edge_points, vertices, vertex_points = [], [], [], [], [], []
    for face in get_primitives(v_shape, TopAbs_FACE, v_remove_half=True):
        v, f = get_triangulations(face, 0.1, 0.1)
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        area = mesh_item.area
        num_samples = min(max(int(v_num_per_m * v_num_per_m * area), 5), 10000)
        pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
        normals = mesh_item.face_normals[id_face]
        faces.append(face)
        face_points.append(np.concatenate((pc_item, normals), axis=1))
    for edge in get_primitives(v_shape, TopAbs_EDGE, v_remove_half=True):
        length = get_curve_length(edge)
        num_samples = min(max(int(v_num_per_m * length), 5), 10000)
        v = get_points_along_edge(edge, num_samples)
        edges.append(edge)
        edge_points.append(v)
    for vertex in get_primitives(v_shape, TopAbs_VERTEX, v_remove_half=True):
        vertices.append(vertex)
        vertex_points.append(np.asarray([BRep_Tool.Pnt(vertex).Coord()]))
    vertex_points = np.stack(vertex_points, axis=0)
    return faces, face_points, edges, edge_points, vertices, vertex_points


def get_chamfer(v_recon_points, v_gt_points):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chamfer_distance = ChamferDistance()
    recon_fp = torch.from_numpy(np.concatenate(v_recon_points, axis=0)).float().to(device)[:, :3]
    gt_fp = torch.from_numpy(np.concatenate(v_gt_points, axis=0)).float().to(device)[:, :3]
    fp_acc_cd = chamfer_distance(recon_fp.unsqueeze(0), gt_fp.unsqueeze(0),
                                 bidirectional=False, point_reduction='mean').cpu().item()
    fp_com_cd = chamfer_distance(gt_fp.unsqueeze(0), recon_fp.unsqueeze(0),
                                 bidirectional=False, point_reduction='mean').cpu().item()
    fp_cd = (fp_acc_cd + fp_com_cd) / 2
    return fp_acc_cd, fp_com_cd, fp_cd


def get_match_ids(v_recon_points, v_gt_points):
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros([len(v_recon_points), len(v_gt_points)]) # recon to gt
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            _, _, cost[i][j] = get_chamfer(
                v_recon_points[i][..., :3][None,...,:3],
                v_gt_points[j][..., :3][None,...,:3]
            )

    recon_indices, recon_to_gt = linear_sum_assignment(cost)

    result_recon2gt = -1 * np.ones(len(v_recon_points), dtype=np.int32)
    result_gt2recon = -1 * np.ones(len(v_gt_points), dtype=np.int32)

    result_recon2gt[recon_indices] = recon_to_gt
    result_gt2recon[recon_to_gt] = recon_indices
    return result_recon2gt, result_gt2recon, cost


def get_detection(id_recon_gt, id_gt_recon, cost_matrix, v_threshold=0.1):
    true_positive = 0
    for i in range(len(id_recon_gt)):
        if id_recon_gt[i] != -1 and cost_matrix[i, id_recon_gt[i]] < v_threshold:
            true_positive += 1
    precision = true_positive / len(id_recon_gt)
    recall = true_positive / len(id_gt_recon)
    return 2 * precision * recall / (precision + recall), precision, recall


def get_topology(faces, edges, vertices):
    recon_face_edge, recon_edge_vertex = {}, {}
    for i_face, face in enumerate(faces):
        face_edge = []
        for edge in get_primitives(face, TopAbs_EDGE):
            face_edge.append(edges.index(edge) if edge in edges else edges.index(edge.Reversed()))
        recon_face_edge[i_face] = face_edge

    for i_edge, edge in enumerate(edges):
        edge_vertex = []
        for vertex in get_primitives(edge, TopAbs_VERTEX):
            edge_vertex.append(vertices.index(vertex) if vertex in vertices else vertices.index(vertex.Reversed()))
        recon_edge_vertex[i_edge] = edge_vertex
    return recon_face_edge, recon_edge_vertex


def get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge):
    positive = 0
    for i_recon_face, edges in recon_face_edge.items():
        i_gt_face = id_recon_gt_face[i_recon_face]
        for i_edge in edges:
            if id_recon_gt_edge[i_edge] in gt_face_edge[i_gt_face]:
                positive += 1
    precision = positive / sum([len(edges) for edges in recon_face_edge.values()])
    recall = positive / sum([len(edges) for edges in gt_face_edge.values()])
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_one(eval_root, gt_root, folder_name, v_num_per_m=100):
    if os.path.exists(eval_root / folder_name / 'error.txt'):
        os.remove(eval_root / folder_name / 'error.txt')
    if os.path.exists(eval_root / folder_name / 'eval.npz'):
        os.remove(eval_root / folder_name / 'eval.npz')

    # At least have fall_back_mesh
    step_name = "recon_brep.step"
    fall_back_mesh = "recon_brep.ply"
    if not os.path.exists(eval_root / folder_name / fall_back_mesh):
        raise ValueError(f"Cannot find the {fall_back_mesh} in {eval_root / folder_name}")

    # Face chamfer distance
    if (eval_root / folder_name / step_name).exists():
        valid, recon_shape = check_step_valid_soild(eval_root / folder_name / step_name, return_shape=True)
    elif (eval_root / folder_name / fall_back_mesh).exists():
        valid, recon_shape = False, trimesh.load(eval_root / folder_name / fall_back_mesh)
    else:
        raise

    # GT
    _, gt_shape = check_step_valid_soild(gt_root / folder_name / "normalized_shape.step", return_shape=True)

    # Get data
    recon_faces, recon_face_points, recon_edges, recon_edge_points, recon_vertices, recon_vertex_points = get_data(
        recon_shape, v_num_per_m)
    gt_faces, gt_face_points, gt_edges, gt_edge_points, gt_vertices, gt_vertex_points = get_data(gt_shape, v_num_per_m)

    # Chamfer
    face_acc_cd, face_com_cd, face_cd = get_chamfer(recon_face_points, gt_face_points)
    edge_acc_cd, edge_com_cd, edge_cd = get_chamfer(recon_edge_points, gt_edge_points)
    vertex_acc_cd, vertex_com_cd, vertex_cd = get_chamfer(recon_vertex_points, gt_vertex_points)

    # Detection
    id_recon_gt_face, id_gt_recon_face, cost_face = get_match_ids(recon_face_points, gt_face_points)
    id_recon_gt_edge, id_gt_recon_edge, cost_edge = get_match_ids(recon_edge_points, gt_edge_points)
    id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices = get_match_ids(recon_vertex_points, gt_vertex_points)

    face_fscore, face_pre, face_rec = get_detection(id_recon_gt_face, id_gt_recon_face, cost_face)
    edge_fscore, edge_pre, edge_rec = get_detection(id_recon_gt_edge, id_gt_recon_edge, cost_edge)
    vertex_fscore, vertex_pre, vertex_rec = get_detection(id_recon_gt_vertex, id_gt_recon_vertex, cost_vertices)

    # Topology
    recon_face_edge, recon_edge_vertex = get_topology(recon_faces, recon_edges, recon_vertices)
    gt_face_edge, gt_edge_vertex = get_topology(gt_faces, gt_edges, gt_vertices)

    fe_fscore, fe_pre, fe_rec = get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge)
    ev_fscore, ev_pre, ev_rec = get_topo_detection(recon_edge_vertex, gt_edge_vertex, id_recon_gt_edge, id_recon_gt_vertex)

    results = {
        "face_cd": face_cd,
        "edge_cd": edge_cd,
        "vertex_cd": vertex_cd,

        "face_fscore": face_fscore,
        "edge_fscore": edge_fscore,
        "vertex_fscore": vertex_fscore,
        "fe_fscore": fe_fscore,
        "ev_fscore": ev_fscore,

        "face_acc_cd": face_acc_cd,
        "edge_acc_cd": edge_acc_cd,
        "vertex_acc_cd": vertex_acc_cd,

        "face_com_cd": face_com_cd,
        "edge_com_cd": edge_com_cd,
        "vertex_com_cd": vertex_com_cd,

        "fe_pre": fe_pre,
        "ev_pre": ev_pre,
        "fe_rec": fe_rec,
        "ev_rec": ev_rec,

        "num_recon_face": len(recon_faces),
        "num_gt_face": len(gt_faces),
        "num_recon_edge": len(recon_edges),
        "num_gt_edge": len(gt_edges),
        "num_recon_vertex": len(recon_vertices),
        "num_gt_vertex": len(gt_vertices),
    }
    np.savez_compressed(eval_root / folder_name / 'eval.npz', results=results, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate The Generated Brep')
    parser.add_argument('--eval_root', type=str, default=r"E:\data\img2brep\.43\2024_09_22_21_57_44_0921_pure_out2")
    parser.add_argument('--gt_root', type=str, default=r"E:\data\img2brep\deepcad_whole_v5\deepcad_whole_test_v5")
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--is_cover', type=bool, default=True)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--list', type=str, default='')
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    gt_root = Path(args.gt_root)
    is_use_ray = args.use_ray
    is_cover = args.is_cover
    num_cpus = args.num_cpus
    listfile = args.list

    if not os.path.exists(eval_root):
        raise ValueError(f"Data root path {eval_root} does not exist.")
    if not os.path.exists(gt_root):
        raise ValueError(f"Output root path {gt_root} does not exist.")

    # test_eval_one(eval_root, gt_root)

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(eval_root / folder)]
    ori_length = len(all_folders)
    if listfile != '':
        valid_names = [item.strip() for item in open(listfile, 'r').readlines()]
        all_folders = list(set(all_folders) & set(valid_names))
        all_folders.sort()
    print(f"Total {len(all_folders)}/{ori_length} folders to evaluate")

    if not is_cover:
        print("Filtering the folders that have eval.npz")
        all_folders = [folder for folder in all_folders if not os.path.exists(eval_root / folder / 'eval.npz')]
        print(f"Total {len(all_folders)} folders to evaluate after caching")

    if args.prefix != '':
        eval_one(eval_root, gt_root, args.prefix)

    if not is_use_ray:
        # random.shuffle(self.folder_names)
        for i in tqdm(range(len(all_folders))):
            eval_one(eval_root, gt_root, all_folders[i])
    else:
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            # num_cpus=num_cpus,
            # local_mode=True
        )
        eval_one_remote = ray.remote(max_retries=0)(eval_one)
        tasks = []
        for i in range(len(all_folders)):
            tasks.append(eval_one_remote.remote(eval_root, gt_root, all_folders[i]))
        ray.get(tasks)

    print("Computing statistics...")
    compute_statistics(eval_root, txt_path=eval_root + '_eval_result.txt')
    print("Done")
