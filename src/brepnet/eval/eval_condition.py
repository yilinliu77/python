import time, os, random, traceback, sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def compute_statistics(eval_root):
    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, folder))]
    exception_folders = []
    results = {

    }
    for folder_name in tqdm(all_folders):
        if not os.path.exists(os.path.join(eval_root, folder_name, 'eval.npz')):
            exception_folders.append(folder_name)
            continue

        item = np.load(os.path.join(eval_root, folder_name, 'eval.npz'), allow_pickle=True)['results'].item()
        if item['num_recon_face'] == 0:
            exception_folders.append(folder_name)
            continue

        for key in item:
            if key not in results:
                results[key] = []
            results[key].append(item[key])

    if len(exception_folders) != 0:
        print(f"Found exception folders: {exception_folders}")

    for key in results:
        results[key] = np.array(results[key])

    results_str = ""
    results_str += "Number\n"
    results_str += f"Vertices: {np.mean(results['num_recon_vertex'])}/{np.mean(results['num_gt_vertex'])}\n"
    results_str += f"Edge: {np.mean(results['num_recon_edge'])}/{np.mean(results['num_gt_edge'])}\n"
    results_str += f"Face: {np.mean(results['num_recon_face'])}/{np.mean(results['num_gt_face'])}\n"

    results_str += "Chamfer\n"
    results_str += f"Vertices: {np.mean(results['vertex_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_cd'])}\n"

    results_str += "Detection\n"
    results_str += f"Vertices: {np.mean(results['vertex_fscore'])}\n"
    results_str += f"Edge: {np.mean(results['edge_fscore'])}\n"
    results_str += f"Face: {np.mean(results['face_fscore'])}\n"

    results_str += "Topology\n"
    results_str += f"FE: {np.mean(results['fe_fscore'])}\n"
    results_str += f"EV: {np.mean(results['ev_fscore'])}\n"

    results_str += "Accuracy\n"
    results_str += f"Vertices: {np.mean(results['vertex_acc_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_acc_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_acc_cd'])}\n"
    results_str += f"FE: {np.mean(results['fe_pre'])}\n"
    results_str += f"EV: {np.mean(results['ev_pre'])}\n"

    results_str += "Completeness\n"
    results_str += f"Vertices: {np.mean(results['vertex_com_cd'])}\n"
    results_str += f"Edge: {np.mean(results['edge_com_cd'])}\n"
    results_str += f"Face: {np.mean(results['face_com_cd'])}\n"
    results_str += f"FE: {np.mean(results['fe_rec'])}\n"
    results_str += f"EV: {np.mean(results['ev_rec'])}\n"
    print(results_str)
    print("{:.4f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        np.mean(results['vertex_cd']), np.mean(results['edge_cd']), np.mean(results['face_cd']),
        np.mean(results['vertex_fscore']), np.mean(results['edge_fscore']), np.mean(results['face_fscore']),
        np.mean(results['fe_fscore']), np.mean(results['ev_fscore']),
    ))
    print(
        "{:.0f}/{:.0f} {:.0f}/{:.0f} {:.0f}/{:.0f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            np.mean(results['num_recon_vertex']), np.mean(results['num_gt_vertex']),
            np.mean(results['num_recon_edge']), np.mean(results['num_gt_edge']),
            np.mean(results['num_recon_face']), np.mean(results['num_gt_face']),
            np.mean(results['vertex_acc_cd']), np.mean(results['edge_acc_cd']), np.mean(results['face_acc_cd']),
            np.mean(results['vertex_com_cd']), np.mean(results['edge_com_cd']), np.mean(results['face_com_cd']),
            np.mean(results['vertex_pre']), np.mean(results['edge_pre']), np.mean(results['face_pre']),
            np.mean(results['fe_pre']), np.mean(results['ev_pre']),
            np.mean(results['vertex_rec']), np.mean(results['edge_rec']), np.mean(results['face_rec']),
            np.mean(results['fe_rec']), np.mean(results['ev_rec'])
        ))

    def draw():
        face_chamfer = results['face_cd']
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(face_chamfer, bins=50, range=(0, 0.05), density=True, alpha=0.5, color='b', label='Face')
        ax.set_title('Face Chamfer Distance')
        ax.set_xlabel('Chamfer Distance')
        ax.set_ylabel('Density')
        ax.legend()
        plt.savefig(str(eval_root) + "_face_chamfer.png", dpi=600)
        # plt.show()

    draw()
    pass


def get_data(v_shape, v_num_per_m=100):
    faces, face_points, edges, edge_points, vertices, vertex_points = [], [], [], [], [], []
    for face in get_primitives(v_shape, TopAbs_FACE, v_remove_half=True):
        try:
            v, f = get_triangulations(face, 0.1, 0.1)
        except:
            print("Ignore 1 face")
            continue
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
    fp_cd = fp_acc_cd + fp_com_cd
    return fp_acc_cd, fp_com_cd, fp_cd


def get_match_ids(v_recon_points, v_gt_points):
    from scipy.optimize import linear_sum_assignment

    cost = np.zeros([len(v_recon_points), len(v_gt_points)])  # recon to gt
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            _, _, cost[i][j] = get_chamfer(
                v_recon_points[i][..., :3][None, ..., :3],
                v_gt_points[j][..., :3][None, ..., :3]
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
    precision = true_positive / (len(id_recon_gt) + 1e-6)
    recall = true_positive / (len(id_gt_recon) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def get_topology(faces, edges, vertices):
    recon_face_edge, recon_edge_vertex = {}, {}
    for i_face, face in enumerate(faces):
        face_edge = []
        for edge in get_primitives(face, TopAbs_EDGE):
            face_edge.append(edges.index(edge) if edge in edges else edges.index(edge.Reversed()))
        recon_face_edge[i_face] = list(set(face_edge))

    for i_edge, edge in enumerate(edges):
        edge_vertex = []
        for vertex in get_primitives(edge, TopAbs_VERTEX):
            edge_vertex.append(vertices.index(vertex) if vertex in vertices else vertices.index(vertex.Reversed()))
        recon_edge_vertex[i_edge] = list(set(edge_vertex))
    return recon_face_edge, recon_edge_vertex


def get_topo_detection(recon_face_edge, gt_face_edge, id_recon_gt_face, id_recon_gt_edge):
    positive = 0
    for i_recon_face, edges in recon_face_edge.items():
        i_gt_face = id_recon_gt_face[i_recon_face]
        if i_gt_face == -1:
            continue
        for i_edge in edges:
            if id_recon_gt_edge[i_edge] in gt_face_edge[i_gt_face]:
                positive += 1
    precision = positive / (sum([len(edges) for edges in recon_face_edge.values()]) + 1e-6)
    recall = positive / (sum([len(edges) for edges in gt_face_edge.values()]) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6), precision, recall


def eval_one(eval_root, gt_root, folder_name, v_num_per_m=100):
    if os.path.exists(eval_root / folder_name / 'error.txt'):
        os.remove(eval_root / folder_name / 'error.txt')
    if os.path.exists(eval_root / folder_name / 'eval.npz'):
        os.remove(eval_root / folder_name / 'eval.npz')

    # At least have fall_back_mesh
    step_name = "recon_brep.step"

    # Face chamfer distance
    if (eval_root / folder_name / step_name).exists():
        valid, recon_shape = check_step_valid_soild(eval_root / folder_name / step_name, return_shape=True)
    else:
        print(f"Error: {folder_name} does not have {step_name}")
        return
    if recon_shape is None:
        print(f"Error: {folder_name} 's {step_name} is not valid")
        return
        
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
    ev_fscore, ev_pre, ev_rec = get_topo_detection(recon_edge_vertex, gt_edge_vertex, id_recon_gt_edge,
                                                   id_recon_gt_vertex)

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

        "vertex_pre": vertex_pre,
        "edge_pre": edge_pre,
        "face_pre": face_pre,

        "vertex_rec": vertex_rec,
        "edge_rec": edge_rec,
        "face_rec": face_rec,

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
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--list', type=str, default='')
    parser.add_argument('--from_scratch', action='store_true')
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    gt_root = Path(args.gt_root)
    is_use_ray = args.use_ray
    num_cpus = args.num_cpus
    listfile = args.list
    from_scratch = args.from_scratch

    if not os.path.exists(eval_root):
        raise ValueError(f"Data root path {eval_root} does not exist.")
    if not os.path.exists(gt_root):
        raise ValueError(f"Output root path {gt_root} does not exist.")

    if args.prefix != '':
        eval_one(eval_root, gt_root, args.prefix)
        exit()

    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(eval_root / folder)]
    ori_length = len(all_folders)
    if listfile != '':
        valid_names = [item.strip() for item in open(listfile, 'r').readlines()]
        all_folders = list(set(all_folders) & set(valid_names))
        all_folders.sort()
    print(f"Total {len(all_folders)}/{ori_length} folders to evaluate")

    if not from_scratch:
        print("Filtering the folders that have eval.npz")
        all_folders = [folder for folder in all_folders if not os.path.exists(eval_root / folder / 'eval.npz')]
        print(f"Total {len(all_folders)} folders to evaluate after caching")

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
        eval_one_remote = ray.remote(max_retries=0)(eval_one)
        tasks = []
        for i in range(len(all_folders)):
            tasks.append(eval_one_remote.remote(eval_root, gt_root, all_folders[i]))
        for _ in tqdm(range(len(all_folders))):
            ray.get(tasks.pop(0))

    print("Computing statistics...")
    compute_statistics(eval_root)
    print("Done")
