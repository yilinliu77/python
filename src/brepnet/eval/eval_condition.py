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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

shape_type = ["valid_solid", "invalid_solid", "shell", "compound", "else"]


def is_vertex_close(p1, p2, tol=1e-3):
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


def read_step_and_get_data(step_file_path, NUM_SAMPLE_EDGE_UNIT=100, save_data=False):
    if not os.path.exists(step_file_path):
        return None, None

    shape = read_step_file(step_file_path, verbosity=False)
    if save_data and not os.path.exists(step_file_path.replace('.step', '.stl')):
        write_stl_file(shape, step_file_path.replace('.step', '.stl'), linear_deflection=0.001, angular_deflection=0.5)

    def get_primitives(v_shape, v_type):
        exp = TopExp_Explorer(v_shape, v_type)
        v_list = []
        while exp.More():
            v = exp.Current()
            v_list.append(v)
            exp.Next()
        return v_list

    cached_data = {}
    faces = get_primitives(shape, TopAbs_FACE)
    for face in faces:
        for edge in get_primitives(face, TopAbs_EDGE):
            curve_data = BRep_Tool.Curve(edge)
            if curve_data and len(curve_data) == 3:
                curve_handle, first, last = curve_data
            else:
                continue

            NUM_SAMPLE_EDGE = 3 + int(get_edge_length(edge, NUM_SEGMENTS=32) * NUM_SAMPLE_EDGE_UNIT)
            u_values = np.linspace(first, last, NUM_SAMPLE_EDGE)
            points_on_edge = []
            for u in u_values:
                point = curve_handle.Value(u)
                points_on_edge.append((point.X(), point.Y(), point.Z()))
            if len(points_on_edge) != 0:
                points_on_edge = np.array(points_on_edge)
                points_on_edge = points_on_edge.reshape(-1, 3)

            if edge.Reversed() in cached_data:
                cached_data[edge.Reversed()].append((face, points_on_edge))
            else:
                cached_data[edge] = [(face, points_on_edge)]

    edge_points = []
    for edge in cached_data:
        if len(cached_data[edge]) == 1:
            continue
        if cached_data[edge][0][0].IsEqual(cached_data[edge][1][0]):
            continue
        edge_points.append(cached_data[edge][0][1])

    vertexes_from_edge = []
    for edge in edge_points:
        # check if it is a closed edge, pass the closed edge
        if is_vertex_close(edge[0], edge[-1], 1e-2):
            continue
        if not any([is_vertex_close(edge[0], v, 1e-2) for v in vertexes_from_edge]):
            vertexes_from_edge.append(edge[0])
        if not any([is_vertex_close(edge[-1], v, 1e-2) for v in vertexes_from_edge]):
            vertexes_from_edge.append(edge[-1])

    vertexes_from_shape = get_primitives(shape, TopAbs_VERTEX)
    vertexes_from_shape = [BRep_Tool.Pnt(v).Coord() for v in vertexes_from_shape]
    vertexes = []
    for vertex in vertexes_from_shape:
        if not any([is_vertex_close(vertex, v, 1e-2) for v in vertexes_from_edge]):
            continue
        if not any([is_vertex_close(vertex, v, 1e-2) for v in vertexes]):
            vertexes.append(vertex)

    return faces, vertexes, edge_points


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
    print(f"{sum_recon_face} {sum_gt_face} {sum_recon_face / sum_gt_face} {np.mean(all_face_acc_cd)} {np.mean(all_face_com_cd)} "
          f"{np.mean(all_face_cd)}", file=fp)

    print("\nEdge", file=fp)
    print("Recon: ", sum_recon_edge, file=fp)
    print("GT: ", sum_gt_edge, file=fp)
    print("Average ACC CD: ", np.mean(all_edge_acc_cd), file=fp)
    print("Average COM CD: ", np.mean(all_edge_com_cd), file=fp)
    print("Average CD: ", np.mean(all_edge_cd), file=fp)
    print(f"{sum_recon_edge} {sum_gt_edge} {sum_recon_edge / sum_gt_edge} {np.mean(all_edge_acc_cd)} {np.mean(all_edge_com_cd)} "
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
        print(f"{num_valid_solid} {len(all_folders)} {num_valid_solid / len(all_folders)} {np.mean(valid_solid_acc_cd)} "
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

    print(f"{sum_recon_face} {sum_gt_face} {sum_recon_face / sum_gt_face} {np.mean(all_face_acc_cd)} {np.mean(all_face_com_cd)} "
          f"{np.mean(all_face_cd)}", file=fp)
    print(f"{sum_recon_edge} {sum_gt_edge} {sum_recon_edge / sum_gt_edge} {np.mean(all_edge_acc_cd)} {np.mean(all_edge_com_cd)} "
          f"{np.mean(all_edge_cd)}", file=fp)
    print(f"{sum_recon_vertex} {sum_gt_vertex} {sum_recon_vertex / sum_gt_vertex} {np.mean(all_vertex_acc_cd)} "
          f"{np.mean(all_vertex_com_cd)} {np.mean(all_vertex_cd)}", file=fp)
    print(f"{num_valid_solid} {len(all_folders)} {num_valid_solid / len(all_folders)} {np.mean(valid_solid_acc_cd)} "
          f"{np.mean(valid_solid_com_cd)} {np.mean(valid_solid_cd)}", file=fp)

    fp.close()


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


def get_data(v_shape, v_num_per_m=100):
    faces, face_points, edges, edge_points, vertices, vertex_points = [], [], [], [], [], []
    for face in get_primitives(v_shape, TopAbs_FACE):
        v, f = get_triangulations(face, 0.1, 0.1)
        mesh_item = trimesh.Trimesh(vertices=v, faces=f)
        area = mesh_item.area
        num_samples = min(max(int(v_num_per_m * v_num_per_m * area), 5), 10000)
        pc_item, id_face = trimesh.sample.sample_surface(mesh_item, num_samples)
        normals = mesh_item.face_normals[id_face]
        faces.append(face)
        face_points.append(np.concatenate((pc_item,normals), axis=1))
    for edge in get_primitives(v_shape, TopAbs_EDGE):
        length = get_curve_length(edge)
        num_samples = min(max(int(v_num_per_m * length), 5), 10000)
        v = get_points_along_edge(edge, num_samples)
        edges.append(edge)
        edge_points.append(v)
    for vertex in get_primitives(v_shape, TopAbs_VERTEX):
        vertices.append(vertex)
        vertex_points.append(np.asarray([BRep_Tool.Pnt(vertex).Coord()]))
    vertex_points = np.concatenate(vertex_points, axis=0)
    return faces, face_points, edges, edge_points, vertices, vertex_points

def get_chamfer(v_recon_points, v_gt_points):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chamfer_distance = ChamferDistance()
    recon_fp = torch.from_numpy(np.concatenate(v_recon_points, axis=0)).float().to(device)
    gt_fp = torch.from_numpy(np.concatenate(v_gt_points, axis=0)).float().to(device)
    fp_acc_cd = chamfer_distance(recon_fp.unsqueeze(0), gt_fp.unsqueeze(0),
                              bidirectional=False, point_reduction='mean').cpu().item()
    fp_com_cd = chamfer_distance(gt_fp.unsqueeze(0), recon_fp.unsqueeze(0),
                              bidirectional=False, point_reduction='mean').cpu().item()
    fp_cd = (fp_acc_cd + fp_com_cd) / 2
    return fp_acc_cd, fp_com_cd, fp_cd

def get_detection(v_recon_points, v_gt_points):
    return 0

def eval_one(eval_root, gt_root, folder_name, SAMPLE_NUM=100):
    if os.path.exists(eval_root / folder_name / 'error.txt'):
        os.remove(eval_root / folder_name / 'error.txt')
    if os.path.exists(eval_root / folder_name / 'eval.npz'):
        os.remove(eval_root / folder_name / 'eval.npz')

    # At least have fall_back_mesh
    step_name = "recon_brep.step"
    fall_back_mesh = "recon_brep.ply"
    if not os.path.exists(eval_root / folder_name / fall_back_mesh):
        raise ValueError(f"Cannot find the {fall_back_mesh} in {eval_root / folder_name}")

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
        recon_shape, SAMPLE_NUM)
    gt_faces, gt_face_points, gt_edges, gt_edge_points, gt_vertices, gt_vertex_points = get_data(gt_shape, SAMPLE_NUM)

    # Chamfer
    face_acc_cd, face_com_cd, face_cd = get_chamfer(recon_face_points, gt_face_points)
    edge_acc_cd, edge_com_cd, edge_cd = get_chamfer(recon_edge_points, gt_edge_points)
    vertex_acc_cd, vertex_com_cd, vertex_cd = get_chamfer(recon_vertex_points, gt_vertex_points)

    # Detection
    face_fscore = get_detection(recon_face_points, gt_face_points)
    edge_fscore = get_detection(recon_edge_points, gt_edge_points)
    vertex_fscore = get_detection(recon_vertex_points, gt_vertex_points)

    # Topology
    face_edge_dict = {}
    for face in get_primitives(shape, TopAbs_FACE):
        edges = []
        for edge in get_primitives(face, TopAbs_EDGE):
            edges.append(edge)
        face_edge_dict[face] = edges

    # gt info
    gt_mesh_path = gt_root / folder_name / 'mesh.ply'
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_pc, _ = trimesh.sample.sample_surface(gt_mesh, SAMPLE_NUM)
    gt_pc_tensor = torch.from_numpy(gt_pc).float().to(device)

    data_npz = np.load(os.path.join(gt_root, folder_name, 'data.npz'))
    result['num_gt_face'] = data_npz['sample_points_faces'].shape[0]
    # result['num_gt_edge'] = data_npz['sample_points_lines'].shape[0]
    # result['num_gt_vertex'] = data_npz['sample_points_vertices'].shape[0]

    gt_step_path = os.path.join(gt_root, folder_name, 'normalized_shape.step')
    gt_faces, gt_vertexes, gt_edge_points = read_step_and_get_data(gt_step_path)
    result['num_gt_edge'] = len(gt_edge_points)
    result['num_gt_vertex'] = len(gt_vertexes)
    gt_edge_points = np.concatenate(gt_edge_points, axis=0)

    gt_edge_tensor = torch.from_numpy(gt_edge_points).float().to(device)

    gen_step_name = [f for f in os.listdir(os.path.join(eval_root, folder_name)) if f.endswith('.step')]

    # read the edge and vertex from gen step file
    if os.path.exists(os.path.join(eval_root, folder_name, 'recon_brep.step')):
        step_file_path = os.path.join(eval_root, folder_name, 'recon_brep.step')
        gen_face, gen_vertexes, gen_edge_points = read_step_and_get_data(step_file_path, save_data=True)
    else:
        gen_face, gen_vertexes, gen_edge_points = None, None

    gen_stl_name = [f for f in os.listdir(os.path.join(eval_root, folder_name)) if f.endswith('.stl')]
    gen_face_name = [f for f in os.listdir(os.path.join(eval_root, folder_name)) if f.endswith('.ply')]

    # face
    if len(gen_face_name) != 0:
        result['num_recon_face'] = len(gen_face)
        recon_face_mesh = trimesh.load(os.path.join(eval_root, folder_name, gen_face_name[0]))
        recon_face_pc, _ = trimesh.sample.sample_surface(recon_face_mesh, SAMPLE_NUM)
        recon_face_pc_tensor = torch.from_numpy(recon_face_pc).float().to(device)
        acc_cd = chamfer_distance(recon_face_pc_tensor.unsqueeze(0), gt_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        com_cd = chamfer_distance(gt_pc_tensor.unsqueeze(0), recon_face_pc_tensor.unsqueeze(0),
                                  bidirectional=False, point_reduction='mean').cpu().item()
        cd = (acc_cd + com_cd) / 2
        result['face_acc_cd'].append(acc_cd)
        result['face_com_cd'].append(com_cd)
        result['face_cd'].append(cd)

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
        if len(gen_vertexes) != 0 and len(gt_vertexes) != 0:
            gen_vertexes = np.stack(gen_vertexes, axis=0)
            gt_vertexes = np.stack(gt_vertexes, axis=0)
            gt_vertex_tensor = torch.from_numpy(gt_vertexes).float().to(device)
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
    gen_mesh = trimesh.load(str(os.path.join(eval_root, folder_name, gen_stl_name[0])))

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


    all_folders = [folder for folder in os.listdir(eval_root) if os.path.isdir(eval_root/folder)]
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
