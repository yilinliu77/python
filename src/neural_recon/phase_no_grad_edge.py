import itertools
import sys, os
import time
from copy import copy
from typing import List
import pickle
import scipy.spatial
from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.geometric_util import fit_plane_svd
from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1, loss2, loss3, loss4, loss5

# sys.path.append("thirdparty/sdf_computer/build/")
# import pysdf
# from src.neural_recon.phase12 import Siren2

import math

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
import networkx as nx
from torch_scatter import scatter_add, scatter_min, scatter_mean
import faiss
# import torchsort

# import mcubes
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from tqdm import tqdm, trange
import ray
import platform
import shutil
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img, sample_img_prediction2
from shared.common_utils import *

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
# from src.neural_recon.phase1 import NGPModel
from scipy.spatial import Delaunay
from math import ceil

from src.neural_recon.collision_checker import Collision_checker
from src.neural_recon.sample_utils import sample_points_2d
from src.neural_recon.optimize_planes_batch import optimize_planes_batch, global_assemble, local_assemble
from src.neural_recon.loss_utils import dilate_edge

###################################################################################################
# 1. Input
###################################################################################################
@ray.remote
def read_graph(v_filename, img_size):
    data = [item for item in open(v_filename).readlines()]
    vertices = [item.strip().split(" ")[1:-1] for item in data if item[0] == "v"]
    vertices = np.asarray(vertices).astype(np.float32) / img_size
    faces = [item.strip().split(" ")[1:] for item in data if item[0] == "f"]
    graph = nx.Graph()
    graph.add_nodes_from([(idx, {"pos_2d": item}) for idx, item in enumerate(vertices)])
    new_faces = []  # equal faces - 1 because of the obj format

    for id_face, id_edge_per_face in enumerate(faces):
        id_edge_per_face = (np.asarray(id_edge_per_face).astype(np.int32) - 1).tolist()
        new_faces.append(id_edge_per_face)
        id_edge_per_face = [(id_edge_per_face[idx], id_edge_per_face[idx + 1]) for idx in
                            range(len(id_edge_per_face) - 1)] + [(id_edge_per_face[-1], id_edge_per_face[0])]
        graph.add_edges_from(id_edge_per_face)

    graph.graph["faces"] = new_faces

    # Mark boundary nodes, lines and faces
    for node in graph.nodes():
        graph.nodes[node]["valid_flag"] = graph.nodes[node]["pos_2d"][0] != 0 and \
                                          graph.nodes[node]["pos_2d"][1] != 0 and \
                                          graph.nodes[node]["pos_2d"][0] != 1 and \
                                          graph.nodes[node]["pos_2d"][1] != 1
    for node1, node2 in graph.edges():
        graph.edges[(node1, node2)]["valid_flag"] = graph.nodes[node1]["valid_flag"] and \
                                                    graph.nodes[node1]["valid_flag"]
    face_flags = []
    for id_face, face in enumerate(graph.graph["faces"]):
        face_flags.append(min([graph.nodes[point]["valid_flag"] for point in face]))
        center_point_2d = np.zeros(2, dtype=np.float32)
        for id_point in range(len(face)):
            id_start = id_point
            id_end = (id_start + 1) % len(face)
            if "id_face" not in graph[face[id_start]][face[id_end]]:
                graph[face[id_start]][face[id_end]]["id_face"] = []
            graph[face[id_start]][face[id_end]]["id_face"].append(id_face)
            center_point_2d += graph.nodes[face[id_start]]["pos_2d"]
        center_point_2d /= len(face)

    graph.graph["face_flags"] = np.array(face_flags, dtype=bool)

    return graph


def prepare_dataset_and_model(v_colmap_dir, v_viz_face, v_bounds, v_reconstruct_data=False):
    print("Start to prepare dataset")
    print("1. Read imgs")

    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name) and not v_reconstruct_data:
        print("Found cache ", img_cache_name)
        img_database, points_3d = np.load(img_cache_name, allow_pickle=True)
    else:
        print("Dosen't find cache, read raw img data")
        bound_min = np.array((v_bounds[0], v_bounds[1], v_bounds[2]))
        bound_max = np.array((v_bounds[3], v_bounds[4], v_bounds[5]))
        img_database, points_3d = read_dataset(v_colmap_dir,
                                               [bound_min,
                                                bound_max]
                                               )
        np.save(img_cache_name[:-4], np.asarray([img_database, points_3d], dtype=object))
        print("Save cache to ", img_cache_name)

    graph_cache_name = "output/img_field_test/graph_cache.npy"
    print("2. Build graph")
    if os.path.exists(graph_cache_name) and not v_reconstruct_data:
        graphs = np.load(graph_cache_name, allow_pickle=True)
    else:
        ray.init(
            # local_mode=True
        )
        tasks = [read_graph.remote(
            os.path.join(v_colmap_dir, "wireframe/{}.obj".format(img_database[i_img].img_name)),
            img_database[i_img].img_size
        ) for i_img in range(len(img_database))]
        graphs = ray.get(tasks)
        print("Read {} graphs".format(len(graphs)))
        graphs = np.asarray(graphs, dtype=object)
        np.save(graph_cache_name, graphs, allow_pickle=True)

    points_cache_name = "output/img_field_test/points_cache.npy"
    if os.path.exists(points_cache_name) and not v_reconstruct_data:
        points_from_sfm = np.load(points_cache_name)
    else:
        preserved_points = []
        for point in tqdm(points_3d):
            for track in point.tracks:
                if track[0] in [1, 2]:
                    preserved_points.append(point)
        if len(preserved_points) == 0:
            points_from_sfm = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        else:
            points_from_sfm = np.stack([item.pos for item in preserved_points])
        np.save(points_cache_name, points_from_sfm)

    print("Start to calculate initial wireframe for each image")

    # project the points_3d_from_sfm to points_2d, and filter points(2d && 3d) outside 2d space
    def project_points(v_projection_matrix, points_3d_pos):
        # project the 3d points into 2d space
        projected_points = np.transpose(v_projection_matrix @ np.transpose(np.insert(points_3d_pos, 3, 1, axis=1)))
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        # create a mask to filter points(2d && 3d) outside 2d space
        projected_points_mask = np.logical_and(projected_points[:, 0] > 0, projected_points[:, 1] > 0)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 0] < 1)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 1] < 1)
        points_3d_pos = points_3d_pos[projected_points_mask]
        projected_points = projected_points[projected_points_mask]
        return points_3d_pos, projected_points

    def draw_initial(img, v_graph):
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("1", 1600, 900)
        # cv2.moveWindow("1", 5, 5)
        v_rgb = cv2.imread(img.img_path, cv2.IMREAD_UNCHANGED)
        point_img = v_rgb.copy()
        for point in points_from_sfm_2d:
            cv2.circle(point_img, (point * img.img_size).astype(np.int32), 2, (0, 0, 255), thickness=4)
        print("Draw lines on img1")
        line_img1 = v_rgb.copy()

        # Draw first img
        for idx, (id_start, id_end) in enumerate(v_graph.edges()):
            # print(idx)
            start = (v_graph.nodes[id_start]["pos_2d"] * img.img_size).astype(np.int32)
            end = (v_graph.nodes[id_end]["pos_2d"] * img.img_size).astype(np.int32)
            cv2.line(line_img1, start, end, (0, 0, 255), thickness=1)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()

        # Draw target patch
        for id_edge in v_viz_face:
            (id_start, id_end) = np.asarray(v_graph.edges)[id_edge]
            start = (v_graph.nodes[id_start]["pos_2d"] * img.img_size).astype(np.int32)
            end = (v_graph.nodes[id_end]["pos_2d"] * img.img_size).astype(np.int32)
            cv2.line(line_img1, start, end, (0, 255, 0), thickness=1)
            cv2.circle(line_img1, start, 1, (0, 255, 255), 2)
            cv2.circle(line_img1, end, 1, (0, 255, 255), 2)
        viz_img = np.concatenate((point_img, line_img1), axis=0)
        cv2.imwrite("output/img_field_test/input_img.jpg", viz_img)

    # easy(but may be wrong sometimes) method:
    # 1. to project the points_3d_from_sfm to points_2d
    # 2. find nearist points_2d to init nodes' depth
    # used method:
    # 1. to project the points_3d_from_sfm to points_2d
    # 2. for each node in the graph, find nearist N candidate points_2d(correspondingly get candidate points_3d_camera),
    # then back project nodes to camera coordinates,
    # construct the ray of each node,
    # compute the nearist points_3d_camera from ray in camera coordinates
    def compute_initial(v_graph, v_points_3d, v_points_2d, v_extrinsic, v_intrinsic):
        distance_threshold = 5  # 5m; not used

        v_graph.graph["face_center"] = np.zeros((len(v_graph.graph["faces"]), 2), dtype=np.float32)
        v_graph.graph["ray_c"] = np.zeros((len(v_graph.graph["faces"]), 3), dtype=np.float32)
        v_graph.graph["distance"] = np.zeros((len(v_graph.graph["faces"]),), dtype=np.float32)

        # 1. Calculate the centroid of each faces
        for id_face, id_edge_per_face in enumerate(v_graph.graph["faces"]):
            # Convex assumption
            center_point = np.stack(
                [v_graph.nodes[id_vertex]["pos_2d"] for id_vertex in id_edge_per_face], axis=0).mean(axis=0)
            v_graph.graph["face_center"][id_face] = center_point

        # Query points: (M, 2)
        # points from sfm: (N, 2)
        kd_tree = faiss.IndexFlatL2(2)
        kd_tree.add(v_points_2d.astype(np.float32))

        # Prepare query points: Build an array of query points that
        # contains vertices and centroid of each face in the graph
        vertices_2d = np.asarray([v_graph.nodes[id_node]["pos_2d"] for id_node in v_graph.nodes()])  # (M, 2)
        centroids_2d = v_graph.graph["face_center"]
        query_points = np.concatenate([vertices_2d, centroids_2d], axis=0)
        # 32 nearest neighbors for each query point.
        shortest_distance, index_shortest_distance = kd_tree.search(query_points, 32)  # (M, K)

        points_from_sfm_camera = (v_extrinsic @ np.insert(v_points_3d, 3, 1, axis=1).T).T[:, :3]  # (N, 3)

        # Select the point which is nearest to the actual ray for each endpoints
        # 1. Construct the ray
        # (M, 3); points in camera coordinates
        ray_c = (np.linalg.inv(v_intrinsic) @ np.insert(query_points, 2, 1, axis=1).T).T
        ray_c = ray_c / np.linalg.norm(ray_c + 1e-6, axis=1, keepdims=True)  # Normalize the points(dir)
        nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
        # Compute the shortest distance from the candidate point to the ray for each query point
        # (M, K, 1): K projected distance of the candidate point along each ray
        distance_of_projection = nearest_candidates @ ray_c[:, :, np.newaxis]
        # (M, K, 3): K projected points along the ray
        # 投影距离*单位ray方向 = 投影点坐标
        projected_points_on_ray = distance_of_projection * ray_c[:, np.newaxis, :]
        distance_from_candidate_points_to_ray = np.linalg.norm(
            nearest_candidates - projected_points_on_ray + 1e-6, axis=2)  # (M, 1)

        # 相机坐标系中所有点到其相应的射线的距离，距离最小的称之为最佳投影点
        # (M, 1): Index of the best projected points along the ray
        index_best_projected = distance_from_candidate_points_to_ray.argmin(axis=1)

        chosen_distances = distance_of_projection[np.arange(projected_points_on_ray.shape[0]), index_best_projected]
        valid_mask = distance_from_candidate_points_to_ray[np.arange(
            projected_points_on_ray.shape[0]), index_best_projected] < distance_threshold  # (M, 1)
        # (M, 3): The best projected points along the ray
        initial_points_camera = projected_points_on_ray[
            np.arange(projected_points_on_ray.shape[0]), index_best_projected]
        initial_points_world = (np.linalg.inv(v_extrinsic) @ np.insert(initial_points_camera, 3, 1, axis=1).T).T
        initial_points_world = initial_points_world[:, :3] / initial_points_world[:, 3:4]

        for idx, id_node in enumerate(v_graph.nodes):
            v_graph.nodes[id_node]["pos_world"] = initial_points_world[idx]
            v_graph.nodes[id_node]["distance"] = chosen_distances[idx, 0]
            v_graph.nodes[id_node]["ray_c"] = ray_c[idx]

        for id_face in range(v_graph.graph["face_center"].shape[0]):
            idx = id_face + len(v_graph.nodes)
            v_graph.graph["ray_c"][id_face] = ray_c[idx]
            v_graph.graph["distance"][id_face] = chosen_distances[idx, 0]

        line_coordinates = []
        for edge in v_graph.edges():
            line_coordinates.append(np.concatenate((initial_points_world[edge[0]], initial_points_world[edge[1]])))
        save_line_cloud("output/img_field_test/initial_segments.obj", np.stack(line_coordinates, axis=0))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(initial_points_world[len(v_graph.nodes):])
        o3d.io.write_point_cloud("output/img_field_test/initial_face_centroid.ply", pc)
        return

    for id_img, img in enumerate(img_database):
        points_from_sfm, points_from_sfm_2d = project_points(img.projection, points_from_sfm)
        rgb = cv2.imread(img.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[:, :, None]
        # draw_initial(rgb, graphs[id_img], img)
        compute_initial(graphs[id_img], points_from_sfm, points_from_sfm_2d, img.extrinsic, img.intrinsic)
    draw_initial(img_database[0], graphs[0])

    # Read camera pairs
    camera_pair_txt = open(os.path.join(v_colmap_dir, "pairs.txt")).readlines()
    assert (len(img_database) == int(camera_pair_txt[0]))
    camera_pair_txt.pop(0)
    camera_pair_data = [np.asarray(item.strip().split(" ")[1:], dtype=np.float32).reshape(-1, 2) for item in
                        camera_pair_txt[1::2]]

    return img_database, graphs, camera_pair_data, points_from_sfm



# Remove the redundant face and edges in the graph
# And build the dual graph in order to navigate between patches
def fix_graph(v_graph, is_visualize=False):
    dual_graph = nx.Graph()

    id_original_to_current = {}
    for id_face, face in enumerate(v_graph.graph["faces"]):
        if not v_graph.graph["face_flags"][id_face]:
            continue
        has_black = False
        for idx, id_start in enumerate(face):
            id_end = face[(idx + 1) % len(face)]
            if v_graph.edges[(id_start, id_end)]["is_black"]:
                has_black = True
                break
        if has_black:
            continue

        id_original_to_current[id_face] = len(dual_graph.nodes)
        dual_graph.add_node(len(dual_graph.nodes), id_vertex=face,
                            id_in_original_array=id_face,
                            face_center=v_graph.graph['face_center'][id_face],
                            ray_c=v_graph.graph['ray_c'][id_face])

    for node in dual_graph.nodes():
        faces = dual_graph.nodes[node]["id_vertex"]
        for idx, id_start in enumerate(faces):
            id_end = faces[(idx + 1) % len(faces)]
            t = copy(v_graph.edges[(id_start, id_end)]["id_face"])
            t.remove(dual_graph.nodes[node]["id_in_original_array"])
            if t[0] in id_original_to_current:
                edge = (node, id_original_to_current[t[0]])
                if edge not in dual_graph.edges():
                    id_points_in_another_face = dual_graph.nodes[id_original_to_current[t[0]]]["id_vertex"]
                    adjacent_vertices = []
                    id_cur = idx
                    while True:
                        adjacent_vertices.append(id_start)
                        id_cur += 1
                        if id_cur >= len(faces):
                            adjacent_vertices.append(id_end)
                            break
                        id_start = faces[id_cur]
                        id_end = faces[(id_cur + 1) % len(faces)]
                        if id_end not in id_points_in_another_face:
                            adjacent_vertices.append(id_start)
                            break

                    # adjacent_vertices.append(id_start)
                    dual_graph.add_edge(edge[0], edge[1], adjacent_vertices=adjacent_vertices)

    v_graph.graph["dual_graph"] = dual_graph

    # Visualize
    if is_visualize:
        for idx, id_face in enumerate(dual_graph.nodes):
            print("{}/{}".format(idx, len(dual_graph.nodes)))
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]
            face = dual_graph.nodes[id_face]["id_vertex"]

            for idx, id_start in enumerate(face):
                id_end = face[(idx + 1) % len(face)]
                pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                cv2.line(img11, pos1, pos2, (0, 0, 255), 2)

            for id_another_face in dual_graph[id_face]:
                face = dual_graph.nodes[id_another_face]["id_vertex"]
                for idx, id_start in enumerate(face):
                    id_end = face[(idx + 1) % len(face)]
                    pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                    pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                    cv2.line(img11, pos1, pos2, (0, 255, 255), 1)

                for id_vertex in dual_graph[id_face][id_another_face]["adjacent_vertices"]:
                    pos1 = np.around(v_graph.nodes[id_vertex]["pos_2d"] * shape).astype(np.int64)
                    cv2.circle(img11, pos1, 2, (0, 255, 0), 2)

            cv2.imshow("1", img11)
            cv2.waitKey()
    return


def determine_valid_edges(v_graph, v_img, v_gradient):
    for edge in v_graph.edges():
        pos1 = v_graph.nodes[edge[0]]["pos_2d"]
        pos2 = v_graph.nodes[edge[1]]["pos_2d"]
        pos = torch.from_numpy(np.stack((pos1, pos2), axis=0).astype(np.float32)).to(v_img.device).unsqueeze(0)

        ns, s = sample_points_2d(pos,
                                 torch.tensor([100] * pos.shape[0], dtype=torch.long, device=pos.device),
                                 v_img_width=v_img.shape[1], v_vertical_length=10)
        pixels = sample_img(v_img[None, None, :, :], s[None, :])[0]

        mean_pixels = scatter_mean(pixels,
                                   torch.arange(ns.shape[0], device=pos.device).repeat_interleave(ns),
                                   dim=0)
        v_graph.edges[edge]["is_black"] = mean_pixels < 0.05
        pass

    return


# wireframe src face
def visualize_polygon(points, path):
    with open(path, 'w') as f:
        # 写入顶点
        for point in points:
            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
        # 写入多边形的面
        f.write("f")
        for i in range(1, len(points) + 1):
            f.write(f" {i}")
        f.write("\n")


def visualize_polygon_with_normal(projected_points, normal, path, normal_seg_len=1):
    with open(path, 'w') as f:
        for p in projected_points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        # 写入多边形
        f.write("f")
        for idx in range(len(projected_points)):
            f.write(f" {idx + 1}")  # OBJ 索引从 1 开始
        f.write("\n")

    filename, file_extension = os.path.splitext(os.path.basename(path))
    new_filename = filename + "_normal" + file_extension
    new_path = os.path.join(os.path.dirname(path), new_filename)
    with open(new_path, 'w') as f:
        # normal start and end
        normal = normal / np.linalg.norm(normal) * normal_seg_len
        center = np.mean(projected_points, axis=0)
        normal_end = center + normal
        f.write(f"v {center[0]} {center[1]} {center[2]}\n")
        f.write(f"v {normal_end[0]} {normal_end[1]} {normal_end[2]}\n")

        # 写入法向量线段
        f.write(f"l {1} {2}\n")


def initialize_patches(rays_c, ray_distances_c, v_vertex_id_per_face):
    initialized_vertices = rays_c * ray_distances_c[:, None]
    # abcd
    plane_parameters = []
    for vertex_id in v_vertex_id_per_face:
        pos_vertexes = initialized_vertices[vertex_id]
        assert (len(pos_vertexes) >= 3)
        # a) 3d vertexes of each patch -> fitting plane
        p_abcd = fit_plane_svd(pos_vertexes)
        plane_parameters.append(p_abcd)

    return torch.stack(plane_parameters, dim=0)


def optimize_plane(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    img_src_id = 0
    optimized_abcd_list_v = []

    for id_img1, graph in enumerate(v_graphs):
        # 1. Prepare data
        # prepare some data
        # if id_img1 != 3:
        #     continue
        id_src_imgs = (v_img_pairs[id_img1][:, 0]).astype(np.int64)
        ref_img = cv2.imread(v_img_database[id_img1].img_path, cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path, cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.

        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic

        # pos_in_c1 = v_img_database[int(id_img1)].extrinsic @ v_img_database[1].pos[None, :]

        # transformation store the transformation matrix from ref_img to src_imgs
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        c1_2_c2 = torch.from_numpy(
            v_img_database[int(id_src_imgs[img_src_id])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        ).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        c1_2_c2_list = []
        for i in range(len(id_src_imgs)):
            c1_2_c2_list.append(torch.from_numpy(
                v_img_database[int(id_src_imgs[i])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
            ).to(device).to(torch.float32))

        # Image gradients
        # Do not normalize!
        gy, gx = torch.gradient(imgs[0])
        gradients1 = torch.stack((gx, gy), dim=-1)
        gy, gx = torch.gradient(imgs[img_src_id + 1])
        gradients2 = torch.stack((gx, gy), dim=-1)

        dilated_gradients1 = torch.from_numpy(dilate_edge(gradients1)).to(device)
        dilated_gradients2 = torch.from_numpy(dilate_edge(gradients2)).to(device)

        determine_valid_edges(graph, imgs[0], dilated_gradients1)
        fix_graph(graph)

        # Visualize
        if False:
            mask1 = (torch.linalg.norm(gradients1, dim=-1))
            mask2 = (torch.linalg.norm(dilated_gradients1, dim=-1))
            cv2.imshow("1", torch.cat((mask1, mask2), dim=1).cpu().numpy().astype(np.float32))
            cv2.waitKey()

        # Rays
        rays_c = [None] * len(graph.nodes)
        ray_distances_c = [None] * len(graph.nodes)
        for idx, id_points in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_points]["ray_c"]
            ray_distances_c[idx] = graph.nodes[id_points]["distance"]
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        ray_distances_c = torch.from_numpy(np.stack(ray_distances_c)).to(device).to(torch.float32)

        dual_graph = graph.graph["dual_graph"]
        vertex_id_per_face = [dual_graph.nodes[id_node]["id_vertex"] for id_node in dual_graph.nodes]
        centroid_rays_c = torch.from_numpy(
            np.stack([dual_graph.nodes[id_node]["ray_c"] for id_node in dual_graph], axis=0)).to(device)

        # 2. Initialize planes for each patch
        num_patch = len(dual_graph.nodes)

        initialized_planes = initialize_patches(rays_c, ray_distances_c, vertex_id_per_face)  # (num_patch, 4)

        # 3. Optimize to get abcd_list for current ref-src img pair
        # v_log_root_c = os.path.join(os.path.normpath(v_log_root), str(id_img1))
        # os.makedirs(v_log_root_c, exist_ok=True)
        # optimized_abcd_list = optimize_planes_batch(initialized_planes, rays_c, centroid_rays_c, dual_graph,
        #                                             imgs, transformation, intrinsic, c1_2_c2_list, v_log_root_c)
        # optimized_abcd_list_v.append(optimized_abcd_list)

        if os.path.exists("output/init_optimized_abcd_list.pkl"):
            optimized_abcd_list = pickle.load(open("output/init_optimized_abcd_list.pkl", "rb"))
        else:
            #initialized_planes = pickle.load(open("output/bu2/optimized_abcd_list_merged.pkl", "rb"))
            #initialized_planes = torch.from_numpy(initialized_planes).to(device)
            optimized_abcd_list = optimize_planes_batch(initialized_planes, rays_c, centroid_rays_c, dual_graph,
                                                        imgs, transformation, intrinsic, c1_2_c2_list, v_log_root)
            # save optimized_abcd_list
            # with open("output/init_optimized_abcd_list.pkl", "wb") as f:
            #     pickle.dump(optimized_abcd_list, f)

        # 4. Local assemble
        merged_dual_graph, optimized_abcd_list = local_assemble(optimized_abcd_list, rays_c, centroid_rays_c,
                                                                dual_graph, imgs, dilated_gradients1,
                                                                dilated_gradients2, transformation, intrinsic,
                                                                c1_2_c2, v_log_root)

        centroid_rays_c_new = [merged_dual_graph.nodes[i]['ray_c'].tolist() for i in
                               range(len(merged_dual_graph.nodes))]
        centroid_rays_c_new = torch.from_numpy(np.stack(centroid_rays_c_new)).to(device).to(torch.float32)

        optimized_abcd_list = optimize_planes_batch(copy(optimized_abcd_list), rays_c, centroid_rays_c_new,
                                                    merged_dual_graph, imgs, dilated_gradients1, dilated_gradients2,
                                                    transformation,
                                                    intrinsic,
                                                    c1_2_c2,
                                                    v_log_root
                                                    )

    # 5. Global assemble
    #global_assemble(optimized_abcd_list_v, transformation, intrinsic, c1_2_c2_list, v_log_root)


@hydra.main(config_name="phase6.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    # data = img_database, graphs, camera_pair_data, points_from_sfm
    data = prepare_dataset_and_model(
        v_cfg["dataset"]["colmap_dir"],
        v_cfg["dataset"]["id_viz_face"],
        v_cfg["dataset"]["scene_boundary"],
        v_cfg["dataset"]["v_reconstruct_data"],
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # optimize(data, v_cfg["trainer"]["output"])
    optimize_plane(data, v_cfg["trainer"]["output"])


if __name__ == '__main__':
    main()
