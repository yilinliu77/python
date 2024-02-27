import itertools
import sys, os, yaml
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
from src.neural_recon.graphmatch import PairGraphMatcher, GraphFusion
from src.neural_recon.GraphPlaneOptimization import GraphPlaneOptimiser


def is_convex_polygon(points):
    assert len(points) >= 3

    points_rolled_1 = np.roll(points, -1, axis=0)
    points_rolled_2 = np.roll(points, -2, axis=0)

    vectors_1 = points_rolled_1 - points
    vectors_2 = points_rolled_2 - points_rolled_1

    cross_products = vectors_1[:, 0] * vectors_2[:, 1] - vectors_1[:, 1] * vectors_2[:, 0]
    signs = np.sign(cross_products)

    return np.all(signs == signs[0])


def check_edge_validity(v_img, v_gradient, faces, vertices, vis=False):
    edges_id = []
    for id_face, id_vertices_per_face in enumerate(faces):
        id_edge_per_face = [(id_vertices_per_face[idx - 1], id_vertices_per_face[idx])
                            for idx in range(len(id_vertices_per_face))]
        edges_id.extend(id_edge_per_face)
    edges_id = list(set(tuple(sorted(edge)) for edge in edges_id))
    edges_pos = [(vertices[edge[0]], vertices[edge[1]]) for edge in edges_id]

    black_list = []
    gradient_free_list = []
    short_length_list = []
    img_np = v_img.cpu().numpy()

    for idx, edge in enumerate(edges_pos):
        pos1, pos2 = edge[0], edge[1]
        pos = torch.from_numpy(np.stack((pos1, pos2)).astype(np.float32)).to(v_img.device).unsqueeze(0)
        length = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)

        ns, s = sample_points_2d(pos,
                                 torch.tensor([1000] * pos.shape[0], dtype=torch.long, device=pos.device),
                                 v_img_width=v_img.shape[1], v_vertical_length=0)

        pixels_endpoints = sample_img(v_img[None, None, :, :], pos)[0]
        pixels = sample_img(v_img[None, None, :, :], s[None, :])[0]
        # v_gradient = (torch.linalg.norm(v_gradient, dim=-1))
        # v_gradient[v_gradient >= 0.001], v_gradient[v_gradient < 0.001] = 1, 0
        # v_gradient = v_gradient.unsqueeze(2)
        gradient = sample_img(v_gradient[None, :].permute(0, 3, 1, 2), s[None, :, :])[0]

        mean_pixels = scatter_mean(pixels,
                                   torch.arange(ns.shape[0], device=pos.device).repeat_interleave(ns),
                                   dim=0)
        mean_gradient = (torch.linalg.norm(gradient, dim=-1)).mean()
        black_rate = (pixels.squeeze(1) == 0).sum() / pixels.squeeze(1).shape[0]

        gradient_free_rate = (torch.linalg.norm(gradient, dim=-1) < 0.0001).sum() / gradient.shape[0]

        # dot product
        direction = torch.tensor(pos1 - pos2).to(pos.device).to(torch.float32)
        dot_product = normalize_tensor(gradient) @ normalize_tensor(direction.unsqueeze(0)).T
        mean_dot_product = dot_product.mean().abs()

        black_list.append(edges_id[idx]) if black_rate > 0.5 else None
        gradient_free_list.append(edges_id[idx]) if gradient_free_rate > 0.001 and black_rate < 0.5 else None
        # gradient_free_list.append(edges_id[idx]) if ((mean_dot_product > 0.5 or gradient_free_rate > 0.05)
        #                                              and black_rate < 0.5) else None
        short_length_list.append(edges_id[idx]) if length < (2 / v_img.shape[0]) else None

        if vis:
            def log_transform(tensor):
                tensor = tensor - tensor.min() + 1  # shift to make all elements positive
                return torch.log10(tensor)

            mask1 = (torch.linalg.norm(v_gradient, dim=-1))
            mask1[mask1 > 0.01] = 1
            # mask1 = log_transform(mask1)
            # mask1 = (mask1 - mask1.min()) / (mask1.max() - mask1.min())
            mask1 = mask1.cpu().numpy().astype(np.float32)
            mask1_rgb = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB)

            if gradient_free_rate > 0.01 and black_rate < 0.5:
                color = [0, 0, 255]
            else:
                color = [0, 255, 255]

            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            s_pixel = (s * torch.tensor(v_img.shape[::1], device=s.device)).round().long()
            for point in s_pixel:
                x, y = point.tolist()  # 注意：在图像中，y 对应行数，x 对应列数
                if 0 <= y < img_np.shape[0] and 0 <= x < img_np.shape[1]:  # 确保坐标在图像范围内
                    img_rgb[y, x, :] = color
                    mask1_rgb[y, x, :] = color

            img_show = np.hstack((mask1_rgb, img_rgb))
            cv2.putText(img_show,
                        "mean_dot_product: {:.3f}, gradient_free_rate: {:.3f}".format(mean_dot_product,
                                                                                      gradient_free_rate),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("1", img_show)
            if cv2.waitKey() == 27:
                vis = False

    return black_list, gradient_free_list, short_length_list


def remove_short_edges(faces, short_length_edges, v_gradient, vertices, v_img_rgb, is_vis=False):
    # 1. fix the short_length_edges, any edge will share endpoints with other edges,
    # ex: [(1,2),(2,3)], then we need to merge them into (1,2,3), and consider which one to save in (1,2,3)
    G = nx.Graph()
    G.add_edges_from(short_length_edges)
    merged_list = []
    for node in G.nodes:
        adjacent_nodes = list(G.neighbors(node))
        if len(adjacent_nodes) > 1:
            merged_list.append(tuple(sorted([node] + adjacent_nodes)))

    short_length_edges_removed = copy(short_length_edges)
    for edge in short_length_edges:
        if any(edge[0] in merged and edge[1] in merged for merged in merged_list):
            short_length_edges_removed.remove(edge)
    near_points_list = short_length_edges_removed + merged_list

    # 2. determine which endpoint of a short edge to remove by comparing their edge_gradient
    G = nx.Graph()
    for face in faces:
        for i in range(len(face)):
            G.add_edge(face[i], face[(i - 1) % len(face)])
            G.add_edge(face[i], face[(i + 1) % len(face)])

    def eval_merged(near_p_list, remain_p):
        def compute_edge_gradient(pos1_idx, pos2_idx):
            pos1, pos2 = vertices[pos1_idx], vertices[pos2_idx]
            pos = torch.from_numpy(np.stack((pos1, pos2)).astype(np.float32)).to(v_gradient.device).unsqueeze(0)
            ns, s = sample_points_2d(pos,
                                     torch.tensor([1000] * pos.shape[0], dtype=torch.long, device=pos.device),
                                     v_img_width=v_img_rgb.shape[1], v_vertical_length=0)
            gradient = sample_img(v_gradient[None, :].permute(0, 3, 1, 2), s[None, :, :])[0]
            return torch.linalg.norm(gradient, dim=-1).mean()

        # mapping the points need to be removed to p_idx_c
        removed_p_mapping = {item: remain_p for item in near_p_list if item != remain_p}
        gradient_list = []

        # for the remained point, compute the gradient of its adjacent edges (beside the edge need to be removed)
        for adj_p in list(G.neighbors(remain_p)):
            if adj_p in removed_p_mapping:
                continue
            gradient_edge = compute_edge_gradient(remain_p, adj_p)
            gradient_list.append(gradient_edge)

        # for the points need to be removed, consturct the new edge and compute the gradient
        for remove_p, replace_p in removed_p_mapping.items():
            for adj_p in list(G.neighbors(remove_p)):
                if adj_p in removed_p_mapping or adj_p == replace_p:
                    continue
                # we need to change the remove_p to replace_p
                gradient_edge = compute_edge_gradient(replace_p, adj_p)
                gradient_list.append(gradient_edge)

        return torch.stack(gradient_list)

    replaced_vertices = {}
    for near_points_list_c in near_points_list:
        edge_gradients = []
        for p_idx in near_points_list_c:
            edge_gradients.append(eval_merged(near_points_list_c, p_idx))
        edge_gradients_mean = [gradient.mean() for gradient in edge_gradients]
        min_gradient_idx = edge_gradients_mean.index(min(edge_gradients_mean))
        for i, p_idx in enumerate(near_points_list_c):
            if i != min_gradient_idx:
                replaced_vertices[p_idx] = near_points_list_c[min_gradient_idx]

    # 3. del the point to remove
    new_faces = []
    for face in faces:
        # we replace the second vertex of each short edge with the first vertex
        new_face = []
        for vertex in face:
            if vertex in replaced_vertices:
                vertex = replaced_vertices[vertex]
            new_face.append(vertex)
        # remove the repeated vertex
        new_face = [v for i, v in enumerate(new_face) if v != new_face[(i - 1) % len(new_face)]]
        new_faces.append(new_face)

    if is_vis:
        def vis(faces_c, img_rgb_c):
            v_img = cv2.cvtColor(img_rgb_c, cv2.COLOR_BGR2GRAY)
            for face_c in faces_c:
                edge_cur_face = [(face_c[i], face_c[(i + 1) % len(face_c)]) for i in range(len(face_c))]
                for edge in edge_cur_face:
                    start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb_c, start, end, (0, 255, 255), thickness=1)

                    if edge in short_length_edges or edge[::-1] in short_length_edges:
                        cv2.circle(img_rgb_c, tuple(start), 3, (0, 0, 255), thickness=1)
                        cv2.circle(img_rgb_c, tuple(end), 6, (255, 0, 0), thickness=1)
            return img_rgb_c

        faces_vis_img = vis(faces, copy(v_img_rgb))
        new_faces_vis_img = vis(new_faces, copy(v_img_rgb))
        vis_img = np.hstack([faces_vis_img, new_faces_vis_img])
        cv2.imshow("1", vis_img)
        cv2.waitKey()
    return new_faces


def remove_edges(faces, vertices, gradient_free_edges, v_img_rgb, is_vis=False):
    graph = nx.Graph()
    graph.add_nodes_from([(idx, {"vertices_id": vertices_id, "sub_face_id": [idx]})
                          for idx, vertices_id in enumerate(faces)])
    for i, face1 in enumerate(faces):
        for j, face2 in enumerate(faces[i + 1:], i + 1):
            edges1 = [(face1[i], face1[(i + 1) % len(face1)]) for i in range(len(face1))]
            edges2 = [(face2[i], face2[(i + 1) % len(face2)]) for i in range(len(face2))]

            edges1_set = set(frozenset(edge) for edge in edges1)
            edges2_set = set(frozenset(edge) for edge in edges2)
            share_edges = edges1_set & edges2_set
            share_edges = [sorted(list(edge)) for edge in share_edges]

            # share_edge = sorted(list(set(face1) & set(face2)))
            # assert len(share_edge) <= 2
            if len(share_edges) >= 1:
                graph.add_edge(i, j, share_edges=share_edges)

    merged_target = {}  # key face is merged to value face
    for gradient_free_edge in gradient_free_edges:
        if is_vis:
            img_rgb = copy(v_img_rgb)
            v_img = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY)
            start = (vertices[gradient_free_edge[0]] * v_img.shape[::1]).astype(np.int32)
            end = (vertices[gradient_free_edge[1]] * v_img.shape[::1]).astype(np.int32)
            cv2.line(img_rgb, start, end, (255, 0, 0), thickness=1)
            cv2.imshow("1", img_rgb)
            cv2.waitKey()

        # When v is merged into u, v's other neighbors (ex:k) will be redirected to u,
        # and the edge properties of v and k will be retained on the edge of u and k,
        # but if u and k already have an edge, the edges of u and k will continue to exist with the same attributes,
        # and the edge properties of v and k will be retained as contraction properties between u and k.
        def extract_share_edge(data):
            result = []
            if isinstance(data, dict):
                for key, value in data.items():
                    result.extend(value['share_edges'])  # no need to check, because the share_edge is must exist
                    if 'contraction' in value.keys():
                        result.extend(extract_share_edge(value['contraction']))
            return result

        # find the (only) two faces adjacent to the edge
        edge_to_remove = []
        for e in graph.edges(data=True):
            data = e[2]
            check_edge_list = []
            check_edge_list.extend(data['share_edges'])
            if 'contraction' in data:
                result = extract_share_edge(data['contraction'])
                check_edge_list.extend(result)

            if sorted(list(gradient_free_edge)) in check_edge_list:
                edge_to_remove.append(e)
                break

        # edge_to_remove = [e for e in graph.edges(data=True) if sorted(list(gradient_free_edge)) == e[2]['share_edge']]
        if edge_to_remove:
            u, v, data = edge_to_remove[0]
            share_edge_list = []
            share_edge_list.extend(data['share_edges'])

            if 'contraction' in graph[u][v].keys():
                # for another_edge in graph[u][v]['contraction'].values():
                #     share_edge_list.append(another_edge['share_edge'])
                result = extract_share_edge(graph[u][v]['contraction'])
                share_edge_list.extend(result)

            # merge face
            face1 = graph.nodes[u]["vertices_id"]
            face2 = graph.nodes[v]["vertices_id"]
            if is_vis:
                img_rgb = copy(v_img_rgb)
                v_img = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY)
                # vis face1
                for i in range(len(face1)):
                    start = (vertices[face1[i]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[face1[(i + 1) % len(face1)]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb, tuple(start), tuple(end), (0, 0, 255), thickness=1)
                cv2.imshow("1", img_rgb)
                cv2.waitKey()
                # vis face2
                for i in range(len(face2)):
                    start = (vertices[face2[i]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[face2[(i + 1) % len(face2)]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb, tuple(start), tuple(end), (0, 255, 0), thickness=1)
                cv2.imshow("1", img_rgb)
                cv2.waitKey()
                # vis share_edge
                for share_edge in share_edge_list:
                    start = (vertices[share_edge[0]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[share_edge[1]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb, tuple(start), tuple(end), (255, 0, 0), thickness=1)
                cv2.imshow("1", img_rgb)
                cv2.waitKey()

            face1_edge = [[face1[i], face1[(i + 1) % len(face1)]] for i in range(len(face1))]
            face2_edge = [[face2[i], face2[(i + 1) % len(face2)]] for i in range(len(face2))]
            for share_edge in share_edge_list:
                face1_edge.remove(share_edge) if share_edge in face1_edge else None
                face1_edge.remove(share_edge[::-1]) if share_edge[::-1] in face1_edge else None
                face2_edge.remove(share_edge) if share_edge in face2_edge else None
                face2_edge.remove(share_edge[::-1]) if share_edge[::-1] in face2_edge else None

            if len(face1_edge) == 0 or len(face2_edge) == 0:
                continue

            merged_face_edge = []
            for edge in face1_edge + face2_edge:
                assert (edge not in merged_face_edge and edge[::-1] not in merged_face_edge)
                # if edge in merged_face_edge or edge[::-1] in merged_face_edge:
                #     continue
                merged_face_edge.append(edge)

            # connect the edge to new face
            merged_face = []
            merged_face.extend(merged_face_edge[0])
            for i in range(1, len(merged_face_edge)):
                for j in range(1, len(merged_face_edge)):
                    if merged_face_edge[j][0] == merged_face[-1]:
                        merged_face.append(merged_face_edge[j][1])
                        break

            # check loop and refine
            assert len(merged_face) > 2

            if merged_face[0] != merged_face[-1]:
                print("loop not closed!")
                continue
            merged_face.pop(-1)

            if len(merged_face) != len(merged_face_edge):
                print("multi loop!")
                continue

            sub_face_id_u = copy(list(graph.nodes[u]["sub_face_id"]))
            sub_face_id_v = copy(list(graph.nodes[v]["sub_face_id"]))

            # this will del v, and the edge connect to v will reconnect to u
            graph = nx.contracted_nodes(graph, u, v, self_loops=False)
            graph.nodes[u]["vertices_id"] = merged_face
            graph.nodes[u]["sub_face_id"] = list(set(sub_face_id_u + sub_face_id_v))
            # record: v is merged to u
            merged_target[v] = u

            if is_vis:
                img_rgb = copy(v_img_rgb)
                v_img = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY)
                edge_cur_face = [(merged_face[i], merged_face[(i + 1) % len(merged_face)]) for i in
                                 range(len(merged_face))]
                for edge in edge_cur_face:
                    start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb, start, end, (0, 255, 255), thickness=1)
                for vertex in vertices[merged_face]:
                    cv2.circle(img_rgb, (int(vertex[0] * v_img.shape[1]), int(vertex[1] * v_img.shape[0])),
                               1, (0, 0, 255), thickness=2)
                for share_edge in share_edge_list:
                    start = (vertices[share_edge[0]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[share_edge[1]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(img_rgb, start, end, (255, 0, 0), thickness=1)
                    cv2.circle(img_rgb, start, 1, (255, 0, 0), thickness=2)
                    cv2.circle(img_rgb, end, 1, (255, 0, 0), thickness=2)
                # cv2.putText(img_rgb, str(share_edge), start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("1", img_rgb)
                if cv2.waitKey() == 27:
                    is_vis = False

    merged_faces = [list(data['vertices_id']) for _, data in graph.nodes(data=True)]
    sub_faces_id = [list(data['sub_face_id']) for _, data in graph.nodes(data=True)]
    sub_faces_vertices_id = [[faces[i] for i in sub_face_id] for sub_face_id in sub_faces_id]
    return merged_faces, sub_faces_id, sub_faces_vertices_id


def remove_invaild_faces(faces, vertices, v_img_rgb, is_vis=False):
    def check_invaildity(face_vertices, tol=3):
        edges_c = [face_vertices[i] - face_vertices[i - 1] for i in range(len(face_vertices))]
        angles = []
        for i in range(len(edges_c)):
            dot_product = np.dot(edges_c[i], edges_c[i - 1])
            norms = np.linalg.norm(edges_c[i]) * np.linalg.norm(edges_c[i - 1])
            angle = np.arccos(dot_product / (norms + 1e-6))
            angles.append(np.degrees(angle))

        if False:
            v_img_grey = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY)
            img_rgb = copy(v_img_rgb)
            edges_c = [(face_vertices[i], face_vertices[i - 1]) for i in range(len(face_vertices))]
            color_ = (0, 0, 255) if all(np.abs(a - 180) < tol or np.abs(a) < tol for a in angles) else (0, 255, 0)
            for edge in edges_c:
                start = (edge[0] * v_img_grey.shape[::1]).astype(np.int32)
                end = (edge[1] * v_img_grey.shape[::1]).astype(np.int32)
                cv2.line(img_rgb, start, end, color_, thickness=1)
                cv2.circle(img_rgb, start, 1, (0, 255, 255), thickness=2)
                cv2.circle(img_rgb, end, 1, (0, 255, 255), thickness=2)

            cv2.putText(img_rgb, str(angles), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("1", img_rgb)
            cv2.waitKey()

        return all(np.abs(a - 180) < tol or np.abs(a) < tol for a in angles)

    valid_faces = []
    removed_faces = []
    removed_faces_id = []
    for idx, face in enumerate(faces):
        if check_invaildity(vertices[face]):
            removed_faces_id.append(idx)
            removed_faces.append(face)
        else:
            valid_faces.append(face)
    if is_vis:
        # transform to grey
        v_img_grey = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY)
        for face in faces:
            if face in removed_faces:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                continue
            if not is_vis:
                break
            img_rgb = copy(v_img_rgb)
            edges = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
            for edge in edges:
                start = (vertices[edge[0]] * v_img_grey.shape[::1]).astype(np.int32)
                end = (vertices[edge[1]] * v_img_grey.shape[::1]).astype(np.int32)
                cv2.line(img_rgb, start, end, color, thickness=1)
                cv2.circle(img_rgb, start, 1, (0, 255, 255), thickness=2)
                cv2.circle(img_rgb, end, 1, (0, 255, 255), thickness=2)
            cv2.imshow("1", img_rgb)
            if cv2.waitKey() == 27:
                is_vis = False

    return removed_faces_id, removed_faces


###################################################################################################
# 1. Input
###################################################################################################
# @ray.remote
def read_graph(v_filename, img_path, img_size, device, is_vis=False):
    v_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    v_img_rgb_src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    v_img = torch.from_numpy(v_img).to(device).to(torch.float32) / 255.
    gy, gx = torch.gradient(v_img)
    gradients1 = torch.stack((gx, gy), dim=-1)
    dilated_gradients1 = torch.from_numpy(dilate_edge(gradients1, 1)).to(device)

    data = [item for item in open(v_filename).readlines()]
    vertices = [item.strip().split(" ")[1:-1] for item in data if item[0] == "v"]
    vertices = np.asarray(vertices).astype(np.float32) / img_size
    faces = [item.strip().split(" ")[1:] for item in data if item[0] == "f"]
    faces = [(np.asarray(face).astype(np.int32) - 1).tolist() for face in faces]  # faces - 1 because of the obj format

    # check edge validity and fix faces
    black_edge_list, gradient_free_list, short_length_list = check_edge_validity(v_img, dilated_gradients1, faces,
                                                                                 vertices)
    # if not pre_process
    # black_edge_list, gradient_free_list, short_length_list = [], [], []

    src_check_result = copy(v_img_rgb_src)
    if is_vis:
        for face in faces:
            edge_cur_face = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
            for edge in edge_cur_face:
                start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                if edge in black_edge_list or edge[::-1] in black_edge_list:
                    color = (0, 255, 0)
                elif edge in gradient_free_list or edge[::-1] in gradient_free_list:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)
                cv2.line(src_check_result, start, end, color, thickness=1)

    if short_length_list:
        faces = remove_short_edges(faces, short_length_list, dilated_gradients1, vertices, copy(v_img_rgb_src))
        faces = [f for f in faces if len(f) >= 3]
        black_edge_list, gradient_free_list, _ = check_edge_validity(v_img, dilated_gradients1, faces, vertices)

    # remove the invalid (almost a line) faces, we just need to remove one of its edges, the whole face will be removed
    removed_faces_id, removed_faces = remove_invaild_faces(faces, vertices, copy(v_img_rgb_src))
    invalid_faces_edge = []
    for removed_face in removed_faces:
        edges = [(removed_face[i], removed_face[(i + 1) % len(removed_face)]) for i in range(len(removed_face))]
        invalid_faces_edge.extend([edges[0]])

    removed_edge_list = gradient_free_list + invalid_faces_edge
    removed_edge_list = list(set(tuple(sorted(edge)) for edge in removed_edge_list))

    src_faces = copy(faces)
    merged_faces, sub_faces_id, sub_faces = remove_edges(faces, vertices, removed_edge_list, copy(v_img_rgb_src),
                                                         is_vis)

    if is_vis:
        v_img_rgb = copy(v_img_rgb_src)
        for face in faces:
            edge_cur_face = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
            for edge in edge_cur_face:
                start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                if edge in black_edge_list or edge[::-1] in black_edge_list:
                    color = (0, 255, 0)
                elif edge in gradient_free_list or edge[::-1] in gradient_free_list:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)
                cv2.line(v_img_rgb, start, end, color, thickness=1)
        # cv2.arrowedLine(v_img_rgb, start, end, color, thickness=1, line_type=cv2.LINE_AA)
        # cv2.imshow("1", v_img_rgb)
        # cv2.waitKey()

        for idx, face in enumerate(merged_faces):
            v_img_rgb_merged = copy(v_img_rgb_src)
            cv2.putText(v_img_rgb_merged, str(face), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            edge_cur_face = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
            for edge in edge_cur_face:
                start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                if edge in black_edge_list or edge[::-1] in black_edge_list:
                    color = (0, 255, 0)
                elif edge in gradient_free_list or edge[::-1] in gradient_free_list:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)
                cv2.arrowedLine(v_img_rgb_merged, start, end, color, thickness=1, line_type=cv2.LINE_AA)
                cv2.circle(v_img_rgb_merged, start, 1, (255, 0, 0), thickness=2)
                cv2.circle(v_img_rgb_merged, end, 1, (255, 0, 0), thickness=2)

            sub_faces_c = sub_faces[idx]
            v_img_rgb_sub_contex = copy(v_img_rgb_src)
            for sub_face in sub_faces_c:
                edge_cur_face = [(sub_face[i], sub_face[(i + 1) % len(sub_face)]) for i in range(len(sub_face))]
                for edge in edge_cur_face:
                    start = (vertices[edge[0]] * v_img.shape[::1]).astype(np.int32)
                    end = (vertices[edge[1]] * v_img.shape[::1]).astype(np.int32)
                    cv2.line(v_img_rgb_sub_contex, start, end, (0, 255, 255), thickness=1)
            # cv2.imshow("1", v_img_rgb_sub_contex)
            # cv2.waitKey()

            # v_img_rgb_top = np.hstack((src_check_result, v_img_rgb))
            # v_img_rgb_bottom = np.hstack((v_img_rgb_merged, v_img_rgb_sub_contex))
            # v_img_rgb_merged = np.vstack((v_img_rgb_top, v_img_rgb_bottom))
            v_img_rgb_merged = np.hstack((v_img_rgb, v_img_rgb_merged, v_img_rgb_sub_contex))
            scale_factor = 10 / 16
            new_size = (int(v_img_rgb_merged.shape[1] * scale_factor), int(v_img_rgb_merged.shape[0] * scale_factor))
            v_img_rgb_merged = cv2.resize(v_img_rgb_merged, new_size)
            cv2.imshow("1", v_img_rgb_merged)
            if cv2.waitKey() == 27:
                break

    faces = merged_faces

    graph = nx.Graph()
    graph.add_nodes_from([(idx, {"pos_2d": item}) for idx, item in enumerate(vertices)])
    for id_face, id_vertices_per_face in enumerate(faces):
        id_edge_per_face = [(id_vertices_per_face[idx - 1], id_vertices_per_face[idx]) for idx in
                            range(len(id_vertices_per_face))]
        graph.add_edges_from(id_edge_per_face)

    graph.graph["faces"] = faces
    graph.graph['sub_faces_id'] = sub_faces_id
    graph.graph["sub_faces"] = sub_faces
    graph.graph["src_faces"] = src_faces

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


def prepare_dataset_and_model(v_colmap_dir, v_viz_face, v_bounds, v_reconstruct_data,
                              v_max_error_for_initial_sfm
                              ):
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

    points_cache_name = "output/img_field_test/points_cache.npy"
    if os.path.exists(points_cache_name) and not v_reconstruct_data:
        points_from_sfm = np.load(points_cache_name)
    else:
        preserved_points = []
        for point in tqdm(points_3d):
            if point.error < v_max_error_for_initial_sfm and len(point.tracks) > 3:
                preserved_points.append(point)
        if len(preserved_points) == 0:
            points_from_sfm = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        else:
            points_from_sfm = np.stack([item.pos for item in preserved_points])
        export_point_cloud("output/img_field_test/filter_sfm_points.ply", points_from_sfm)
        np.save(points_cache_name, points_from_sfm)

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

    final_graph_cache_name = "output/img_field_test/final_graph_cache.npy"
    if os.path.exists(final_graph_cache_name) and not v_reconstruct_data and False:
        graphs = np.load(final_graph_cache_name, allow_pickle=True)
    else:
        graph_cache_name = "output/img_field_test/graph_cache.npy"
        print("2. Build graph")
        if os.path.exists(graph_cache_name) and not v_reconstruct_data:
            graphs = np.load(graph_cache_name, allow_pickle=True)
        else:
            graphs = []
            for i_img in tqdm(range(len(img_database))):
                # if i_img != 39:
                #     continue
                graphs.append(
                        read_graph(os.path.join(v_colmap_dir, "wireframe/{}.obj".format(img_database[i_img].img_name)),
                                   img_database[i_img].img_path,
                                   img_database[i_img].img_size,
                                   device=torch.device("cuda")))

            # ray.init(
            #         #local_mode=True
            #         )
            # tasks = [read_graph.remote(
            #         os.path.join(v_colmap_dir, "wireframe/{}.obj".format(img_database[i_img].img_name)),
            #         img_database[i_img].img_path,
            #         img_database[i_img].img_size,
            #         device=torch.device("cuda")
            #         ) for i_img in range(len(img_database))]
            # graphs = ray.get(tasks)
            print("Read {} graphs".format(len(graphs)))
            graphs = np.asarray(graphs, dtype=object)
            np.save(graph_cache_name, graphs, allow_pickle=True)

        print("Start to calculate initial wireframe for each image")

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

            src_faces = v_graph.graph["src_faces"]
            v_graph.graph["src_face_center"] = np.zeros((len(src_faces), 2), dtype=np.float32)
            v_graph.graph["src_face_ray_c"] = np.zeros((len(src_faces), 3), dtype=np.float32)
            v_graph.graph["src_face_distance"] = np.zeros((len(src_faces),), dtype=np.float32)

            # 1. Calculate the centroid of each face
            # merged faces and all src faces
            for id_face, id_vertex_per_face in enumerate(src_faces):
                # Convex assumption
                center_point = np.stack(
                        [v_graph.nodes[id_vertex]["pos_2d"] for id_vertex in id_vertex_per_face]).mean(axis=0)
                v_graph.graph["src_face_center"][id_face] = center_point

            for id_face, id_vertex_per_face in enumerate(v_graph.graph["faces"]):
                # without Convex assumption, using the centroid of first sub-face
                id_vertex_sub_face = v_graph.graph["sub_faces"][id_face][0]
                center_point = np.stack(
                        [v_graph.nodes[id_vertex]["pos_2d"] for id_vertex in id_vertex_sub_face]).mean(axis=0)
                v_graph.graph["face_center"][id_face] = center_point

            points_from_sfm_camera = (v_extrinsic @ np.insert(v_points_3d, 3, 1, axis=1).T).T[:, :3]  # (N, 3)
            # filter out the point behind the camera
            valid_mask = points_from_sfm_camera[:, 2] > 0
            points_from_sfm_camera = points_from_sfm_camera[valid_mask]
            v_points_2d = v_points_2d[valid_mask]
            v_points_3d = v_points_3d[valid_mask]

            # Query points: (M, 2)
            # points from sfm: (N, 2)
            kd_tree = faiss.IndexFlatL2(2)
            kd_tree.add(v_points_2d.astype(np.float32))

            # Prepare query points: Build an array of query points that
            # contains vertices and centroid of each face in the graph
            vertices_2d = np.asarray([v_graph.nodes[id_node]["pos_2d"] for id_node in v_graph.nodes()])  # (M, 2)
            centroids_2d_merged_faces = v_graph.graph["face_center"]
            centroids_2d_src_faces = v_graph.graph["src_face_center"]
            query_points = np.concatenate([vertices_2d, centroids_2d_merged_faces, centroids_2d_src_faces], axis=0)

            # 32 nearest neighbors for each query point.
            shortest_distance, index_shortest_distance = kd_tree.search(query_points, 32)  # (M, K)

            # Select the point which is nearest to the actual ray for each endpoints
            # 1. Construct the ray
            # (M, 3); points in camera coordinates
            ray_c = (np.linalg.inv(v_intrinsic) @ np.insert(query_points, 2, 1, axis=1).T).T
            ray_c = ray_c / np.linalg.norm(ray_c + 1e-6, axis=1, keepdims=True)  # Normalize the points(dir)
            nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
            # Compute the shortest distance from the candidate point to the ray for each query point
            # (M, K, 1): K projected distance of the candidate point along each ray
            distance_of_projection = nearest_candidates @ ray_c[:, :, np.newaxis]
            distance_of_projection[distance_of_projection < 0] = np.inf

            # (M, K, 3): K projected points along the ray
            # 投影距离*单位ray方向 = 投影点坐标
            projected_points_on_ray = distance_of_projection * ray_c[:, np.newaxis, :]
            distance_from_candidate_points_to_ray = np.linalg.norm(
                    nearest_candidates - projected_points_on_ray + 1e-6, axis=2)  # (M, 1)

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

            for id_face in range(len(src_faces)):
                idx = id_face + len(v_graph.nodes) + v_graph.graph["face_center"].shape[0]
                v_graph.graph["src_face_ray_c"][id_face] = ray_c[idx]
                v_graph.graph["src_face_distance"][id_face] = chosen_distances[idx, 0]

            line_coordinates = []
            for edge in v_graph.edges():
                line_coordinates.append(np.concatenate((initial_points_world[edge[0]], initial_points_world[edge[1]])))
            save_line_cloud("output/img_field_test/initial_segments.obj", np.stack(line_coordinates, axis=0))
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(initial_points_world[len(v_graph.nodes):])
            o3d.io.write_point_cloud("output/img_field_test/initial_face_centroid.ply", pc)
            return

        for id_img, img in enumerate(tqdm(img_database)):
            points_from_sfm_local, points_from_sfm_2d = project_points(img.projection, points_from_sfm)
            rgb = cv2.imread(img.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[:, :, None]
            # draw_initial(rgb, graphs[id_img], img)
            compute_initial(graphs[id_img], points_from_sfm_local, points_from_sfm_2d, img.extrinsic, img.intrinsic)
        np.save(final_graph_cache_name, graphs)

    def draw_initial(img, v_graph):
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("1", 1600, 900)
        # cv2.moveWindow("1", 5, 5)
        v_rgb = cv2.imread(img.img_path, cv2.IMREAD_UNCHANGED)
        point_img = v_rgb.copy()
        points_from_sfm_local, points_from_sfm_2d = project_points(img.projection, points_from_sfm)
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

    draw_initial(img_database[0], graphs[0])

    # Read camera pairs
    camera_pair_txt = open(os.path.join(v_colmap_dir, "pairs.txt")).readlines()
    assert (len(img_database) == int(camera_pair_txt[0]))
    camera_pair_txt.pop(0)
    camera_pair_data = [np.asarray(item.strip().split(" ")[1:], dtype=np.float32).reshape(-1, 2) for item in
                        camera_pair_txt[1::2]]

    return img_database, graphs, camera_pair_data, points_from_sfm


def determine_valid_edges(v_graph, v_img, v_gradient):
    img_np = v_img.cpu().numpy()
    gradient_np = v_gradient.cpu().numpy()

    for edge in v_graph.edges():
        pos1 = v_graph.nodes[edge[0]]["pos_2d"]
        pos2 = v_graph.nodes[edge[1]]["pos_2d"]
        pos = torch.from_numpy(np.stack((pos1, pos2), axis=0).astype(np.float32)).to(v_img.device).unsqueeze(0)
        pixels_endpoints = sample_img(v_img[None, None, :, :], pos)[0]
        v_graph.nodes[edge[0]]["is_black"] = (pixels_endpoints[0] < 0.05).all()
        v_graph.nodes[edge[1]]["is_black"] = (pixels_endpoints[1] < 0.05).all()

        length = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)

        ns, s = sample_points_2d(pos,
                                 torch.tensor([1000] * pos.shape[0], dtype=torch.long, device=pos.device),
                                 v_img_width=v_img.shape[1], v_vertical_length=1)
        pixels = sample_img(v_img[None, None, :, :], s[None, :])[0]
        gradient = sample_img(v_gradient[None, :].permute(0, 3, 1, 2), s[None, :, :])[0]
        mean_pixels = scatter_mean(pixels,
                                   torch.arange(ns.shape[0], device=pos.device).repeat_interleave(ns),
                                   dim=0)
        mean_gradient = (torch.linalg.norm(gradient, dim=-1)).mean()
        black_rate = (pixels.squeeze(1) == 0).sum() / pixels.squeeze(1).shape[0]
        gradient_free_rate = (gradient.mean(dim=1) == 0).sum() / gradient.mean(dim=1).shape[0]
        # v_graph.edges[edge]["is_black"] = mean_pixels < 0.2
        v_graph.edges[edge]["is_black"] = black_rate > 0.5
        v_graph.edges[edge]["is_gradient_free"] = gradient_free_rate > 0.01
        v_graph.edges[edge]["gradient"] = mean_gradient
        v_graph.edges[edge]["gradient_free_rate"] = gradient_free_rate
        v_graph.edges[edge]["is_short_length"] = length < 0.01

        if False:
            s_pixel = (s * torch.tensor(v_img.shape[::1], device=s.device)).round().long()
            for point in s_pixel:
                x, y = point.tolist()  # 注意：在图像中，y 对应行数，x 对应列数
                if 0 <= y < img_np.shape[0] and 0 <= x < img_np.shape[1]:  # 确保坐标在图像范围内
                    img_np[y, x] = 0  # BGR 格式，将点设为绿色

            cv2.imshow("1", img_np)
            cv2.waitKey()
        pass

    return


# Remove the redundant (1.any vertices connect with image boundary 2.any edge in black) face and edges in the graph
# And build the dual graph in order to navigate between patches
# v_graph.nodes[i]['valid_flag']: 0 if the vertex is in the image boundary
# v_graph.edges[(i, j)]['valid_flag']: 0 if any endpoint of the edge is in the image boundary
# v_graph.graph["face_flags"][i]: 0 if any node of the face is in the image boundary
# v_graph.edges[(i, j)]['is_black']: 1 if the edge is in the black region
def fix_graph(v_graph, ref_img, is_visualize=False):
    dual_graph = nx.Graph()

    id_original_to_current = {}
    for id_face, face in enumerate(v_graph.graph["faces"]):
        # v_graph.graph["face_flags"][id_face] is set to 0 if any vertices in the image boundary
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
        dual_graph.add_node(len(dual_graph.nodes),
                            id_vertex=face,
                            sub_faces_id=v_graph.graph["sub_faces_id"][id_face],
                            sub_faces=v_graph.graph["sub_faces"][id_face],
                            id_in_original_array=id_face,
                            face_center=v_graph.graph['face_center'][id_face],
                            ray_c=v_graph.graph['ray_c'][id_face])

    for node in dual_graph.nodes():
        faces = dual_graph.nodes[node]["id_vertex"]
        # for each edge in current face
        for idx, id_start in enumerate(faces):
            id_end = faces[(idx + 1) % len(faces)]
            # get the adjacent face of the edge (only one beyond the current idx face)
            t = copy(v_graph.edges[(id_start, id_end)]["id_face"])
            t.remove(dual_graph.nodes[node]["id_in_original_array"])
            # no need to consider the face not in the dual_graph
            if t[0] in id_original_to_current:
                # add a edge between current face and adjacent face to dual graph
                edge = (node, id_original_to_current[t[0]])
                # check, merge if the edge is gradient free
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

    # There are some very close neighboring points in a patch that need to be merged
    # for idx_f, id_face in enumerate(dual_graph.nodes):
    #     face = dual_graph.nodes[id_face]["id_vertex"]
    #     if len(face) == 3:
    #         continue
    #     remove_p_id = []
    #     for idx_p, id_start in enumerate(face):
    #         id_end = face[(idx_p + 1) % len(face)]
    #         pos1 = v_graph.nodes[id_start]["pos_2d"]
    #         pos2 = v_graph.nodes[id_end]["pos_2d"]
    #
    #         if np.linalg.norm(pos1-pos2) < 0.01:
    #             adjacent_vertices = []
    #             for adj in list(dual_graph.adj[id_face].values()):
    #                 adjacent_vertices.extend(adj['adjacent_vertices'])
    #             if id_start not in adjacent_vertices:
    #                 remove_p_id.append(id_start)
    #                 break
    #             elif id_end not in adjacent_vertices:
    #                 remove_p_id.append(id_end)
    #                 break
    #
    #     if len(remove_p_id) == 1:
    #         face.remove(remove_p_id[0])
    #     elif len(remove_p_id) > 1:
    #         assert False

    # Visualize

    if is_visualize:
        for idx, id_face in enumerate(dual_graph.nodes):
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]
            face = dual_graph.nodes[id_face]["id_vertex"]
            print("{}/{}, points_num: {}, sub_faces: {}".format(idx,
                                                                len(dual_graph.nodes),
                                                                len(face),
                                                                dual_graph.nodes[id_face]["sub_faces_id"]))

            # draw current merged face
            gradient_list = []
            for idx, id_start in enumerate(face):
                id_end = face[(idx + 1) % len(face)]
                pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                print("({}, {}) pos1: {}, pos2: {}, length: {} gradient: {} gradient_free_rate: {}".format(
                        id_start, id_end, pos1, pos2, np.linalg.norm(pos1 - pos2),
                        v_graph.edges[(id_start, id_end)]["gradient"],
                        v_graph.edges[(id_start, id_end)]["gradient_free_rate"]))
                gradient_list.append(v_graph.edges[(id_start, id_end)]["gradient"])
                if v_graph.edges[(id_start, id_end)]["is_gradient_free"]:
                    cv2.line(img11, pos1, pos2, (255, 0, 0), 2)
                else:
                    cv2.line(img11, pos1, pos2, (0, 0, 255), 2)
                cv2.circle(img11, pos1, 2, (0, 0, 255), 2)
                cv2.circle(img11, pos2, 2, (0, 0, 255), 2)

            # draw sub_face of merged face
            for face in dual_graph.nodes[id_face]["sub_faces"]:
                # draw a sub_face
                for idx, id_start in enumerate(face):
                    id_end = face[(idx + 1) % len(face)]
                    pos1 = np.around(v_graph.nodes[id_start]["pos_2d"] * shape).astype(np.int64)
                    pos2 = np.around(v_graph.nodes[id_end]["pos_2d"] * shape).astype(np.int64)
                    cv2.line(img11, pos1, pos2, (0, 0, 255), 1)

            # draw neighbor faces
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

            cv2.circle(img11, np.around(dual_graph.nodes[id_face]['face_center'] * shape).astype(np.int64), 2,
                       (0, 0, 255), 2)

            cv2.imshow("1", img11)
            cv2.waitKey()
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

    if True:
        points = initialized_vertices.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud('output/initialized_vertices.ply', pcd)

    # abcd
    plane_parameters = []
    for vertex_id in v_vertex_id_per_face:
        pos_vertexes = initialized_vertices[vertex_id]
        # assert (len(pos_vertexes) >= 3)
        # # a) 3d vertexes of each patch -> fitting plane
        # p_abcd = fit_plane_svd(pos_vertexes)
        abc = torch.mean(pos_vertexes, dim=0)
        d = -torch.dot(abc, abc)
        p_abcd = torch.cat((abc, d.unsqueeze(0)), dim=-1)
        plane_parameters.append(p_abcd)

    return torch.stack(plane_parameters, dim=0)


def viz_merge_plane(v_img_database):
    import trimesh
    def double_faces(mesh):
        reversed_faces = mesh.faces[:, ::-1]  # 翻转每个面的顶点顺序
        double_faces_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                            faces=np.concatenate([mesh.faces, reversed_faces]))
        return double_faces_mesh

    mesh_paths = [r'D:\StructDescription\python\src\neural_recon\outputs\id_img1=5\optimized.ply',
                  r'D:\StructDescription\python\src\neural_recon\outputs\id_img1=11\optimized.ply',
                  r'D:\StructDescription\python\src\neural_recon\outputs\id_img1=22\optimized.ply',
                  r'D:\StructDescription\python\src\neural_recon\outputs\id_img1=23\optimized.ply', ]

    transforms = [None,
                  v_img_database[5].extrinsic @ np.linalg.inv(v_img_database[11].extrinsic),
                  v_img_database[5].extrinsic @ np.linalg.inv(v_img_database[22].extrinsic),
                  v_img_database[5].extrinsic @ np.linalg.inv(v_img_database[23].extrinsic), ]

    if len(mesh_paths) != len(transforms):
        raise ValueError("The number of meshes and transforms must be the same.")

    count = 0
    meshes = []
    for mesh_path, transform in zip(mesh_paths, transforms):
        mesh = trimesh.load(mesh_path)
        if transform is not None:
            mesh.apply_transform(transform)
        meshes.append(mesh)
        mesh.export('./outputs/' + str(count) + '.ply')
        count += 1

    # 合并所有模型
    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh += mesh
    merged_mesh.export('./outputs/merged_mesh.ply')
    pass


def optimize_plane(v_data, v_log_root, plane_vertex_pos_gt=None):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    if plane_vertex_pos_gt is not None:
        for graph in v_graphs:
            graph.graph["plane_vertex_pos_gt"] = torch.from_numpy(plane_vertex_pos_gt).to(device).to(torch.float32)

    if os.path.exists(os.path.join(v_log_root, "optimized_graph.npy")):
        print("Found optimized graph, loading...")
        v_graphs = np.load(os.path.join(os.path.normpath(v_log_root), "optimized_graph.npy"), allow_pickle=True)
        return

    print("1. Plane Optimizing")
    for id_img1, graph in enumerate(v_graphs):
        # if id_img1 != 0:
        #     continue
        v_log_root_c = os.path.join(os.path.normpath(v_log_root), '1.PlaneOptimizing', str(id_img1))
        os.makedirs(v_log_root_c, exist_ok=True)

        if os.path.exists(os.path.join(v_log_root_c, "optimized_graph_c.npy")) and False:
            v_graphs[id_img1] = np.load(os.path.join(v_log_root_c, "optimized_graph_c.npy"), allow_pickle=True)[0]
        else:
            id_src_imgs = (v_img_pairs[id_img1][:, 0]).astype(np.int64)
            src_Images = [v_img_database[id_src] for id_src in id_src_imgs]
            graph_optimiser = GraphPlaneOptimiser(id_img1, graph, v_img_database[id_img1], src_Images,
                                                  v_log_root_c, device, use_cache=False, use_group=False)
            v_graphs[id_img1] = graph_optimiser.run()

    graphs = np.asarray(v_graphs, dtype=object)
    graph_cache_name = os.path.join(v_log_root, "optimized_graph.npy")
    np.save(graph_cache_name, graphs, allow_pickle=True)


def optimize_plane_bk(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    img_src_id = 0
    optimized_abcd_list_v = []
    # viz_merge_plane(v_img_database)

    if os.path.exists(os.path.join(v_log_root, "optimized_graph.npy")):
        print("Found optimized graph, loading...")
        v_graphs = np.load(os.path.join(os.path.normpath(v_log_root), "optimized_graph.npy"), allow_pickle=True)

    # img1_group_idx = 34
    # debug_img1 = [img1_group_idx] + list(v_img_pairs[img1_group_idx][:, 0].astype(np.int64))
    if False:
        for id_img1, graph in enumerate(v_graphs):
            # Visualize id_img1 using opencv
            ref_img = cv2.imread(v_img_database[id_img1].img_path, cv2.IMREAD_GRAYSCALE)
            cv2.putText(ref_img, str(id_img1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Choose Image', ref_img)

            key = cv2.waitKey(0)
            if key == ord(' '):
                continue
            elif key == ord('\r'):
                debug_img1 = id_img1
                cv2.destroyAllWindows()
                break

    print("1. Plane Optimizing")
    for id_img1, graph in enumerate(v_graphs):
        # checking the graph to avoid repeat computing
        is_pass = True
        if "dual_graph" not in graph.graph:
            is_pass = False
        else:
            dual_graph = graph.graph["dual_graph"]
            if "optimized_abcd_list" not in dual_graph.graph:
                is_pass = False
            for node in dual_graph.nodes:
                if "loss" not in dual_graph.nodes[node]:
                    is_pass = False
                    break
            if is_pass:
                print("\nLoading the graph {}".format(str(id_img1)))
                continue

        v_log_root_c = os.path.join(os.path.normpath(v_log_root), '1.PlaneOptimizing', str(id_img1))
        os.makedirs(v_log_root_c, exist_ok=True)

        # 1. Prepare data
        # prepare some data
        print("\nBegin to optimize the graph {}".format(str(id_img1)))
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
                v_img_database[int(id_src_imgs[img_src_id])].extrinsic @ np.linalg.inv(
                        v_img_database[id_img1].extrinsic)
                ).to(device).to(torch.float32)
        extrinsic_ref_cam = torch.from_numpy(v_img_database[id_img1].extrinsic).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        c1_2_c2_list = []
        for i in range(len(id_src_imgs)):
            c1_2_c2_list.append(torch.from_numpy(
                    v_img_database[int(id_src_imgs[i])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
                    ).to(device).to(torch.float32))

        def sharpen_image(image):
            laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().unsqueeze(0).unsqueeze(0)
            channels = image.shape[1]
            laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
            output_image = F.conv2d(image, laplacian_kernel.to(image.device), padding=1)
            output_image = image + output_image
            return output_image

        # imgs[0] = sharpen_image(imgs[0][None, None, :, :])[0, 0, :, :]

        if False:
            # Image gradients
            # Do not normalize!
            imgs_sharp = sharpen_image(imgs[0][None, None, :, :])[0, 0, :, :]
            cv2.imshow("1", torch.cat((imgs[0], imgs_sharp), dim=1).cpu().numpy().astype(np.float32))
            cv2.waitKey()

        gy, gx = torch.gradient(imgs[0])
        gradients1 = torch.stack((gx, gy), dim=-1)
        gy, gx = torch.gradient(imgs[img_src_id + 1])
        gradients2 = torch.stack((gx, gy), dim=-1)

        dilated_gradients1 = torch.from_numpy(dilate_edge(gradients1, 1)).to(device)
        dilated_gradients2 = torch.from_numpy(dilate_edge(gradients2, 1)).to(device)

        # Visualize
        if False:
            def log_transform(tensor):
                tensor = tensor - tensor.min() + 1  # shift to make all elements positive
                return torch.log10(tensor)

            mask1 = (torch.linalg.norm(gradients1, dim=-1))
            mask2 = (torch.linalg.norm(dilated_gradients1, dim=-1))
            mask1[mask1 > 0.01] = 1
            mask2[mask2 > 0.01] = 1
            # mask1 = log_transform(mask1)
            # mask2 = log_transform(mask2)
            # mask1 = (mask1 - mask1.min()) / (mask1.max() - mask1.min())
            # mask2 = (mask2 - mask2.min()) / (mask2.max() - mask2.min())

            cv2.imshow("1", torch.cat((mask1, mask2), dim=1).cpu().numpy().astype(np.float32))
            cv2.waitKey()

        determine_valid_edges(graph, imgs[0], gradients1)
        fix_graph(graph, ref_img, is_visualize=False)

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
        centroid_rays_c = torch.from_numpy(np.stack([dual_graph.nodes[id_node]["ray_c"] for id_node in dual_graph])).to(
                device)
        src_faces_centroid_rays_c = torch.from_numpy(graph.graph["src_face_ray_c"]).to(device)

        # 2. Initialize planes for each patch
        initialized_planes = initialize_patches(rays_c, ray_distances_c, vertex_id_per_face)  # (num_patch, 4)
        # initialized_planes = pickle.load(open("output/optimized_abcd_list.pkl", "rb"))
        # initialized_planes = torch.from_numpy(initialized_planes).to(device)

        if os.path.exists(os.path.join(v_log_root_c, "optimized_abcd_list.pkl")):
            print("Found existing optimized_abcd_list.pkl, loading...")
            optimized_abcd_list = pickle.load(open(os.path.join(v_log_root_c, "optimized_abcd_list.pkl"), "rb"))
            optimize_planes_batch(initialized_planes, rays_c, centroid_rays_c,
                                  src_faces_centroid_rays_c, dual_graph,
                                  imgs, transformation, extrinsic_ref_cam,
                                  intrinsic, c1_2_c2_list, v_log_root_c, optimized_abcd_list)
        else:
            optimized_abcd_list = optimize_planes_batch(initialized_planes, rays_c, centroid_rays_c,
                                                        src_faces_centroid_rays_c, dual_graph,
                                                        imgs, transformation, extrinsic_ref_cam,
                                                        intrinsic, c1_2_c2_list, v_log_root_c)
            # save optimized_abcd_list
            with open(os.path.join(v_log_root_c, "optimized_abcd_list.pkl"), "wb") as f:
                pickle.dump(optimized_abcd_list, f)
                print("Graph {} is saved to {}".format(str(id_img1),
                                                       os.path.join(v_log_root_c, "optimized_abcd_list.pkl")))

        dual_graph.graph['optimized_abcd_list'] = optimized_abcd_list

    graphs = np.asarray(v_graphs, dtype=object)
    graph_cache_name = os.path.join(v_log_root, "optimized_graph.npy")
    np.save(graph_cache_name, graphs, allow_pickle=True)
    return


def graph_matching(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    # use the cached optimized_graph
    v_graphs = np.load(os.path.join(os.path.normpath(v_log_root), "optimized_graph.npy"), allow_pickle=True)
    v_data = (v_img_database, v_graphs, v_img_pairs, v_points_sfm)

    v_log_root_c = os.path.join(os.path.normpath(v_log_root), '2.GraphMatching')
    os.makedirs(v_log_root_c, exist_ok=True)

    # img_ref_id = 4
    # img_src_id = 2
    # v_log_root_cc = os.path.join(v_log_root_c, str(img_ref_id) + "_" + str(img_src_id))
    # PGM = PairGraphMatcher(v_graphs[img_ref_id], v_graphs[img_src_id], v_log_root_cc, device)
    # PGM.run()

    # topN = 10
    # for group_idx in sort_idx[0: topN]:
    #     group_graph_idxs = [group_idx] + list(v_img_pairs[group_idx][:, 0].astype(np.int64))
    #     # v_log_root_cc = os.path.join(v_log_root_c, str(group_idx))
    #     MGM = GraphFusion(v_graphs[group_graph_idxs], v_log_root_c, device)
    #     MGM.run()

    MGM = GraphFusion(v_data, v_log_root_c, device)
    MGM.run()

    return


def get_plane_vertex_pos_gt(path):
    mesh_path = os.path.join(path, "mesh.ply")
    feature_path = os.path.join(path, "features.yml")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    faces_vertice_idx = np.asarray(mesh.triangles)

    with open(feature_path, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.CLoader)

    plane_vertex_pos = []
    for surface in data['surfaces']:
        assert surface['type'] == 'Plane'
        tri_idx = faces_vertice_idx[random.sample(surface['face_indices'], 1)].reshape(-1)
        tri_pos = vertices[tri_idx]
        plane_vertex_pos.append(tri_pos)

    print("Found {} plane".format(len(plane_vertex_pos)))
    plane_vertex_pos = np.stack(plane_vertex_pos)
    return plane_vertex_pos  # n * 3 * 3


def save_as_wireframe_obj(triangles, file_name):
    with open(file_name, 'w') as file:
        # 写入顶点数据
        for tri in triangles:
            for vertex in tri:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # 写入线段数据
        vertex_count = 1
        for _ in range(len(triangles)):
            # 对于每个三角形，我们需要定义其边缘作为线段
            file.write(f"l {vertex_count} {vertex_count + 1}\n")
            file.write(f"l {vertex_count + 1} {vertex_count + 2}\n")
            file.write(f"l {vertex_count + 2} {vertex_count}\n")
            vertex_count += 3


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
            v_cfg["dataset"]["max_error_for_initial_sfm"],
            )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_11_23_10_47_04"
    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_11_28_08_07_42"
    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_11_29_22_20_13"
    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_11_29_22_48_45"

    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_12_19_22_41_39"
    # tmp_path = r"G:\workplace\python\src\neural_recon\outputs\2023_12_20_02_16_53"

    tmp_path = r"C:/repo/python/outputs/2024_02_21_12_24_49"
    # plane_vertex_pos_gt = get_plane_vertex_pos_gt(v_cfg["dataset"]["colmap_dir"])
    # save_as_wireframe_obj(plane_vertex_pos_gt, os.path.join(v_cfg["trainer"]["output"], "mesh_gt.obj"))

    # optimize_plane(data, tmp_path, plane_vertex_pos_gt=None)
    optimize_plane(data, v_cfg["trainer"]["output"], plane_vertex_pos_gt=None)

    graph_matching(data, tmp_path)


if __name__ == '__main__':
    main()
