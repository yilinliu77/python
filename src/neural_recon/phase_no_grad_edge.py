import itertools
import sys, os
import time
from typing import List

from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1, loss2, loss4, loss5

# sys.path.append("thirdparty/sdf_computer/build/")
# import pysdf
from src.neural_recon.phase12 import Siren2

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

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img, sample_img_prediction2
from shared.common_utils import debug_imgs, to_homogeneous, save_line_cloud, to_homogeneous_vector, normalize_tensor, \
    to_homogeneous_mat_tensor, to_homogeneous_tensor, normalized_torch_img_to_numpy, padding, \
    vector_to_sphere_coordinate, sphere_coordinate_to_vector, caculate_align_mat, normalize_vector, \
    pad_and_enlarge_along_y, refresh_timer, get_line_mesh

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
from src.neural_recon.phase1 import NGPModel


def sample_new_distance(v_original_distances, num_sample=100, scale_factor=1.6, v_max=10, ):
    num_vertices = v_original_distances.shape[0]
    device = v_original_distances.device
    # (B, S)
    new_distance = -torch.ones((num_vertices, num_sample - 1), device=device, dtype=v_original_distances.dtype)
    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, S)
    repeated_vertices_distances = v_original_distances[:, None].tile((1, num_sample - 1))
    while not torch.all(sample_distance_mask):
        t_ = new_distance[~sample_distance_mask]
        a = repeated_vertices_distances[~sample_distance_mask] + \
            scale_factor * torch.distributions.utils._standard_normal(
            t_.shape[0],
            device=device,
            dtype=t_.dtype)
        new_distance[~sample_distance_mask] = a
        sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, (S + 1))
    new_distance = torch.cat((v_original_distances[:, None], new_distance), dim=1)
    return new_distance


def get_up_vector(start_points, end_points, t):
    # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
    cur_dir = end_points - start_points
    a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
    up_c = torch.stack((
        -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
        a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
        torch.sqrt(a * a + b * b) * torch.sin(t)
    ), dim=1)
    return normalize_tensor(up_c)


def sample_new_ups(v_edge_points):
    num_edges = v_edge_points.shape[0]
    t = torch.linspace(0, 1, steps=100, device=v_edge_points.device) * torch.pi * 2
    num_sample = t.shape[0]

    sampled_ups = get_up_vector(
        v_edge_points[:, None, 0, :].tile(1, num_sample, 1).view(-1, 3),
        v_edge_points[:, None, 1, :].tile(1, num_sample, 1).view(-1, 3),
        t[None, :].tile(num_edges, 1).view(-1),
    ).view(num_edges, num_sample, 3)

    return sampled_ups


def sample_edge(num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
    times = [0 for _ in range(10)]
    cur_time = time.time()
    length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
    num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
    num_edge_points_ = num_edge_points.roll(1)
    num_edge_points_[0] = 0
    times[1] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = torch.arange(num_edge_points.sum(), device=cur_dir.device) - num_edge_points_.cumsum(
        dim=0).repeat_interleave(num_edge_points)
    times[2] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
    times[3] += time.time() - cur_time
    cur_time = time.time()
    sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                          + start_point.repeat_interleave(num_edge_points, dim=0)
    times[4] += time.time() - cur_time
    cur_time = time.time()
    return num_edge_points, sampled_edge_points


def sample_triangles(num_per_m, p1, p2, p3, num_max_sample=500):
    d1 = p2 - p1
    d2 = p3 - p2
    area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

    num_edge_points, edge_points = sample_edge(num_per_m,
                                               torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                               # torch.stack((d1,), dim=1).reshape(-1, 3),
                                               torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                               # torch.stack((p1,), dim=1).reshape(-1, 3),
                                               num_max_sample=num_max_sample)
    num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)
    # num_edge_points = num_edge_points.reshape(-1, 1).sum(dim=1)

    num_per_m2 = num_per_m * num_per_m
    num_tri_samples = torch.clamp((area * num_per_m2).to(torch.long), 1, num_max_sample * 4)
    samples = torch.rand(num_tri_samples.sum(), 2, device=p1.device)
    _t1 = torch.sqrt(samples[:, 0:1] + 1e-6)
    sampled_polygon_points = (1 - _t1) * p1.repeat_interleave(num_tri_samples, dim=0) + \
                             _t1 * (1 - samples[:, 1:2]) * p2.repeat_interleave(num_tri_samples, dim=0) + \
                             _t1 * samples[:, 1:2] * p3.repeat_interleave(num_tri_samples, dim=0)

    # Only use the code below for debug
    if True:
        num_total_points = num_edge_points + num_tri_samples
        num_total_points_cumsum = num_total_points.cumsum(0).roll(1)
        num_total_points_cumsum[0] = 0
        sampled_total_points = torch.zeros((num_total_points.sum(), 3), device=p1.device, dtype=torch.float32)
        num_edge_points_ = num_edge_points.cumsum(0).roll(1)
        num_edge_points_[0] = 0
        num_tri_points_ = num_tri_samples.cumsum(0).roll(1)
        num_tri_points_[0] = 0
        edge_index = torch.arange(num_edge_points.sum(), device=p1.device) \
                     - (num_edge_points_ - num_total_points_cumsum).repeat_interleave(num_edge_points)
        tri_index = torch.arange(num_tri_samples.sum(), device=p1.device) \
                    - (num_tri_points_ - num_total_points_cumsum - num_edge_points).repeat_interleave(
            num_tri_samples)
        sampled_total_points[edge_index] = edge_points
        sampled_total_points[tri_index] = sampled_polygon_points
        return num_total_points, sampled_total_points
    return None, torch.cat((edge_points, sampled_polygon_points), dim=0)


def sample_points_2d(v_edge_points, v_num_horizontal, v_img_width=800, v_vertical_length=10, v_max_points=500):
    device = v_edge_points.device
    cur_dir = v_edge_points[:, 1] - v_edge_points[:, 0]
    cur_length = torch.linalg.norm(cur_dir, dim=-1) + 1e-6

    cur_dir_h = torch.cat((cur_dir, torch.zeros_like(cur_dir[:, 0:1])), dim=1)
    z_axis = torch.zeros_like(cur_dir_h)
    z_axis[:, 2] = 1
    edge_up = normalize_tensor(torch.cross(cur_dir_h, z_axis, dim=1)[:, :2]) * v_vertical_length / v_img_width
    # The vertical length is 10 -> 10/6000 = 0.00167

    # Compute interpolated point
    num_horizontal = v_num_horizontal
    num_half_vertical = 10
    num_coordinates_per_edge = num_horizontal * num_half_vertical * 2

    begin_idxes = num_horizontal.cumsum(dim=0)
    total_num_x_coords = begin_idxes[-1]
    begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
    begin_idxes[0] = 0  # (M,)
    dx = torch.arange(num_horizontal.sum(), device=device) - \
         begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
    dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal)
    dy = torch.arange(num_half_vertical, device=device) / (num_half_vertical - 1)
    dy = torch.cat((torch.flip(-dy, dims=[0]), dy))

    # Meshgrid
    total_num_coords = total_num_x_coords * dy.shape[0]
    coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * dy.shape[0])  # (total_num_coords,)
    coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
    coords = torch.stack((coords_x, coords_y), dim=1)

    interpolated_coordinates = \
        cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
        edge_up.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
        v_edge_points[:, 0].repeat_interleave(num_coordinates_per_edge, dim=0)

    return num_coordinates_per_edge, interpolated_coordinates


def compute_similarity_wrapper(start_points, end_points,
                               imgs, transformations, intrinsic, v_vertical_length=10):
    times = [0 for _ in range(10)]
    cur_time = time.time()
    num_src_imgs = imgs.shape[0] - 1
    points_c = torch.stack([start_points, end_points], dim=1).reshape(-1, 3)
    times[0] += time.time() - cur_time
    cur_time = time.time()

    edge_points = (intrinsic @ points_c.T).T
    edge_points = edge_points[:, :2] / (edge_points[:, 2:3] + 1e-6)
    edge_points = edge_points.reshape(-1, 2, 2)

    # sample step=0.01
    num_horizontal = torch.clamp((torch.linalg.norm(end_points - start_points, dim=-1) / 0.01).to(torch.long),
                                 2, 1000)

    num_per_edge1, points_2d1 = sample_points_2d(edge_points, num_horizontal, imgs[0].shape[1],
                                                 v_vertical_length=v_vertical_length)

    valid_mask1 = torch.logical_and(points_2d1 > 0, points_2d1 < 1)
    valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
    points_2d1 = torch.clamp(points_2d1, 0, 0.999999)

    edge_points = (transformations @ to_homogeneous_tensor(points_c).T).transpose(1, 2)
    edge_points = edge_points[:, :, :2] / (edge_points[:, :, 2:3] + 1e-6)
    edge_points = edge_points.reshape(num_src_imgs, -1, 2, 2)

    num_per_edge2, points_2d2 = sample_points_2d(edge_points.reshape(-1, 2, 2),
                                                 num_horizontal.tile(num_src_imgs), imgs[0].shape[1],
                                                 v_vertical_length=v_vertical_length)
    num_per_edge2 = num_per_edge2.reshape(num_src_imgs, -1)
    points_2d2 = points_2d2.reshape(num_src_imgs, -1, 2)

    valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
    valid_mask2 = torch.logical_and(valid_mask2[:, :, 0], valid_mask2[:, :, 1])
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    # 4. Sample pixel color
    sample_imgs1 = sample_img(imgs[0:1, None, :, :], points_2d1[None, :, :])[0]
    sample_imgs2 = sample_img(imgs[1:, None], points_2d2)
    times[1] += time.time() - cur_time
    cur_time = time.time()

    similarity_loss, black_area_in_img1 = loss4(sample_imgs1, sample_imgs2, num_per_edge1)
    similarity_mask = torch.logical_and(valid_mask1[None, :].tile([valid_mask2.shape[0], 1]), valid_mask2)
    similarity_mask = scatter_min(similarity_mask.to(torch.long), torch.arange(
        num_per_edge1.shape[0], device=similarity_loss.device).repeat_interleave(num_per_edge1), dim=1)[0]
    times[3] += time.time() - cur_time
    cur_time = time.time()
    return similarity_loss, similarity_mask.to(torch.bool), black_area_in_img1, [points_2d1, points_2d2]


def get_initial_normal(v_center_ray):
    return v_center_ray


def sample_new_directions_batch(v_input, num_sample, v_variance):
    batch_size = v_input.shape[0]
    device = v_input.device
    v_input = v_input / (v_input.norm(dim=1, keepdim=True) + 1e-6)

    # Generate N random unit vectors for each input vector
    random_vectors = torch.randn(batch_size, num_sample, 3, device=device)
    random_vectors = random_vectors / (torch.norm(random_vectors, dim=2, keepdim=True) + 1e-6)

    # Project random vectors onto the plane orthogonal to the input vectors
    projections = torch.matmul(random_vectors, v_input.unsqueeze(2))
    orthogonal_vectors = random_vectors - projections * v_input.unsqueeze(1)

    # Generate Gaussian distributed angles
    angles = torch.randn(batch_size, num_sample, device=device) * math.sqrt(v_variance)

    # Calculate the new directions
    new_directions = torch.cos(angles).unsqueeze(2) * v_input.unsqueeze(1) + torch.sin(angles).unsqueeze(
        2) * orthogonal_vectors

    new_directions = normalize_tensor(new_directions)
    return new_directions


def sample_new_distance_and_normals(v_center_distance, v_init_normal, num_sample=100, v_max=10., scale_factor=1.6, ):
    num_vertices = v_center_distance.shape[0]
    device = v_center_distance.device
    dtype = v_center_distance.dtype
    # (B, S)
    new_distance = -torch.ones((num_vertices, num_sample - 1), device=device, dtype=dtype)
    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, S)
    repeated_vertices_distances = v_center_distance[:, None].tile((1, num_sample - 1))
    while not torch.all(sample_distance_mask):
        t_ = new_distance[~sample_distance_mask]
        a = repeated_vertices_distances[~sample_distance_mask] + \
            scale_factor * torch.distributions.utils._standard_normal(
            t_.shape[0],
            device=device,
            dtype=t_.dtype)
        new_distance[~sample_distance_mask] = a
        sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < v_max)
    # (B, (S + 1))
    new_distance = torch.cat((v_center_distance[:, None], new_distance), dim=1)

    new_normals = -torch.ones((num_vertices, num_sample - 1, 3), device=device, dtype=dtype)
    sample_normals_mask = torch.zeros_like(new_normals[:, :, 0], dtype=torch.bool)
    while not torch.all(sample_normals_mask):
        a = sample_new_directions_batch(v_init_normal, num_sample - 1, 0.3)
        new_normals[~sample_normals_mask] = a[~sample_normals_mask]
        sample_normals_mask = (v_init_normal[:, None, :] * new_normals).sum(dim=-1) > 0

    new_normals = sample_new_directions_batch(v_init_normal, num_sample - 1, 0.3)
    new_normals = torch.cat((v_init_normal[:, None, :], new_normals), dim=1)
    return new_distance, new_normals


def get_edge_vertices(v_center_point, v_normal, src_rays):
    # Calculate the distance from the origin to the plane along the normal vector
    d = torch.sum(v_center_point * v_normal, dim=2)
    # Calculate the dot product between the rays (a) and the normal vectors
    a_dot_v_normal = torch.matmul(src_rays, v_normal.transpose(1, 2))
    # Calculate the t values for each ray-plane intersection
    t = d[:, None, :] / a_dot_v_normal
    collision_points = src_rays[:, :, None, :] * t[:, :, :, None]
    return collision_points


def evaluate_candidates(v_edge_points, v_intrinsic, v_transformation,
                        v_img1, v_img2):
    num_edge, _, _ = v_edge_points.shape

    num_horizontal = torch.clamp((torch.linalg.norm(v_edge_points[:,0] - v_edge_points[:,1], dim=-1) / 0.01).to(torch.long),
                                 2, 1000)

    points_2d1 = (v_intrinsic @ v_edge_points.view(-1, 3).T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-6)
    points_2d1 = points_2d1.reshape(num_edge, 2, 2)

    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_edge_points.view(-1, 3)).T).transpose(0, 1)
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-6)
    points_2d2 = points_2d2.reshape(num_edge, 2, 2)
    valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1]) # And in two edge points
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1]) # And in x and y
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    num_sampled_points1, sampled_points1 = sample_points_2d(points_2d1.reshape(-1,2,2),
                                                            num_horizontal.reshape(-1),
                                                            v_img_width=v_img1.shape[1])
    _, sampled_points2 = sample_points_2d(points_2d2.reshape(-1,2,2),
                                                            num_horizontal.reshape(-1),
                                                            v_img_width=v_img1.shape[1])

    # 4. Sample pixel color
    sample_imgs1 = sample_img(v_img1[None, None, :], sampled_points1[None, :, :])[0]
    sample_imgs2 = sample_img(v_img2[None, None, :], sampled_points2[None, :, :])[0]

    similarity_loss = loss4(sample_imgs1, sample_imgs2[None,:], num_sampled_points1)[0]
    similarity_loss = similarity_loss[0]
    similarity_loss = similarity_loss.reshape(num_edge,)
    return similarity_loss, valid_mask2


def determine_valid_edges(v_edge_rays_c, v_edge_distances_c, v_img, v_intrinsic):
    edge_points_c = v_edge_rays_c * v_edge_distances_c[:, :, None]
    p = (v_intrinsic @ edge_points_c.reshape(-1, 3).T).T
    p = p[:, :2] / p[:, 2:3]
    p = p.reshape(-1, 2, 2)

    ns, s = sample_points_2d(p,
                             torch.tensor([100] * p.shape[0], dtype=torch.long, device=p.device),
                             v_img_width=v_img.shape[1], v_vertical_length=10)
    pixels = sample_img(v_img[None, None, :, :], s[None, :])[0]

    mean_pixels = scatter_mean(pixels,
                               torch.arange(ns.shape[0], device=p.device).repeat_interleave(ns),
                               dim=0)

    return mean_pixels > 0.05


def optimize1(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vertical_length = 5

    for id_img1, graph in enumerate(v_graphs):
        all_face_ids = np.asarray(graph.graph["faces"], dtype=object)

        rays_c = [None] * len(graph.nodes)
        distances = [None] * len(graph.nodes)
        id_end = []
        for idx, id_node in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_node]["ray_c"]
            distances[idx] = graph.nodes[id_node]["distance"]
            id_end.append(list(graph[0].keys()))

        id_src_imgs = v_img_pairs[id_img1][:, 0]
        ref_img = cv2.imread(v_img_database[id_img1].img_path,
                             cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path,
                               cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Vertex
        num_vertex = len(rays_c)
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        distances = torch.from_numpy(np.stack(distances)).to(device).to(torch.float32)
        # Face
        face_flags = torch.tensor(graph.graph["face_flags"], dtype=torch.bool, device=device)
        centers_ray_c = torch.from_numpy(graph.graph["ray_c"]).to(device).to(torch.float32)
        center_distances = torch.from_numpy(graph.graph["distance"]).to(device).to(torch.float32)

        point_id_per_edge=[]
        edge_id_dict = {}
        for edge in graph.edges():
            point_id_per_edge.append(edge)
            edge_id_dict[(edge[0],edge[1])] = len(point_id_per_edge)
            edge_id_dict[(edge[1],edge[0])] = -len(point_id_per_edge)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        # Edges
        num_edge = len(graph.edges())
        num_max_edge_per_vertex = max([len(graph[item]) for item in graph.nodes])
        edge_rays_c = rays_c[point_id_per_edge]
        edge_distances_c = distances[point_id_per_edge]
        edge_is_not_black = determine_valid_edges(edge_rays_c, edge_distances_c, imgs[0], intrinsic)[:, 0]
        to_non_black_edge_id = -torch.ones(num_edge,device=device,dtype=torch.long)
        to_non_black_edge_id[edge_is_not_black] = torch.arange(edge_is_not_black.sum(),device=device)

        edge_point_id_per_face=[]
        for face_ids,face_flag in zip(all_face_ids,face_flags):
            if not face_flag:
                continue
            local_id_edges = []
            for i,id_start in enumerate(face_ids):
                id_end = face_ids[(i+1)%len(face_ids)]
                id_edge = edge_id_dict[(id_start,id_end)]
                mask = 1 if id_edge>0 else -1
                local_id_edges.append((to_non_black_edge_id[abs(id_edge)-1]+1) * mask)
            a = torch.tensor(local_id_edges,dtype=torch.long,device=device)
            b = torch.roll(a,-1)
            edge_point_id_per_face.append(torch.stack((a,b),dim=1))

        edge_rays_c = edge_rays_c[edge_is_not_black]
        edge_distances_c = edge_distances_c[edge_is_not_black]
        num_edge = edge_distances_c.shape[0]
        num_sample = 100

        # edge_rays_c = edge_rays_c[1:2]
        # edge_distances_c = edge_distances_c[1:2]
        # num_edge = 1

        cur_iter = 0
        while True:
            distances_candidate = sample_new_distance(edge_distances_c.view(-1), num_sample=num_sample).view(
                num_edge, 2, num_sample, )
            # num_edge, 2, num_sample, 3
            edge_points_candidate = edge_rays_c[:, :, None, :] * distances_candidate[:, :, :, None]

            edge_points_candidate = edge_rays_c[:, :, None, :] * distances_candidate[:, :, :, None]
            ncc_loss, ncc_loss_mask = evaluate_candidates(edge_points_candidate, intrinsic, transformation[0], imgs[0], imgs[1])
            ncc_loss[~ncc_loss_mask] = torch.inf # Set it to 0 in order to get the samples below

            # Consistent loss
            sampled_index = torch.multinomial(torch.exp(-ncc_loss**4), 99, replacement=True)
            sampled_index = torch.cat((torch.zeros_like(sampled_index[:,0:1]),sampled_index),dim=1)
            sampled_loss = torch.gather(ncc_loss,dim=1,index=sampled_index).mean(dim=0)
            sampled_edges = torch.gather(edge_points_candidate, dim=2,
                                         index=sampled_index.view(num_edge, 1, 100, 1).tile(1, 2, 1, 3))
            sampled_edges = sampled_edges.permute(2,0,1,3)
            consistency = []
            for face in edge_point_id_per_face:
                true_face = face[torch.all(face!=0,dim=1)]
                if true_face.shape[0]==0:
                    continue
                face_mask = true_face < 0
                chosen_points = sampled_edges[:,torch.abs(true_face)-1]
                chosen_points[:,face_mask] = torch.flip(chosen_points[:,face_mask], dims=[2])
                distance = torch.linalg.norm(chosen_points[:,:,0,1]-chosen_points[:,:,1,0],dim=2).mean(dim=1)
                consistency.append(distance)
            consistency = torch.stack(consistency,dim=1).mean(dim=1)

            alpha = 1
            loss = sampled_loss * alpha + consistency * (1-alpha)
            id_best = sampled_index[:,loss.argmin(dim=0)]
            chosen_distance = torch.gather(distances_candidate, dim=2, index=id_best.view(-1,1,1).tile(1,2,1))[:,:,0]
            chosen_edge_points = torch.gather(edge_points_candidate, dim=2, index=id_best.view(-1,1,1,1).tile(1,2,1,3))[:,:,0]
            chosen_loss = torch.gather(loss, dim=0, index=loss.argmin(dim=0))
            # chosen_edge_points = edge_points_candidate[:,:,0]
            edge_distances_c = chosen_distance
            # Log
            if True:
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                start_points_c = chosen_edge_points[:,0]
                end_points_c = chosen_edge_points[:,1]
                sp_2d1 = ((intrinsic @ start_points_c.T).T).cpu().numpy()
                ep_2d1 = ((intrinsic @ end_points_c.T).T).cpu().numpy()
                sp_2d2 = (transformation[0] @ to_homogeneous_tensor(start_points_c).T).T.cpu().numpy()
                ep_2d2 = (transformation[0] @ to_homogeneous_tensor(end_points_c).T).T.cpu().numpy()
                sp_2d1 = sp_2d1[:, :2] / sp_2d1[:, 2:3]
                ep_2d1 = ep_2d1[:, :2] / ep_2d1[:, 2:3]
                sp_2d2 = sp_2d2[:, :2] / sp_2d2[:, 2:3]
                ep_2d2 = ep_2d2[:, :2] / ep_2d2[:, 2:3]

                shape = img11.shape[:2][::-1]
                sp_2d1 = np.around(sp_2d1 * shape).astype(np.int64)
                ep_2d1 = np.around(ep_2d1 * shape).astype(np.int64)
                sp_2d2 = np.around(sp_2d2 * shape).astype(np.int64)
                ep_2d2 = np.around(ep_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(sp_2d1.shape[0]):
                    cv2.line(img11, sp_2d1[i_edge], ep_2d1[i_edge], line_color, line_thickness)
                    cv2.line(img21, sp_2d2[i_edge], ep_2d2[i_edge], line_color, line_thickness)
                for i_point in range(sp_2d1.shape[0]):
                    cv2.circle(img11, sp_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img11, ep_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, sp_2d2[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, ep_2d2[i_point], 1, point_color, point_thickness)

                cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                            np.concatenate([
                                np.concatenate([img11, img21], axis=1),
                                np.concatenate([img12, img22], axis=1),
                            ], axis=0)
                            )
                print("{:3d}:{:.4f}".format(cur_iter,
                                            chosen_loss.mean().cpu().item()))

            cur_iter+=1
        exit()

    pass

def optimize(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vertical_length = 5

    for id_img1, graph in enumerate(v_graphs):
        all_face_ids = np.asarray(graph.graph["faces"], dtype=object)

        rays_c = [None] * len(graph.nodes)
        distances = [None] * len(graph.nodes)
        id_end = []
        for idx, id_points in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_points]["ray_c"]
            distances[idx] = graph.nodes[id_points]["distance"]
            id_end.append(list(graph[0].keys()))

        id_src_imgs = v_img_pairs[id_img1][:, 0]
        ref_img = cv2.imread(v_img_database[id_img1].img_path,
                             cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path,
                               cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Vertex
        num_vertex = len(rays_c)
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        distances = torch.from_numpy(np.stack(distances)).to(device).to(torch.float32)
        # Face
        face_flags = torch.tensor(graph.graph["face_flags"], dtype=torch.bool, device=device)
        centers_ray_c = torch.from_numpy(graph.graph["ray_c"]).to(device).to(torch.float32)
        center_distances = torch.from_numpy(graph.graph["distance"]).to(device).to(torch.float32)

        point_id_per_edge=[]
        edge_id_dict = {}
        for edge in graph.edges():
            point_id_per_edge.append(edge)
            edge_id_dict[(edge[0],edge[1])] = len(point_id_per_edge)
            edge_id_dict[(edge[1],edge[0])] = -len(point_id_per_edge)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        # Edges
        num_edge = len(graph.edges())
        num_max_edge_per_vertex = max([len(graph[item]) for item in graph.nodes])

        edge_rays_c = rays_c[point_id_per_edge]
        edge_distances_c = distances[point_id_per_edge]
        edge_is_not_black = determine_valid_edges(edge_rays_c, edge_distances_c, imgs[0], intrinsic)[:, 0]
        to_non_black_edge_id = -torch.ones(num_edge,device=device,dtype=torch.long)
        to_non_black_edge_id[edge_is_not_black] = torch.arange(edge_is_not_black.sum(),device=device)

        edge_point_id_per_face=[]
        for face_ids,face_flag in zip(all_face_ids,face_flags):
            if not face_flag:
                continue
            local_id_edges = []
            for i,id_start in enumerate(face_ids):
                id_end = face_ids[(i+1)%len(face_ids)]
                id_edge = edge_id_dict[(id_start,id_end)]
                mask = 1 if id_edge>0 else -1
                local_id_edges.append((to_non_black_edge_id[abs(id_edge)-1]+1) * mask)
            a = torch.tensor(local_id_edges,dtype=torch.long,device=device)
            b = torch.roll(a,-1)
            edge_point_id_per_face.append(torch.stack((a,b),dim=1))

        edge_rays_c = edge_rays_c[edge_is_not_black]
        edge_distances_c = edge_distances_c[edge_is_not_black]
        num_edge = edge_distances_c.shape[0]
        num_sample = 100

        # edge_rays_c = edge_rays_c[1:2]
        # edge_distances_c = edge_distances_c[1:2]
        # num_edge = 1

        target_point_id_per_vertex = []
        magic_id1 = []
        magic_id2 = []
        num_total_edges = 0
        for id_face, id_points in enumerate(all_face_ids):
            if not face_flags[id_face]:
                continue

            has_black_edge = False
            local_target_points = []
            for id_iter in range(len(id_points)):
                id_start = id_points[id_iter]
                id_end = id_points[(id_iter+1)%len(id_points)]
                id_prev = id_points[(id_iter-1)%len(id_points)]
                if not edge_is_not_black[abs(edge_id_dict[(id_start,id_end)])-1]:
                    has_black_edge=True
                    break
                local_target_points.append((id_start,id_prev,id_end))

            if has_black_edge:
                continue
            target_point_id_per_vertex.extend(local_target_points)
            magic_id1.append(torch.roll(
                num_total_edges+torch.arange(len(local_target_points),device=device,dtype=torch.long), -1))
            magic_id2.append(torch.roll(
                num_total_edges+torch.arange(len(local_target_points),device=device,dtype=torch.long), 1))
            num_total_edges+=len(local_target_points)
        target_point_id_per_vertex = torch.tensor(target_point_id_per_vertex,dtype=torch.long,device=device)
        magic_id1 = torch.cat(magic_id1)
        magic_id2 = torch.cat(magic_id2)

        # Training
        num_vertex = target_point_id_per_vertex.shape[0]
        id_start_points = target_point_id_per_vertex[:, 0]
        id_end_points = target_point_id_per_vertex[:, 2]
        id_prev_points = target_point_id_per_vertex[:, 1]
        original_start_distances = distances[id_start_points]
        end_distances = original_start_distances.clone()[magic_id1]
        prev_distances = original_start_distances.clone()[magic_id2]

        unique_target_vertex_id = torch.unique(id_start_points)
        magic_id3 = [torch.where(id_start_points==idx)[0] for idx in unique_target_vertex_id]

        def generate_combinations(n):
            indices = torch.arange(n,device=device,dtype=torch.long)
            grid_x, grid_y = torch.meshgrid(indices, indices)
            combinations = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
            combinations = combinations[combinations[:, 0] != combinations[:, 1]]
            return combinations

        cur_iter = 0
        while True:
            start_distances_candidate = sample_new_distance(original_start_distances, num_sample=num_sample).view(
                num_vertex, num_sample, )
            start_points_c = start_distances_candidate[:,:,None] * rays_c[id_start_points][:,None,:]
            end_points_c = (end_distances[:,None] * rays_c[id_end_points])[:,None,:].tile(1,100,1)
            prev_points_c = (prev_distances[:,None] * rays_c[id_prev_points])[:,None,:].tile(1,100,1)

            # num_edges * 2, num_sample, 2, 3
            edges_points = torch.stack((
                torch.stack((start_points_c,end_points_c),dim=-2),
                torch.stack((start_points_c,prev_points_c),dim=-2),
            ), dim=1).reshape(2 * num_vertex, num_sample, 2, 3)

            ncc_loss, ncc_loss_mask = evaluate_candidates(edges_points.reshape(-1,2,3),
                                                          intrinsic, transformation[0], imgs[0], imgs[1])
            ncc_loss[~ncc_loss_mask] = torch.inf # Set it to 0 in order to get the samples below
            ncc_loss = ncc_loss.reshape(num_vertex, 2, num_sample)
            ncc_loss = ncc_loss.mean(dim=1)

            total_losses = []
            for id_vertex in range(unique_target_vertex_id.shape[0]):
                num_local_vertex = magic_id3[id_vertex].shape[0]
                ncc = ncc_loss[magic_id3[id_vertex]]
                target_id = magic_id3[id_vertex]
                assert num_local_vertex<=3
                if num_local_vertex==1:
                    id_best = ncc.argmin(dim=1)[0]
                    original_start_distances[magic_id3[id_vertex]] = start_distances_candidate[magic_id3[id_vertex],id_best]
                    total_losses.append(torch.stack((
                        ncc[0,id_best],
                        ncc[0,id_best],
                        torch.zeros_like(ncc[0,id_best]),
                    )))
                elif num_local_vertex>=2:
                    all_combinations = generate_combinations(num_local_vertex)

                    distance_loss = torch.abs(start_distances_candidate[target_id][all_combinations[:, 0]] - \
                        original_start_distances[target_id][all_combinations[:, 1]].view(-1,1))
                    distance_loss = distance_loss.view(num_local_vertex-1, num_local_vertex, num_sample)

                    if num_local_vertex>2:
                        comb_ori = torch.combinations(torch.arange(num_local_vertex - 1, device=device))
                        comb_ori = comb_ori[None, :].tile(num_local_vertex, 1, 1)
                        indicator = torch.arange(
                            num_local_vertex, device=device)[:, None, None].tile(1,comb_ori.shape[1], 2)
                        comb_ori[comb_ori >= indicator] += 1
                        origin_dis_com = original_start_distances[target_id][comb_ori]
                        origin_dis_loss = torch.abs(
                            origin_dis_com[:, :, 0] - origin_dis_com[:, :, 1]).permute(0,1).view(1, -1, 1)
                    else:
                        origin_dis_loss = torch.zeros((1,2,1),device=device,dtype=torch.float32)

                    all_distance_loss = torch.cat((distance_loss,origin_dis_loss.tile(1,1,num_sample)),dim=0)

                    loss_d = all_distance_loss.min(dim=0)[0] + 0.1 * all_distance_loss.max(dim=0)[0]

                    weighted_consistency_loss = loss_d * 20 / 10
                    total_loss = ncc + weighted_consistency_loss
                    id_best = total_loss.argmin(dim=-1)
                    original_start_distances[magic_id3[id_vertex]] = start_distances_candidate[magic_id3[id_vertex],id_best]
                    ncc_loss_ = torch.gather(ncc,1,id_best.view(-1,1)).mean()
                    consistency_loss_ = torch.gather(weighted_consistency_loss, 1, id_best.view(-1,1)).mean()
                    loss_ = torch.gather(total_loss,1,id_best.view(-1,1)).mean()
                    total_losses.append((
                        torch.stack((loss_,ncc_loss_,consistency_loss_))
                    ))
            # id_best = ncc_loss.argmin(dim=1)
            # chosen_distance = torch.gather(start_distances_candidate, dim=1, index=id_best.view(-1,1))[:,0]
            # chosen_start_points_c = torch.gather(start_points_c, dim=1, index=id_best.view(-1,1,1).tile(1,1,3))[:,0]
            # chosen_end_points_c = end_points_c[:,0]
            # chosen_loss = torch.gather(ncc_loss, dim=1, index=id_best.view(-1,1))
            # original_start_distances = chosen_distance
            end_distances = original_start_distances.clone()[magic_id1]
            prev_distances = original_start_distances.clone()[magic_id2]
            total_losses = torch.stack(total_losses)

            chosen_start_points_c = original_start_distances[:,None] * rays_c[id_start_points]
            chosen_end_points_c = end_distances[:,None] * rays_c[id_end_points]

            # Log
            if True:
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                start_points_c = chosen_start_points_c
                end_points_c = chosen_end_points_c
                sp_2d1 = ((intrinsic @ start_points_c.T).T).cpu().numpy()
                ep_2d1 = ((intrinsic @ end_points_c.T).T).cpu().numpy()
                sp_2d2 = (transformation[0] @ to_homogeneous_tensor(start_points_c).T).T.cpu().numpy()
                ep_2d2 = (transformation[0] @ to_homogeneous_tensor(end_points_c).T).T.cpu().numpy()
                sp_2d1 = sp_2d1[:, :2] / sp_2d1[:, 2:3]
                ep_2d1 = ep_2d1[:, :2] / ep_2d1[:, 2:3]
                sp_2d2 = sp_2d2[:, :2] / sp_2d2[:, 2:3]
                ep_2d2 = ep_2d2[:, :2] / ep_2d2[:, 2:3]

                shape = img11.shape[:2][::-1]
                sp_2d1 = np.around(sp_2d1 * shape).astype(np.int64)
                ep_2d1 = np.around(ep_2d1 * shape).astype(np.int64)
                sp_2d2 = np.around(sp_2d2 * shape).astype(np.int64)
                ep_2d2 = np.around(ep_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(sp_2d1.shape[0]):
                    cv2.line(img11, sp_2d1[i_edge], ep_2d1[i_edge], line_color, line_thickness)
                    cv2.line(img21, sp_2d2[i_edge], ep_2d2[i_edge], line_color, line_thickness)
                for i_point in range(sp_2d1.shape[0]):
                    cv2.circle(img11, sp_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img11, ep_2d1[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, sp_2d2[i_point], 1, point_color, point_thickness)
                    cv2.circle(img21, ep_2d2[i_point], 1, point_color, point_thickness)

                cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                            np.concatenate([
                                np.concatenate([img11, img21], axis=1),
                                np.concatenate([img12, img22], axis=1),
                            ], axis=0)
                            )
                print("{:3d}:{:.4f}; {:.4f}; {:.4f}".format(cur_iter,
                                            total_losses[:,0].mean().cpu().item(),
                                            total_losses[:,1].mean().cpu().item(),
                                            total_losses[:,2].mean().cpu().item(),
                                            ))

            cur_iter+=1
        exit()

    pass

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

    graph.graph["face_flags"] = face_flags
    return graph


def prepare_dataset_and_model(v_colmap_dir, v_viz_face, v_bounds):
    print("Start to prepare dataset")
    print("1. Read imgs")

    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name) and False:
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
    if os.path.exists(graph_cache_name) and False:
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
    if os.path.exists(points_cache_name) and False:
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

    def project_points(v_projection_matrix, points_3d_pos):
        projected_points = np.transpose(v_projection_matrix @ np.transpose(np.insert(points_3d_pos, 3, 1, axis=1)))
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
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

    def compute_initial(v_graph, v_points_3d, v_points_2d, v_extrinsic, v_intrinsic):
        distance_threshold = 5  # 5m; not used

        v_graph.graph["face_center"] = np.zeros((len(v_graph.graph["faces"]), 2), dtype=np.float32)
        v_graph.graph["ray_c"] = np.zeros((len(v_graph.graph["faces"]), 3), dtype=np.float32)
        v_graph.graph["distance"] = np.zeros((len(v_graph.graph["faces"]),), dtype=np.float32)
        for id_face, id_edge_per_face in enumerate(v_graph.graph["faces"]):
            # Convex assumption
            center_point = np.stack(
                [v_graph.nodes[id_vertex]["pos_2d"] for id_vertex in id_edge_per_face], axis=0).mean(axis=0)
            v_graph.graph["face_center"][id_face] = center_point

        # Query points: (M, 2)
        # points from sfm: (N, 2)
        kd_tree = faiss.IndexFlatL2(2)
        kd_tree.add(v_points_2d.astype(np.float32))
        vertices_2d = np.asarray([v_graph.nodes[id_node]["pos_2d"] for id_node in v_graph.nodes()])  # (M, 2)
        centroids_2d = v_graph.graph["face_center"]
        query_points = np.concatenate([vertices_2d, centroids_2d], axis=0)
        shortest_distance, index_shortest_distance = kd_tree.search(query_points, 32)  # (M, K)

        points_from_sfm_camera = (v_extrinsic @ np.insert(v_points_3d, 3, 1, axis=1).T).T[:, :3]  # (N, 3)

        # Select the point which is nearest to the actual ray for each endpoints
        # 1. Construct the ray
        # (M, 2); points in camera coordinates
        ray_c = (np.linalg.inv(v_intrinsic) @ np.insert(query_points, 2, 1, axis=1).T).T
        ray_c = ray_c / np.linalg.norm(ray_c + 1e-6, axis=1, keepdims=True)  # Normalize the points
        nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
        # Compute the shortest distance from the candidate point to the ray for each query point
        # (M, K, 1): K projected distance of the candidate point along each ray
        distance_of_projection = nearest_candidates @ ray_c[:, :, np.newaxis]
        # (M, K, 3): K projected points along the ray
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


@hydra.main(config_name="phase6.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))
    data = prepare_dataset_and_model(
        v_cfg["dataset"]["colmap_dir"],
        v_cfg["dataset"]["id_viz_face"],
        v_cfg["dataset"]["scene_boundary"],
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    optimize(data, v_cfg["trainer"]["output"])

    model = Phase5(v_cfg, data)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        default_root_dir=log_dir,
        # precision=16,
        # gradient_clip_val=0.5
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
