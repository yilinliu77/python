import itertools
import sys, os
import time
from copy import copy
from typing import List

import scipy.spatial
from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

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

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img, sample_img_prediction2
from shared.common_utils import debug_imgs, to_homogeneous, save_line_cloud, to_homogeneous_vector, normalize_tensor, \
    to_homogeneous_mat_tensor, to_homogeneous_tensor, normalized_torch_img_to_numpy, padding, \
    vector_to_sphere_coordinate, sphere_coordinate_to_vector, caculate_align_mat, normalize_vector, \
    pad_and_enlarge_along_y, refresh_timer, get_line_mesh, ray_line_intersection1, ray_line_intersection2

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
# from src.neural_recon.phase1 import NGPModel
from scipy.spatial import Delaunay
from math import ceil

from src.neural_recon.phase_no_grad_edge_util import Collision_checker


def sample_new_distance(v_original_distances,
                        num_sample=100, scale_factor=1.6,
                        v_max=10, ):
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


def sample_points_2d(v_edge_points, v_num_horizontal,
                     v_img_width=800, v_vertical_length=10,
                     v_max_points=500):
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
                        v_img1, v_img2,v_sample_density=0.01):
    num_edge, _, _ = v_edge_points.shape

    num_horizontal = torch.clamp(
        (torch.linalg.norm(v_edge_points[:, 0] - v_edge_points[:, 1], dim=-1) / v_sample_density).to(torch.long),
        2, 1000)

    points_2d1 = (v_intrinsic @ v_edge_points.view(-1, 3).T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-6)
    points_2d1 = points_2d1.reshape(num_edge, 2, 2)

    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_edge_points.view(-1, 3)).T).transpose(0, 1)
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-6)
    points_2d2 = points_2d2.reshape(num_edge, 2, 2)
    valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])  # And in two edge points
    valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])  # And in x and y
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    num_sampled_points1, sampled_points1 = sample_points_2d(points_2d1.reshape(-1, 2, 2),
                                                            num_horizontal.reshape(-1),
                                                            v_vertical_length=5,
                                                            v_img_width=v_img1.shape[1])
    _, sampled_points2 = sample_points_2d(points_2d2.reshape(-1, 2, 2),
                                          num_horizontal.reshape(-1),
                                          v_img_width=v_img2.shape[1])

    # 4. Sample pixel color
    sample_imgs1 = sample_img(v_img1[None, None, :], sampled_points1[None, :, :])[0]
    sample_imgs2 = sample_img(v_img2[None, None, :], sampled_points2[None, :, :])[0]

    similarity_loss = loss4(sample_imgs1, sample_imgs2[None, :], num_sampled_points1)[0]
    similarity_loss = similarity_loss[0]
    similarity_loss = similarity_loss.reshape(num_edge, )

    # Visualize
    is_viz=False
    if is_viz:
        num_samples = num_sampled_points1.reshape(36,100,3)
        for id_point in range(num_samples.shape[0]):
            img1 = cv2.cvtColor((v_img1.cpu().numpy()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor((v_img2.cpu().numpy()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            id_start = num_sampled_points1[:100*3*id_point].sum()
            coords1=sampled_points1[id_start:id_start+num_samples[id_point,0].sum()]
            coords1 = (coords1 * 800).cpu().numpy().astype(np.int64)
            coords1=np.clip(coords1,0,799)
            img1[coords1[:,1],coords1[:,0]]=(0,0,255)
            coords2 = sampled_points2[id_start:id_start + num_samples[id_point, 0].sum()]
            coords2 = (coords2 * 800).cpu().numpy().astype(np.int64)
            coords2=np.clip(coords2,0,799)
            img2[coords2[:,1],coords2[:,0]]=(0,0,255)

            similarity_loss_ = similarity_loss.reshape(36,100,3).cpu().numpy()

            print("{:.2f}, {:.2f}, {:.2f}; {}, {}, {}".format(
                similarity_loss_[id_point,0,0], similarity_loss_[id_point,0,1], similarity_loss_[id_point,0,2],
                num_samples[id_point,0,0],num_samples[id_point,0,1],num_samples[id_point,0,2]
            ))

            for i in range(3):
                img11 = sample_imgs1[id_start:id_start+num_samples[id_point, 0, i]].reshape(-1, 20).cpu().numpy()
                img11=(img11*255).astype(np.uint8)
                img22 = sample_imgs2[id_start:id_start+num_samples[id_point, 0, i]].reshape(-1, 20).cpu().numpy()
                img22=(img22*255).astype(np.uint8)
                cv2.imwrite("output/{}.jpg".format(i), np.concatenate((img11, img22), axis=1))
                id_start = id_start+num_samples[id_point, 0, i]

            cv2.imshow("1", np.concatenate((img1,img2), axis=1))
            cv2.waitKey(0)



        pass

    return similarity_loss, valid_mask2, num_sampled_points1


def determine_valid_edges(v_graph, v_img):
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
    return


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

        point_id_per_edge = []
        edge_id_dict = {}
        for edge in graph.edges():
            point_id_per_edge.append(edge)
            edge_id_dict[(edge[0], edge[1])] = len(point_id_per_edge)
            edge_id_dict[(edge[1], edge[0])] = -len(point_id_per_edge)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        # Edges
        num_edge = len(graph.edges())
        num_max_edge_per_vertex = max([len(graph[item]) for item in graph.nodes])
        edge_rays_c = rays_c[point_id_per_edge]
        edge_distances_c = distances[point_id_per_edge]
        edge_is_not_black = determine_valid_edges(edge_rays_c, edge_distances_c, imgs[0], intrinsic)[:, 0]
        to_non_black_edge_id = -torch.ones(num_edge, device=device, dtype=torch.long)
        to_non_black_edge_id[edge_is_not_black] = torch.arange(edge_is_not_black.sum(), device=device)

        edge_point_id_per_face = []
        for face_ids, face_flag in zip(all_face_ids, face_flags):
            if not face_flag:
                continue
            local_id_edges = []
            for i, id_start in enumerate(face_ids):
                id_end = face_ids[(i + 1) % len(face_ids)]
                id_edge = edge_id_dict[(id_start, id_end)]
                mask = 1 if id_edge > 0 else -1
                local_id_edges.append((to_non_black_edge_id[abs(id_edge) - 1] + 1) * mask)
            a = torch.tensor(local_id_edges, dtype=torch.long, device=device)
            b = torch.roll(a, -1)
            edge_point_id_per_face.append(torch.stack((a, b), dim=1))

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
            ncc_loss, ncc_loss_mask = evaluate_candidates(edge_points_candidate, intrinsic,
                                                          transformation[0], imgs[0], imgs[1])
            ncc_loss[~ncc_loss_mask] = torch.inf  # Set it to 0 in order to get the samples below

            # Consistent loss
            sampled_index = torch.multinomial(torch.exp(-ncc_loss ** 4), 99, replacement=True)
            sampled_index = torch.cat((torch.zeros_like(sampled_index[:, 0:1]), sampled_index), dim=1)
            sampled_loss = torch.gather(ncc_loss, dim=1, index=sampled_index).mean(dim=0)
            sampled_edges = torch.gather(edge_points_candidate, dim=2,
                                         index=sampled_index.view(num_edge, 1, 100, 1).tile(1, 2, 1, 3))
            sampled_edges = sampled_edges.permute(2, 0, 1, 3)
            consistency = []
            for face in edge_point_id_per_face:
                true_face = face[torch.all(face != 0, dim=1)]
                if true_face.shape[0] == 0:
                    continue
                face_mask = true_face < 0
                chosen_points = sampled_edges[:, torch.abs(true_face) - 1]
                chosen_points[:, face_mask] = torch.flip(chosen_points[:, face_mask], dims=[2])
                distance = torch.linalg.norm(chosen_points[:, :, 0, 1] - chosen_points[:, :, 1, 0], dim=2).mean(dim=1)
                consistency.append(distance)
            consistency = torch.stack(consistency, dim=1).mean(dim=1)

            alpha = 1
            loss = sampled_loss * alpha + consistency * (1 - alpha)
            id_best = sampled_index[:, loss.argmin(dim=0)]
            chosen_distance = torch.gather(distances_candidate, dim=2, index=id_best.view(-1, 1, 1).tile(1, 2, 1))[:, :,
                              0]
            chosen_edge_points = torch.gather(edge_points_candidate, dim=2,
                                              index=id_best.view(-1, 1, 1, 1).tile(1, 2, 1, 3))[:, :, 0]
            chosen_loss = torch.gather(loss, dim=0, index=loss.argmin(dim=0))
            # chosen_edge_points = edge_points_candidate[:,:,0]
            edge_distances_c = chosen_distance
            # Log
            if True:
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                start_points_c = chosen_edge_points[:, 0]
                end_points_c = chosen_edge_points[:, 1]
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

            cur_iter += 1
        exit()

    pass


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


def sample_img_window(v_rgb, v_coor_2d):
    MAX_ITEM_PER_BATCH = 262144
    window_size = 7  # 窗口大小
    padding = (window_size - 1) // 2  # 计算填充大小
    v_rgb_padded = torch.nn.functional.pad(v_rgb, (padding, padding, padding, padding), mode='constant',
                                           value=0)  # 对图像进行填充

    # 计算归一化后的窗口大小，用于torch.grid_sample的输入
    img_height, img_width = v_rgb.shape[-2:]
    window_size_normalized = window_size / (img_width - 1), window_size / (img_height - 1)

    # 创建网格以获得7x7窗口
    grid_x, grid_y = torch.meshgrid(torch.linspace(-window_size_normalized[0], window_size_normalized[0], window_size),
                                    torch.linspace(-window_size_normalized[1], window_size_normalized[1], window_size))
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # 将x和y网格合并为一个网格张量
    grid = grid.to(v_coor_2d.device)

    coor_2d = v_coor_2d
    results = []
    for i in range(ceil(coor_2d.shape[1] / MAX_ITEM_PER_BATCH)):
        batch_coor_2d = coor_2d[:,
                        i * MAX_ITEM_PER_BATCH:min(coor_2d.shape[1], (i + 1) * MAX_ITEM_PER_BATCH)]  # 获取当前批次的坐标
        batch_coor_2d = batch_coor_2d.unsqueeze(2).unsqueeze(3).tile(1, 1, window_size, window_size,
                                                                     1)  # (n, num_points, window_size, window_size, 2)
        # TODO: fix the sample points in the edge of img(now it's clampped)
        batch_grid = (grid + batch_coor_2d * 2 - 1).clamp(-1, 1)  # 将网格加到坐标上，并将范围限制在[-1, 1]
        batch_grid = batch_grid.view(batch_grid.shape[0], batch_grid.shape[1], -1,
                                     2)  # (n, num_points, window_size*window_size, 2)
        # 使用grid_sample从输入图像中采样7x7窗口
        sampled_pixels = torch.nn.functional.grid_sample(v_rgb_padded, batch_grid,
                                                         align_corners=True)  # 输出维度为(1, 1, k, 49)
        results.append(sampled_pixels)

    sampled_pixels = torch.cat(results, dim=2).reshape(1, -1, window_size, window_size)  # 调整输出张量维度为(1, k, 7, 7)
    return sampled_pixels


def optimize(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vertical_length = 5

    for id_img1, graph in enumerate(v_graphs):
        # The vertex id for each face
        all_face_ids = np.asarray(graph.graph["faces"], dtype=object)

        # The ray of each node in camera coordinate
        rays_c = [None] * len(graph.nodes)
        # The distance of each node along the ray direction in camera coordinate
        ray_distances_c = [None] * len(graph.nodes)
        # id_end = []
        for idx, id_points in enumerate(graph.nodes):
            rays_c[idx] = graph.nodes[id_points]["ray_c"]
            ray_distances_c[idx] = graph.nodes[id_points]["distance"]
            # id_end.append(list(graph[0].keys()))

        # prepare some data
        id_src_imgs = v_img_pairs[id_img1][:, 0]
        ref_img = cv2.imread(v_img_database[id_img1].img_path, cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path, cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.
        # transformation store the transformation matrix from ref_img to src_imgs
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        c1_2_c2 = torch.from_numpy(
            v_img_database[int(id_src_imgs[0])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        ).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Vertex
        num_vertex = len(rays_c)
        # rays_c = query_points(nodes)' pos in camera coordinate
        rays_c = torch.from_numpy(np.stack(rays_c)).to(device).to(torch.float32)
        ray_distances_c = torch.from_numpy(np.stack(ray_distances_c)).to(device).to(torch.float32)
        # Face
        # face_flag record if it's a valid face
        face_flags = torch.tensor(graph.graph["face_flags"], dtype=torch.bool, device=device)
        centroid_rays_c = torch.from_numpy(graph.graph["ray_c"]).to(device).to(torch.float32)
        centroid_ray_distances_c = torch.from_numpy(graph.graph["distance"]).to(device).to(torch.float32)

        def prepare_edge_data(v_graph):
            determine_valid_edges(v_graph, imgs[0])
            # The id of the vertex for all edges
            point_id_per_edge = []
            # The id of a specific edge in the `point_id_per_edge`
            edge_id_dict = {}
            for edge in v_graph.edges():
                if not v_graph.edges[edge]["valid_flag"] or v_graph.edges[edge]["is_black"]:
                    continue
                point_id_per_edge.append(edge)
                # + and - to distinguish whether the edge is flip
                edge_id_dict[(edge[0], edge[1])] = len(point_id_per_edge)
                edge_id_dict[(edge[1], edge[0])] = -len(point_id_per_edge)
            return point_id_per_edge, edge_id_dict

        point_id_per_edge, edge_id_dict = prepare_edge_data(graph)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        def prepare_node_data(v_point_id_per_edge, v_device):
            unique_point_ids, counts = torch.unique(v_point_id_per_edge, return_counts=True)
            unique_point_ids = unique_point_ids[counts > 1]
            num_point = unique_point_ids.shape[0]
            max_counts = counts.max()

            node_data = -torch.ones((num_point, max_counts + 1), device=v_device, dtype=torch.long)
            for idx, id_point in enumerate(unique_point_ids):
                coords = torch.stack(torch.where(v_point_id_per_edge == id_point), dim=1)
                num_valid = coords.shape[0]
                coords[:, 1] = 1 - coords[:, 1]  # Index another point
                node_data[idx, 0] = id_point
                node_data[idx, 1:num_valid + 1] = v_point_id_per_edge[coords[:, 0], coords[:, 1]]
            return node_data

        # Index the valid nodes and the corresponding edges
        # The node_data is in the shape of [M,K],
        # where M is the number of valid node, K is the maximum edges (K=2) means every node has at most 1 edge
        # invalid edges are padded with -1
        node_data = prepare_node_data(point_id_per_edge, device)

        # Visualize the nodes that to be optimized
        if False:
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]

            # Draw edges
            for node in node_data:
                id_start = int(node[0].cpu().item())
                for id_end in node[1:]:
                    id_end = int(id_end.cpu().item())
                    if id_end == -1:
                        break
                    start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                    end_pos = np.around((graph.nodes[id_end]["pos_2d"]) * shape).astype(np.int64)
                    cv2.line(img11, start_pos, end_pos, (0, 0, 255), 2)

            # Draw nodes
            for node in node_data:
                id_start = int(node[0].cpu().item())
                start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                cv2.circle(img11, start_pos, 2, (0, 255, 255), 2)

            cv2.imshow("1", img11)
            cv2.waitKey(0)

        def optimize_node(v_node_data, v_rays_c, v_ray_distances_c):
            id_start_points = v_node_data[:, 0]
            id_end_points = v_node_data[:, 1:]
            # M
            num_vertex = id_start_points.shape[0]
            # N
            num_max_edge_per_vertex = id_end_points.shape[1]

            optimized_distance = v_ray_distances_c.clone()
            # (M, N)
            end_distances = optimized_distance[id_end_points]
            # K
            num_sample = 100

            # (M, K, N)
            edge_valid_flag = (id_end_points == -1)[:, None, :].tile(1, num_sample, 1)

            cur_iter = 0
            last_loss = None
            num_tolerence = 100
            while True:
                # (M, K)
                start_distances_candidate = sample_new_distance(optimized_distance[id_start_points],
                                                                num_sample=num_sample).view(
                    num_vertex, num_sample, )
                # (M, K, 3)
                start_points_c = start_distances_candidate[:, :, None] * v_rays_c[id_start_points][:, None, :]
                # (M, K, N, 3)
                start_points_c = start_points_c[:, :, None, :].tile(1, 1, num_max_edge_per_vertex, 1)
                # (M, K, N, 3)
                end_points_c = (end_distances[:, :, None] * v_rays_c[id_end_points])[:, None, :, :].tile(1, 100, 1, 1)

                # (M, K, N, 2, 3)
                edges_points = torch.stack((start_points_c, end_points_c), dim=-2)

                ncc_loss, ncc_loss_mask, num_samples_per_edge = evaluate_candidates(edges_points.reshape(-1, 2, 3),
                                                              intrinsic, transformation[0], imgs[0], imgs[1],
                                                              v_sample_density=0.01)
                ncc_loss[~ncc_loss_mask] = torch.inf
                ncc_loss = ncc_loss.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                num_samples_per_edge = num_samples_per_edge.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                ncc_loss[edge_valid_flag] = 0
                num_samples_per_edge[edge_valid_flag] = 0
                # ncc_loss = ncc_loss * num_samples_per_edge/num_samples_per_edge.sum(dim=-1,keepdim=True)
                ncc_loss = ncc_loss.mean(dim=-1)
                id_best = ncc_loss.argmin(dim=1)

                optimized_distance[id_start_points] = torch.gather(start_distances_candidate, 1, id_best[:, None])[:, 0]
                end_distances = optimized_distance[id_end_points]

                total_loss = torch.gather(ncc_loss, 1, id_best[:, None]).mean()

                if last_loss is None:
                    last_loss = total_loss
                else:
                    delta = last_loss - total_loss

                    print("{:3d}:{:.4f}; Delta:{:.4f}".format(cur_iter,
                                                              total_loss.cpu().item(),
                                                              delta.cpu().item()
                                                              ))

                    if delta < 1e-5:
                        num_tolerence -= 1
                    else:
                        num_tolerence = 100
                        last_loss = total_loss
                    if num_tolerence <= 0:
                        break

                # Visualize
                if True:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                    img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                    start_points_c = optimized_distance[id_start_points][:, None] * v_rays_c[id_start_points]
                    start_points_c = start_points_c[:, None, :, ].tile(1, num_max_edge_per_vertex, 1)
                    end_points_c = optimized_distance[id_end_points][:, :, None] * v_rays_c[id_end_points]

                    all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                    all_points_c = all_points_c[~edge_valid_flag[:, 0]]
                    all_points_c = all_points_c.reshape(-1, 3)
                    p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                    p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                    p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                    p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                    p_2d1 = p_2d1.reshape(-1, 2, 2)
                    p_2d2 = p_2d2.reshape(-1, 2, 2)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                    p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                    line_color = (0, 0, 255)
                    line_thickness = 2
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_edge in range(p_2d1.shape[0]):
                        cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                        cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                    cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                                np.concatenate([
                                    np.concatenate([img11, img21], axis=1),
                                    np.concatenate([img12, img22], axis=1),
                                ], axis=0)
                                )

                cur_iter += 1
            return optimized_distance

        optmized_distance_cache = "output/optimized_distance.tar"
        if os.path.exists(optmized_distance_cache) and False:
            optimized_distance = torch.load(optmized_distance_cache).to(device)
        else:
            optimized_distance = optimize_node(node_data, rays_c, ray_distances_c)
            torch.save(optimized_distance, optmized_distance_cache)

        # Remove the redundant face and edges in the graph
        # And build the dual graph in order to navigate between patches
        def fix_graph(v_graph, is_visualize=False):
            dual_graph = nx.Graph()

            id_original_to_current = {}
            for id_face, face in enumerate(graph.graph["faces"]):

                if not graph.graph["face_flags"][id_face]:
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
                                    face_center=graph.graph['face_center'][id_face],
                                    ray_c=graph.graph['ray_c'][id_face])

            for node in dual_graph.nodes():
                faces = dual_graph.nodes[node]["id_vertex"]
                for idx, id_start in enumerate(faces):
                    id_end = faces[(idx + 1) % len(faces)]
                    t = copy(graph.edges[(id_start, id_end)]["id_face"])
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

        fix_graph(graph)

        # fun defined by xdt
        def fit_plane_svd(points: torch.Tensor) -> torch.Tensor:
            centroid = torch.mean(points, axis=0)
            centered_points = points - centroid
            u, s, vh = torch.linalg.svd(centered_points)
            d = -torch.dot(vh[-1], centroid)
            abcd = torch.cat((vh[-1], torch.tensor([d]).to(device)))
            return abcd

        # TODO: using ray_c
        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def ncc_matching_cost(img1, img2):
            mean1 = torch.mean(img1)
            mean2 = torch.mean(img2)
            std1 = torch.std(img1)
            std2 = torch.std(img2)
            norm_img1 = (img1 - mean1) / std1
            norm_img2 = (img2 - mean2) / std2
            ncc = torch.sum(norm_img1 * norm_img2) / (img1.numel() - 1)
            matching_cost = 1 - ncc
            return matching_cost

        def vectors_to_angles(normal_vectors):
            normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
            x, y, z = normal_vectors.unbind(dim=1)
            phi = torch.atan2(y, x)
            theta = torch.acos(z)
            return torch.stack([phi, theta], dim=1)

        def angles_to_vectors(angles):
            phi, theta = angles.unbind(dim=1)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            return torch.stack([x, y, z], dim=1)

        # 对应位置求交点
        def intersection_of_ray_and_plane(planes, rays_direction):
            # planes: n*4 ray: n*3
            n = planes[:, :3]
            d = planes[:, 3]
            denominator = torch.sum(n * rays_direction, dim=1)
            t = -d / torch.sum(n * rays_direction, dim=1)
            intersection_points = torch.unsqueeze(t, 1) * rays_direction
            valid_intersection = (denominator != 0) & (t >= 0)
            # n * _
            return valid_intersection, intersection_points

        def bilateral_ncc_(v_img1, v_img2):
            batch_size = v_img1.shape[0]
            window_size = v_img1.shape[1]
            device = v_img1.device
            # Add weights
            sigma_spatial = 3
            spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
            spatial_weights = torch.stack(torch.meshgrid(
                torch.arange(window_size, device=device, dtype=torch.float32),
                torch.arange(window_size, device=device, dtype=torch.float32),
                indexing="xy"
            ), dim=2) - window_size // 2
            spatial_weights = torch.linalg.norm(spatial_weights, dim=-1)
            spatial_weights = spatial_weights ** 2 * spatial_normalization_
            spatial_weights = torch.exp(-spatial_weights)
            spatial_weights = spatial_weights / spatial_weights.sum()

            v_img1 = (v_img1 * spatial_weights[None, :, :])
            v_img2 = (v_img2 * spatial_weights[None, :, :])

            norm_img1 = v_img1 - v_img1.mean(dim=[1, 2])[:, None, None]
            norm_img2 = v_img2 - v_img2.mean(dim=[1, 2])[:, None, None]

            ncc1 = torch.sum(norm_img1 * norm_img2, dim=[1, 2])
            ncc2 = torch.sqrt(torch.sum(norm_img1 ** 2, dim=[1, 2]) * torch.sum(norm_img2 ** 2, dim=[1, 2]))
            ncc2 = torch.clamp_min(ncc2, 1e-6)
            ncc = 1 - ncc1 / ncc2
            return ncc

        def bilateral_ncc(v_points_c, v_normal_c, v_intrinsic, v_transformation, v_img1, v_img2,
                          v_window_size=7):
            device = v_points_c.device
            num_points = v_points_c.shape[0]
            world_up_vector = torch.zeros_like(v_normal_c)
            world_up_vector[:, 2] = 1
            right_vector = normalize_tensor(torch.cross(world_up_vector, v_normal_c))
            up_vector = normalize_tensor(torch.cross(v_normal_c, right_vector))

            height, width = v_img1.shape[:2]
            resolution = 1 / min(height, width) * torch.linalg.norm(v_points_c, dim=-1)

            index = torch.arange(v_window_size, device=device) - v_window_size // 2
            index = torch.stack(torch.meshgrid(index, index, indexing="xy"), dim=2)
            window_points = \
                (index[:, :, 0].view(1, -1, 1) * right_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3) + \
                (index[:, :, 1].view(1, -1, 1) * up_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3)
            window_points = window_points + v_points_c[:, None, None, :]

            points_2d1 = (v_intrinsic @ window_points.reshape(-1, 3).T).T
            points_2d1 = points_2d1[:, :2] / points_2d1[:, 2:3]
            points_2d2 = (v_transformation @ to_homogeneous_tensor(window_points.reshape(-1, 3)).T).T
            points_2d2 = points_2d2[:, :2] / points_2d2[:, 2:3]

            sample_imgs1 = sample_img(v_img1[None, None, :], points_2d1[None, :, :])[0]
            sample_imgs2 = sample_img(v_img2[None, None, :], points_2d2[None, :, :])[0]

            sample_imgs1 = sample_imgs1.reshape(num_points, v_window_size, v_window_size)
            sample_imgs2 = sample_imgs2.reshape(num_points, v_window_size, v_window_size)

            ncc = bilateral_ncc_(sample_imgs1, sample_imgs2)

            # Visualize
            if False:
                points_2d1 = points_2d1.reshape(num_points, v_window_size * v_window_size, 2)
                points_2d2 = points_2d2.reshape(num_points, v_window_size * v_window_size, 2)

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                point_color = (0, 255, 255)
                point_thickness = 3

                for i in range(num_points):
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                    img11[p_2d1[i, :, 1], p_2d1[i, :, 0]] = point_color
                    img21[p_2d2[i, :, 1], p_2d2[i, :, 0]] = point_color

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            return ncc

        # img data pre
        intrinsic1 = v_img_database[id_img1].intrinsic
        intrinsic2 = v_img_database[1].intrinsic
        intrinsic1 = torch.from_numpy(intrinsic1).to(device).to(torch.float32)
        intrinsic2 = torch.from_numpy(intrinsic2).to(device).to(torch.float32)
        v_img1 = imgs[0]
        v_img2 = imgs[1]

        # optimized data pre
        optimized_points_pos = optimized_distance[:, None, None] * rays_c[:, None, :]
        patches_list = graph.graph["dual_graph"].nodes  # each patch = id_vertexes
        patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]

        # polygon -> edge (using the centroid)
        # TODO：1
        edge_list = []  # num_tri * 2
        for patch in patch_vertexes_id:
            num_vertex = len(patch)
            edge_id = [[patch[i], patch[(i + 1) % num_vertex]] for i in range(num_vertex)]
            edge_list.extend(edge_id)
        num_points_per_face = torch.tensor([len(item) for item in patch_vertexes_id], dtype=torch.long, device=device)
        edge_list = torch.tensor(edge_list).to(device).to(torch.long)
        edge = optimized_points_pos[edge_list].squeeze(dim=2)
        # patch_id of edge belongs to
        edges_indices = [torch.tensor([i] * len(patch_vertexes)) for i, patch_vertexes in enumerate(patch_vertexes_id)]
        edges_indices = torch.cat(edges_indices).to(device)

        # torch.cat((scatter_mean(edge[:, 0], edges_indices, dim=0)[:, None, :].repeat_interleave(num_points_per_face,

        def initialize_plane_hypothesis(v_rays_c, v_optimized_points_pos, v_patches_list):
            # 1. calculate ncc loss of each patch
            # for each patch do:
            # a) 3d vertexes of each patch -> fitting plane
            # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
            # c) sample points on projected polygon
            # d) project sample points to img1 and img2 and get pixel
            # e) calculate the ncc loss
            poly_abcd_list = []
            ncc_list = []
            # sample_points_list = []
            # projected_vertexes_list = []
            # sample_points_2d1 = []
            # sample_points_2d2 = []
            # centroid_list = []
            for patch_idx in range(len(v_patches_list)):
                id_vertexes = v_patches_list[patch_idx]['id_vertex']
                pos_vertexes = v_optimized_points_pos[id_vertexes].view(-1, 3)
                assert (len(pos_vertexes) >= 3)
                if patch_idx == 8:
                    print("debug")
                # a) 3d vertexes of each patch -> fitting plane
                p_abcd = fit_plane_svd(pos_vertexes)
                poly_abcd_list.append(p_abcd)

                # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
                projected_points = ray_line_intersection2(p_abcd,
                                                          torch.zeros_like(v_rays_c[id_vertexes]),
                                                          v_rays_c[id_vertexes])
                # projected_points = project_points_to_plane(pos_vertexes, p_abcd)
                # projected_vertexes_list.append(projected_points)

                # c) sample points on projected polygon
                # polygon -> triangles (using the centroid)
                num_vertex = projected_points.shape[0]
                edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                centroid = torch.mean(projected_points, dim=0)
                # centroid_list.append(centroid)
                edges = projected_points[torch.tensor(edges_idx)]
                triangles = torch.cat((edges, centroid.tile(num_vertex, 1)[:, None, :]), dim=1)

                # sample points on the fitting plane
                num_samples, sample_points_on_face = \
                    sample_triangles(100, triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :])
                # sample_points_list.append(sample_points_on_face)

                triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                               triangles[:, 1, :] - triangles[:, 2, :]))
                triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                ncc = bilateral_ncc(sample_points_on_face, triangle_normal,
                                    intrinsic1, transformation[0], v_img1, v_img2).mean()

                # # d) project sample points to img1 and img2
                # points_2d1 = (intrinsic1 @ sample_points_on_face.T).T
                # points_2d1 = points_2d1[:, :2] / points_2d1[:, 2:3]
                # points_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_on_face).T).T
                # points_2d2 = points_2d2[:, :2] / points_2d2[:, 2:3]
                # # sample_points_2d1.append(points_2d1)
                # # sample_points_2d2.append(points_2d2)
                #
                # # get pixel
                # sample_imgs1 = sample_img(v_img1[None, None, :], points_2d1[None, :, :])[0]
                # sample_imgs2 = sample_img(v_img2[None, None, :], points_2d2[None, :, :])[0]
                #
                # # e) calculate the ncc loss
                # ncc = ncc_matching_cost(sample_imgs1, sample_imgs2)
                ncc_list.append(ncc)

                # Visualize
                if False:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                    p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point], 1, point_color, point_thickness)

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            # 2. sorted patches according the ncc loss
            ncc_list = torch.stack(ncc_list).view(-1)
            poly_abcd_list = torch.stack(poly_abcd_list).view(-1, 4)
            # centroid_list = torch.stack(centroid_list).view(-1, 3)
            return poly_abcd_list, ncc_list

        # For each patch, initialize plane hypothesis
        poly_abcd_list, ncc = initialize_plane_hypothesis(rays_c, optimized_points_pos, patches_list)

        def save_plane(v_abcds, rays_c, patch_vertexes_id, file_path="output/init.ply"):
            vertices = []
            polygons = []
            acc_num_vertices = 0
            for id_patch in range(len(patch_vertexes_id)):
                intersection_points = ray_line_intersection2(v_abcds[id_patch],
                                                             torch.zeros_like(rays_c[patch_vertexes_id[id_patch]]),
                                                             rays_c[patch_vertexes_id[id_patch]])
                vertices.append(intersection_points)
                polygons.append(np.arange(intersection_points.shape[0]) + acc_num_vertices)
                acc_num_vertices += intersection_points.shape[0]
                pass
            vertices = torch.cat(vertices, dim=0).cpu().numpy()
            with open(file_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex {}\nproperty float x\nproperty float y\nproperty float z\n".format(
                    acc_num_vertices))
                f.write("element face {}\nproperty list uchar int vertex_index\n".format(len(polygons)))
                f.write("end_header\n")
                for ver in vertices:
                    f.write("{} {} {}\n".format(ver[0], ver[1], ver[2]))
                for polygon in polygons:
                    f.write("{}".format(len(polygon)))
                    for item in polygon:
                        f.write(" {}".format(item))
                    f.write("\n")
                pass
            print("Save done")
            return

        def visualize_plane(ncc, patch_vertexes_id):
            for idx in range(len(patch_vertexes_id)):
                id_vertices = patch_vertexes_id[idx]
                ncc_ = ncc[idx]
                print("{:2d}: ncc={:.2f} vertex_num={:2d}".format(idx, ncc_, len(id_vertices)))

                start_points_c = optimized_points_pos[id_vertices].view(-1, 3)
                end_points_c = torch.roll(start_points_c, -1, dims=[0])

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                all_points_c = all_points_c.reshape(-1, 3)
                p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                p_2d1 = p_2d1.reshape(-1, 2, 2)
                p_2d2 = p_2d2.reshape(-1, 2, 2)

                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(p_2d1.shape[0]):
                    cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                    cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                for i_point in range(p_2d1.shape[0]):
                    cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                cv2.imshow("1", np.concatenate([
                    np.concatenate([img11, img21], axis=1),
                    np.concatenate([img12, img22], axis=1),
                ], axis=0))
                cv2.waitKey()

        save_plane(poly_abcd_list, rays_c, patch_vertexes_id)

        if False:
            visualize_plane(ncc, patch_vertexes_id)

        def sample_depth_and_angle(depth, angle, num_sample=100):
            # sample depth
            sample_depth = torch.normal(depth.repeat(num_sample, 1))
            # sample angle
            sample_angle = torch.normal(angle.repeat(num_sample, 1),
                                        torch.full_like(angle, 2 * math.pi / 6).repeat(num_sample, 1))
            sample_angle = torch.clamp(sample_angle, min=-2 * math.pi, max=2 * math.pi)
            # add itself
            sample_depth = torch.cat((depth[None, :], sample_depth), dim=0)
            sample_angle = torch.cat((angle, sample_angle), dim=0)
            return sample_depth, sample_angle

        def compute_plane_abcd(patch_ray, ray_depth, normal):
            intersection = patch_ray * ray_depth
            d = -torch.sum(intersection * normal, dim=1, keepdim=True)
            plane_abcd = torch.cat([normal, d], dim=1)
            return intersection, plane_abcd

        def intersection_of_ray_and_all_plane(planes_abcd, vertexes_ray):
            # planes: n*4 ray: m*3
            n_planes = planes_abcd.size(0)
            m_rays = vertexes_ray.size(0)

            # 将平面参数和射线向量扩展为广播兼容的形状
            planes_abcd_expanded = planes_abcd.unsqueeze(1).expand(n_planes, m_rays, 4)
            vertexes_ray_expanded = vertexes_ray.unsqueeze(0).expand(n_planes, m_rays, 3)

            # 计算射线与平面的交点
            numerator = -planes_abcd_expanded[..., -1].unsqueeze(-1)  # 分子：-(D)
            denominator = torch.sum(planes_abcd_expanded[..., :3] * vertexes_ray_expanded, dim=-1).unsqueeze(
                -1)  # 分母：(A * v_ray.x + B * v_ray.y + C * v_ray.z)
            t = numerator / denominator  # t = -(D) / (A * v_ray.x + B * v_ray.y + C * v_ray.z)
            intersection_points = t * vertexes_ray_expanded  # p = p0 + t * v_ray，这里我们假设 p0 为原点 (0, 0, 0)
            # n*m*3
            return intersection_points

        def points_outside_polygon(sample_points, polygon_points, transformation_):
            sample_points = (intrinsic @ sample_points.T).T
            sample_points = (transformation_ @ to_homogeneous_tensor(sample_points).T).T
            sample_points = sample_points[:, :2] / sample_points[:, 2:3]
            polygon_points = (intrinsic @ polygon_points.T).T
            polygon_points = (transformation_ @ to_homogeneous_tensor(polygon_points).T).T
            polygon_points = polygon_points[:, :2] / polygon_points[:, 2:3]
            # 多边形的边
            poly_edges_start = polygon_points
            poly_edges_end = torch.roll(polygon_points, -1, dims=0)
            # 判断点是否在多边形外部
            cross_products = (sample_points[:, None, 0] - poly_edges_start[:, 0]) * (
                    poly_edges_end[:, 1] - poly_edges_start[:, 1]) - (
                                     sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                                     poly_edges_end[:, 0] - poly_edges_start[:, 0])
            y_diff_sign = (sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                    sample_points[:, None, 1] - poly_edges_end[:, 1])
            # 计算每个点与多边形的交点数
            intersection_count = torch.sum((cross_products > 0) & (y_diff_sign <= 0), dim=1)
            # 交点数为奇数的点在多边形内部，交点数为偶数的点在多边形外部
            outside_polygon_mask = intersection_count % 2 == 0
            return outside_polygon_mask

        def points_in_adj_patch(sample_points, v_polygon_points, transformation_):
            mask = torch.zeros(sample_points.shape[0], dtype=torch.bool).to(device)
            for polygon_points in v_polygon_points:
                mask = torch.logical_or(mask, ~points_outside_polygon(sample_points, polygon_points, transformation_))
            return mask

        def optimize_plane_hypothesis(poly_abcd_list, ncc, v_patches_list, v_dual_graph, v_c1_2_c2):
            # 1. get the intersection of the fitting plane and ray_c
            rays_c_valid_patch = np.stack([v_patches_list[i]['ray_c'] for i in range(len(v_patches_list))],axis=0)
            rays_c_valid_patch = torch.from_numpy(rays_c_valid_patch).to(device).to(torch.float32)
            # isValid, intersection = intersection_of_ray_and_plane(poly_abcd_list, rays_c_valid_patch)
            # assert (torch.sum(isValid) == isValid.shape[0])
            # depth = torch.linalg.norm(intersection, dim=-1)
            # angle = vectors_to_angles(poly_abcd_list[:, 0:3])

            sorted_values, sorted_indices = torch.sort(ncc, descending=False)

            collision_checker = Collision_checker()
            optimized_abcd_list = poly_abcd_list.clone()
            # 2. process the patches in the order of ncc loss
            for i in range(len(sorted_indices)):
                patch_id, init_ncc_loss = sorted_indices[i], sorted_values[i]
                isValid, intersection = intersection_of_ray_and_plane(optimized_abcd_list[patch_id].unsqueeze(0),
                                                                      rays_c_valid_patch[patch_id].unsqueeze(0))
                assert (torch.sum(isValid) == isValid.shape[0])
                depth = torch.linalg.norm(intersection, dim=-1)
                angle = vectors_to_angles(optimized_abcd_list[patch_id, 0:3].unsqueeze(0))

                vertexes_id = torch.tensor(patch_vertexes_id[patch_id]).to(device).to(torch.long)
                vertexes_pos = optimized_points_pos[vertexes_id]
                vertexes_ray = rays_c[vertexes_id]
                patch_ray = torch.tensor(v_patches_list[int(patch_id)]['ray_c']).to(device).to(torch.float32)

                # optimize the patch
                cur_iter = 0
                ncc_loss_optimized = init_ncc_loss
                id_best = 0  # init patch will be place at 0 index
                num_plane_sample = 100
                while True:
                    # 1. Sample candidate plane
                    samples_depth, samples_angle = sample_depth_and_angle(depth, angle, num_plane_sample)
                    samples_abc = angles_to_vectors(samples_angle)
                    samples_planes_intersection, samples_abcd = compute_plane_abcd(patch_ray, samples_depth,
                                                                                   samples_abc)

                    # num_samples_plane * num_vertexes_cur_patch * 3
                    samples_planes_vertexes_pos = intersection_of_ray_and_all_plane(samples_abcd, vertexes_ray)

                    # 2. Get intersection points
                    # candidate_triangle_points (N, 3, 3)
                    num_vertex = vertexes_id.shape[0]
                    edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                    centroid = torch.mean(samples_planes_vertexes_pos, dim=1)
                    # centroid_list.append(centroid)
                    edges = samples_planes_vertexes_pos[:, torch.tensor(edges_idx)]
                    triangles = torch.cat((edges, centroid[:, None, None, :].tile(1, num_vertex, 1, 1)), dim=2)
                    triangles = triangles.view(-1, 3, 3)  # (101*num_tri,3,3)

                    # 3. calculate loss (point ncc loss and line loss)
                    # sample points on the fitting plane
                    # get each sample point's ncc loss, then aggregate to each patch
                    num_samples, sample_points_on_face = sample_triangles(100, triangles[:, 0, :],
                                                                          triangles[:, 1, :], triangles[:, 2, :])

                    triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                                   triangles[:, 1, :] - triangles[:, 2, :]))
                    triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                    ncc_loss_points = bilateral_ncc(sample_points_on_face, triangle_normal, intrinsic,
                                                    transformation[0], v_img1, v_img2)

                    # get the edge point of each sample patch plane
                    ncc_loss_edge, ncc_loss_mask = evaluate_candidates(edges.reshape(-1, 2, 3),
                                                                       intrinsic, transformation[0], v_img1, v_img2)

                    # filter the points int adj patch (in img2 candidate)
                    filter_points = True
                    if filter_points:
                        sample_points_on_face_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            sample_points_on_face).transpose(0, 1)).transpose(0, 1)[:, :3]
                        point_mask = collision_checker.check_ray(
                            torch.zeros_like(sample_points_on_face_c2),
                            sample_points_on_face_c2)
                        edges_shape = edges.shape
                        edges_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            edges.reshape(-1, 3)).transpose(0, 1)).transpose(0, 1)[:, :3]
                        edge_mask = collision_checker.check_ray(
                            torch.zeros_like(edges_c2),
                            edges_c2).reshape(*edges_shape[:-1])
                        edge_mask = torch.all(edge_mask, dim=-1)

                        # adj_patch_id_list = [edge[1] for edge in list(v_dual_graph.edges(int(patch_id)))]
                        # adj_patch_vertex_pos = []
                        # for adj_patch_id in adj_patch_id_list:
                        #     vertexes_id_adj = torch.tensor(patch_vertexes_id[adj_patch_id]).to(device).to(torch.long)
                        #     vertexes_ray_adj = rays_c[vertexes_id_adj]
                        #     valid_mask, vertexes_pos_adj = intersection_of_ray_and_plane(
                        #         poly_abcd_list[patch_id].tile(vertexes_ray_adj.shape[0], 1),
                        #         vertexes_ray_adj)
                        #     adj_patch_vertex_pos.append(vertexes_pos_adj)
                        # mask_points_in_adj_patch = points_in_adj_patch(sample_points_on_face,
                        #                                                adj_patch_vertex_pos,
                        #                                                transformation[0])
                        # ncc_loss_points = ncc_loss_points[~mask_points_in_adj_patch]
                        ncc_loss_points = ncc_loss_points[~point_mask]
                        num_samples = scatter_add((~point_mask).to(torch.int64),
                                                  torch.arange(num_samples.shape[0], device=device)
                                                  .repeat_interleave(num_samples))

                        ncc_loss_edge

                    ncc_loss_edge = torch.mean(ncc_loss_edge.view(num_plane_sample + 1, -1), dim=1)
                    # 5. Find best
                    points_to_tri = torch.arange(num_samples.shape[0]).to(device).repeat_interleave(num_samples)
                    ncc_loss_triangle = scatter_mean(ncc_loss_points, points_to_tri, dim=0)
                    if filter_points:
                        pass
                        # points_to_tri = points_to_tri[~mask_points_in_adj_patch]
                        # tri_idx = torch.arange(num_samples.shape[0]).to(device)
                        # missing_tri_idx = tri_idx[~torch.isin(tri_idx, torch.unique(points_to_tri))]
                        # ncc_loss_triangle[missing_tri_idx] = 0
                    ncc_loss_triangle = ncc_loss_triangle.view(num_plane_sample + 1, -1)
                    ncc_loss_patch = torch.mean(ncc_loss_triangle, dim=1)
                    ncc_loss_patch = ncc_loss_patch + ncc_loss_edge
                    id_best = torch.argmin(ncc_loss_patch, dim=-1)
                    best_abcd = samples_abcd[id_best]
                    # update
                    optimized_abcd_list[patch_id] = best_abcd
                    ncc_loss_optimized = ncc_loss_patch[id_best]
                    depth = torch.linalg.norm(samples_planes_intersection[id_best].unsqueeze(0), dim=-1)
                    angle = vectors_to_angles(best_abcd[0:3].unsqueeze(0))

                    # 6. Detect neighbour
                    num_sample_points = num_samples.view(num_plane_sample + 1, -1).sum(dim=-1)
                    num_sample_points = torch.cumsum(num_sample_points, dim=0)
                    if id_best == 0:
                        sample_points_c = sample_points_on_face[0:num_sample_points[id_best]]
                    else:
                        sample_points_c = sample_points_on_face[
                                          num_sample_points[id_best - 1]:num_sample_points[id_best]]
                    all_points_c = samples_planes_vertexes_pos[id_best].view(-1, 3)

                    def vis_(all_points_c_, sample_points_c_, patch_id_, cur_iter_):
                        img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                        img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                        shape = img11.shape[:2][::-1]
                        # vertex
                        p_2d1 = (intrinsic @ all_points_c_.T).T.cpu().numpy()
                        p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c_).T).T.cpu().numpy()
                        p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                        p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
                        p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                        p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                        # sample points
                        s_2d1 = (intrinsic @ sample_points_c_.T).T.cpu().numpy()
                        s_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_c_).T).T.cpu().numpy()
                        s_2d1 = s_2d1[:, :2] / s_2d1[:, 2:3]
                        s_2d2 = s_2d2[:, :2] / s_2d2[:, 2:3]
                        s_2d1 = np.around(s_2d1 * shape).astype(np.int64)
                        s_2d2 = np.around(s_2d2 * shape).astype(np.int64)

                        line_color = (0, 0, 255)
                        line_thickness = 2
                        vertex_point_color = (0, 0, 255)
                        sample_point_color = (0, 255, 0)
                        point_thickness = 3

                        for i_point in range(s_2d1.shape[0]):
                            cv2.circle(img11, s_2d1[i_point], 1, sample_point_color, 2)
                            cv2.circle(img21, s_2d2[i_point], 1, sample_point_color, 2)

                        for i_point in range(p_2d1.shape[0]):
                            cv2.circle(img11, p_2d1[i_point], 1, vertex_point_color, point_thickness)
                            cv2.circle(img21, p_2d2[i_point], 1, vertex_point_color, point_thickness)

                        cv2.imwrite(os.path.join(v_log_root, "patch_{}_iter_{}.jpg").format(patch_id_, cur_iter_),
                                    np.concatenate([
                                        np.concatenate([img11, img21], axis=1),
                                        np.concatenate([img12, img22], axis=1),
                                    ], axis=0)
                                    )

                    if cur_iter == 0:
                        all_points_c_init = samples_planes_vertexes_pos[0].view(-1, 3)
                        sample_points_c_init = sample_points_on_face[0:num_sample_points[0]]
                        vis_(all_points_c_init, sample_points_c_init, patch_id, cur_iter)

                    vis_(all_points_c, sample_points_c, patch_id, cur_iter + 1)

                    if ncc_loss_optimized <= 0.01 or cur_iter >= 5:
                        print(
                            "End: patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        break
                    else:
                        print("patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        cur_iter = cur_iter + 1

                # Update triangles
                final_edge_points = all_points_c[torch.tensor(edges_idx)]
                final_centroid = torch.mean(all_points_c, dim=0)
                final_triangles = torch.cat(
                    (final_edge_points, final_centroid[None, None, :].tile(num_vertex, 1, 1)), dim=1)

                final_triangles_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                    final_triangles).transpose(1,2)).transpose(1,2)[:,:,:3]
                collision_checker.add_triangles(final_triangles_c2)
            return optimized_abcd_list

        # Optimize plane hypothesis based on: 1) NCC; 2) Distance to the optimized vertices
        optimized_abcd_list = optimize_plane_hypothesis(poly_abcd_list, ncc, patches_list, graph.graph["dual_graph"],
                                                        c1_2_c2)

        save_plane(optimized_abcd_list, rays_c, patch_vertexes_id, file_path="output/optimized.ply")
        return
        # Determine the final location of vertices
        determine_vertices_location()

        exit()


def optimize_plane(v_data, v_log_root):
    v_img_database: list[Image] = v_data[0]
    v_graphs: np.ndarray[nx.Graph] = v_data[1]
    v_img_pairs: list[np.ndarray] = v_data[2]
    v_points_sfm = v_data[3]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    for id_img1, graph in enumerate(v_graphs):
        # 1. Prepare data

        # prepare some data
        id_src_imgs = (v_img_pairs[id_img1][:, 0]).astype(np.int64)
        ref_img = cv2.imread(v_img_database[id_img1].img_path, cv2.IMREAD_GRAYSCALE)
        src_imgs = [cv2.imread(v_img_database[int(item)].img_path, cv2.IMREAD_GRAYSCALE) for item in id_src_imgs]
        imgs = torch.from_numpy(np.concatenate(([ref_img], src_imgs), axis=0)).to(device).to(torch.float32) / 255.

        projection2 = np.stack([v_img_database[int(id_img)].projection for id_img in id_src_imgs], axis=0)
        intrinsic = v_img_database[id_img1].intrinsic
        # transformation store the transformation matrix from ref_img to src_imgs
        transformation = projection2 @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        transformation = torch.from_numpy(transformation).to(device).to(torch.float32)
        c1_2_c2 = torch.from_numpy(
            v_img_database[int(id_src_imgs[0])].extrinsic @ np.linalg.inv(v_img_database[id_img1].extrinsic)
        ).to(device).to(torch.float32)
        intrinsic = torch.from_numpy(intrinsic).to(device).to(torch.float32)

        # Remove the redundant face and edges in the graph
        # And build the dual graph in order to navigate between patches
        def fix_graph(v_graph, is_visualize=False):
            dual_graph = nx.Graph()

            id_original_to_current = {}
            for id_face, face in enumerate(graph.graph["faces"]):
                if not graph.graph["face_flags"][id_face]:
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
                                    face_center=graph.graph['face_center'][id_face],
                                    ray_c=graph.graph['ray_c'][id_face])

            for node in dual_graph.nodes():
                faces = dual_graph.nodes[node]["id_vertex"]
                for idx, id_start in enumerate(faces):
                    id_end = faces[(idx + 1) % len(faces)]
                    t = copy(graph.edges[(id_start, id_end)]["id_face"])
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

        determine_valid_edges(graph, imgs[0])
        fix_graph(graph)

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

        # fun defined by xdt
        def fit_plane_svd(points: torch.Tensor) -> torch.Tensor:
            centroid = torch.mean(points, axis=0)
            centered_points = points - centroid
            u, s, vh = torch.linalg.svd(centered_points)
            d = -torch.dot(vh[-1], centroid)
            abcd = torch.cat((vh[-1], torch.tensor([d]).to(device)))
            return abcd

        def initialize_patches(rays_c, ray_distances_c, v_vertex_id_per_face):
            initialized_vertices = rays_c*ray_distances_c[:,None]
            # abcd
            plane_parameters = []
            for vertex_id in v_vertex_id_per_face:
                pos_vertexes = initialized_vertices[vertex_id]
                assert (len(pos_vertexes) >= 3)
                # a) 3d vertexes of each patch -> fitting plane
                p_abcd = fit_plane_svd(pos_vertexes)
                plane_parameters.append(p_abcd)

            return torch.stack(plane_parameters, dim=0)

        initialized_planes = initialize_patches(rays_c, ray_distances_c, vertex_id_per_face) # (num_patch, 4)

        def vectors_to_angles(normal_vectors):
            normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
            x, y, z = normal_vectors.unbind(dim=1)
            phi = torch.atan2(y, x)
            theta = torch.acos(z)
            return torch.stack([phi, theta], dim=1)

        def angles_to_vectors(angles):
            phi, theta = angles.unbind(dim=1)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            return torch.stack([x, y, z], dim=1)

        def intersection_of_ray_and_plane(planes, rays_direction):
            # planes: n*4 ray: n*3
            n = planes[:, :3]
            d = planes[:, 3]
            denominator = torch.sum(n * rays_direction, dim=1)
            t = -d / torch.sum(n * rays_direction, dim=1)
            intersection_points = torch.unsqueeze(t, 1) * rays_direction
            valid_intersection = (denominator != 0) & (t >= 0)
            # n * _
            return valid_intersection, intersection_points

        def sample_depth_and_angle(depth, angle, num_sample=100):
            # sample depth
            sample_depth = sample_new_distance(depth)

            # sample angle
            sample_angle = torch.normal(angle[:,None,:].repeat(1, num_sample, 1),
                                        torch.full_like(angle, 2 * math.pi / 6)[:,None,:].repeat(1, num_sample, 1))
            # sample_angle = torch.clamp(sample_angle, min=-2 * math.pi, max=2 * math.pi)
            # add itself
            # sample_depth = torch.cat((depth[None, :], sample_depth), dim=0)
            # sample_angle = torch.cat((angle, sample_angle), dim=0)
            return sample_depth, sample_angle

        def sample_new_planes(v_original_parameters, v_centroid_rays_c, v_dual_graph):
            id_neighbour_patches = [list(v_dual_graph[id_node].keys()) for id_node in v_dual_graph.nodes]

            plane_angles = vectors_to_angles(v_original_parameters[:, :3])
            initial_centroids = intersection_of_ray_and_plane(v_original_parameters, v_centroid_rays_c)[1]
            init_depth = torch.linalg.norm(initial_centroids, dim=-1)

            sample_depth, sample_angle = sample_depth_and_angle(init_depth, plane_angles, 100)

            return

        def optimize_planes(initialized_planes, v_rays_c, v_centroid_rays_c, dual_graph):
            # 1. Sample new hypothesis from 1) propagation 2) random perturbation
            sample_new_planes(initialized_planes, v_centroid_rays_c, dual_graph) # [15, 100, 4]

            # 2. Compute validity from 1) ncc 2) edge fitness
            # compute ncc

            # compute edge fitness

            # 3. Finish iteration and visualize
            pass

        # 3. Propagate and sample plane hypothesises
        optimize_planes(initialized_planes, rays_c, centroid_rays_c, dual_graph)

        def prepare_edge_data(v_graph):
            determine_valid_edges(v_graph, imgs[0])
            # The id of the vertex for all edges
            point_id_per_edge = []
            # The id of a specific edge in the `point_id_per_edge`
            edge_id_dict = {}
            for edge in v_graph.edges():
                if not v_graph.edges[edge]["valid_flag"] or v_graph.edges[edge]["is_black"]:
                    continue
                point_id_per_edge.append(edge)
                # + and - to distinguish whether the edge is flip
                edge_id_dict[(edge[0], edge[1])] = len(point_id_per_edge)
                edge_id_dict[(edge[1], edge[0])] = -len(point_id_per_edge)
            return point_id_per_edge, edge_id_dict

        point_id_per_edge, edge_id_dict = prepare_edge_data(graph)
        point_id_per_edge = torch.tensor(point_id_per_edge, dtype=torch.long, device=device)

        def prepare_node_data(v_point_id_per_edge, v_device):
            unique_point_ids, counts = torch.unique(v_point_id_per_edge, return_counts=True)
            unique_point_ids = unique_point_ids[counts > 1]
            num_point = unique_point_ids.shape[0]
            max_counts = counts.max()

            node_data = -torch.ones((num_point, max_counts + 1), device=v_device, dtype=torch.long)
            for idx, id_point in enumerate(unique_point_ids):
                coords = torch.stack(torch.where(v_point_id_per_edge == id_point), dim=1)
                num_valid = coords.shape[0]
                coords[:, 1] = 1 - coords[:, 1]  # Index another point
                node_data[idx, 0] = id_point
                node_data[idx, 1:num_valid + 1] = v_point_id_per_edge[coords[:, 0], coords[:, 1]]
            return node_data

        # Index the valid nodes and the corresponding edges
        # The node_data is in the shape of [M,K],
        # where M is the number of valid node, K is the maximum edges (K=2) means every node has at most 1 edge
        # invalid edges are padded with -1
        node_data = prepare_node_data(point_id_per_edge, device)

        # Visualize the nodes that to be optimized
        if False:
            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]

            # Draw edges
            for node in node_data:
                id_start = int(node[0].cpu().item())
                for id_end in node[1:]:
                    id_end = int(id_end.cpu().item())
                    if id_end == -1:
                        break
                    start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                    end_pos = np.around((graph.nodes[id_end]["pos_2d"]) * shape).astype(np.int64)
                    cv2.line(img11, start_pos, end_pos, (0, 0, 255), 2)

            # Draw nodes
            for node in node_data:
                id_start = int(node[0].cpu().item())
                start_pos = np.around((graph.nodes[id_start]["pos_2d"]) * shape).astype(np.int64)
                cv2.circle(img11, start_pos, 2, (0, 255, 255), 2)

            cv2.imshow("1", img11)
            cv2.waitKey(0)

        def optimize_node(v_node_data, v_rays_c, v_ray_distances_c):
            id_start_points = v_node_data[:, 0]
            id_end_points = v_node_data[:, 1:]
            # M
            num_vertex = id_start_points.shape[0]
            # N
            num_max_edge_per_vertex = id_end_points.shape[1]

            optimized_distance = v_ray_distances_c.clone()
            # (M, N)
            end_distances = optimized_distance[id_end_points]
            # K
            num_sample = 100

            # (M, K, N)
            edge_valid_flag = (id_end_points == -1)[:, None, :].tile(1, num_sample, 1)

            cur_iter = 0
            last_loss = None
            num_tolerence = 100
            while True:
                # (M, K)
                start_distances_candidate = sample_new_distance(optimized_distance[id_start_points],
                                                                num_sample=num_sample).view(
                    num_vertex, num_sample, )
                # (M, K, 3)
                start_points_c = start_distances_candidate[:, :, None] * v_rays_c[id_start_points][:, None, :]
                # (M, K, N, 3)
                start_points_c = start_points_c[:, :, None, :].tile(1, 1, num_max_edge_per_vertex, 1)
                # (M, K, N, 3)
                end_points_c = (end_distances[:, :, None] * v_rays_c[id_end_points])[:, None, :, :].tile(1, 100, 1, 1)

                # (M, K, N, 2, 3)
                edges_points = torch.stack((start_points_c, end_points_c), dim=-2)

                ncc_loss, ncc_loss_mask, num_samples_per_edge = evaluate_candidates(edges_points.reshape(-1, 2, 3),
                                                              intrinsic, transformation[0], imgs[0], imgs[1],
                                                              v_sample_density=0.01)
                ncc_loss[~ncc_loss_mask] = torch.inf
                ncc_loss = ncc_loss.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                num_samples_per_edge = num_samples_per_edge.reshape(num_vertex, num_sample, num_max_edge_per_vertex)
                ncc_loss[edge_valid_flag] = 0
                num_samples_per_edge[edge_valid_flag] = 0
                # ncc_loss = ncc_loss * num_samples_per_edge/num_samples_per_edge.sum(dim=-1,keepdim=True)
                ncc_loss = ncc_loss.mean(dim=-1)
                id_best = ncc_loss.argmin(dim=1)

                optimized_distance[id_start_points] = torch.gather(start_distances_candidate, 1, id_best[:, None])[:, 0]
                end_distances = optimized_distance[id_end_points]

                total_loss = torch.gather(ncc_loss, 1, id_best[:, None]).mean()

                if last_loss is None:
                    last_loss = total_loss
                else:
                    delta = last_loss - total_loss

                    print("{:3d}:{:.4f}; Delta:{:.4f}".format(cur_iter,
                                                              total_loss.cpu().item(),
                                                              delta.cpu().item()
                                                              ))

                    if delta < 1e-5:
                        num_tolerence -= 1
                    else:
                        num_tolerence = 100
                        last_loss = total_loss
                    if num_tolerence <= 0:
                        break

                # Visualize
                if True:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                    img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                    start_points_c = optimized_distance[id_start_points][:, None] * v_rays_c[id_start_points]
                    start_points_c = start_points_c[:, None, :, ].tile(1, num_max_edge_per_vertex, 1)
                    end_points_c = optimized_distance[id_end_points][:, :, None] * v_rays_c[id_end_points]

                    all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                    all_points_c = all_points_c[~edge_valid_flag[:, 0]]
                    all_points_c = all_points_c.reshape(-1, 3)
                    p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                    p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                    p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                    p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                    p_2d1 = p_2d1.reshape(-1, 2, 2)
                    p_2d2 = p_2d2.reshape(-1, 2, 2)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                    p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                    line_color = (0, 0, 255)
                    line_thickness = 2
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_edge in range(p_2d1.shape[0]):
                        cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                        cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                    cv2.imwrite(os.path.join(v_log_root, "{}.jpg").format(cur_iter),
                                np.concatenate([
                                    np.concatenate([img11, img21], axis=1),
                                    np.concatenate([img12, img22], axis=1),
                                ], axis=0)
                                )

                cur_iter += 1
            return optimized_distance

        optmized_distance_cache = "output/optimized_distance.tar"
        if os.path.exists(optmized_distance_cache) and False:
            optimized_distance = torch.load(optmized_distance_cache).to(device)
        else:
            optimized_distance = optimize_node(node_data, rays_c, ray_distances_c)
            torch.save(optimized_distance, optmized_distance_cache)



        # fun defined by xdt
        def fit_plane_svd(points: torch.Tensor) -> torch.Tensor:
            centroid = torch.mean(points, axis=0)
            centered_points = points - centroid
            u, s, vh = torch.linalg.svd(centered_points)
            d = -torch.dot(vh[-1], centroid)
            abcd = torch.cat((vh[-1], torch.tensor([d]).to(device)))
            return abcd

        # TODO: using ray_c
        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def project_points_to_plane(points: torch.Tensor, abcd: torch.Tensor) -> torch.Tensor:
            normal, d = abcd[0:3], abcd[3]
            t = -(torch.matmul(points, normal) + d) / torch.dot(normal, normal)
            projected_points = points + t.unsqueeze(1) * normal
            return projected_points

        def ncc_matching_cost(img1, img2):
            mean1 = torch.mean(img1)
            mean2 = torch.mean(img2)
            std1 = torch.std(img1)
            std2 = torch.std(img2)
            norm_img1 = (img1 - mean1) / std1
            norm_img2 = (img2 - mean2) / std2
            ncc = torch.sum(norm_img1 * norm_img2) / (img1.numel() - 1)
            matching_cost = 1 - ncc
            return matching_cost

        def vectors_to_angles(normal_vectors):
            normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
            x, y, z = normal_vectors.unbind(dim=1)
            phi = torch.atan2(y, x)
            theta = torch.acos(z)
            return torch.stack([phi, theta], dim=1)

        def angles_to_vectors(angles):
            phi, theta = angles.unbind(dim=1)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            return torch.stack([x, y, z], dim=1)

        # 对应位置求交点
        def intersection_of_ray_and_plane(planes, rays_direction):
            # planes: n*4 ray: n*3
            n = planes[:, :3]
            d = planes[:, 3]
            denominator = torch.sum(n * rays_direction, dim=1)
            t = -d / torch.sum(n * rays_direction, dim=1)
            intersection_points = torch.unsqueeze(t, 1) * rays_direction
            valid_intersection = (denominator != 0) & (t >= 0)
            # n * _
            return valid_intersection, intersection_points

        def bilateral_ncc_(v_img1, v_img2):
            batch_size = v_img1.shape[0]
            window_size = v_img1.shape[1]
            device = v_img1.device
            # Add weights
            sigma_spatial = 3
            spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
            spatial_weights = torch.stack(torch.meshgrid(
                torch.arange(window_size, device=device, dtype=torch.float32),
                torch.arange(window_size, device=device, dtype=torch.float32),
                indexing="xy"
            ), dim=2) - window_size // 2
            spatial_weights = torch.linalg.norm(spatial_weights, dim=-1)
            spatial_weights = spatial_weights ** 2 * spatial_normalization_
            spatial_weights = torch.exp(-spatial_weights)
            spatial_weights = spatial_weights / spatial_weights.sum()

            v_img1 = (v_img1 * spatial_weights[None, :, :])
            v_img2 = (v_img2 * spatial_weights[None, :, :])

            norm_img1 = v_img1 - v_img1.mean(dim=[1, 2])[:, None, None]
            norm_img2 = v_img2 - v_img2.mean(dim=[1, 2])[:, None, None]

            ncc1 = torch.sum(norm_img1 * norm_img2, dim=[1, 2])
            ncc2 = torch.sqrt(torch.sum(norm_img1 ** 2, dim=[1, 2]) * torch.sum(norm_img2 ** 2, dim=[1, 2]))
            ncc2 = torch.clamp_min(ncc2, 1e-6)
            ncc = 1 - ncc1 / ncc2
            return ncc

        def bilateral_ncc(v_points_c, v_normal_c, v_intrinsic, v_transformation, v_img1, v_img2,
                          v_window_size=7):
            device = v_points_c.device
            num_points = v_points_c.shape[0]
            world_up_vector = torch.zeros_like(v_normal_c)
            world_up_vector[:, 2] = 1
            right_vector = normalize_tensor(torch.cross(world_up_vector, v_normal_c))
            up_vector = normalize_tensor(torch.cross(v_normal_c, right_vector))

            height, width = v_img1.shape[:2]
            resolution = 1 / min(height, width) * torch.linalg.norm(v_points_c, dim=-1)

            index = torch.arange(v_window_size, device=device) - v_window_size // 2
            index = torch.stack(torch.meshgrid(index, index, indexing="xy"), dim=2)
            window_points = \
                (index[:, :, 0].view(1, -1, 1) * right_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3) + \
                (index[:, :, 1].view(1, -1, 1) * up_vector[:, None, :] * resolution[:, None, None]).view(
                    num_points, v_window_size, v_window_size, 3)
            window_points = window_points + v_points_c[:, None, None, :]

            points_2d1 = (v_intrinsic @ window_points.reshape(-1, 3).T).T
            points_2d1 = points_2d1[:, :2] / points_2d1[:, 2:3]
            points_2d2 = (v_transformation @ to_homogeneous_tensor(window_points.reshape(-1, 3)).T).T
            points_2d2 = points_2d2[:, :2] / points_2d2[:, 2:3]

            sample_imgs1 = sample_img(v_img1[None, None, :], points_2d1[None, :, :])[0]
            sample_imgs2 = sample_img(v_img2[None, None, :], points_2d2[None, :, :])[0]

            sample_imgs1 = sample_imgs1.reshape(num_points, v_window_size, v_window_size)
            sample_imgs2 = sample_imgs2.reshape(num_points, v_window_size, v_window_size)

            ncc = bilateral_ncc_(sample_imgs1, sample_imgs2)

            # Visualize
            if False:
                points_2d1 = points_2d1.reshape(num_points, v_window_size * v_window_size, 2)
                points_2d2 = points_2d2.reshape(num_points, v_window_size * v_window_size, 2)

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                point_color = (0, 255, 255)
                point_thickness = 3

                for i in range(num_points):
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                    img11[p_2d1[i, :, 1], p_2d1[i, :, 0]] = point_color
                    img21[p_2d2[i, :, 1], p_2d2[i, :, 0]] = point_color

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            return ncc

        # img data pre
        intrinsic1 = v_img_database[id_img1].intrinsic
        intrinsic2 = v_img_database[1].intrinsic
        intrinsic1 = torch.from_numpy(intrinsic1).to(device).to(torch.float32)
        intrinsic2 = torch.from_numpy(intrinsic2).to(device).to(torch.float32)
        v_img1 = imgs[0]
        v_img2 = imgs[1]

        # optimized data pre
        optimized_points_pos = optimized_distance[:, None, None] * rays_c[:, None, :]
        patches_list = graph.graph["dual_graph"].nodes  # each patch = id_vertexes
        patch_vertexes_id = [patches_list[i]['id_vertex'] for i in range(len(patches_list))]

        # polygon -> edge (using the centroid)
        # TODO：1
        edge_list = []  # num_tri * 2
        for patch in patch_vertexes_id:
            num_vertex = len(patch)
            edge_id = [[patch[i], patch[(i + 1) % num_vertex]] for i in range(num_vertex)]
            edge_list.extend(edge_id)
        num_points_per_face = torch.tensor([len(item) for item in patch_vertexes_id], dtype=torch.long, device=device)
        edge_list = torch.tensor(edge_list).to(device).to(torch.long)
        edge = optimized_points_pos[edge_list].squeeze(dim=2)
        # patch_id of edge belongs to
        edges_indices = [torch.tensor([i] * len(patch_vertexes)) for i, patch_vertexes in enumerate(patch_vertexes_id)]
        edges_indices = torch.cat(edges_indices).to(device)

        # torch.cat((scatter_mean(edge[:, 0], edges_indices, dim=0)[:, None, :].repeat_interleave(num_points_per_face,

        def initialize_plane_hypothesis(v_rays_c, v_optimized_points_pos, v_patches_list):
            # 1. calculate ncc loss of each patch
            # for each patch do:
            # a) 3d vertexes of each patch -> fitting plane
            # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
            # c) sample points on projected polygon
            # d) project sample points to img1 and img2 and get pixel
            # e) calculate the ncc loss
            poly_abcd_list = []
            ncc_list = []
            # sample_points_list = []
            # projected_vertexes_list = []
            # sample_points_2d1 = []
            # sample_points_2d2 = []
            # centroid_list = []
            for patch_idx in range(len(v_patches_list)):
                id_vertexes = v_patches_list[patch_idx]['id_vertex']
                pos_vertexes = v_optimized_points_pos[id_vertexes].view(-1, 3)
                assert (len(pos_vertexes) >= 3)
                if patch_idx == 8:
                    print("debug")
                # a) 3d vertexes of each patch -> fitting plane
                p_abcd = fit_plane_svd(pos_vertexes)
                poly_abcd_list.append(p_abcd)

                # b) 3d vertexes -> 3d vertexes projected to fitting plane(projected polygon)
                projected_points = ray_line_intersection2(p_abcd,
                                                          torch.zeros_like(v_rays_c[id_vertexes]),
                                                          v_rays_c[id_vertexes])
                # projected_points = project_points_to_plane(pos_vertexes, p_abcd)
                # projected_vertexes_list.append(projected_points)

                # c) sample points on projected polygon
                # polygon -> triangles (using the centroid)
                num_vertex = projected_points.shape[0]
                edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                centroid = torch.mean(projected_points, dim=0)
                # centroid_list.append(centroid)
                edges = projected_points[torch.tensor(edges_idx)]
                triangles = torch.cat((edges, centroid.tile(num_vertex, 1)[:, None, :]), dim=1)

                # sample points on the fitting plane
                num_samples, sample_points_on_face = \
                    sample_triangles(100, triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :])
                # sample_points_list.append(sample_points_on_face)

                triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                               triangles[:, 1, :] - triangles[:, 2, :]))
                triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                ncc = bilateral_ncc(sample_points_on_face, triangle_normal,
                                    intrinsic1, transformation[0], v_img1, v_img2).mean()

                # # d) project sample points to img1 and img2
                # points_2d1 = (intrinsic1 @ sample_points_on_face.T).T
                # points_2d1 = points_2d1[:, :2] / points_2d1[:, 2:3]
                # points_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_on_face).T).T
                # points_2d2 = points_2d2[:, :2] / points_2d2[:, 2:3]
                # # sample_points_2d1.append(points_2d1)
                # # sample_points_2d2.append(points_2d2)
                #
                # # get pixel
                # sample_imgs1 = sample_img(v_img1[None, None, :], points_2d1[None, :, :])[0]
                # sample_imgs2 = sample_img(v_img2[None, None, :], points_2d2[None, :, :])[0]
                #
                # # e) calculate the ncc loss
                # ncc = ncc_matching_cost(sample_imgs1, sample_imgs2)
                ncc_list.append(ncc)

                # Visualize
                if False:
                    img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                    img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                    shape = img11.shape[:2][::-1]
                    p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
                    p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
                    point_color = (0, 255, 255)
                    point_thickness = 3
                    for i_point in range(p_2d1.shape[0]):
                        cv2.circle(img11, p_2d1[i_point], 1, point_color, point_thickness)
                        cv2.circle(img21, p_2d2[i_point], 1, point_color, point_thickness)

                    cv2.imshow("1", np.concatenate([
                        np.concatenate([img11, img21], axis=1),
                    ], axis=0))
                    cv2.waitKey()

            # 2. sorted patches according the ncc loss
            ncc_list = torch.stack(ncc_list).view(-1)
            poly_abcd_list = torch.stack(poly_abcd_list).view(-1, 4)
            # centroid_list = torch.stack(centroid_list).view(-1, 3)
            return poly_abcd_list, ncc_list

        # For each patch, initialize plane hypothesis
        poly_abcd_list, ncc = initialize_plane_hypothesis(rays_c, optimized_points_pos, patches_list)

        def save_plane(v_abcds, rays_c, patch_vertexes_id, file_path="output/init.ply"):
            vertices = []
            polygons = []
            acc_num_vertices = 0
            for id_patch in range(len(patch_vertexes_id)):
                intersection_points = ray_line_intersection2(v_abcds[id_patch],
                                                             torch.zeros_like(rays_c[patch_vertexes_id[id_patch]]),
                                                             rays_c[patch_vertexes_id[id_patch]])
                vertices.append(intersection_points)
                polygons.append(np.arange(intersection_points.shape[0]) + acc_num_vertices)
                acc_num_vertices += intersection_points.shape[0]
                pass
            vertices = torch.cat(vertices, dim=0).cpu().numpy()
            with open(file_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex {}\nproperty float x\nproperty float y\nproperty float z\n".format(
                    acc_num_vertices))
                f.write("element face {}\nproperty list uchar int vertex_index\n".format(len(polygons)))
                f.write("end_header\n")
                for ver in vertices:
                    f.write("{} {} {}\n".format(ver[0], ver[1], ver[2]))
                for polygon in polygons:
                    f.write("{}".format(len(polygon)))
                    for item in polygon:
                        f.write(" {}".format(item))
                    f.write("\n")
                pass
            print("Save done")
            return

        def visualize_plane(ncc, patch_vertexes_id):
            for idx in range(len(patch_vertexes_id)):
                id_vertices = patch_vertexes_id[idx]
                ncc_ = ncc[idx]
                print("{:2d}: ncc={:.2f} vertex_num={:2d}".format(idx, ncc_, len(id_vertices)))

                start_points_c = optimized_points_pos[id_vertices].view(-1, 3)
                end_points_c = torch.roll(start_points_c, -1, dims=[0])

                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                all_points_c = torch.stack((start_points_c, end_points_c), dim=-2)
                all_points_c = all_points_c.reshape(-1, 3)
                p_2d1 = ((intrinsic @ all_points_c.T).T).cpu().numpy()
                p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c).T).T.cpu().numpy()
                p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]

                p_2d1 = p_2d1.reshape(-1, 2, 2)
                p_2d2 = p_2d2.reshape(-1, 2, 2)

                shape = img11.shape[:2][::-1]
                p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                line_color = (0, 0, 255)
                line_thickness = 2
                point_color = (0, 255, 255)
                point_thickness = 3
                for i_edge in range(p_2d1.shape[0]):
                    cv2.line(img11, p_2d1[i_edge, 0], p_2d1[i_edge, 1], line_color, line_thickness)
                    cv2.line(img21, p_2d2[i_edge, 0], p_2d2[i_edge, 1], line_color, line_thickness)
                for i_point in range(p_2d1.shape[0]):
                    cv2.circle(img11, p_2d1[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img11, p_2d1[i_point, 1], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 0], 1, point_color, point_thickness)
                    cv2.circle(img21, p_2d2[i_point, 1], 1, point_color, point_thickness)

                cv2.imshow("1", np.concatenate([
                    np.concatenate([img11, img21], axis=1),
                    np.concatenate([img12, img22], axis=1),
                ], axis=0))
                cv2.waitKey()

        save_plane(poly_abcd_list, rays_c, patch_vertexes_id)

        if False:
            visualize_plane(ncc, patch_vertexes_id)

        def sample_depth_and_angle(depth, angle, num_sample=100):
            # sample depth
            sample_depth = torch.normal(depth.repeat(num_sample, 1))
            # sample angle
            sample_angle = torch.normal(angle.repeat(num_sample, 1),
                                        torch.full_like(angle, 2 * math.pi / 6).repeat(num_sample, 1))
            sample_angle = torch.clamp(sample_angle, min=-2 * math.pi, max=2 * math.pi)
            # add itself
            sample_depth = torch.cat((depth[None, :], sample_depth), dim=0)
            sample_angle = torch.cat((angle, sample_angle), dim=0)
            return sample_depth, sample_angle

        def compute_plane_abcd(patch_ray, ray_depth, normal):
            intersection = patch_ray * ray_depth
            d = -torch.sum(intersection * normal, dim=1, keepdim=True)
            plane_abcd = torch.cat([normal, d], dim=1)
            return intersection, plane_abcd

        def intersection_of_ray_and_all_plane(planes_abcd, vertexes_ray):
            # planes: n*4 ray: m*3
            n_planes = planes_abcd.size(0)
            m_rays = vertexes_ray.size(0)

            # 将平面参数和射线向量扩展为广播兼容的形状
            planes_abcd_expanded = planes_abcd.unsqueeze(1).expand(n_planes, m_rays, 4)
            vertexes_ray_expanded = vertexes_ray.unsqueeze(0).expand(n_planes, m_rays, 3)

            # 计算射线与平面的交点
            numerator = -planes_abcd_expanded[..., -1].unsqueeze(-1)  # 分子：-(D)
            denominator = torch.sum(planes_abcd_expanded[..., :3] * vertexes_ray_expanded, dim=-1).unsqueeze(
                -1)  # 分母：(A * v_ray.x + B * v_ray.y + C * v_ray.z)
            t = numerator / denominator  # t = -(D) / (A * v_ray.x + B * v_ray.y + C * v_ray.z)
            intersection_points = t * vertexes_ray_expanded  # p = p0 + t * v_ray，这里我们假设 p0 为原点 (0, 0, 0)
            # n*m*3
            return intersection_points

        def points_outside_polygon(sample_points, polygon_points, transformation_):
            sample_points = (intrinsic @ sample_points.T).T
            sample_points = (transformation_ @ to_homogeneous_tensor(sample_points).T).T
            sample_points = sample_points[:, :2] / sample_points[:, 2:3]
            polygon_points = (intrinsic @ polygon_points.T).T
            polygon_points = (transformation_ @ to_homogeneous_tensor(polygon_points).T).T
            polygon_points = polygon_points[:, :2] / polygon_points[:, 2:3]
            # 多边形的边
            poly_edges_start = polygon_points
            poly_edges_end = torch.roll(polygon_points, -1, dims=0)
            # 判断点是否在多边形外部
            cross_products = (sample_points[:, None, 0] - poly_edges_start[:, 0]) * (
                    poly_edges_end[:, 1] - poly_edges_start[:, 1]) - (
                                     sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                                     poly_edges_end[:, 0] - poly_edges_start[:, 0])
            y_diff_sign = (sample_points[:, None, 1] - poly_edges_start[:, 1]) * (
                    sample_points[:, None, 1] - poly_edges_end[:, 1])
            # 计算每个点与多边形的交点数
            intersection_count = torch.sum((cross_products > 0) & (y_diff_sign <= 0), dim=1)
            # 交点数为奇数的点在多边形内部，交点数为偶数的点在多边形外部
            outside_polygon_mask = intersection_count % 2 == 0
            return outside_polygon_mask

        def points_in_adj_patch(sample_points, v_polygon_points, transformation_):
            mask = torch.zeros(sample_points.shape[0], dtype=torch.bool).to(device)
            for polygon_points in v_polygon_points:
                mask = torch.logical_or(mask, ~points_outside_polygon(sample_points, polygon_points, transformation_))
            return mask

        def optimize_plane_hypothesis(poly_abcd_list, ncc, v_patches_list, v_dual_graph, v_c1_2_c2):
            # 1. get the intersection of the fitting plane and ray_c
            rays_c_valid_patch = np.stack([v_patches_list[i]['ray_c'] for i in range(len(v_patches_list))],axis=0)
            rays_c_valid_patch = torch.from_numpy(rays_c_valid_patch).to(device).to(torch.float32)
            # isValid, intersection = intersection_of_ray_and_plane(poly_abcd_list, rays_c_valid_patch)
            # assert (torch.sum(isValid) == isValid.shape[0])
            # depth = torch.linalg.norm(intersection, dim=-1)
            # angle = vectors_to_angles(poly_abcd_list[:, 0:3])

            sorted_values, sorted_indices = torch.sort(ncc, descending=False)

            collision_checker = Collision_checker()
            optimized_abcd_list = poly_abcd_list.clone()
            # 2. process the patches in the order of ncc loss
            for i in range(len(sorted_indices)):
                patch_id, init_ncc_loss = sorted_indices[i], sorted_values[i]
                isValid, intersection = intersection_of_ray_and_plane(optimized_abcd_list[patch_id].unsqueeze(0),
                                                                      rays_c_valid_patch[patch_id].unsqueeze(0))
                assert (torch.sum(isValid) == isValid.shape[0])
                depth = torch.linalg.norm(intersection, dim=-1)
                angle = vectors_to_angles(optimized_abcd_list[patch_id, 0:3].unsqueeze(0))

                vertexes_id = torch.tensor(patch_vertexes_id[patch_id]).to(device).to(torch.long)
                vertexes_pos = optimized_points_pos[vertexes_id]
                vertexes_ray = rays_c[vertexes_id]
                patch_ray = torch.tensor(v_patches_list[int(patch_id)]['ray_c']).to(device).to(torch.float32)

                # optimize the patch
                cur_iter = 0
                ncc_loss_optimized = init_ncc_loss
                id_best = 0  # init patch will be place at 0 index
                num_plane_sample = 100
                while True:
                    # 1. Sample candidate plane
                    samples_depth, samples_angle = sample_depth_and_angle(depth, angle, num_plane_sample)
                    samples_abc = angles_to_vectors(samples_angle)
                    samples_planes_intersection, samples_abcd = compute_plane_abcd(patch_ray, samples_depth,
                                                                                   samples_abc)

                    # num_samples_plane * num_vertexes_cur_patch * 3
                    samples_planes_vertexes_pos = intersection_of_ray_and_all_plane(samples_abcd, vertexes_ray)

                    # 2. Get intersection points
                    # candidate_triangle_points (N, 3, 3)
                    num_vertex = vertexes_id.shape[0]
                    edges_idx = [[i, (i + 1) % num_vertex] for i in range(num_vertex)]
                    centroid = torch.mean(samples_planes_vertexes_pos, dim=1)
                    # centroid_list.append(centroid)
                    edges = samples_planes_vertexes_pos[:, torch.tensor(edges_idx)]
                    triangles = torch.cat((edges, centroid[:, None, None, :].tile(1, num_vertex, 1, 1)), dim=2)
                    triangles = triangles.view(-1, 3, 3)  # (101*num_tri,3,3)

                    # 3. calculate loss (point ncc loss and line loss)
                    # sample points on the fitting plane
                    # get each sample point's ncc loss, then aggregate to each patch
                    num_samples, sample_points_on_face = sample_triangles(100, triangles[:, 0, :],
                                                                          triangles[:, 1, :], triangles[:, 2, :])

                    triangle_normal = normalize_tensor(torch.cross(triangles[:, 0, :] - triangles[:, 1, :],
                                                                   triangles[:, 1, :] - triangles[:, 2, :]))
                    triangle_normal = triangle_normal.repeat_interleave(num_samples, dim=0)

                    ncc_loss_points = bilateral_ncc(sample_points_on_face, triangle_normal, intrinsic,
                                                    transformation[0], v_img1, v_img2)

                    # get the edge point of each sample patch plane
                    ncc_loss_edge, ncc_loss_mask = evaluate_candidates(edges.reshape(-1, 2, 3),
                                                                       intrinsic, transformation[0], v_img1, v_img2)

                    # filter the points int adj patch (in img2 candidate)
                    filter_points = True
                    if filter_points:
                        sample_points_on_face_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            sample_points_on_face).transpose(0, 1)).transpose(0, 1)[:, :3]
                        point_mask = collision_checker.check_ray(
                            torch.zeros_like(sample_points_on_face_c2),
                            sample_points_on_face_c2)
                        edges_shape = edges.shape
                        edges_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                            edges.reshape(-1, 3)).transpose(0, 1)).transpose(0, 1)[:, :3]
                        edge_mask = collision_checker.check_ray(
                            torch.zeros_like(edges_c2),
                            edges_c2).reshape(*edges_shape[:-1])
                        edge_mask = torch.all(edge_mask, dim=-1)

                        # adj_patch_id_list = [edge[1] for edge in list(v_dual_graph.edges(int(patch_id)))]
                        # adj_patch_vertex_pos = []
                        # for adj_patch_id in adj_patch_id_list:
                        #     vertexes_id_adj = torch.tensor(patch_vertexes_id[adj_patch_id]).to(device).to(torch.long)
                        #     vertexes_ray_adj = rays_c[vertexes_id_adj]
                        #     valid_mask, vertexes_pos_adj = intersection_of_ray_and_plane(
                        #         poly_abcd_list[patch_id].tile(vertexes_ray_adj.shape[0], 1),
                        #         vertexes_ray_adj)
                        #     adj_patch_vertex_pos.append(vertexes_pos_adj)
                        # mask_points_in_adj_patch = points_in_adj_patch(sample_points_on_face,
                        #                                                adj_patch_vertex_pos,
                        #                                                transformation[0])
                        # ncc_loss_points = ncc_loss_points[~mask_points_in_adj_patch]
                        ncc_loss_points = ncc_loss_points[~point_mask]
                        num_samples = scatter_add((~point_mask).to(torch.int64),
                                                  torch.arange(num_samples.shape[0], device=device)
                                                  .repeat_interleave(num_samples))

                        ncc_loss_edge

                    ncc_loss_edge = torch.mean(ncc_loss_edge.view(num_plane_sample + 1, -1), dim=1)
                    # 5. Find best
                    points_to_tri = torch.arange(num_samples.shape[0]).to(device).repeat_interleave(num_samples)
                    ncc_loss_triangle = scatter_mean(ncc_loss_points, points_to_tri, dim=0)
                    if filter_points:
                        pass
                        # points_to_tri = points_to_tri[~mask_points_in_adj_patch]
                        # tri_idx = torch.arange(num_samples.shape[0]).to(device)
                        # missing_tri_idx = tri_idx[~torch.isin(tri_idx, torch.unique(points_to_tri))]
                        # ncc_loss_triangle[missing_tri_idx] = 0
                    ncc_loss_triangle = ncc_loss_triangle.view(num_plane_sample + 1, -1)
                    ncc_loss_patch = torch.mean(ncc_loss_triangle, dim=1)
                    ncc_loss_patch = ncc_loss_patch + ncc_loss_edge
                    id_best = torch.argmin(ncc_loss_patch, dim=-1)
                    best_abcd = samples_abcd[id_best]
                    # update
                    optimized_abcd_list[patch_id] = best_abcd
                    ncc_loss_optimized = ncc_loss_patch[id_best]
                    depth = torch.linalg.norm(samples_planes_intersection[id_best].unsqueeze(0), dim=-1)
                    angle = vectors_to_angles(best_abcd[0:3].unsqueeze(0))

                    # 6. Detect neighbour
                    num_sample_points = num_samples.view(num_plane_sample + 1, -1).sum(dim=-1)
                    num_sample_points = torch.cumsum(num_sample_points, dim=0)
                    if id_best == 0:
                        sample_points_c = sample_points_on_face[0:num_sample_points[id_best]]
                    else:
                        sample_points_c = sample_points_on_face[
                                          num_sample_points[id_best - 1]:num_sample_points[id_best]]
                    all_points_c = samples_planes_vertexes_pos[id_best].view(-1, 3)

                    def vis_(all_points_c_, sample_points_c_, patch_id_, cur_iter_):
                        img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img12 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                        img22 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)

                        shape = img11.shape[:2][::-1]
                        # vertex
                        p_2d1 = (intrinsic @ all_points_c_.T).T.cpu().numpy()
                        p_2d2 = (transformation[0] @ to_homogeneous_tensor(all_points_c_).T).T.cpu().numpy()
                        p_2d1 = p_2d1[:, :2] / p_2d1[:, 2:3]
                        p_2d2 = p_2d2[:, :2] / p_2d2[:, 2:3]
                        p_2d1 = np.around(p_2d1 * shape).astype(np.int64)
                        p_2d2 = np.around(p_2d2 * shape).astype(np.int64)
                        # sample points
                        s_2d1 = (intrinsic @ sample_points_c_.T).T.cpu().numpy()
                        s_2d2 = (transformation[0] @ to_homogeneous_tensor(sample_points_c_).T).T.cpu().numpy()
                        s_2d1 = s_2d1[:, :2] / s_2d1[:, 2:3]
                        s_2d2 = s_2d2[:, :2] / s_2d2[:, 2:3]
                        s_2d1 = np.around(s_2d1 * shape).astype(np.int64)
                        s_2d2 = np.around(s_2d2 * shape).astype(np.int64)

                        line_color = (0, 0, 255)
                        line_thickness = 2
                        vertex_point_color = (0, 0, 255)
                        sample_point_color = (0, 255, 0)
                        point_thickness = 3

                        for i_point in range(s_2d1.shape[0]):
                            cv2.circle(img11, s_2d1[i_point], 1, sample_point_color, 2)
                            cv2.circle(img21, s_2d2[i_point], 1, sample_point_color, 2)

                        for i_point in range(p_2d1.shape[0]):
                            cv2.circle(img11, p_2d1[i_point], 1, vertex_point_color, point_thickness)
                            cv2.circle(img21, p_2d2[i_point], 1, vertex_point_color, point_thickness)

                        cv2.imwrite(os.path.join(v_log_root, "patch_{}_iter_{}.jpg").format(patch_id_, cur_iter_),
                                    np.concatenate([
                                        np.concatenate([img11, img21], axis=1),
                                        np.concatenate([img12, img22], axis=1),
                                    ], axis=0)
                                    )

                    if cur_iter == 0:
                        all_points_c_init = samples_planes_vertexes_pos[0].view(-1, 3)
                        sample_points_c_init = sample_points_on_face[0:num_sample_points[0]]
                        vis_(all_points_c_init, sample_points_c_init, patch_id, cur_iter)

                    vis_(all_points_c, sample_points_c, patch_id, cur_iter + 1)

                    if ncc_loss_optimized <= 0.01 or cur_iter >= 5:
                        print(
                            "End: patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        break
                    else:
                        print("patch{:2d} iter={:2d}: ncc={:.2f}".format(patch_id, cur_iter + 1, ncc_loss_optimized))
                        cur_iter = cur_iter + 1

                # Update triangles
                final_edge_points = all_points_c[torch.tensor(edges_idx)]
                final_centroid = torch.mean(all_points_c, dim=0)
                final_triangles = torch.cat(
                    (final_edge_points, final_centroid[None, None, :].tile(num_vertex, 1, 1)), dim=1)

                final_triangles_c2 = (v_c1_2_c2 @ to_homogeneous_tensor(
                    final_triangles).transpose(1,2)).transpose(1,2)[:,:,:3]
                collision_checker.add_triangles(final_triangles_c2)
            return optimized_abcd_list

        # Optimize plane hypothesis based on: 1) NCC; 2) Distance to the optimized vertices
        optimized_abcd_list = optimize_plane_hypothesis(poly_abcd_list, ncc, patches_list, graph.graph["dual_graph"],
                                                        c1_2_c2)

        save_plane(optimized_abcd_list, rays_c, patch_vertexes_id, file_path="output/optimized.ply")
        return
        # Determine the final location of vertices
        determine_vertices_location()

        exit()


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


def prepare_dataset_and_model(v_colmap_dir, v_viz_face, v_bounds):
    print("Start to prepare dataset")
    print("1. Read imgs")

    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name):
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
    if os.path.exists(graph_cache_name):
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
    if os.path.exists(points_cache_name):
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


@hydra.main(config_name="phase6.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    # data = img_database, graphs, camera_pair_data, points_from_sfm
    data = prepare_dataset_and_model(
        v_cfg["dataset"]["colmap_dir"],
        v_cfg["dataset"]["id_viz_face"],
        v_cfg["dataset"]["scene_boundary"],
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # optimize(data, v_cfg["trainer"]["output"])
    optimize_plane(data, v_cfg["trainer"]["output"])

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
