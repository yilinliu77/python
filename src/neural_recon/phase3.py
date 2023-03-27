import itertools
import sys, os
import time

from torch.distributions.utils import _standard_normal

sys.path.append("thirdparty/sdf_computer/build/")
import pysdf

import math

import faiss
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import networkx as nx

import mcubes
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from tqdm import tqdm
import platform
import shutil

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img
from shared.common_utils import debug_imgs, to_homogeneous, save_line_cloud, to_homogeneous_vector, normalize_tensor, \
    to_homogeneous_mat_tensor, to_homogeneous_tensor, normalized_torch_img_to_numpy, padding, \
    vector_to_sphere_coordinate, sphere_coordinate_to_vector, caculate_align_mat, normalize_vector, \
    pad_and_enlarge_along_y, refresh_timer, get_line_mesh

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
from src.neural_recon.phase1 import NGPModel


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_length, v_id_target):
        super(Dummy_dataset, self).__init__()
        self.length = v_length
        self.id_target = v_id_target
        pass

    def __getitem__(self, index):
        if self.id_target != -1:
            return torch.tensor(self.id_target, dtype=torch.long)
        else:
            return torch.tensor(index, dtype=torch.long)

    def __len__(self):
        if self.id_target != -1:
            return 1
        else:
            return self.length


# Normal loss and Similarity loss using calculated up vector and gaussian distribution
class LModel14(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel14, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        v_up = []
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)
        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        # self.gaussian_std = nn.Parameter(torch.log(self.gaussian[:, 1]), requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root
        self.loss_weights = v_weights
        self.img_method = v_img_method

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0,0,1),),device=device,dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long),
                                     2, 1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long),
                                   2, 1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up_c[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_polygon(self, edge_points, v_dis, id_epoch):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point

        # Sample points on edges
        length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        num_per_edge_m2 = 10
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 500)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)

        # Sample points within triangle
        num_per_half_m2 = 100
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir)+1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 1000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1]+1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3]+1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3]+1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        # num_repeat = 100
        # id_end = num_edge_points.cumsum(0)
        # id_start = num_edge_points.cumsum(0)
        # id_start = id_start.roll(1)
        # id_start[0]=0
        # loss = []
        # for id_repeat in range(num_repeat):
        #     id_edge = 0 + id_repeat * 9
        #     img1 = torch.cat((
        #         sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs1[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     img2 = torch.cat((
        #         sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
        #         sample_imgs2[
        #         num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
        #                                                                                            :id_edge + 1].sum()],
        #     ), dim=0)
        #     loss.append((img1-img2).mean())

        # Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        # Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        is_debug=False
        if is_debug:
            id_edge_base = 0
            for id_repeat in range(100):
                id_edge = id_edge_base + id_repeat * 9
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_repeat, id_epoch)),
                            np.concatenate((img1, img2), axis=0))

                img1 = torch.cat((
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs1[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                img2 = torch.cat((
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                print("{}_{:.3f}_{:.1f}_{:.1f}".format(id_repeat, (img1-img2).mean(),
                      v_dis.reshape(-1,4)[id_edge,0].item(),
                      v_dis.reshape(-1,4)[id_edge,1].item(),)
                      )
            pass

        # 9: Viz
        if False and self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"output/img_field_test/imgs_log/2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(r"output/img_field_test/imgs_log/3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()

        return similarity_loss, normal_loss, normalization_loss

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = 1 if is_log else 100
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        repeat_point_index = point_index * num_repeat
        ray_c = self.ray_c[repeat_point_index].reshape((-1, 4, 3))

        # Target
        target_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,0:1]
        eps = _standard_normal(target_dis.shape, target_dis.dtype, target_dis.device)
        target_dis = (target_dis + self.gaussian_std[repeat_point_index].reshape(-1,4)[:,0:1] * eps) * self.seg_distance_normalizer
        # Reference
        ref_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,1:].detach()
        seg_distance = torch.cat((target_dis,ref_dis), dim=1)
        point_pos_c = ray_c * seg_distance[:, :, None]
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, seg_distance, v_id_epoch)

        losses = torch.stack(losses).reshape((-1,3))

        id_target = 0
        total_loss = losses[id_target, 0] * self.loss_weights[0] + \
                     losses[id_target, 1] * self.loss_weights[1] + \
                     losses[id_target, 2] * self.loss_weights[2]
        # total_loss = losses[id_target, 0]
        time_profile[9], timer = refresh_timer(timer)

        if is_log:
            # Log 2D
            img1_base = self.rgb1.copy()
            shape = img1_base.shape[:2][::-1]
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            img1_base = cv2.polylines(img1_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, losses[id_target]

    def debug_save(self, v_index):
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_normal = normalize_tensor(torch.cross(total_edge_points[:, 1] - total_edge_points[:, 0], \
                            total_edge_points[:, 3] - total_edge_points[:, 1]))
            sign_flag = torch.sum(total_normal * torch.tensor(((0, 0, 1),),
                                                              device=point_pos_c.device, dtype=torch.float32),dim=1) > 0
            total_normal[sign_flag] = -total_normal[sign_flag]
            total_up = torch.cross(
                total_normal,
                total_edge_points[:, 1] - total_edge_points[:, 0]
            )

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,:3]\
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index)), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1,4))[:,0].mean()

    def len(self):
        return len(self.graph1.graph["faces"])


# Normal loss and Similarity loss using calculated up vector and gaussian distribution
class LModel15(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel15, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)

        # Batch index
        v_up = []
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        self.gaussian = [[] for _ in self.graph1.nodes()]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            distances = []
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)
                distances.append(self.graph1.nodes[id_start]["distance"])
                v_up.append(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch])
            min_dis = min(distances)
            max_dis = max(distances)
            for id_node in face_ids:
                self.gaussian[id_node].append(min_dis)
                self.gaussian[id_node].append(max_dis)
        for id_node in range(len(self.graph1.nodes)):
            cur_dis = self.graph1.nodes[id_node]["distance"]
            scale = max(abs(max(self.gaussian[id_node]) - cur_dis), abs(min(self.gaussian[id_node]) - cur_dis))
            self.gaussian[id_node] = (cur_dis, scale)
            pass
        self.gaussian = torch.tensor(self.gaussian, dtype=torch.float32) / self.seg_distance_normalizer
        self.gaussian_mean = nn.Parameter(self.gaussian[:, 0], requires_grad=True)
        # self.gaussian_std = nn.Parameter(torch.log(self.gaussian[:, 1]), requires_grad=True)
        self.gaussian_std = nn.Parameter(self.gaussian[:, 1], requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.v_up = torch.tensor(np.stack(v_up, axis=0)).float()
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root
        self.loss_weights = v_weights
        self.img_method = v_img_method

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0,0,1),),device=device,dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long),
                                     2, 1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long),
                                   2, 1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up_c[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:,0])) # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.5).to(device)  # m
        half_window_size_step = 0.05

        # Compute interpolated point
        # Num edges: M
        # Number of sample points for each edge (M edges); The total number of sample points is num_horizontal * num_vertical
        num_horizontal = torch.clamp((half_window_size_meter_horizontal // half_window_size_step).to(torch.long), 2,
                                     1000)  # (M,)
        num_vertical = torch.clamp((half_window_size_meter_vertical // half_window_size_step).to(torch.long), 2,
                                   1000)  # (9,); fixed
        num_coordinates_per_edge = num_horizontal * num_vertical

        begin_idxes = num_horizontal.cumsum(dim=0)
        total_num_x_coords = begin_idxes[-1]
        begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
        begin_idxes[0] = 0  # (M,)
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - \
             begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal) * \
             half_window_size_meter_horizontal.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        interpolated_coordinates_camera = \
            cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            cur_up[:, :].repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)

        cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        next_up = torch.cross(edge_up_c[:, 1], next_dir)
        next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        prev_dir = edge_points[:, 0] - edge_points[:, 2]
        prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss1 + normal_loss2) / 2

        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_polygon(self, edge_points, v_dis, id_epoch, v_num_repeat):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point

        # Sample points on edges
        length = torch.linalg.norm(cur_dir+1e-6, dim=1)
        num_per_edge_m2 = 10
        num_edge_points = torch.clamp((length * num_per_edge_m2).to(torch.long), 1, 500)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)

        # Sample points within triangle
        num_per_half_m2 = 100
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir)+1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 1000)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)

        _t1 = torch.sqrt(sample_points1[:, 0:1]+1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)

        coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3]+1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[4], timer = refresh_timer(timer)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3]+1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2, reduce=False)

        total_edge_points = num_edge_points.sum()
        edge_index = torch.arange(num_edge_points.shape[0], device=device).repeat_interleave(num_edge_points)
        face_index = torch.arange(num_polygon_points.shape[0], device=device).repeat_interleave(num_polygon_points)
        edge_similarity = similarity_loss[:total_edge_points]
        face_similarity = similarity_loss[total_edge_points:]
        from torch_scatter import scatter_add
        edge_similarity = scatter_add(edge_similarity,edge_index,dim=0) / num_edge_points
        face_similarity = scatter_add(face_similarity,face_index,dim=0) / num_polygon_points

        total_similarity = torch.mean(edge_similarity+face_similarity, dim=1).reshape(-1, v_num_repeat)

        num_repeat = 100
        id_end = num_edge_points.cumsum(0)
        id_start = num_edge_points.cumsum(0)
        id_start = id_start.roll(1)
        id_start[0]=0
        loss = []
        for id_repeat in range(num_repeat):
            id_edge = 0 + id_repeat * 9
            img1 = torch.cat((
                sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                sample_imgs1[
                num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                   :id_edge + 1].sum()],
            ), dim=0)
            img2 = torch.cat((
                sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                sample_imgs2[
                num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[
                                                                                                   :id_edge + 1].sum()],
            ), dim=0)
            loss.append((img1-img2).mean())

        # Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]

        # Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        is_debug=False
        if is_debug:
            id_edge_base = 0
            for id_repeat in range(100):
                id_edge = id_edge_base + id_repeat * 9
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge+1].sum()],
                    roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_repeat, id_epoch)),
                            np.concatenate((img1, img2), axis=0))

                img1 = torch.cat((
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs1[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                img2 = torch.cat((
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0)
                print("{}_{:.3f}_{:.1f}_{:.1f}".format(id_repeat, (img1-img2).mean(),
                      v_dis.reshape(-1,4)[id_edge,0].item(),
                      v_dis.reshape(-1,4)[id_edge,1].item(),)
                      )
            pass

        # 9: Viz
        if False and self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # 2D RoI
            line_img1 = line_img1_base.copy()
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            line_img2 = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)

            cv2.imwrite(r"output/img_field_test/imgs_log/2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((line_img1, line_img2), axis=0))
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                cv2.waitKey()

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_edge = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d2_numpy = roi_coor_2d_img2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()

                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(r"output/img_field_test/imgs_log/3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((line_img1, line_img2), axis=0))
                    cv2.waitKey()

        return similarity_loss, normal_loss, normalization_loss

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0: Unpack data
        num_repeat = 100
        # num_repeat = 1 if is_log else 100
        point_index = list(itertools.chain(*[self.edge_point_index[item] * num_repeat for item in v_index]))
        point_index = np.asarray(point_index).reshape(-1,4)
        repeat_point_index = np.tile(point_index[:,None,:], [1, num_repeat, 1],).flatten().tolist()
        ray_c = self.ray_c[repeat_point_index].reshape((-1, 4, 3))

        # Target
        target_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,0:1]
        eps = _standard_normal(target_dis.shape, target_dis.dtype, target_dis.device)
        target_dis = (target_dis + self.gaussian_std[repeat_point_index].reshape(-1,4)[:,0:1] * eps) * self.seg_distance_normalizer
        # Reference
        ref_dis = self.gaussian_mean[repeat_point_index].reshape(-1,4)[:,1:].detach()
        seg_distance = torch.cat((target_dis,ref_dis), dim=1)
        point_pos_c = ray_c * seg_distance[:, :, None]
        edge_points = point_pos_c

        losses = []
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        losses += self.sample_points_based_on_polygon(edge_points, seg_distance, v_id_epoch)

        losses = torch.stack(losses).reshape((-1,3))

        id_target = 0
        total_loss = losses[id_target, 0] * self.loss_weights[0] + \
                     losses[id_target, 1] * self.loss_weights[1] + \
                     losses[id_target, 2] * self.loss_weights[2]
        # total_loss = losses[id_target, 0]
        time_profile[9], timer = refresh_timer(timer)

        if is_log:
            # Log 2D
            img1_base = self.rgb1.copy()
            shape = img1_base.shape[:2][::-1]
            roi_c = edge_points[:, 0]
            roi_2d1 = (self.intrinsic1 @ roi_c.T).T
            roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            img1_base = cv2.polylines(img1_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{:05d}.jpg".format(v_id_epoch),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, losses[id_target]

    def debug_save(self, v_index):
        seg_distance = self.gaussian_mean * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_normal = normalize_tensor(torch.cross(total_edge_points[:, 1] - total_edge_points[:, 0], \
                            total_edge_points[:, 3] - total_edge_points[:, 1]))
            sign_flag = torch.sum(total_normal * torch.tensor(((0, 0, 1),),
                                                              device=point_pos_c.device, dtype=torch.float32),dim=1) > 0
            total_normal[sign_flag] = -total_normal[sign_flag]
            total_up = torch.cross(
                total_normal,
                total_edge_points[:, 1] - total_edge_points[:, 0]
            )

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                               cylinder_height=0.0005, cone_height=0.0005,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow
            colors = np.zeros_like(np.asarray(arrows.vertices))
            colors[:,0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # Visualize target patch
        arrows = get_arrow(self.edge_point_index[self.id_viz_patch])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index)), arrows)
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,:3]\
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index)), start_point_w, edge_index)

        # Visualize whole patch
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(r"output/img_field_test/imgs_log/total_{}_line.obj".format(v_index), start_point_w, edge_index)
        pass

        return self.gaussian_std[self.edge_point_index[self.id_viz_patch]].reshape((-1,4))[:,0].mean()

    def len(self):
        return len(self.graph1.graph["faces"])

class Phase3(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Phase3, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        if not os.path.exists(self.hydra_conf["trainer"]["output"]):
            os.makedirs(self.hydra_conf["trainer"]["output"])

        self.data = v_data
        # self.model = LModel0(self.data)
        self.model = LModel15(self.data, self.hydra_conf["trainer"]["loss_weight"],
                              self.hydra_conf["trainer"]["img_model"], self.hydra_conf["trainer"]["output"])
        # self.model = LModel31(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])
        # self.model = LModel12(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])

    def train_dataloader(self):
        self.train_dataset = Dummy_dataset(self.model.len(), self.hydra_conf["dataset"]["only_train_target"])
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        id_target = self.hydra_conf["dataset"]["only_train_target"]
        self.valid_dataset = Dummy_dataset(self.model.len() if id_target == -1 else 1, id_target)
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        total_loss, losses = self.model(batch, self.current_epoch, False)

        self.log("Training_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)
        self.log("Training_Normal_Loss", losses[0].detach(), prog_bar=True, logger=True, on_step=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=1)
        self.log("Training_Similarity_Loss", losses[1].detach(), prog_bar=True, logger=True, on_step=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=1)
        self.log("Training_Normalization_Loss", losses[2].detach(), prog_bar=True, logger=True, on_step=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=1)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, losses = self.model(batch, self.current_epoch, True)

        self.log("Validation_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)

        return total_loss

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        mean_std = self.model.debug_save(self.current_epoch)
        self.log("Validation_Std", mean_std.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)

        if self.trainer.sanity_checking:
            return

    # def on_after_backward(self) -> None:
    #     """
    #     Skipping updates in case of unstable gradients
    #     https://github.com/Lightning-AI/lightning/issues/4956
    #     """
    #     valid_gradients = True
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #             if not valid_gradients:
    #                 break
    #     if not valid_gradients:
    #         print(f'detected inf or nan values in gradients. not updating model parameters')
    #         self.zero_grad()

def prepare_dataset_and_model(v_colmap_dir, v_only_train_target):
    print("Start to prepare dataset")
    print("1. Read imgs")
    bound_min = np.array((-40, -40, -5))
    bound_max = np.array((130, 150, 60))
    bounds_center = (bound_min + bound_max) / 2
    bounds_size = (bound_max - bound_min).max()

    imgs, points_3d = read_dataset(v_colmap_dir,
                                   [bound_min,
                                    bound_max]
                                   )
    print("2. Build graph")
    if True:
        graphs = []
        for i in range(1, 3):
            data = [item for item in open(
                os.path.join(v_colmap_dir, "wireframes/wireframe{}.obj".format(
                    i))).readlines()]
            vertices = [item.strip().split(" ")[1:-1] for item in data if item[0] == "v"]
            vertices = np.asarray(vertices).astype(np.float32) / np.array((1499, 999), dtype=np.float32)
            faces = [item.strip().split(" ")[1:] for item in data if item[0] == "f"]
            graph = nx.Graph()
            graph.add_nodes_from([(idx, {"pos_2d": item}) for idx, item in enumerate(vertices)])
            new_faces = []  # equal faces - 1 because of the obj format
            for face in faces:
                face = (np.asarray(face).astype(np.int32) - 1).tolist()
                new_faces.append(face)
                face = [(face[idx], face[idx + 1]) for idx in range(len(face) - 1)] + [(face[-1], face[0])]
                graph.add_edges_from(face)
            graph.graph["faces"] = new_faces
            print("Read {}/{} vertices".format(vertices.shape[0], len(graph.nodes)))
            print("Read {} faces".format(len(faces)))
            graphs.append(graph)
        pass

    preserved_points = []
    for point in tqdm(points_3d):
        for track in point.tracks:
            if track[0] in [1, 2]:
                preserved_points.append(point)
    points_from_sfm = np.stack([item.pos for item in preserved_points])

    img1 = imgs[1]
    img2 = imgs[2]
    graph1 = graphs[0]
    graph2 = graphs[1]
    print("3. Project points on img1")

    def project_points(points_3d_pos):
        projected_points = np.transpose(img1.projection @ np.transpose(np.insert(points_3d_pos, 3, 1, axis=1)))
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        projected_points_mask = np.logical_and(projected_points[:, 0] > 0, projected_points[:, 1] > 0)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 0] < 1)
        projected_points_mask = np.logical_and(projected_points_mask, projected_points[:, 1] < 1)
        points_3d_pos = points_3d_pos[projected_points_mask]
        projected_points = projected_points[projected_points_mask]
        return points_3d_pos, projected_points

    points_from_sfm, points_from_sfm_2d = project_points(points_from_sfm)

    rgb1 = cv2.imread(img1.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    rgb2 = cv2.imread(img2.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    shape = rgb1.shape[:2][::-1]

    id_patch = v_only_train_target
    print("Will visualize patch {} during the training".format(id_patch))
    print("4. Draw input on img1")

    def draw_initial():
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("1", 1600, 900)
        # cv2.moveWindow("1", 5, 5)
        point_img = rgb1.copy()
        for point in points_from_sfm_2d:
            cv2.circle(point_img, (point * shape).astype(np.int32), 5, (0, 0, 255), thickness=10)
        print("Draw lines on img1")
        line_img1 = rgb1.copy()
        line_img2 = rgb2.copy()

        # Draw first img
        for idx, face in enumerate(graph1.graph["faces"]):
            # print(idx)
            vertices = [graph1.nodes[id_node]["pos_2d"] for id_node in face]
            cv2.polylines(line_img1, [(np.asarray(vertices) * shape).astype(np.int32)], True, (0, 0, 255),
                          thickness=3, lineType=cv2.LINE_AA)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()

        # Draw target patch
        if True:
            vertices_t = [graph1.nodes[id_node]["pos_2d"] for id_node in graph1.graph["faces"][id_patch]]
            cv2.polylines(line_img1, [(np.asarray(vertices_t) * shape).astype(np.int32)], True, (0, 255, 0),
                          thickness=5,
                          lineType=cv2.LINE_AA)
            for item in vertices_t:
                cv2.circle(line_img1, (item * shape).astype(np.int32), 7, (0, 255, 255), 7)

        # Draw second img
        for idx, face in enumerate(graph2.graph["faces"]):
            # print(idx)
            vertices = [graph2.nodes[id_node]["pos_2d"] for id_node in face]
            cv2.polylines(line_img2, [(np.asarray(vertices) * shape).astype(np.int32)], True, (0, 0, 255),
                          thickness=3, lineType=cv2.LINE_AA)

        viz_img = np.concatenate((point_img, line_img1, line_img2), axis=0)
        cv2.imwrite("output/img_field_test/input_img1.jpg", viz_img)

    draw_initial()

    print("5. Build kd tree and compute initial line clouds")

    def compute_initial():
        distance_threshold = 5  # 5m; not used

        # Query points: (M, 2)
        # points from sfm: (N, 2)
        kd_tree = faiss.IndexFlatL2(2)
        kd_tree.add(points_from_sfm_2d.astype(np.float32))
        query_points = np.asarray([graph1.nodes[id_node]["pos_2d"] for id_node in graph1.nodes()])  # (M, 2)
        shortest_distance, index_shortest_distance = kd_tree.search(query_points, 8)  # (M, K)

        points_from_sfm_camera = np.transpose(
            img1.extrinsic @ np.transpose(np.insert(points_from_sfm, 3, 1, axis=1)))[:, :3]  # (N, 3)

        # Select the point which is nearest to the actual ray for each endpoints
        # 1. Construct the ray
        ray_c = np.transpose(np.linalg.inv(img1.intrinsic) @ np.transpose(
            np.insert(query_points, 2, 1, axis=1)))  # (M, 2); points in camera coordinates
        ray_c = ray_c / np.linalg.norm(ray_c+1e-6, axis=1, keepdims=True)  # Normalize the points
        nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
        # Compute the shortest distance from the candidate point to the ray for each query point
        distance_of_projection = nearest_candidates @ ray_c[:, :,
                                                      np.newaxis]  # (M, K, 1): K projected distance of the candidate point along each ray
        projected_points_on_ray = distance_of_projection * ray_c[:, np.newaxis,
                                                           :]  # (M, K, 3): K projected points along the ray
        distance_from_candidate_points_to_ray = np.linalg.norm(nearest_candidates - projected_points_on_ray+1e-6,
                                                               axis=2)  # (M, 1)
        index_best_projected = distance_from_candidate_points_to_ray.argmin(
            axis=1)  # (M, 1): Index of the best projected points along the ray

        chosen_distances = distance_of_projection[np.arange(projected_points_on_ray.shape[0]), index_best_projected]
        valid_mask = distance_from_candidate_points_to_ray[np.arange(
            projected_points_on_ray.shape[0]), index_best_projected] < distance_threshold  # (M, 1)
        initial_points_camera = projected_points_on_ray[np.arange(projected_points_on_ray.shape[
                                                                      0]), index_best_projected]  # (M, 3): The best projected points along the ray
        initial_points_world = np.transpose(
            np.linalg.inv(img1.extrinsic) @ np.transpose(np.insert(initial_points_camera, 3, 1, axis=1)))
        initial_points_world = initial_points_world[:, :3] / initial_points_world[:, 3:4]

        for id_node in range(initial_points_world.shape[0]):
            graph1.nodes[id_node]["pos_world"] = initial_points_world[id_node]
            graph1.nodes[id_node]["distance"] = chosen_distances[id_node, 0]
            graph1.nodes[id_node]["ray_c"] = ray_c[id_node]

        # line_coordinates = initial_points_world.reshape(-1, 6)  # (M, 6)
        # line_coordinates = line_coordinates[valid_mask]
        # valid_distances = chosen_distances.reshape(-1, 2)
        # valid_distances = valid_distances[valid_mask]
        ## line_coordinates = points_from_sfm[index_shortest_distance[:, 0], :].reshape(-1, 6)
        # id_patch = (valid_mask[:id_patch]).sum()

        line_coordinates = []
        for edge in graph1.edges():
            line_coordinates.append(np.concatenate((initial_points_world[edge[0]], initial_points_world[edge[1]])))
        save_line_cloud("output/img_field_test/initial_segments.obj", np.stack(line_coordinates, axis=0))
        return

    compute_initial()  # Coordinate of the initial segments in world coordinate

    print("6. Load img models")
    img_model_root_dir = r"output/neural_recon/img_nif_log"

    def load_img_model(img_name):
        if os.path.exists(os.path.join(img_model_root_dir, img_name)):
            checkpoint_name = [item for item in os.listdir(os.path.join(img_model_root_dir, img_name)) if
                               item[-4:] == "ckpt"]
            assert len(checkpoint_name) == 1
            img_model = NGPModel()
            # state_dict = torch.load(os.path.join(img_model_root_dir, img_name, checkpoint_name[0]))
            state_dict = torch.load(os.path.join(img_model_root_dir, img_name, checkpoint_name[0]))["state_dict"]
            img_model.load_state_dict({item[6:]: state_dict[item] for item in state_dict}, strict=True)
            img_model.eval()
            return img_model
        else:
            print("cannot find model for img {}".format(img_name))
            raise

    img_model1 = load_img_model(img1.img_name)
    img_model2 = load_img_model(img2.img_name)

    point_pos2d = np.asarray([graph1.nodes[id_node]["pos_2d"] for id_node in graph1.nodes()])  # (M, 2)
    point_pos3d_w = np.asarray([graph1.nodes[id_node]["pos_world"] for id_node in graph1.nodes()])  # (M, 3)
    distance = np.asarray([graph1.nodes[id_node]["distance"] for id_node in graph1.nodes()])  # (M, 1)
    ray_c = np.asarray([graph1.nodes[id_node]["ray_c"] for id_node in graph1.nodes()])
    points_pos_3d_c = ray_c * distance[:, None]  # (M, 3)

    print("7. Visualize target patch")
    if True:
        with open("output/img_field_test/target_patch.obj", "w") as f:
            for id_point in graph1.graph["faces"][id_patch]:
                f.write("v {} {} {}\n".format(point_pos3d_w[id_point, 0], point_pos3d_w[id_point, 1],
                                              point_pos3d_w[id_point, 2]))
            for id_point in range(len(graph1.graph["faces"][id_patch])):
                if id_point == len(graph1.graph["faces"][id_patch]) - 1:
                    f.write("l {} {}\n".format(id_point + 1, 1))
                else:
                    f.write("l {} {}\n".format(id_point + 1, id_point + 2))

    print("8. Calculate initial normal")
    # compute_initial_normal_based_on_pos(graph1)
    compute_initial_normal_based_on_camera(points_pos_3d_c, graph1)

    print("9. Visualize target patch")
    if True:
        arrows = o3d.geometry.TriangleMesh()
        for id_segment in range(len(graph1.graph["faces"][id_patch])):
            id_start = graph1.graph["faces"][id_patch][id_segment]
            id_end = graph1.graph["faces"][id_patch][(id_segment + 1) % len(graph1.graph["faces"][id_patch])]
            up_vector_c = graph1[id_start][id_end]["up_c"][id_patch]
            center_point_c = (graph1.nodes[id_end]["ray_c"] * graph1.nodes[id_end]["distance"] +
                              graph1.nodes[id_start]["ray_c"] * graph1.nodes[id_start]["distance"]) / 2
            up_point = center_point_c + up_vector_c
            up_vector_w = (np.linalg.inv(img1.extrinsic) @ to_homogeneous_vector(up_point)) - np.linalg.inv(
                img1.extrinsic) @ to_homogeneous_vector(center_point_c)

            center_point = (graph1.nodes[id_end]["pos_world"] + graph1.nodes[id_start]["pos_world"]) / 2
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0001, cone_radius=0.00015,
                                                           cylinder_height=0.001, cone_height=0.001)
            arrow.rotate(caculate_align_mat(normalize_vector(up_vector_w[:3])), center=(0, 0, 0))
            arrow.translate(center_point)
            arrows += arrow
        o3d.io.write_triangle_mesh(r"output/img_field_test/up_vector_arrow_for_patch_{}.ply".format(id_patch),
                                   arrows)

    data = {
        "graph1": graph1,
        "graph2": graph2,
        "id_patch": id_patch,
        "intrinsic1": img1.intrinsic,
        "extrinsic1": img1.extrinsic,
        "intrinsic2": img2.intrinsic,
        "extrinsic2": img2.extrinsic,
        "img_model1": img_model1,
        "img_model2": img_model2,
        "rgb1": rgb1,
        "rgb2": rgb2,
    }

    return data


@hydra.main(config_name="phase3_l7.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)
    data = prepare_dataset_and_model(v_cfg["dataset"]["colmap_dir"], v_cfg["dataset"]["only_train_target"])

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"] )

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e6),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        default_root_dir=log_dir,
        # precision=16,
        # gradient_clip_val=0.5
    )

    model = Phase3(v_cfg, data)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    if False:
        def is_point_inside_polygon(point, vertices):
            # Ray casting algorithm to check if a point is inside a polygon
            # Returns True if the point is inside the polygon, False otherwise
            x, y = point
            x_coords, y_coords = vertices[:, 0], vertices[:, 1]
            # Shift the vertices to create edges
            x_edges = np.roll(x_coords, 1) - x_coords
            y_edges = np.roll(y_coords, 1) - y_coords
            # Check if the point is inside the polygon using the ray casting algorithm
            is_above = (y_coords > y) != (np.roll(y_coords, 1) > y)
            is_left = x < (
                    np.roll(x_edges, 1) * (y - np.roll(y_coords, 1)) / np.roll(y_edges, 1) + np.roll(x_coords, 1))
            return (is_above & is_left).sum() % 2 != 0


        # def is_point_inside_polygon(point, vertices):
        #     # Ray casting algorithm to check if a point is inside a polygon
        #     # Returns True if the point is inside the polygon, False otherwise
        #     x, y = point
        #     inside = False
        #     for i in range(len(vertices)):
        #         j = (i + 1) % len(vertices)
        #         if ((vertices[i][1] > y) != (vertices[j][1] > y)) and \
        #                 (x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) / (
        #                         vertices[j][1] - vertices[i][1]) + vertices[i][0]):
        #             inside = not inside
        #     return inside

        def generate_random_points_inside_polygon(vertices, num_points):
            # Generate random points inside a non-convex polygon defined by its vertices
            # Returns a numpy array of shape (num_points, 2) containing the random points
            x_min, y_min = np.min(vertices, axis=0)
            x_max, y_max = np.max(vertices, axis=0)
            points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))
            inside_points = []
            for point in points:
                if is_point_inside_polygon(point, vertices):
                    inside_points.append(point)
            return np.array(inside_points)


        vertices = np.array([[2, 2], [4, 4], [5, 3], [4.5, 2.5], [5, 2]])
        from matplotlib import pyplot as plt

        points = generate_random_points_inside_polygon(vertices, 1000)
        vertices = np.insert(vertices, 5, vertices[0], axis=0)

        plt.figure()
        plt.plot(vertices[:, 0], vertices[:, 1])
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()  # if you need...

    main()
