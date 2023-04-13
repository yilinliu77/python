import itertools
import sys, os
import time

from torch.distributions import Binomial

from src.neural_recon.init_segments import compute_init_based_on_similarity

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
from torch_scatter import scatter_add, scatter_min
import faiss
import torchsort

import mcubes
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from tqdm import tqdm
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


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_length, v_id_target, v_training_mode):
        super(Dummy_dataset, self).__init__()
        self.length = v_length
        self.id_target = v_id_target
        self.training_mode = v_training_mode
        pass

    def __getitem__(self, index):
        if self.id_target != -1:
            return torch.tensor(self.id_target, dtype=torch.long)
        else:
            return torch.tensor(index, dtype=torch.long)

    def __len__(self):
        if self.id_target != -1:
            if self.training_mode == "training":
                return 1000
            else:
                return 1
        else:
            return self.length


class Edge_dataset(torch.utils.data.Dataset):
    def __init__(self, v_edge_indexes, is_one_target, id_edge, v_training_mode):
        super(Edge_dataset, self).__init__()
        self.id_faces = torch.arange(len(v_edge_indexes)).repeat_interleave(
            torch.tensor([len(item) // 4 for item in v_edge_indexes]))
        self.data = np.concatenate([np.asarray(item).reshape(-1, 4) for item in v_edge_indexes], axis=0)

        self.is_one_target = is_one_target
        self.id_edge = id_edge
        if not self.is_one_target:
            self.length = self.data.shape[0]
        else:
            self.length = 1000
        self.training_mode = v_training_mode
        pass

    def __getitem__(self, index):
        if self.is_one_target or self.training_mode == "validation":
            return self.id_edge, torch.tensor(self.data[self.id_edge], dtype=torch.long)
        else:
            return index, torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        if self.training_mode == "training":
            return self.length
        else:
            return 1


# Normal loss and Similarity loss using calculated up vector without gaussian distribution
class LModel17(nn.Module):
    def __init__(self, v_data, v_weights, v_img_method, v_log_root):
        super(LModel17, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method

        # Graph related
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]

        # Buffer variables
        ray_c_vertex = torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32)
        ray_c_centroid = torch.tensor(self.graph1.graph["face_center"]["ray_c"], dtype=torch.float32)
        self.register_buffer("ray_c", torch.cat((ray_c_vertex, ray_c_centroid), dim=0))  # (M, 2)
        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field1", torch.asarray(v_data["edge_field1"]).permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field2", torch.asarray(v_data["edge_field2"]).permute(2, 0, 1).unsqueeze(0))

        # Image models
        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        # for p in self.img_model2.parameters():
        #     p.requires_grad = False

        # Edge index
        # Start, end, prev, next
        self.edge_point_index = [[] for _ in range(len(self.graph1.graph["faces"]))]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.edge_point_index[id_patch].append(id_start)
                self.edge_point_index[id_patch].append(id_end)
                self.edge_point_index[id_patch].append(id_prev)
                self.edge_point_index[id_patch].append(id_next)

        # Trained parameters
        # Distance parameters
        # Normalized to [0,1]
        self.seg_distance_normalizer = 300
        seg_distance_vertex = torch.tensor([
            self.graph1.nodes[id_node]["distance"].tolist() for id_node in self.graph1.nodes()
        ], dtype=torch.float32) / self.seg_distance_normalizer
        seg_distance_centroid = torch.tensor(self.graph1.graph["face_center"]["distance"],
                                             dtype=torch.float32) / self.seg_distance_normalizer
        self.seg_distance = nn.Parameter(torch.cat((seg_distance_vertex, seg_distance_centroid), dim=0),
                                         requires_grad=True)
        self.id_centroid_start = seg_distance_vertex.shape[0]
        # Up vector
        # Method 1: Define a 3-dimensional vector
        if False:
            v_up = []
            self.v_up_dict = {}
            for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
                for id_segment in range(len(face_ids)):
                    id_start = face_ids[id_segment]
                    id_end = face_ids[(id_segment + 1) % len(face_ids)]
                    if id_start > id_end:
                        t = id_start
                        id_start = id_end
                        id_end = t
                    self.v_up_dict[(id_start, id_end, id_patch)] = len(v_up)
                    v_up.append(
                        torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32))
            self.v_up = torch.stack(v_up, dim=0)

        # Method 2: Define a 1 parameter up vector which is normalized by 2*pi
        num_edges_per_face = [len(item) for item in self.graph1.graph["faces"]]
        face_to_up = torch.tensor(np.insert(np.cumsum(num_edges_per_face), 0, 0), dtype=torch.long)
        self.register_buffer("face_to_up", face_to_up)

        self.edge_to_up = {}
        v_up = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                if id_start > id_end:
                    t = id_start
                    id_start = id_end
                    id_end = t
                self.edge_to_up[(id_start, id_end, id_patch)] = len(v_up)
                v_up.append(torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32))

        all_edge_points = (self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer)[
            list(itertools.chain(*self.edge_point_index))].reshape(-1, 4, 3)
        start_points, end_points = all_edge_points[:, 0], all_edge_points[:, 1]
        t_candidate = torch.arange(0, 1, 0.1)
        start_points = start_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        end_points = end_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        all_t_candidate = t_candidate.tile(all_edge_points.shape[0]) * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(all_t_candidate) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            a * torch.cos(all_t_candidate) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate)
        ), dim=1)
        up_c = normalize_tensor(up_c)
        up_c = up_c.reshape(-1, t_candidate.shape[0], 3)
        distance = (up_c * torch.stack(v_up, dim=0)[:, None, :]).sum(dim=-1)
        id_best = distance.argmax(dim=1)
        self.v_up = t_candidate[id_best]

        # Register parameters
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        # Visualization
        viz_shape = (6000, 4000)
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]
        self.log_root = v_log_root

        # Accurate initialization in patch 1522
        id_vertices = np.asarray(self.edge_point_index[1522]).reshape(-1, 4)[:, 0]
        self.seg_distance.data[id_vertices] = torch.tensor(
            [0.3040, 0.3033, 0.3030, 0.3021, 0.3110, 0.3067, 0.3063, 0.3057, 0.3045])
        # self.seg_distance.data[id_vertices] = torch.tensor([0.3040,0.3033,0.3030,0.3021,0.3115,0.3067,0.3063,0.3057,0.3045])
        self.seg_distance.data[self.id_centroid_start + 1522] = 0.3030

    def sample_points_based_on_vertices(self, edge_points):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        cur_dir = edge_points[:, 1] - start_point
        next_dir = edge_points[:, 3] - edge_points[:, 1]
        prev_dir = edge_points[:, 0] - edge_points[:, 2]

        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
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
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [80, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_up(self, edge_points, edge_up_c, v_id_epoch, is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))
        cur_up = normalize_tensor(edge_up_c)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.2).to(device)  # m
        half_window_size_step = 0.01

        # Compute interpolated point
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
            cur_up.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        similarity_loss = self.compute_similarity(interpolated_coordinates_camera,
                                                  self.intrinsic1, self.intrinsic2, self.extrinsic1, self.extrinsic2)

        # Normal Loss
        # Input: start_point and cur_up
        p1 = start_point
        p2 = start_point + cur_up
        p = torch.cat([p1, p2], dim=0)
        A = torch.stack([p[:, 0], p[:, 1], torch.ones_like(p[:, 0])], dim=1)
        solution = torch.linalg.lstsq(A, p[:, 2:3])
        normal_loss = (((A * solution.solution[:, 0]).sum(axis=1, keepdims=True) - p[:,
                                                                                   2:3]) ** 2).mean() / self.seg_distance_normalizer
        # cur_normal = normalize_tensor(torch.cross(cur_dir, cur_up))
        # next_dir = edge_points[:, 3] - edge_points[:, 1]
        # next_up = torch.cross(edge_up_c[:, 1], next_dir)
        # next_normal = normalize_tensor(torch.cross(next_dir, next_up))
        # prev_dir = edge_points[:, 0] - edge_points[:, 2]
        # prev_up = normalize_tensor(torch.cross(edge_up_c[:, 2], prev_dir))
        # prev_normal = normalize_tensor(torch.cross(prev_dir, prev_up))
        # normal_loss1 = (1 - (cur_normal * next_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        # normal_loss2 = (1 - (cur_normal * prev_normal).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        # normal_loss = (normal_loss1 + normal_loss2) / 2
        #
        # observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        # should_not_perpendicular = torch.min(torch.sum(cur_up * observing_normal, dim=1).abs(),
        #                                      0.5 * torch.ones_like(cur_up[:,0])) # [60, 90] degrees
        # normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)

        if is_log and self.id_viz_patch != -1:
            line_thickness = 1
            point_thickness = 2
            point_radius = 1
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ \
                             torch.inverse(self.extrinsic1)

            polygon_points_2d_1 = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
            polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_2 = (transformation @ to_homogeneous_tensor(interpolated_coordinates_camera).T).T
            polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()

            for id_edge, _ in enumerate(num_coordinates_per_edge):
                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                polygon_2d1 = (self.intrinsic1 @ start_point.T).T
                polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
                polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.polylines(line_img1, [polygon_2d1],
                              isClosed=True, color=(0, 255, 0), thickness=line_thickness, lineType=cv2.LINE_AA)
                for point in polygon_2d1:
                    cv2.circle(line_img1, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = polygon_points_2d_1[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]]
                roi_coor_2d1_numpy = np.clip(roi_coor_2d1_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                # Image 2
                line_img2 = self.rgb2.copy()
                shape = line_img2.shape[:2][::-1]

                polygon_2d2 = (transformation @ to_homogeneous_tensor(start_point).T).T
                polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
                polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.polylines(line_img2, [polygon_2d2],
                              isClosed=True, color=(0, 255, 0), thickness=line_thickness, lineType=cv2.LINE_AA)
                for point in polygon_2d2:
                    cv2.circle(line_img2, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d2_numpy = polygon_points_2d_2[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]]
                roi_coor_2d2_numpy = np.clip(roi_coor_2d2_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))

        return similarity_loss

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)
        return num_edge_points, sampled_edge_points

    def sample_polygon(self, num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point):
        area = torch.linalg.norm(torch.cross(cur_dir, next_dir) + 1e-6, dim=1).abs()

        num_polygon_points = torch.clamp((area * num_per_half_m2).to(torch.long), 1, 500)
        sample_points1 = torch.rand(num_polygon_points.sum(), 2).to(cur_dir.device)
        _t1 = torch.sqrt(sample_points1[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * start_point.repeat_interleave(num_polygon_points, dim=0) + \
                                 _t1 * (1 - sample_points1[:, 1:2]) * end_point.repeat_interleave(num_polygon_points,
                                                                                                  dim=0) + \
                                 _t1 * sample_points1[:, 1:2] * next_point.repeat_interleave(num_polygon_points, dim=0)
        return num_polygon_points, sampled_polygon_points

    def sample_triangles(self, num_per_m, p1, p2, p3, num_max_sample=500):
        d1 = p2 - p1
        d2 = p3 - p2
        area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

        num_edge_points, edge_points = self.sample_edge(num_per_m,
                                                        torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)

        num_per_m2 = num_per_m * num_per_m
        num_tri_samples = torch.clamp((area * num_per_m2).to(torch.long), 1, num_max_sample * 4)
        samples = torch.rand(num_tri_samples.sum(), 2).to(p1.device)
        _t1 = torch.sqrt(samples[:, 0:1] + 1e-6)
        sampled_polygon_points = (1 - _t1) * p1.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * (1 - samples[:, 1:2]) * p2.repeat_interleave(num_tri_samples, dim=0) + \
                                 _t1 * samples[:, 1:2] * p3.repeat_interleave(num_tri_samples, dim=0)

        num_total_points = num_edge_points + num_tri_samples
        num_total_points_cumsum = num_total_points.cumsum(0).roll(1)
        num_total_points_cumsum[0] = 0
        sampled_total_points = torch.zeros((num_total_points.sum(), 3), device=p1.device, dtype=torch.float32)
        num_edge_points_ = num_edge_points.cumsum(0).roll(1)
        num_edge_points_[0] = 0
        num_tri_points_ = num_tri_samples.cumsum(0).roll(1)
        num_tri_points_[0] = 0
        edge_index = torch.arange(num_edge_points.sum(), device=p1.device) \
                     - num_edge_points_.repeat_interleave(num_edge_points) \
                     + num_total_points_cumsum.repeat_interleave(num_edge_points)
        tri_index = torch.arange(num_tri_samples.sum(), device=p1.device) \
                    - num_tri_points_.repeat_interleave(num_tri_samples) \
                    + num_total_points_cumsum.repeat_interleave(num_tri_samples) \
                    + num_edge_points.repeat_interleave(num_tri_samples)
        sampled_total_points[edge_index] = edge_points
        sampled_total_points[tri_index] = sampled_polygon_points
        return num_total_points, sampled_total_points

    def sample_points_based_on_polygon(self, edge_points, id_epoch, v_is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0. Unpack data
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point
        time_profile[0], timer = refresh_timer(timer)

        # 1. Sample points on edges
        num_per_edge_m = 100
        num_edge_points, sampled_edge_points = self.sample_edge(num_per_edge_m, cur_dir, start_point)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample points within triangle
        # num_per_half_m2 = 50
        # num_polygon_points, sampled_polygon_points = self.sample_polygon(num_per_half_m2, cur_dir, next_dir, start_point, end_point, next_point)
        # time_profile[2], timer = refresh_timer(timer)

        # 3. Calculate pixel coordinate
        # coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)
        coordinates = sampled_edge_points

        roi_coor_2d = (self.intrinsic1 @ coordinates.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / (roi_coor_2d[:, 2:3] + 1e-6)
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 0.999999)
        time_profile[3], timer = refresh_timer(timer)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d[None, :, :])[0]
        time_profile[4], timer = refresh_timer(timer)

        # 5. Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
            self.extrinsic1)
        roi_coor_2d_img2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / (roi_coor_2d_img2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 0.999999)
        time_profile[6], timer = refresh_timer(timer)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        # 7. Similarity loss
        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

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

        # 8. Normal loss
        cur_normal_c = normalize_tensor(torch.cross(cur_dir, next_dir))
        sign_flag = torch.sum(cur_normal_c * torch.tensor(((0, 0, 1),), device=device, dtype=torch.float32), dim=1) > 0
        cur_normal_c[sign_flag] = -cur_normal_c[sign_flag]
        cur_up_c = normalize_tensor(torch.cross(cur_normal_c, cur_dir))
        prev_normal_c = normalize_tensor(torch.cross(prev_dir, cur_dir))
        normal_loss = (1 - (cur_normal_c * prev_normal_c).sum(dim=1)).mean() / 2  # [0, 2] -> [0, 1]
        time_profile[8], timer = refresh_timer(timer)

        # 9. Regularization loss
        observing_normal = normalize_tensor(torch.cross(edge_points[:, 0], edge_points[:, 1]))
        should_not_perpendicular = torch.min(torch.sum(cur_up_c * observing_normal, dim=1).abs(),
                                             0.5 * torch.ones_like(cur_up_c[:, 0]))  # [60, 90] degrees
        normalization_loss = torch.mean(1 - should_not_perpendicular / 0.5)
        time_profile[9], timer = refresh_timer(timer)

        is_debug = False
        if is_debug:
            for id_edge in range(1):
                img1 = self.rgb1.copy()
                img2 = self.rgb2.copy()
                shape = img1.shape[:2][::-1]
                p1_2d = (torch.cat((
                    roi_coor_2d[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    # roi_coor_2d[num_edge_points.sum()+num_polygon_points[:id_edge].sum():num_edge_points.sum()+num_polygon_points[:id_edge+1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                p2_2d = (torch.cat((
                    roi_coor_2d_img2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    # roi_coor_2d_img2[num_edge_points.sum() + num_polygon_points[:id_edge].sum():num_edge_points.sum() + num_polygon_points[:id_edge + 1].sum()],
                ), dim=0).detach().cpu().numpy() * shape).astype(np.int32)
                img1[p1_2d[:, 1], p1_2d[:, 0]] = (0, 0, 255)
                img2[p2_2d[:, 1], p2_2d[:, 0]] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            np.concatenate((img1, img2), axis=0))
                sampled_imgs = (torch.stack([
                    sample_imgs1[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                    sample_imgs2[num_edge_points[:id_edge].sum():num_edge_points[:id_edge + 1].sum()],
                ], dim=0).detach().cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
                cv2.imwrite(os.path.join(self.log_root, "s_{}_{:05d}.jpg".format(id_edge, id_epoch)),
                            sampled_imgs)
            pass

        # 9: Viz
        if False and v_is_log:
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
        return similarity_loss, normal_loss, normalization_loss

    def sample_points_based_on_convex(self, edge_points, centroid_c, v_id_epoch, v_is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0. Unpack data
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point
        time_profile[0], timer = refresh_timer(timer)

        # 1. Sample points on edges
        num_per_edge_m = 100
        num_edge_points, sampled_edge_points = self.sample_edge(num_per_edge_m, cur_dir, start_point)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample points within triangle
        num_per_m = 20
        num_polygon_points, sampled_polygon_points = self.sample_triangles(num_per_m,
                                                                           start_point, end_point, centroid_c,
                                                                           num_max_sample=500)
        time_profile[2], timer = refresh_timer(timer)

        # 3. Calculate pixel coordinate
        # coordinates = torch.cat([sampled_edge_points, sampled_polygon_points], dim=0)
        coordinates = sampled_edge_points
        num_coordinates = num_edge_points
        similarity_loss = self.compute_similarity(coordinates,
                                                  self.intrinsic1, self.intrinsic2, self.extrinsic1, self.extrinsic2)
        time_profile[3], timer = refresh_timer(timer)
        # 9: Viz
        if v_is_log and self.id_viz_patch != -1:
            line_thickness = 1
            point_thickness = 1
            point_radius = 2
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ \
                             torch.inverse(self.extrinsic1)

            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]
            line_img2 = self.rgb2.copy()

            polygon_points_2d_1 = (self.intrinsic1 @ coordinates.T).T
            polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_2 = (transformation @ to_homogeneous_tensor(coordinates).T).T
            polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()
            polygon_points_2d_1 = (np.clip(polygon_points_2d_1, 0, 0.99999) * shape).astype(np.int32)
            polygon_points_2d_2 = (np.clip(polygon_points_2d_2, 0, 0.99999) * shape).astype(np.int32)

            polygon_2d1 = (self.intrinsic1 @ start_point.T).T
            polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
            polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon_2d1],
                          isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            for point in polygon_2d1:
                cv2.circle(line_img1, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

            # Image 2
            polygon_2d2 = (transformation @ to_homogeneous_tensor(start_point).T).T
            polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
            polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
            cv2.polylines(line_img2, [polygon_2d2],
                          isClosed=True, color=(0, 255, 0), thickness=line_thickness)
            for point in polygon_2d2:
                cv2.circle(line_img2, point, radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

            for id_edge in range(start_point.shape[0]):
                img1 = line_img1.copy()
                img2 = line_img2.copy()

                id_polygon_coord = num_coordinates[:id_edge].sum()
                for point in polygon_points_2d_1[id_polygon_coord:id_polygon_coord + num_coordinates[id_edge]]:
                    cv2.circle(img1, point, radius=point_radius, color=(0, 0, 255), thickness=point_thickness)

                id_polygon_coord = num_coordinates[:id_edge].sum()
                for point in polygon_points_2d_2[id_polygon_coord:id_polygon_coord + num_coordinates[id_edge]]:
                    cv2.circle(img2, point, radius=point_radius, color=(0, 0, 255), thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(id_edge, v_id_epoch)),
                            np.concatenate((img1, img2), axis=0))

        return similarity_loss

    def sample_points_2d_(self, edge_points_c, v_intrinsic, v_is_log=False, v_img=None):
        device = edge_points_c.device
        edge_points_2d = (v_intrinsic @ edge_points_c.reshape(-1, 3).T).T.reshape(edge_points_c.shape)
        edge_points_2d = edge_points_2d[:, :, :2] / edge_points_2d[:, :, 2:]
        cur_dir_2d = edge_points_2d[:, 1] - edge_points_2d[:, 0]
        z_vector = to_homogeneous_tensor(torch.zeros_like(cur_dir_2d))  # The z axis of the image
        up_2d = normalize_tensor(
            torch.cross(to_homogeneous_tensor(cur_dir_2d.reshape(-1, 2)), z_vector)[:, :2].reshape(cur_dir_2d.shape))

        length = torch.linalg.norm(cur_dir_2d + 1e-6, dim=1)
        num_horizontal = 100
        num_vertical = 20
        horizontal_step = length / num_horizontal
        vertical_step = 1 / 4000
        num_coordinates_per_edge = torch.ones_like(edge_points_2d[:, 0, 0],
                                                   dtype=torch.long) * num_horizontal * num_vertical

        dx = (torch.arange(num_horizontal).to(device) / (num_horizontal - 1 + 1e-6) * length[:, None]).reshape(-1)
        dy = (torch.arange(num_vertical) - num_vertical / 2).to(device) / (
                num_vertical - 1 + 1e-6) * vertical_step * num_vertical

        # Meshgrid
        total_num_coords = dx.shape[0] * dy.shape[0]
        coords_x = dx.repeat_interleave(dy.shape[0])  # (total_num_coords,)
        coords_y = torch.tile(dy, (dx.shape[0],))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)

        normalized_cur_dir_2d = normalize_tensor(cur_dir_2d)
        roi_coor_2d = \
            normalized_cur_dir_2d.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
            up_2d.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            (edge_points_2d[:, 0].repeat_interleave(num_coordinates_per_edge, dim=0))

        viz_data = []
        # 9: Viz
        if v_is_log:
            for id_edge, _ in enumerate(num_coordinates_per_edge):
                line_img1 = v_img.copy()
                shape = line_img1.shape[:2][::-1]

                id_coord = num_coordinates_per_edge[:id_edge].sum()
                roi_coor_2d1_numpy = roi_coor_2d[
                                     id_coord:id_coord + num_coordinates_per_edge[id_edge]].detach().cpu().numpy()
                roi_coor_2d1_numpy = np.clip(roi_coor_2d1_numpy, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                viz_data.append(line_img1)

        return roi_coor_2d, viz_data

    def sample_points_2d(self, edge_points, id_epoch, v_is_log):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        # 0. Unpack data
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        prev_point = edge_points[:, 2]
        next_point = edge_points[:, 3]
        cur_dir = end_point - start_point
        next_dir = next_point - end_point
        prev_dir = start_point - prev_point
        time_profile[0], timer = refresh_timer(timer)

        # 1. Calculate projected edges and sample points within rectangle
        roi_coor_2d1, viz_data1 = self.sample_points_2d_(edge_points, self.intrinsic1, v_is_log, self.rgb1)
        time_profile[1], timer = refresh_timer(timer)

        # 2. Sample pixel color
        roi_coor_2d1 = torch.clamp(roi_coor_2d1, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, roi_coor_2d1[None, :, :])[0]
        time_profile[2], timer = refresh_timer(timer)

        # 3. Second img
        edge_points_img2 = (self.extrinsic2 @ torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(
            edge_points.reshape(-1, 3)).T).T[:, :3].reshape(edge_points.shape)
        roi_coor_2d2, viz_data2 = self.sample_points_2d_(edge_points_img2, self.intrinsic2, v_is_log, self.rgb2)
        roi_coor_2d2 = torch.clamp(roi_coor_2d2, 0, 0.999999)
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, roi_coor_2d2[None, :, :])[0]
        time_profile[3], timer = refresh_timer(timer)

        # 4. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2)
        time_profile[4], timer = refresh_timer(timer)

        # 5. viz
        if v_is_log:
            for idx in range(len(viz_data1)):
                cv2.imwrite(os.path.join(self.log_root, "3d_{}_{:05d}.jpg".format(idx, id_epoch)),
                            np.concatenate((viz_data1[idx], viz_data2[idx]), axis=0))
        return similarity_loss, similarity_loss, similarity_loss

    def get_up_vector1(self, v_face_index):
        up_index = list(itertools.chain(
            *[np.insert(np.asarray(self.edge_point_index[face_id]).reshape(-1, 4)[:, :2], 2, face_id.cpu().item(),
                        axis=1)
              for face_id in v_face_index]))
        up_index = np.asarray(up_index)
        # Make sure the x index is lower than y
        up_index[up_index[:, 0] > up_index[:, 1]] = up_index[up_index[:, 0] > up_index[:, 1]][:, [1, 0, 2]]
        up_index = [self.v_up_dict[tuple(key.tolist())] for key in up_index]
        return up_index

    def get_up_vector2(self, v_face_index, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t = torch.cat([self.v_up[self.face_to_up[id_face]:self.face_to_up[id_face + 1]] for id_face in v_face_index],
                      dim=0)
        t = t * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            torch.sqrt(a * a + b * b) * torch.sin(t)
        ), dim=1)
        return normalize_tensor(up_c)

    def compute_similarity(self, coords_c,
                           v_intrinsic1, v_intrinsic2, v_extrinsic1, v_extrinsic2):
        coods2d_1 = (v_intrinsic1 @ coords_c.T).T
        coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        valid_mask1 = torch.logical_and(coods2d_1 > 0, coods2d_1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, coods2d_1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, coods2d_1[None, :, :])[0]

        # 5. Second img
        transformation = to_homogeneous_mat_tensor(v_intrinsic2) @ v_extrinsic2 @ torch.inverse(v_extrinsic1)
        coods2d_2 = (transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, coods2d_2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, coods2d_2[None, :, :])[0]

        # 7. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2)

        return similarity_loss

    def compute_edge_response(self, v_start_point, v_end_point):
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        num_edge_points, coords_c = self.sample_edge(100, v_end_point - v_start_point, v_start_point, 2000)
        # First image
        # start_2d1 = (self.intrinsic1 @ v_start_point.T).T
        # start_2d1 = start_2d1[:,:2] / start_2d1[:,2:3]
        # end_2d1 = (self.intrinsic1 @ v_end_point.T).T
        # end_2d1 = end_2d1[:,:2] / end_2d1[:,2:3]
        #
        # coods2d_1 = (self.intrinsic1 @ coords_c.T).T
        # coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        # coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # # Sample edge response
        # coordinate_tensor = coods2d_1.unsqueeze(0).unsqueeze(0)
        # sampled_edge1 = torch.nn.functional.grid_sample(self.edge_field1, coordinate_tensor * 2 - 1,
        #                                               align_corners=True)[0]  # [0,1] -> [-1,1]
        # dir_2d1 = normalize_tensor(end_2d1 - start_2d1)
        # edge_direction1 = normalize_tensor(sampled_edge1[0:2,0].T)
        # edge_weights1 = torch.abs((edge_direction1 * dir_2d1.repeat_interleave(num_edge_points,dim=0)).sum(dim=1))
        # edge_magnitude1 = sampled_edge1[2,0]

        # 5. Second img
        start_2d2 = (transformation @ to_homogeneous_tensor(v_start_point).T).T
        start_2d2 = start_2d2[:, :2] / start_2d2[:, 2:3]
        end_2d2 = (transformation @ to_homogeneous_tensor(v_end_point).T).T
        end_2d2 = end_2d2[:, :2] / end_2d2[:, 2:3]

        coods2d_2 = (transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        coordinate_tensor = coods2d_2.unsqueeze(0).unsqueeze(0)
        sampled_edge2 = torch.nn.functional.grid_sample(self.edge_field2, coordinate_tensor * 2 - 1,
                                                        align_corners=True)[0]  # [0,1] -> [-1,1]

        dir_2d2 = normalize_tensor(end_2d2 - start_2d2)

        edge_direction2 = normalize_tensor(sampled_edge2[0:2, 0].T)
        edge_weights2 = 1 - torch.abs((edge_direction2 * dir_2d2.repeat_interleave(num_edge_points, dim=0)).sum(dim=1))
        edge_magnitude2 = sampled_edge2[2, 0]

        # debug_va = scatter_add(edge_weights2, torch.arange(num_edge_points.shape[0],device=v_start_point.device).repeat_interleave(num_edge_points))
        # edge_response = (edge_weights1 * edge_magnitude1 + edge_weights2 * edge_magnitude2).sum()
        if True:
            edge_response = (edge_weights2 * edge_magnitude2).sum()
        else:
            edge_response_ = edge_weights2 * edge_magnitude2
            edge_response = scatter_add(edge_response_, torch.arange(
                num_edge_points.shape[0], device=v_start_point.device).repeat_interleave(num_edge_points))
            edge_response = (edge_response / num_edge_points).mean()

        return edge_response

    def forward(self, v_face_index, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1
        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_face_index]))
        point_index = np.asarray(point_index).reshape(-1, 4)[4:5].tolist()
        ray_c = self.ray_c[point_index].reshape((-1, 4, 3))
        seg_distance = self.seg_distance[point_index].reshape((-1, 4, 1)) * self.seg_distance_normalizer
        edge_points = ray_c * seg_distance
        up_c = self.get_up_vector2(v_face_index, edge_points[:, 0], edge_points[:, 1])
        # edge_points[:,1:] = edge_points[:,1:].detach()
        # centroid
        id_centroid = torch.tensor(list(itertools.chain(*[
            [item + self.id_centroid_start] * (len(self.edge_point_index[item]) // 4) for item in v_face_index])),
                                   device=v_face_index.device)
        id_centroid = id_centroid[4:5]
        centroid_point_c = (self.seg_distance[id_centroid] * self.seg_distance_normalizer)[:, None] * \
                           self.ray_c[id_centroid]

        edge_loss = self.compute_edge_response(v_start_point=edge_points[:, 0], v_end_point=edge_points[:, 1])
        similarity_loss = self.sample_points_based_on_convex(edge_points, centroid_point_c, v_id_epoch, is_log)
        # similarity_loss = self.sample_points_based_on_up(edge_points, up_c, v_id_epoch, is_log)

        # losses = []
        # losses += self.sample_points_based_on_convex(edge_points, centroid_point_c, v_id_epoch, is_log)
        # losses += self.sample_points_based_on_vertices(edge_points)
        # losses += self.sample_points_based_on_up(edge_points, edge_up_c)
        # losses += self.sample_points_based_on_polygon(edge_points, v_id_epoch, is_log)
        # losses += self.sample_points_2d(edge_points, v_id_epoch, is_log)
        # losses += self.sample_points_based_on_up(edge_points, up_c, v_id_epoch, is_log)
        # losses = torch.stack(losses).reshape((-1,3))

        weighted_similarity = similarity_loss * 1
        weighted_edge = edge_loss * (-1 / 10000)
        # weighted_edge = edge_loss * (-1)
        total_loss = weighted_similarity

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
                                      thickness=1)
            for point in roi_2d_numpy:
                cv2.circle(img1_base, (point * shape).astype(np.int32), radius=1,
                           color=(0, 255, 255),
                           thickness=2)
            transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(
                self.extrinsic1)
            roi_2d2 = (transformation @ to_homogeneous_tensor(roi_c).T).T
            roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
            img2_base = self.rgb2.copy()
            roi_2d_numpy = roi_2d2.detach().cpu().numpy()
            img2_base = cv2.polylines(img2_base, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (0, 0, 255),
                                      thickness=1)
            for point in roi_2d_numpy:
                cv2.circle(img2_base, (point * shape).astype(np.int32), radius=1,
                           color=(0, 255, 255),
                           thickness=2)
            cv2.imwrite(os.path.join(self.log_root, r"2d_{:05d}.jpg".format(v_id_epoch)),
                        np.concatenate((img1_base, img2_base), axis=0))
        pass

        return total_loss, [weighted_similarity, weighted_edge, total_loss]

    def debug_save(self, v_index):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_points, v_up_c):
            total_edge_points = v_edge_points

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + v_up_c

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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        # id_target = torch.tensor((119,),dtype=torch.long, device=point_pos_c.device)
        id_target = torch.tensor((1522,), dtype=torch.long, device=point_pos_c.device)
        # Visualize target patch
        edge_points = point_pos_c[self.edge_point_index[id_target]].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(id_target, edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c)
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(v_index + 1)), arrows)
        id_points = np.asarray(self.edge_point_index[id_target]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(v_index + 1)), start_point_w, edge_index)

        # Visualize whole patch
        # edge_points = point_pos_c[list(itertools.chain(*self.edge_point_index))].reshape(-1,4,3)
        # up_c = self.get_up_vector2(
        #     torch.arange(len(self.edge_point_index), device=self.ray_c.device), edge_points[:,0], edge_points[:,1])
        # arrows = get_arrow(edge_points, up_c)
        # o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(v_index)), arrows)
        # start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
        #     .cpu().numpy()
        # edge_index = np.asarray(list(self.graph1.edges()))
        # get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(v_index)), start_point_w, edge_index)
        pass

        return 0

    def len(self):
        return len(self.graph1.graph["faces"])


class LModel18(nn.Module):
    def __init__(self, v_data, v_viz_patch, v_viz_edge, v_weights, v_img_method, v_log_root):
        super(LModel18, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method
        self.log_root = v_log_root

        self.init_regular_variables(v_data)
        self.register_variables(v_data)
        self.register_img_model(v_data)
        self.calculate_index()
        self.register_distances(True)
        self.register_up_vectors(True)

        # Debug
        self.id_viz_face = v_viz_patch
        self.id_viz_edge = v_viz_edge

        # Accurate initialization in patch 1476
        id_vertices = np.asarray(self.batched_points_per_patch[1476]).reshape(-1, 4)[:, 0]
        # self.seg_distance.data[id_vertices] = torch.tensor(
        #     [0.3040, 0.3033, 0.3030, 0.3026, 0.3107, 0.3067, 0.3063, 0.3057, 0.3045])
        # self.seg_distance.data[id_vertices] = torch.tensor([0.3040,0.3033,0.3030,0.3021,0.3115,0.3067,0.3063,0.3057,0.3045])
        # self.seg_distance.data[self.id_centroid_start + 1522] = 0.3030

    # Init-related methods
    def init_regular_variables(self, v_data):
        # Graph related
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]

        # Visualization
        # viz_shape = (6000, 4000)
        self.rgb1 = v_data["rgb1"]
        self.rgb2 = v_data["rgb2"]
        # self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        # self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

    def register_variables(self, v_data):
        transformation = to_homogeneous(v_data["intrinsic2"]) @ v_data["extrinsic2"] \
                         @ np.linalg.inv(v_data["extrinsic1"])
        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("transformation", torch.as_tensor(transformation).float())
        self.register_buffer("edge_field1", torch.asarray(v_data["edge_field1"]).permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("edge_field2", torch.asarray(v_data["edge_field2"]).permute(2, 0, 1).unsqueeze(0))

    def register_img_model(self, v_data):
        self.register_buffer("o_rgb1", torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("o_rgb2", torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.) \
                             .permute(2, 0, 1).unsqueeze(0))
        # Image models
        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

    def calculate_index(self):
        # Edge index
        # Start, end, prev, next
        self.batched_points_per_patch = [[] for _ in self.graph1.graph["faces"]]
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]
                self.batched_points_per_patch[id_patch].append(id_start)
                self.batched_points_per_patch[id_patch].append(id_end)
                self.batched_points_per_patch[id_patch].append(id_prev)
                self.batched_points_per_patch[id_patch].append(id_next)

    def register_distances(self, train_distance=True):
        self.seg_distance_normalizer = 300
        distances_scale = torch.tensor([self.graph1.nodes[id_node]["scale"] for id_node in self.graph1.nodes()],
                                       dtype=torch.float32) / self.seg_distance_normalizer
        self.register_buffer("scale", distances_scale)

        ray_c_vertex = torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32)
        ray_c_centroid = torch.tensor(self.graph1.graph["ray_c"], dtype=torch.float32)
        self.register_buffer("ray_c", ray_c_vertex)
        self.register_buffer("center_ray_c", ray_c_centroid)

        seg_distance_vertex = torch.tensor([
            self.graph1.nodes[id_node]["distance"].tolist() for id_node in self.graph1.nodes()
        ], dtype=torch.float32) / self.seg_distance_normalizer
        self.seg_distance = nn.Parameter(seg_distance_vertex, requires_grad=train_distance)
        self.id_centroid_start = seg_distance_vertex.shape[0]

    def register_up_vectors(self, train_up_vectors):
        num_total_edge = np.sum([len(face) for face in self.graph1.graph["faces"]])

        id_edge_to_id_face = torch.zeros((num_total_edge, 2), dtype=torch.long)
        self.id_edge_to_id_up_dict = {}
        id_edge_to_id_up = []
        v_up = []

        # These three variables are used to initialize the up vector
        all_edge_points = self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer
        start_points = []
        end_points = []
        id_total_seg = 0
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                id_edge_to_id_up.append([])
                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_start, id_end)])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
                    id_edge_to_id_up[-1].append(len(v_up))
                    if len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 1:
                        v_up.append(-up_c)
                    else:
                        assert len(self.graph1.edges[(id_end, id_start)]["up_c"]) == 2
                        for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                            if id_patch == key:
                                continue
                            v_up.append(torch.tensor(
                                self.graph1.edges[(id_end, id_start)]["up_c"][key], dtype=torch.float32))
                    start_points.append(all_edge_points[id_end])
                    end_points.append(all_edge_points[id_start])
                else:
                    id_edge_to_id_up[-1].append(self.id_edge_to_id_up_dict[(id_end, id_start)])

                id_edge_to_id_face[id_total_seg, 0] = id_patch
                key2 = -1
                for key in self.graph1.edges[(id_end, id_start)]["up_c"]:
                    if id_patch == key:
                        continue
                    key2 = key
                id_edge_to_id_face[id_total_seg, 1] = key2
                id_total_seg += 1

        self.register_buffer("id_edge_to_id_up", torch.tensor(id_edge_to_id_up))
        self.register_buffer("id_edge_to_id_face", id_edge_to_id_face)

        # Initialization
        start_points = torch.stack(start_points, dim=0)
        end_points = torch.stack(end_points, dim=0)
        t_candidate = torch.arange(0, 1, 0.1)
        start_points = start_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        end_points = end_points.tile(t_candidate.shape[0]).reshape(-1, 3)
        all_t_candidate = t_candidate.tile(len(v_up)) * 2 * math.pi
        cur_dir = end_points - start_points
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(all_t_candidate) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            a * torch.cos(all_t_candidate) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate),
            torch.sqrt(a * a + b * b) * torch.sin(all_t_candidate)
        ), dim=1)
        up_c = normalize_tensor(up_c)
        up_c = up_c.reshape(-1, t_candidate.shape[0], 3)
        distance = (up_c * torch.stack(v_up, dim=0)[:, None, :]).sum(dim=-1)
        id_best = distance.argmax(dim=1)
        self.v_up = t_candidate[id_best]

        self.v_up = nn.Parameter(self.v_up, requires_grad=train_up_vectors)
        return

    def sample_points_based_on_up(self, start_point, end_point, edge_up_c):
        time_profile = [0 for _ in range(10)]
        timer = time.time()
        device = self.ray_c.device

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # cur_up = normalize_tensor(torch.cross(edge_up_c[:, 0], cur_dir))
        # cur_up = normalize_tensor(edge_up_c)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = torch.tensor(0.2).to(device)  # m
        half_window_size_step = 0.01

        # Compute interpolated point
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
            edge_up_c.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
            start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        return total_num_coords, interpolated_coordinates_camera

    def sample_triangles(self, num_per_m, p1, p2, p3, num_max_sample=500):
        d1 = p2 - p1
        d2 = p3 - p2
        area = torch.linalg.norm(torch.cross(d1, d2) + 1e-6, dim=1).abs() / 2

        num_edge_points, edge_points = self.sample_edge(num_per_m,
                                                        torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)

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
                        - (num_tri_points_-num_total_points_cumsum-num_edge_points).repeat_interleave(num_tri_samples)
            sampled_total_points[edge_index] = edge_points
            sampled_total_points[tri_index] = sampled_polygon_points
            return num_total_points, sampled_total_points
        return None, torch.cat((edge_points, sampled_polygon_points), dim=0)

    def sample_points_based_on_up_and_centroid(self, start_point, end_point, edge_up_c, v_centroid_ray):
        device = self.ray_c.device

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        plane_normal = torch.cross(cur_dir, edge_up_c)
        plane_normal = normalize_tensor(plane_normal)

        intersection_t = (start_point * plane_normal).sum(dim=1) / (v_centroid_ray * plane_normal).sum(dim=1)
        intersection_point = intersection_t[:, None] * v_centroid_ray

        num_samples, samples = self.sample_triangles(100, start_point, end_point, intersection_point)

        return num_samples, samples

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
        length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        num_edge_points = torch.clamp((length * num_per_edge_m).to(torch.long), 1, 2000)
        num_edge_points_ = num_edge_points.roll(1)
        num_edge_points_[0] = 0
        sampled_edge_points = torch.arange(num_edge_points.sum()).to(cur_dir.device) - num_edge_points_.cumsum(
            dim=0).repeat_interleave(num_edge_points)
        sampled_edge_points = sampled_edge_points / ((num_edge_points - 1 + 1e-8).repeat_interleave(num_edge_points))
        sampled_edge_points = cur_dir.repeat_interleave(num_edge_points, dim=0) * sampled_edge_points[:, None] \
                              + start_point.repeat_interleave(num_edge_points, dim=0)
        return num_edge_points, sampled_edge_points

    def get_up_vector(self, v_id_edges, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t1 = self.v_up[self.id_edge_to_id_up[v_id_edges][:,0]]
        t2 = self.v_up[self.id_edge_to_id_up[v_id_edges][:,1]]

        t = torch.stack([t1, t2], dim=0).reshape(-1)
        t = t * 2 * math.pi
        cur_dir = torch.stack((end_points - start_points, start_points - end_points), dim=0).reshape(-1, 3)
        a, b, c = cur_dir[:, 0], cur_dir[:, 1], cur_dir[:, 2]
        up_c = torch.stack((
            -b * torch.cos(t) - (a * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            a * torch.cos(t) - (b * c) / torch.sqrt(a * a + b * b) * torch.sin(t),
            torch.sqrt(a * a + b * b) * torch.sin(t)
        ), dim=1)
        return (up_c / torch.linalg.norm(up_c.detach(), dim=-1, keepdim=True)).reshape(2, -1, 3).permute(1, 0, 2)

    def compute_similarity(self, coords_c,
                           v_intrinsic1, v_transformation):
        coods2d_1 = (v_intrinsic1 @ coords_c.T).T
        coods2d_1 = coods2d_1[:, :2] / (coods2d_1[:, 2:3] + 1e-6)
        valid_mask1 = torch.logical_and(coods2d_1 > 0, coods2d_1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        coods2d_1 = torch.clamp(coods2d_1, 0, 0.999999)
        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, coods2d_1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, coods2d_1[None, :, :])[0]

        # 5. Second img
        coods2d_2 = (v_transformation @ to_homogeneous_tensor(coords_c).T).T
        coods2d_2 = coods2d_2[:, :2] / (coods2d_2[:, 2:3] + 1e-6)
        valid_mask2 = torch.logical_and(coods2d_2 > 0, coods2d_2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        coods2d_2 = torch.clamp(coods2d_2, 0, 0.999999)
        # 6. Second img
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, coods2d_2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, coods2d_2[None, :, :])[0]

        # 7. Similarity loss
        similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2, reduction="none")
        return similarity_loss, torch.logical_and(valid_mask1, valid_mask2)

    def compute_similarity_wrapper(self, edge_points, up_c, v_centroid1, v_centroid2, v_mask):
        num_coors_per_edge, coords_per_edge = self.sample_points_based_on_up_and_centroid(edge_points[:, 0],
                                                                                          edge_points[:, 1],
                                                                                          up_c[:, 0], v_centroid1)
        similarity_loss11, similarity_mask1 = self.compute_similarity(coords_per_edge, self.intrinsic1,
                                                                      self.transformation)
        similarity_loss11 = scatter_add(similarity_loss11, torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss11.device).repeat_interleave(num_coors_per_edge), dim=0)
        similarity_loss11 = (similarity_loss11 / num_coors_per_edge[:, None]).mean(dim=1)
        # similarity_loss11 = similarity_loss11 * v_mask.to(torch.long)
        similarity_mask1 = scatter_min(similarity_mask1.to(torch.long), torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss11.device).repeat_interleave(num_coors_per_edge),
                                       dim=0)[0]

        num_coors_per_edge2, coords_per_edge2 = self.sample_points_based_on_up_and_centroid(edge_points[:, 1],
                                                                                            edge_points[:, 0],
                                                                                            up_c[:, 1], v_centroid2)
        similarity_loss12, similarity_mask2 = self.compute_similarity(coords_per_edge2, self.intrinsic1,
                                                                      self.transformation)
        similarity_loss12 = scatter_add(similarity_loss12, torch.arange(
            num_coors_per_edge2.shape[0], device=similarity_loss12.device).repeat_interleave(num_coors_per_edge2),
                                        dim=0)
        similarity_loss12 = (similarity_loss12 / num_coors_per_edge2[:, None]).mean(dim=1)
        similarity_loss12 = similarity_loss12 * v_mask.to(torch.long)
        similarity_mask2 = scatter_min(similarity_mask2.to(torch.long), torch.arange(
            num_coors_per_edge2.shape[0], device=similarity_loss12.device).repeat_interleave(num_coors_per_edge2),
                                       dim=0)[0]

        return similarity_loss11 + similarity_loss12, torch.cat((coords_per_edge, coords_per_edge2), dim=0), \
            torch.logical_and(similarity_mask1.to(torch.bool),similarity_mask2.to(torch.bool))

    def random_search(self, v_id_edges, v_id_points, v_centroid1, v_centroid2, v_mask):
        with torch.no_grad():
            seg_distance_detach = self.seg_distance.detach()
            ray_c = self.ray_c[v_id_points].detach()
            selected_seg_distance = seg_distance_detach[v_id_points]

            seg_distance = selected_seg_distance * self.seg_distance_normalizer
            edge_points = ray_c * seg_distance[:, :, None]
            up_c = self.get_up_vector(v_id_edges, edge_points[:, 0], edge_points[:, 1]).detach()

            similarity_loss1, _, similarity_mask1 = self.compute_similarity_wrapper(
                edge_points, up_c, v_centroid1, v_centroid2, v_mask)
            similarity_loss1[~similarity_mask1] = torch.inf

            scale_factor = 1
            selected_seg_distance[:, 0] = -1
            while not torch.all(torch.logical_and(selected_seg_distance[:, 0] > 0, selected_seg_distance[:, 0] < 1)):
                a = seg_distance_detach[v_id_points][:, 0] + scale_factor * self.scale[v_id_points[:, 0]] * \
                    torch.distributions.utils._standard_normal(selected_seg_distance[:, 0].shape,
                                                               device=selected_seg_distance.device,
                                                               dtype=selected_seg_distance.dtype)
                mask = torch.logical_and(a > 0, a < 1)
                if mask.sum() > 0:
                    selected_seg_distance[:, 0][mask] = a[mask]

            seg_distance = selected_seg_distance * self.seg_distance_normalizer
            edge_points = ray_c * seg_distance[:, :, None]
            up_c = self.get_up_vector(v_id_edges, edge_points[:, 0], edge_points[:, 1]).detach()

            similarity_loss2, _, similarity_mask2 = self.compute_similarity_wrapper(
                edge_points, up_c, v_centroid1, v_centroid2, v_mask)
            similarity_loss2[~similarity_mask2] = torch.inf

            sucecced_search = similarity_loss2 < similarity_loss1
            self.seg_distance.data[v_id_points[sucecced_search]] = selected_seg_distance[sucecced_search]
        return

    def forward(self, v_ids, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1

        v_id_edge = v_ids[0]  # (num_edge, )
        v_id_edge_points = v_ids[1]  # (num_edge, 4)

        v_id_faces = self.id_edge_to_id_face[v_id_edge]

        centroid_ray1 = self.center_ray_c[v_id_faces[:, 0]]
        centroid_ray2 = self.center_ray_c[v_id_faces[:, 1]]
        mask = v_id_faces[:, 1] != -1
        # Random search
        if self.training:
            self.random_search(v_id_edge, v_id_edge_points, centroid_ray1, centroid_ray2, mask)

        ray_c = self.ray_c[v_id_edge_points]
        seg_distance = self.seg_distance[v_id_edge_points] * self.seg_distance_normalizer
        edge_points = ray_c * seg_distance[:, :, None]
        up_c = self.get_up_vector(v_id_edge, edge_points[:, 0].detach(), edge_points[:, 1].detach())
        # edge_points[:,1:] = edge_points[:,1:].detach()

        similarity_loss, coords_per_edge, similarity_mask = self.compute_similarity_wrapper(
            edge_points, up_c, centroid_ray1, centroid_ray2, mask)
        similarity_loss = similarity_loss.mean()
        if is_log and self.id_viz_edge in v_id_edge:
            with torch.no_grad():
                line_thickness = 1
                point_thickness = 2
                point_radius = 1

                polygon_points_2d_1 = (self.intrinsic1 @ coords_per_edge.T).T
                polygon_points_2d_1 = (polygon_points_2d_1[:, :2] / polygon_points_2d_1[:, 2:3]).detach().cpu().numpy()
                polygon_points_2d_2 = (self.transformation @ to_homogeneous_tensor(coords_per_edge).T).T
                polygon_points_2d_2 = (polygon_points_2d_2[:, :2] / polygon_points_2d_2[:, 2:3]).detach().cpu().numpy()

                line_img1 = self.rgb1.copy()
                line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
                shape = line_img1.shape[:2][::-1]

                roi_coor_2d1_numpy = np.clip(polygon_points_2d_1, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                polygon_2d1 = (self.intrinsic1 @ edge_points[0].T).T
                polygon_2d1 = polygon_2d1[:, :2] / polygon_2d1[:, 2:3]
                polygon_2d1 = (polygon_2d1.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.line(line_img1, polygon_2d1[0], polygon_2d1[1],
                         color=(0, 255, 0), thickness=line_thickness)
                cv2.circle(line_img1, polygon_2d1[0], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)
                cv2.circle(line_img1, polygon_2d1[1], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                # Image 2
                line_img2 = self.rgb2.copy()
                line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                shape = line_img2.shape[:2][::-1]

                roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                polygon_2d2 = (self.transformation @ to_homogeneous_tensor(edge_points[0]).T).T
                polygon_2d2 = polygon_2d2[:, :2] / polygon_2d2[:, 2:3]
                polygon_2d2 = (polygon_2d2.detach().cpu().numpy() * shape).astype(np.int32)
                cv2.line(line_img2, polygon_2d2[0], polygon_2d2[1],
                         color=(0, 255, 0), thickness=line_thickness)
                cv2.circle(line_img2, polygon_2d2[0], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)
                cv2.circle(line_img2, polygon_2d2[1], radius=point_radius, color=(0, 255, 255), thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))

        weighted_similarity = similarity_loss * 1
        # weighted_edge = edge_loss * (-1)
        total_loss = weighted_similarity

        return total_loss, [None, None, None]

    def debug_save(self, v_index):
        id_epoch = v_index + 1
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_arrow(v_edge_points, v_up_c):
            total_edge_points = v_edge_points

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + v_up_c

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
            colors[:, 0] = 1
            arrows.vertex_colors = o3d.utility.Vector3dVector(colors)
            return arrows

        id_patch = torch.tensor((self.id_viz_face,), dtype=torch.long, device=point_pos_c.device)
        # Visualize target patch
        edge_points = point_pos_c[self.batched_points_per_patch[id_patch]].reshape(-1, 4, 3)
        up_c = self.get_up_vector(np.arange(self.id_viz_edge, self.id_viz_edge+edge_points.shape[0]),
                                  edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "target_{}_arrow.obj".format(id_epoch)), arrows)
        id_points = np.asarray(self.batched_points_per_patch[id_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "target_{}_line.obj".format(id_epoch)), start_point_w, edge_index)

        # Visualize whole patch
        edge_points = point_pos_c[list(itertools.chain(*self.batched_points_per_patch))].reshape(-1, 4, 3)
        up_c = self.get_up_vector(np.arange(np.sum([len(item)//4 for item in self.batched_points_per_patch])),
                                  edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)), start_point_w, edge_index)
        pass

        return 0

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
        # self.model = LModel15(self.data, self.hydra_conf["trainer"]["loss_weight"],
        #                       self.hydra_conf["trainer"]["img_model"],
        #                       self.hydra_conf["trainer"]["output"],
        #                       self.hydra_conf["trainer"]["num_sample"])
        self.model = LModel18(self.data,
                              self.hydra_conf["dataset"]["id_viz_face"],
                              self.hydra_conf["dataset"]["id_viz_edge"],
                              self.hydra_conf["trainer"]["loss_weight"],
                              self.hydra_conf["trainer"]["img_model"],
                              self.hydra_conf["trainer"]["output"])
        # self.model = LModel31(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])
        # self.model = LModel12(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])

    def train_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_edge = self.hydra_conf["dataset"]["id_viz_edge"]
        self.train_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "training")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_edge = self.hydra_conf["dataset"]["id_viz_edge"]
        self.valid_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "validation")
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          num_workers=self.hydra_conf["trainer"]["num_worker"])

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": [self.model.seg_distance], 'lr': self.learning_rate},
            {"params": [self.model.v_up], 'lr': 1e-2},
        ]

        # optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )
        optimizer = SGD(grouped_parameters, lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        total_loss, losses = self.model(batch, self.current_epoch, False)

        self.log("Training_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, losses = self.model(batch, self.current_epoch if not self.trainer.sanity_checking else -1, True)

        self.log("Validation_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)

        return total_loss

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        self.model.debug_save(self.current_epoch if not self.trainer.sanity_checking else -1)

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


def prepare_dataset_and_model(v_colmap_dir, v_img_model_dir, v_viz_face):
    print("Start to prepare dataset")
    print("1. Read imgs")

    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name):
        print("Found cache ", img_cache_name)
        imgs, points_3d = np.load(img_cache_name, allow_pickle=True)
    else:
        print("Dosen't find cache, read raw img data")
        bound_min = np.array((-40, -40, -5))
        bound_max = np.array((130, 150, 60))
        bounds_center = (bound_min + bound_max) / 2
        bounds_size = (bound_max - bound_min).max()
        imgs, points_3d = read_dataset(v_colmap_dir,
                                       [bound_min,
                                        bound_max]
                                       )
        np.save(img_cache_name[:-4], np.asarray([imgs, points_3d], dtype=object))
        print("Save cache to ", img_cache_name)

    graph_cache_name = "output/img_field_test/graph_cache.npy"
    print("2. Build graph")
    if os.path.exists(graph_cache_name):
        graphs = np.load(graph_cache_name, allow_pickle=True)
    else:
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

            for id_face, id_edge_per_face in enumerate(faces):
                id_edge_per_face = (np.asarray(id_edge_per_face).astype(np.int32) - 1).tolist()
                new_faces.append(id_edge_per_face)
                id_edge_per_face = [(id_edge_per_face[idx], id_edge_per_face[idx + 1]) for idx in
                                    range(len(id_edge_per_face) - 1)] + [(id_edge_per_face[-1], id_edge_per_face[0])]
                graph.add_edges_from(id_edge_per_face)

            graph.graph["faces"] = new_faces
            print("Read {}/{} vertices".format(vertices.shape[0], len(graph.nodes)))
            print("Read {} faces".format(len(faces)))
            graphs.append(graph)
        graphs = np.asarray(graphs, dtype=object)
        np.save(graph_cache_name, graphs, allow_pickle=True)

    # Delete border line
    for id_graph in range(graphs.shape[0]):
        nodes_flag = np.asarray([
            graphs[id_graph].nodes[node]["pos_2d"][0] == 0 or graphs[id_graph].nodes[node]["pos_2d"][1] == 0 or
            graphs[id_graph].nodes[node]["pos_2d"][0] == 1 or graphs[id_graph].nodes[node]["pos_2d"][1] == 1
            for node in graphs[id_graph].nodes])
        removed_points = np.asarray(graphs[id_graph].nodes)[nodes_flag]
        graphs[id_graph].remove_nodes_from(removed_points)
        remap = dict(zip(np.asarray(graphs[id_graph].nodes), np.arange(graphs[id_graph].number_of_nodes())))
        graphs[id_graph] = nx.relabel_nodes(graphs[id_graph], remap)
        face_flag = []
        for idx, face in enumerate(graphs[id_graph].graph["faces"]):
            is_remain = True
            for point in face:
                if point in removed_points:
                    is_remain = False
                    break
            if is_remain:
                graphs[id_graph].graph["faces"][idx] = [remap[item] for item in graphs[id_graph].graph["faces"][idx]]
            face_flag.append(is_remain)
        graphs[id_graph].graph["faces"] = np.asarray(graphs[id_graph].graph["faces"], dtype=object)[face_flag].tolist()

    graphs[0].graph["face_center"] = np.zeros((len(graphs[0].graph["faces"]), 2), dtype=np.float32)
    graphs[0].graph["ray_c"] = np.zeros((len(graphs[0].graph["faces"]), 3), dtype=np.float32)
    for id_face, id_edge_per_face in enumerate(graphs[0].graph["faces"]):
        # Convex assumption
        center_point = np.stack(
            [graphs[0].nodes[id_vertex]["pos_2d"] for id_vertex in id_edge_per_face], axis=0).mean(axis=0)
        graphs[0].graph["face_center"][id_face] = center_point

    points_cache_name = "output/img_field_test/points_cache.npy"
    if os.path.exists(points_cache_name):
        points_from_sfm = np.load(points_cache_name)
    else:
        preserved_points = []
        for point in tqdm(points_3d):
            for track in point.tracks:
                if track[0] in [1, 2]:
                    preserved_points.append(point)
        points_from_sfm = np.stack([item.pos for item in preserved_points])
        np.save(points_cache_name, points_from_sfm)

    imgs = imgs[1:3]

    img1 = imgs[0]
    img2 = imgs[1]
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

    # rgb1 = cv2.resize(rgb1, (1500,1000),interpolation=cv2.INTER_AREA)
    # rgb2 = cv2.resize(rgb2, (1500,1000),interpolation=cv2.INTER_AREA)

    rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)[:, :, None]
    rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)[:, :, None]

    shape = rgb1.shape[:2][::-1]

    id_patch = v_viz_face
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

    if True:
        draw_initial()

    print("5. Load img models")

    # img_model_root_dir = r"output/neural_recon/img_nif_log"

    def load_img_model2():
        checkpoint_name = os.path.join(v_img_model_dir, "union.ckpt")

        if os.path.exists(checkpoint_name):
            img_model = Siren2({img.img_name: img.img_path for img in imgs})
            state_dict = torch.load(checkpoint_name)["state_dict"]
            img_model.load_state_dict({item[6:]: state_dict[item] for item in state_dict}, strict=True)
            img_model.eval()
            return img_model
        else:
            print("cannot find model {}".format(checkpoint_name))
            raise

    def load_img_model1(img_name):
        checkpoint_name = os.path.join(v_img_model_dir, img_name + ".ckpt")

        if os.path.exists(checkpoint_name):
            img_model = NGPModel()
            # img_model = Siren2()
            # state_dict = torch.load(os.path.join(img_model_root_dir, img_name, checkpoint_name[0]))
            state_dict = torch.load(checkpoint_name)["state_dict"]
            img_model.load_state_dict({item[6:]: state_dict[item] for item in state_dict}, strict=True)
            # img_model.net = img_model.net[:-1]
            img_model.eval()
            return img_model
        else:
            print("cannot find model for img {}".format(img_name))
            raise

    img_model1 = load_img_model1(img1.img_name)
    img_model2 = load_img_model1(img2.img_name)

    print("6. Compute initial line clouds")

    def compute_initial():
        distance_threshold = 5  # 5m; not used

        # Query points: (M, 2)
        # points from sfm: (N, 2)
        kd_tree = faiss.IndexFlatL2(2)
        kd_tree.add(points_from_sfm_2d.astype(np.float32))
        vertices_2d = np.asarray([graph1.nodes[id_node]["pos_2d"] for id_node in graph1.nodes()])  # (M, 2)
        centroids_2d = graph1.graph["face_center"]
        query_points = np.concatenate([vertices_2d, centroids_2d], axis=0)
        shortest_distance, index_shortest_distance = kd_tree.search(query_points, 32)  # (M, K)

        points_from_sfm_camera = (img1.extrinsic @ np.insert(points_from_sfm, 3, 1, axis=1).T).T[:, :3]  # (N, 3)

        # Select the point which is nearest to the actual ray for each endpoints
        # 1. Construct the ray
        ray_c = (np.linalg.inv(img1.intrinsic) @ np.insert(query_points, 2, 1,
                                                           axis=1).T).T  # (M, 2); points in camera coordinates
        ray_c = ray_c / np.linalg.norm(ray_c + 1e-6, axis=1, keepdims=True)  # Normalize the points
        nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
        # Compute the shortest distance from the candidate point to the ray for each query point
        distance_of_projection = nearest_candidates @ ray_c[:, :,
                                                      np.newaxis]  # (M, K, 1): K projected distance of the candidate point along each ray
        projected_points_on_ray = distance_of_projection * ray_c[:, np.newaxis,
                                                           :]  # (M, K, 3): K projected points along the ray
        distance_from_candidate_points_to_ray = np.linalg.norm(nearest_candidates - projected_points_on_ray + 1e-6,
                                                               axis=2)  # (M, 1)
        index_best_projected = distance_from_candidate_points_to_ray.argmin(
            axis=1)  # (M, 1): Index of the best projected points along the ray

        chosen_distances = distance_of_projection[np.arange(projected_points_on_ray.shape[0]), index_best_projected]
        valid_mask = distance_from_candidate_points_to_ray[np.arange(
            projected_points_on_ray.shape[0]), index_best_projected] < distance_threshold  # (M, 1)
        initial_points_camera = projected_points_on_ray[np.arange(projected_points_on_ray.shape[
                                                                      0]), index_best_projected]  # (M, 3): The best projected points along the ray
        initial_points_world = (np.linalg.inv(img1.extrinsic) @ np.insert(initial_points_camera, 3, 1, axis=1).T).T
        initial_points_world = initial_points_world[:, :3] / initial_points_world[:, 3:4]

        distance_scale = (distance_of_projection.max(axis=1) - distance_of_projection.min(axis=1))

        for idx, id_node in enumerate(graph1.nodes):
            graph1.nodes[id_node]["pos_world"] = initial_points_world[idx]
            graph1.nodes[id_node]["distance"] = chosen_distances[idx, 0]
            graph1.nodes[id_node]["ray_c"] = ray_c[idx]
            graph1.nodes[id_node]["scale"] = distance_scale[idx, 0]

        for id_face in range(graph1.graph["face_center"].shape[0]):
            idx = id_face + len(graph1.nodes)
            # graph1.graph["face_center"]["pos_world"][id_face] = initial_points_world[idx]
            # graph1.graph["face_center"]["distance"][id_face] = chosen_distances[idx, 0]
            graph1.graph["ray_c"][id_face] = ray_c[idx]

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
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(initial_points_world[len(graph1.nodes):])
        o3d.io.write_point_cloud("output/img_field_test/initial_face_centroid.ply", pc)
        return

    compute_initial()  # Coordinate of the initial segments in world coordinate
    if False:
        distances_cache_file = "output/img_field_test/distances_cache.npy"
        if os.path.exists(distances_cache_file):
            distances = np.concatenate(np.load(distances_cache_file, allow_pickle=True))
        else:
            ray.init(
                dashboard_port=15002,
                dashboard_host="0.0.0.0"
            )
            num_edges = len(graph1.edges())
            num_clusters = 7
            step = (num_edges // num_clusters) + 1
            ray_c = np.asarray(
                [(graph1.nodes[id_node1]["ray_c"], graph1.nodes[id_node2]["ray_c"]) for id_node1, id_node2 in
                 graph1.edges()])
            futures = [compute_init_based_on_similarity.remote(
                ray_c[i * step: min((i + 1) * step, num_edges)],
                img1.intrinsic, img2.intrinsic,
                img1.extrinsic, img2.extrinsic,
                img_model1, img_model2,
                rgb1, rgb2
            ) for i in range(num_clusters)]
            distances = np.asarray(ray.get(futures))
            np.save(distances_cache_file, distances)
        distance_candidate = [[] for _ in graph1.nodes()]
        for id_edge, edge in enumerate(tqdm(graph1.edges())):
            distance_candidate[edge[0]].append(distances[id_edge * 2 + 0])
            distance_candidate[edge[1]].append(distances[id_edge * 2 + 1])

        for id_node in graph1.nodes():
            if distance_candidate[id_node][0] != -1:
                graph1.nodes[id_node]["distance"] = distance_candidate[id_node][0]
                # graph1.nodes[id_node]["distance"] = np.mean(distance_candidate[id_node])
                pos_c = (graph1.nodes[id_node]["ray_c"] * graph1.nodes[id_node]["distance"])
                pos_world = np.linalg.inv(img1.extrinsic) @ to_homogeneous_vector(pos_c)
                graph1.nodes[id_node]["pos_world"] = pos_world[:3]

        line_coordinates = []
        for edge in graph1.edges():
            line_coordinates.append(
                np.concatenate((graph1.nodes[edge[0]]["pos_world"], graph1.nodes[edge[1]]["pos_world"])))
        save_line_cloud("output/img_field_test/initial_segments_after_rectify.obj", np.stack(line_coordinates, axis=0))

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

    print("8. Another idea comes, I must leave hhh. Take care")
    if True:
        for id_edge_per_face, id_edge_per_face in enumerate(graph1.graph["faces"]):
            pass

    print("9. Calculate initial normal")
    # compute_initial_normal_based_on_pos(graph1)
    compute_initial_normal_based_on_camera(points_pos_3d_c, graph1)

    print("10. Visualize target patch normal")
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

    # cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    # cv2.moveWindow("1",0,0)
    # cv2.resizeWindow("1", 1600, 900)
    #
    # mask_img = np.zeros_like(img1.line_field[:,:,0])
    # mask_img[img1.line_field[:,:,2] > 0.005] = 1 # 0.005 -> gradient > 1
    # from matplotlib import pyplot as plt
    # plt.imshow(mask_img)
    # plt.show()

    data = {
        "graph1": graph1,
        "graph2": graph2,
        "intrinsic1": img1.intrinsic,
        "extrinsic1": img1.extrinsic,
        "intrinsic2": img2.intrinsic,
        "extrinsic2": img2.extrinsic,
        "img_model1": img_model1,
        "img_model2": img_model2,
        "rgb1": rgb1,
        "rgb2": rgb2,
        "edge_field1": img1.line_field,
        "edge_field2": img2.line_field,
    }

    return data


@hydra.main(config_name="phase3_l7.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))
    data = prepare_dataset_and_model(v_cfg["dataset"]["colmap_dir"], v_cfg["model"]["img_model_dir"],
                                     v_cfg["dataset"]["id_viz_face"])

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

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
