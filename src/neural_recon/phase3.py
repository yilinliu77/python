import itertools
import sys, os
import time

from torch.distributions import Binomial

from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1

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
import torchsort

import mcubes
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


class Singel_node_dataset(torch.utils.data.Dataset):
    def __init__(self, v_only_train_target, v_id_batched_points, v_batched_total_points, v_id_target_face,
                 v_training_mode):
        super(Singel_node_dataset, self).__init__()
        self.only_train_target = v_only_train_target
        self.training_mode = v_training_mode

        id_target_points = np.unique(np.concatenate([v_id_batched_points[item][::4] for item in v_id_target_face]))
        self.validation_data = np.where([points[0, 0] in id_target_points for points in v_batched_total_points])[0]
        if self.training_mode == "validation" or self.only_train_target:
            self.length = len(self.validation_data)
        else:
            self.length = len(v_batched_total_points)

    def __getitem__(self, index):
        if self.training_mode == "validation" or self.only_train_target:
            return self.validation_data[index]
        else:
            return index

    def __len__(self):
        return self.length


def offline_optimize(model, v_batch_size=500):
    model.cuda()
    with torch.no_grad():
        num_step = model.id_point_to_id_up_and_face.shape[0] // v_batch_size + 1
        idxs = np.arange(model.id_point_to_id_up_and_face.shape[0])
        np.random.shuffle(idxs)
        idid = torch.tensor([item[0, 0] for item in model.id_point_to_id_up_and_face],
                            dtype=torch.long, device=model.ray_c.device)

        loss = []
        for id_batch in range(num_step):
            idx = idxs[
                  id_batch * v_batch_size:min(model.id_point_to_id_up_and_face.shape[0], (id_batch + 1) * v_batch_size)]
            id_points = torch.from_numpy(
                np.concatenate(model.id_point_to_id_up_and_face[idx], axis=0)
            ).to(torch.long).cuda()
            id_length = torch.from_numpy(
                model.id_point_to_id_up_and_face_length[idx]
            ).to(torch.long).cuda()

            id_start_point = id_points[:, 0]
            id_end_point = id_points[:, 1]
            id_up = id_points[:, 2:4]
            id_face = id_points[:, 4:6]
            start_rays = model.ray_c[id_start_point]
            end_rays = model.ray_c[id_end_point]
            start_points = model.seg_distance[id_start_point][:, None] * model.seg_distance_normalizer * start_rays
            end_points = model.seg_distance[id_end_point][:, None] * model.seg_distance_normalizer * end_rays
            v_ups = model.get_up_vector2(id_up, start_points, end_points)
            centroid_ray1 = model.center_ray_c[id_face[:, 0]]
            centroid_ray2 = model.center_ray_c[id_face[:, 1]]
            mask1 = id_face[:, 0] != -1
            mask2 = id_face[:, 1] != -1

            similarity_loss1, batched_mask1, similarity_mask1, _ = model.compute_similarity_wrapper(
                start_rays, end_rays, model.seg_distance[id_start_point], model.seg_distance[id_end_point],
                v_ups, centroid_ray1, centroid_ray2, mask1, mask2
            )
            id_mask = torch.arange(id_length.shape[0],
                                   device=id_length.device).repeat_interleave(id_length, dim=0).repeat(2)
            similarity_loss1 = scatter_mean(similarity_loss1, id_mask, dim=0)
            batched_mask1 = (scatter_min(batched_mask1.to(torch.long), id_mask)[0]).to(torch.bool)
            similarity_mask1 = (scatter_min(similarity_mask1.to(torch.long), id_mask)[0]).to(torch.bool)
            similarity_loss1[~batched_mask1] = torch.inf
            similarity_loss1[~similarity_mask1] = torch.inf
            loss.append(similarity_loss1)
        loss = torch.cat(loss, dim=0)
        loss1 = loss.clone()
        loss1[idxs] = loss

        tqdmrange = trange(1000000)
        for iter in tqdmrange:
            loss = []
            new_distance = -torch.ones_like(model.seg_distance)
            while not torch.all(torch.logical_and(new_distance > 0, new_distance < 1)):
                a = model.seg_distance + model.scale \
                    * torch.distributions.utils._standard_normal(1,
                                                                 device=model.seg_distance.device,
                                                                 dtype=model.seg_distance.dtype)
                m_ = torch.logical_and(a > 0, a < 1)
                new_distance[m_] = a[m_]
            for id_batch in range(num_step):
                idx = idxs[id_batch * v_batch_size:min(model.id_point_to_id_up_and_face.shape[0],
                                                       (id_batch + 1) * v_batch_size)]
                id_points = torch.from_numpy(
                    np.concatenate(model.id_point_to_id_up_and_face[idx], axis=0)
                ).to(torch.long).cuda()
                id_length = torch.from_numpy(
                    model.id_point_to_id_up_and_face_length[idx]
                ).to(torch.long).cuda()

                id_start_point = id_points[:, 0]
                id_end_point = id_points[:, 1]
                id_up = id_points[:, 2:4]
                id_face = id_points[:, 4:6]
                start_rays = model.ray_c[id_start_point]
                end_rays = model.ray_c[id_end_point]
                start_points = model.seg_distance[id_start_point][:, None] * model.seg_distance_normalizer * start_rays
                end_points = model.seg_distance[id_end_point][:, None] * model.seg_distance_normalizer * end_rays
                v_ups = model.get_up_vector2(id_up, start_points, end_points)
                centroid_ray1 = model.center_ray_c[id_face[:, 0]]
                centroid_ray2 = model.center_ray_c[id_face[:, 1]]
                mask1 = id_face[:, 0] != -1
                mask2 = id_face[:, 1] != -1

                scale_factor = 1
                id_start = torch.cumsum(id_length, dim=0).roll(1)
                id_start[0] = 0
                new_distance_item = new_distance[id_start_point[id_start]]
                id_mask = torch.arange(id_length.shape[0],
                                       device=id_length.device).repeat_interleave(id_length, dim=0).repeat(2)
                similarity_loss2, batched_mask2, similarity_mask2, _ = model.compute_similarity_wrapper(
                    start_rays, end_rays, new_distance_item.repeat_interleave(id_length),
                    model.seg_distance[id_end_point],
                    v_ups, centroid_ray1, centroid_ray2, mask1, mask2
                )
                similarity_loss2 = scatter_mean(similarity_loss2, id_mask, dim=0)
                batched_mask2 = (scatter_min(batched_mask2.to(torch.long), id_mask)[0]).to(torch.bool)
                similarity_mask2 = (scatter_min(similarity_mask2.to(torch.long), id_mask)[0]).to(torch.bool)
                similarity_loss2[~batched_mask2] = torch.inf
                similarity_loss2[~similarity_mask2] = torch.inf
                loss.append(similarity_loss2)

            loss = torch.cat(loss, dim=0)
            loss2 = loss.clone()
            loss2[idxs] = loss

            succeed_mask = loss2 < loss1
            model.seg_distance.data[idid[succeed_mask]] = new_distance[idid[succeed_mask]]
            loss1[succeed_mask] = loss2[succeed_mask]
            cur_loss = loss1[~torch.isinf(loss1)].mean()
            num_inf = torch.isinf(loss1).sum()
            tqdmrange.set_description('ML (loss={:.4f}, num_inf={})'.format(cur_loss, num_inf))
            np.random.shuffle(idxs)


class LModel19(nn.Module):
    def __init__(self, v_data, v_is_regress_normal, v_viz_patch, v_viz_edge, v_weights, v_img_method, v_log_root):
        super(LModel19, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method
        self.log_root = v_log_root
        self.is_regress_normal = v_is_regress_normal

        self.init_regular_variables(v_data)
        self.register_variables(v_data)
        self.register_img_model(v_data)
        self.calculate_index()
        self.register_distances(True)
        self.register_up_vectors2(v_is_regress_normal)

        # Debug
        self.id_viz_face = v_viz_patch
        self.id_viz_edge = v_viz_edge

        # Accurate initialization in patch 1476
        # id_vertices = np.asarray(self.batched_points_per_patch[1476]).reshape(-1, 4)[:, 0]
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

    def register_up_vectors2(self, train_up_vectors):
        num_total_edge = np.sum([len(face) for face in self.graph1.graph["faces"]])

        id_edge_to_id_face = torch.zeros((num_total_edge, 2), dtype=torch.long)
        self.id_edge_to_id_up_dict = {}  # (id_start_point, id_end_point) -> id_up
        self.id_edge_to_id_face_dict = {}  # (id_start_point, id_end_point) -> id_face
        v_up = []

        # These three variables are used to initialize the up vector
        all_edge_points = self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer
        start_points = []
        end_points = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for idx in range(len(face_ids)):
                id_start = face_ids[idx]
                id_end = face_ids[(idx + 1) % len(face_ids)]
                id_prev = face_ids[(idx + 2) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
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

                self.id_edge_to_id_face_dict[(id_start, id_end)] = id_patch

        id_point_to_id_up_and_face = []
        for id_start_node in self.graph1.nodes():
            data = []
            for id_end_node in self.graph1[id_start_node]:
                if len(self.graph1[id_start_node][id_end_node]) == 0:
                    continue
                data.append(id_start_node)
                data.append(id_end_node)
                data.append(self.id_edge_to_id_up_dict[(id_start_node, id_end_node)])
                data.append(self.id_edge_to_id_up_dict[(id_end_node, id_start_node)])
                if (id_start_node, id_end_node) not in self.id_edge_to_id_face_dict:
                    data.append(-1)
                else:
                    data.append(self.id_edge_to_id_face_dict[(id_start_node, id_end_node)])
                if (id_end_node, id_start_node) not in self.id_edge_to_id_face_dict:
                    data.append(-1)
                else:
                    data.append(self.id_edge_to_id_face_dict[(id_end_node, id_start_node)])
            if len(data) == 0:
                continue
            id_point_to_id_up_and_face.append(data)

        self.id_point_to_id_up_and_face = np.asarray(
            [np.asarray(item).reshape(-1, 6) for item in id_point_to_id_up_and_face], dtype=object)
        self.id_point_to_id_up_and_face_length = np.asarray([len(item) for item in self.id_point_to_id_up_and_face])

        v_up = torch.stack(v_up, dim=0)
        start_points = torch.stack(start_points, dim=0)
        end_points = torch.stack(end_points, dim=0)

        v_normal = torch.cross(end_points - start_points, v_up)
        self.v_up = nn.Parameter(v_normal, requires_grad=train_up_vectors)
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
                                                        # torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((d1,), dim=1).reshape(-1, 3),
                                                        # torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1,), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        # num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)
        num_edge_points = num_edge_points.reshape(-1, 1).sum(dim=1)

        if not self.is_regress_normal:
            return num_edge_points, edge_points  # Debug only

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

    def sample_points_based_on_up_and_centroid(self, start_point, end_point, edge_up_c, v_centroid_ray):
        device = self.ray_c.device
        times = [0 for _ in range(10)]
        cur_time = time.time()

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir + 1e-6, dim=1)
        cur_dir = cur_dir / cur_length[:, None]

        # Method 1
        # https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
        v_perpendicular = normalize_tensor(torch.cross(v_centroid_ray, cur_dir))
        n2 = normalize_tensor(torch.cross(cur_dir, v_perpendicular))
        intersection_t = (start_point * n2).sum(dim=1) / ((v_centroid_ray * n2).sum(dim=1) + 1e-6)
        intersection_point = intersection_t[:, None] * v_centroid_ray
        # Check if the intersection point lies on the outside of the segment
        inside_flag1 = (cur_dir * normalize_tensor(intersection_point - start_point)).sum(dim=1) < 0
        inside_flag2 = (-cur_dir * normalize_tensor(intersection_point - end_point)).sum(dim=1) < 0
        intersection_point[inside_flag1] = (end_point[inside_flag1] * v_centroid_ray[inside_flag1]) \
                                               .sum(dim=1, keepdims=True) * v_centroid_ray[inside_flag1]
        intersection_point[inside_flag2] = (start_point[inside_flag2] * v_centroid_ray[inside_flag2]) \
                                               .sum(dim=1, keepdims=True) * v_centroid_ray[inside_flag2]

        # Method 2
        # plane_normal = torch.cross(cur_dir, edge_up_c)
        # plane_normal = normalize_tensor(plane_normal)
        # intersection_t = (start_point * plane_normal).sum(dim=1) / (v_centroid_ray * plane_normal + 1e-6).sum(dim=1)
        # intersection_point = intersection_t[:, None] * v_centroid_ray
        times[0] += time.time() - cur_time
        cur_time = time.time()

        is_log = False
        if is_log:
            for i in range(start_point.shape[0]):
                with open("output/1.obj", "w") as f:
                    s = start_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    s = end_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    f.write("l 1 2\n")
                    f.write("v 0 0 0\n")
                    s = v_centroid_ray[i].detach().cpu().numpy() * 300
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    f.write("l 3 4\n")
                    s = end_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    s = intersection_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    f.write("l 5 6\n")
                    s = start_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    s = intersection_point[i].detach().cpu().numpy()
                    f.write("v {} {} {}\n".format(s[0], s[1], s[2]))
                    f.write("l 7 8\n")

        num_samples, samples = self.sample_triangles(100, start_point, end_point, intersection_point)
        times[1] += time.time() - cur_time
        cur_time = time.time()

        return num_samples, samples

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
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

    def get_up_vector1(self, v_id_edges, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t1 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 0]]
        t2 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 1]]

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

    def get_up_vector2(self, v_id_edges, start_points, end_points):
        t1 = self.v_up[v_id_edges[:, 0].reshape(-1)]
        t2 = self.v_up[v_id_edges[:, 1].reshape(-1)]

        v_up1 = torch.cross(end_points - start_points, t1)
        v_up2 = torch.cross(start_points - end_points, t2)
        v_up = torch.stack([v_up1, v_up2], dim=0)

        return (v_up / torch.linalg.norm(v_up.detach(), dim=-1, keepdim=True)).reshape(2, -1, 3).permute(1, 0, 2)

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

    def compute_similarity_wrapper(self, start_rays, end_rays, start_distance, end_distances, v_ups,
                                   v_centroid1_rays1, v_centroid1_rays2, v_mask1, v_mask2):
        times = [0 for _ in range(10)]
        cur_time = time.time()
        start_points = start_distance[:, None] * self.seg_distance_normalizer * start_rays
        end_points = end_distances[:, None] * self.seg_distance_normalizer * end_rays
        batched_start_points = torch.cat((start_points, end_points), dim=0)
        batched_end_points = torch.cat((end_points, start_points), dim=0)
        batched_up = v_ups.permute(1, 0, 2).reshape(-1, 3)
        batched_centroid = torch.cat((v_centroid1_rays1, v_centroid1_rays2), dim=0)
        batched_mask = torch.cat((v_mask1, v_mask2), dim=0)
        times[0] += time.time() - cur_time
        cur_time = time.time()

        num_coors_per_edge, coords_per_edge = self.sample_points_based_on_up_and_centroid(batched_start_points,
                                                                                          batched_end_points,
                                                                                          batched_up,
                                                                                          batched_centroid)
        times[1] += time.time() - cur_time
        cur_time = time.time()
        similarity_loss, similarity_mask = self.compute_similarity(coords_per_edge, self.intrinsic1,
                                                                   self.transformation)
        times[2] += time.time() - cur_time
        cur_time = time.time()
        similarity_loss = scatter_add(similarity_loss, torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss.device).repeat_interleave(num_coors_per_edge),
                                      dim=0)
        similarity_loss = (similarity_loss / num_coors_per_edge[:, None]).mean(dim=1)
        similarity_mask = scatter_min(similarity_mask.to(torch.long), torch.arange(
            num_coors_per_edge.shape[0], device=similarity_loss.device).repeat_interleave(num_coors_per_edge),
                                      dim=0)[0]
        times[3] += time.time() - cur_time
        cur_time = time.time()
        return similarity_loss, batched_mask, similarity_mask.to(torch.bool), coords_per_edge

    def random_search(self, start_rays, end_rays, id_start_point, id_end_point, v_ups, centroid_ray1, centroid_ray2,
                      mask1, mask2,  # The mask indicates that centroids are not valid. We won't consider the loss then
                      v_scale, id_length):
        similarity_loss1, batched_mask1, similarity_mask1, _ = self.compute_similarity_wrapper(
            start_rays, end_rays, self.seg_distance[id_start_point], self.seg_distance[id_end_point],
            v_ups, centroid_ray1, centroid_ray2, mask1, mask2
        )
        id_mask = torch.arange(id_length.shape[0],
                               device=id_length.device).repeat_interleave(id_length, dim=0).repeat(2)
        similarity_loss1 = scatter_mean(similarity_loss1, id_mask, dim=0)
        batched_mask1 = (scatter_min(batched_mask1.to(torch.long), id_mask)[0]).to(torch.bool)
        similarity_mask1 = (scatter_min(similarity_mask1.to(torch.long), id_mask)[0]).to(torch.bool)
        similarity_loss1[~batched_mask1] = torch.inf
        similarity_loss1[~similarity_mask1] = torch.inf

        scale_factor = 0.16
        id_start = torch.cumsum(id_length, dim=0)
        id_start = torch.roll(id_start, 1)
        id_start[0] = 0

        # id_points_unique = torch.unique(torch.concatenate([id_start_point,id_end_point]))
        # new_distance = -torch.ones_like(self.seg_distance[id_points_unique])
        # while not torch.all(torch.logical_and(new_distance > 0, new_distance < 1)):
        #     a = self.seg_distance[id_points_unique] + scale_factor \
        #         * torch.distributions.utils._standard_normal(new_distance.shape[0],
        #                                                      device=self.seg_distance.device,
        #                                                      dtype=self.seg_distance.dtype)
        #     m_ = torch.logical_and(a > 0, a < 1)
        #     new_distance[m_] = a[m_]
        # new_distance_ = torch.zeros_like(self.seg_distance)
        # new_distance_[id_points_unique] = new_distance
        #
        # similarity_loss2, batched_mask2, similarity_mask2, _ = self.compute_similarity_wrapper(
        #     start_rays, end_rays, new_distance_[id_start_point], new_distance_[id_end_point],
        #     v_ups, centroid_ray1, centroid_ray2, mask1, mask2
        # )

        new_distance = -torch.ones_like(self.seg_distance[id_start_point[id_start]])
        while not torch.all(torch.logical_and(new_distance > 0, new_distance < 1)):
            a = self.seg_distance[id_start_point[id_start]] + scale_factor \
                * torch.distributions.utils._standard_normal(v_scale[id_start].shape[0],
                                                             device=self.seg_distance.device,
                                                             dtype=self.seg_distance.dtype)
            m_ = torch.logical_and(a > 0, a < 1)
            new_distance[m_] = a[m_]

        similarity_loss2, batched_mask2, similarity_mask2, _ = self.compute_similarity_wrapper(
            start_rays, end_rays, new_distance.repeat_interleave(id_length), self.seg_distance[id_end_point],
            v_ups, centroid_ray1, centroid_ray2, mask1, mask2
        )

        similarity_loss2 = scatter_mean(similarity_loss2, id_mask, dim=0)
        batched_mask2 = (scatter_min(batched_mask2.to(torch.long), id_mask)[0]).to(torch.bool)
        similarity_mask2 = (scatter_min(similarity_mask2.to(torch.long), id_mask)[0]).to(torch.bool)
        similarity_loss2[~batched_mask2] = torch.inf
        similarity_loss2[~similarity_mask2] = torch.inf

        succeed_mask = similarity_loss2 < similarity_loss1
        self.seg_distance.data[id_start_point[id_start[succeed_mask]]] = new_distance[succeed_mask]
        # succeed_mask = succeed_mask.repeat_interleave(id_length)
        # self.seg_distance.data[id_end_point[succeed_mask]] = new_distance_[id_end_point[succeed_mask]]

    def forward(self, idxs, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1
        times = [0 for _ in range(10)]
        cur_time = time.time()

        # id_points = torch.from_numpy(np.concatenate([self.id_point_to_id_up_and_face[idx] for idx in idxs], axis=0),
        #                              ).to(device=idxs.device).to(torch.long)
        index = idxs.cpu().numpy()
        id_points = torch.from_numpy(
            np.concatenate(self.id_point_to_id_up_and_face[index], axis=0)
        ).to(device=idxs.device).to(torch.long)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        id_length = torch.from_numpy(
            self.id_point_to_id_up_and_face_length[index]
        ).to(device=idxs.device).to(torch.long)
        times[1] += time.time() - cur_time
        cur_time = time.time()

        id_start_point = id_points[:, 0]
        id_end_point = id_points[:, 1]

        id_up = id_points[:, 2:4]
        id_face = id_points[:, 4:6]

        start_ray = self.ray_c[id_start_point]
        end_ray = self.ray_c[id_end_point]

        start_points = self.seg_distance[id_start_point][:, None] * self.seg_distance_normalizer * start_ray
        end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
        v_up = self.get_up_vector2(id_up, start_points, end_points)

        centroid_ray1 = self.center_ray_c[id_face[:, 0]]
        centroid_ray2 = self.center_ray_c[id_face[:, 1]]

        mask1 = id_face[:, 0] != -1
        mask2 = id_face[:, 1] != -1
        times[2] += time.time() - cur_time
        cur_time = time.time()
        # Random search
        if self.training:
            with torch.no_grad():
                for _ in range(100):
                    self.random_search(
                        start_ray, end_ray, id_start_point, id_end_point, v_up, centroid_ray1, centroid_ray2,
                        mask1, mask2,
                        self.scale[id_start_point], id_length
                    )
        times[3] += time.time() - cur_time
        cur_time = time.time()
        similarity_loss, batched_mask, similarity_mask, coords_per_edge = self.compute_similarity_wrapper(
            start_ray, end_ray, self.seg_distance[id_start_point], self.seg_distance[id_end_point],
            v_up, centroid_ray1, centroid_ray2, mask1, mask2
        )
        similarity_loss[~batched_mask] = 0
        similarity_loss[~similarity_mask] = 0
        times[4] += time.time() - cur_time
        cur_time = time.time()
        if is_log:
            with torch.no_grad():
                self.debug_face(self.id_viz_face, v_id_epoch)
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

                # Image 2
                line_img2 = self.rgb2.copy()
                line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                shape = line_img2.shape[:2][::-1]

                roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                # cv2.imwrite(os.path.join(self.log_root, "3d_{:05d}.jpg".format(v_id_epoch)),
                #             np.concatenate((line_img1, line_img2), axis=1))

                start_points_1 = (self.intrinsic1 @ start_points.T).T
                start_points_1 = ((start_points_1[:, :2] / start_points_1[:, 2:3]).detach().cpu().numpy() * shape
                                  ).astype(np.int32)
                end_points_1 = (self.intrinsic1 @ end_points.T).T
                end_points_1 = ((end_points_1[:, :2] / end_points_1[:, 2:3]).detach().cpu().numpy() * shape
                                ).astype(np.int32)
                start_points_2 = (self.transformation @ to_homogeneous_tensor(start_points).T).T
                start_points_2 = ((start_points_2[:, :2] / start_points_2[:, 2:3]).detach().cpu().numpy() * shape
                                  ).astype(np.int32)
                end_points_2 = (self.transformation @ to_homogeneous_tensor(end_points).T).T
                end_points_2 = ((end_points_2[:, :2] / end_points_2[:, 2:3]).detach().cpu().numpy() * shape
                                ).astype(np.int32)

                # line_img1 = self.rgb1.copy()
                # line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
                # line_img2 = self.rgb2.copy()
                # line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                # for idx in range(start_points_1.shape[0]):
                #     cv2.line(line_img1, start_points_1[idx], end_points_1[idx],
                #              color=(0, 0, 255), thickness=line_thickness)
                #     cv2.line(line_img2, start_points_2[idx], end_points_2[idx],
                #              color=(0, 0, 255), thickness=line_thickness)
                for idx in range(start_points_1.shape[0]):
                    cv2.circle(line_img1, end_points_1[idx], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.circle(line_img2, end_points_2[idx], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)

                for idx in range(start_points_1.shape[0]):
                    cv2.circle(line_img1, start_points_1[idx], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(line_img2, start_points_2[idx], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "3d_{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=1))

        return torch.mean(similarity_loss), [None, None, None]

    def forwardb(self, id_points, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1

        similarity_losses = []
        for idx in id_points:
            id_start_point = self.id_point_to_id_up_and_face[idx][0, 0]
            id_end_point = self.id_point_to_id_up_and_face[idx][:, 1]

            id_up = self.id_point_to_id_up_and_face[idx][:, 2:4]
            id_face = self.id_point_to_id_up_and_face[idx][:, 4:6]

            start_ray = self.ray_c[id_start_point].repeat(id_end_point.shape[0]).reshape(id_end_point.shape[0], 3)
            end_ray = self.ray_c[id_end_point]

            start_points = self.seg_distance[id_start_point] * self.seg_distance_normalizer * start_ray
            end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
            v_up = self.get_up_vector2(id_up, start_points, end_points)

            centroid_ray1 = self.center_ray_c[id_face[:, 0]]
            centroid_ray2 = self.center_ray_c[id_face[:, 1]]

            mask1 = torch.tensor(id_face[:, 0] != -1, device=centroid_ray1.device)
            mask2 = torch.tensor(id_face[:, 1] != -1, device=centroid_ray1.device)
            # Random search
            if self.training:
                with torch.no_grad():
                    self.seg_distance.data[id_start_point] = self.random_search(
                        start_ray, end_ray, id_start_point, id_end_point, v_up, centroid_ray1, centroid_ray2, mask1,
                        mask2,
                        self.scale[id_start_point]
                    )
            similarity_loss, batched_mask, similarity_mask = self.compute_similarity_wrapper(
                start_ray, end_ray, self.seg_distance[id_start_point], self.seg_distance[id_end_point],
                v_up, centroid_ray1, centroid_ray2, mask1, mask2
            )
            # similarity_loss[~batched_mask] = 0
            # similarity_loss[~similarity_mask] = 0
            similarity_losses.append(similarity_loss.mean())

        return torch.mean(torch.stack(similarity_losses)), [None, None, None]

        if is_log and self.id_viz_edge in id_point:
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
                cv2.circle(line_img1, polygon_2d1[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img1, polygon_2d1[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

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
                cv2.circle(line_img2, polygon_2d2[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img2, polygon_2d2[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))
        return total_loss, [None, None, None]

    def debug_save(self, v_index):
        id_epoch = v_index + 1
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        id_points = torch.from_numpy(np.concatenate([
            self.id_point_to_id_up_and_face[idx] for idx in np.arange(len(self.id_point_to_id_up_and_face))], axis=0),
        ).to(device=seg_distance.device).to(torch.long)

        id_start_point = id_points[:, 0]
        id_end_point = id_points[:, 1]

        id_up = id_points[:, 2:4]
        id_face = id_points[:, 4:6]

        start_ray = self.ray_c[id_start_point]
        end_ray = self.ray_c[id_end_point]

        start_points = self.seg_distance[id_start_point][:, None] * self.seg_distance_normalizer * start_ray
        end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
        v_up = self.get_up_vector2(id_up, start_points, end_points)

        centroid_ray1 = self.center_ray_c[id_face[:, 0]]
        centroid_ray2 = self.center_ray_c[id_face[:, 1]]

        mask1 = id_face[:, 0] != -1
        mask2 = id_face[:, 1] != -1

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

        arrows = get_arrow(torch.stack((start_points, end_points), dim=1), v_up[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(start_points).T).T)[:, :3] \
            .cpu().numpy()
        end_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(end_points).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), np.arange(start_point_w.shape[0]) + start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)),
                      np.concatenate((start_point_w, end_point_w), axis=0), edge_index)
        return

    def debug_save_(self, v_index):
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
        up_c = self.get_up_vector2(np.arange(self.id_viz_edge, self.id_viz_edge + edge_points.shape[0]),
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
        return 0

        # Visualize whole patch
        edge_points = point_pos_c[list(itertools.chain(*self.batched_points_per_patch))].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(np.arange(np.sum([len(item) // 4 for item in self.batched_points_per_patch])),
                                   edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)), start_point_w, edge_index)
        pass

        return 0

    def debug_face(self, v_id_faces, v_id_epoch):
        line_thickness = 1
        point_thickness = 2
        point_radius = 1

        line_img1 = self.rgb1.copy()
        line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
        shape = line_img1.shape[:2][::-1]
        line_img2 = self.rgb2.copy()
        line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)

        for id_face in v_id_faces:
            id_points = self.batched_points_per_patch[id_face][::4]
            edges = np.concatenate([item for item in self.id_point_to_id_up_and_face if item[0, 0] in id_points])
            start_points_c = self.seg_distance[edges[:, 0]][:, None] * self.ray_c[edges[:, 0]] \
                             * self.seg_distance_normalizer
            end_points_c = self.seg_distance[edges[:, 1]][:, None] * self.ray_c[edges[:, 1]] \
                           * self.seg_distance_normalizer
            start_points_2d1 = (self.intrinsic1 @ start_points_c.T).T
            start_points_2d1 = ((start_points_2d1[:, :2] / start_points_2d1[:, 2:3]).detach().cpu().numpy()
                                * shape).astype(np.int32)
            end_points_2d1 = (self.intrinsic1 @ end_points_c.T).T
            end_points_2d1 = ((end_points_2d1[:, :2] / end_points_2d1[:, 2:3]).detach().cpu().numpy()
                              * shape).astype(np.int32)
            start_points_2d2 = (self.transformation @ to_homogeneous_tensor(start_points_c).T).T
            start_points_2d2 = ((start_points_2d2[:, :2] / start_points_2d2[:, 2:3]).detach().cpu().numpy()
                                * shape).astype(np.int32)
            end_points_2d2 = (self.transformation @ to_homogeneous_tensor(end_points_c).T).T
            end_points_2d2 = ((end_points_2d2[:, :2] / end_points_2d2[:, 2:3]).detach().cpu().numpy()
                              * shape).astype(np.int32)

            for idx in range(edges.shape[0]):
                cv2.line(line_img1, start_points_2d1[idx], end_points_2d1[idx], (255, 0, 0), thickness=line_thickness)
                cv2.line(line_img2, start_points_2d2[idx], end_points_2d2[idx], (255, 0, 0), thickness=line_thickness)
            for idx in range(edges.shape[0]):
                cv2.circle(line_img1, start_points_2d1[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img1, end_points_2d1[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img2, start_points_2d2[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img2, end_points_2d2[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
            for id_edge in range(edges.shape[0]):
                if edges[id_edge, 0] in id_points and edges[id_edge, 1] in id_points:
                    cv2.circle(line_img1, start_points_2d1[id_edge], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.circle(line_img2, start_points_2d2[id_edge], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.line(line_img1, start_points_2d1[id_edge], end_points_2d1[id_edge], (0, 0, 255),
                             thickness=line_thickness)
                    cv2.line(line_img2, start_points_2d2[id_edge], end_points_2d2[id_edge], (0, 0, 255),
                             thickness=line_thickness)

        cv2.imwrite(os.path.join(self.log_root, "2d_{:05d}.jpg".format(v_id_epoch)),
                    np.concatenate((line_img1, line_img2), axis=1))

    def len(self):
        return len(self.graph1.graph["faces"])


class LModel20(nn.Module):
    def __init__(self, v_data, v_is_regress_normal, v_viz_patch, v_viz_edge, v_weights, v_img_method, v_log_root):
        super(LModel20, self).__init__()
        self.loss_weights = v_weights
        self.img_method = v_img_method
        self.log_root = v_log_root
        self.is_regress_normal = v_is_regress_normal

        self.init_regular_variables(v_data)
        self.register_variables(v_data)
        self.register_img_model(v_data)
        self.calculate_index()
        self.register_distances(True)
        self.register_up_vectors2(v_is_regress_normal)

        # Debug
        self.id_viz_face = v_viz_patch
        self.id_viz_edge = v_viz_edge

        # Accurate initialization in patch 1476
        # id_vertices = np.asarray(self.batched_points_per_patch[1476]).reshape(-1, 4)[:, 0]
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

    def register_up_vectors2(self, train_up_vectors):
        num_total_edge = np.sum([len(face) for face in self.graph1.graph["faces"]])

        id_edge_to_id_face = torch.zeros((num_total_edge, 2), dtype=torch.long)
        self.id_edge_to_id_up_dict = {}  # (id_start_point, id_end_point) -> id_up
        self.id_edge_to_id_face_dict = {}  # (id_start_point, id_end_point) -> id_face
        v_up = []

        # These three variables are used to initialize the up vector
        all_edge_points = self.ray_c * self.seg_distance[:, None] * self.seg_distance_normalizer
        start_points = []
        end_points = []
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for idx in range(len(face_ids)):
                id_start = face_ids[idx]
                id_end = face_ids[(idx + 1) % len(face_ids)]
                id_prev = face_ids[(idx + 2) % len(face_ids)]

                up_c = torch.tensor(self.graph1.edges[(id_start, id_end)]["up_c"][id_patch], dtype=torch.float32)

                if (id_start, id_end) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_start, id_end)] = len(v_up)
                    v_up.append(up_c)
                    start_points.append(all_edge_points[id_start])
                    end_points.append(all_edge_points[id_end])

                if (id_end, id_start) not in self.id_edge_to_id_up_dict:
                    self.id_edge_to_id_up_dict[(id_end, id_start)] = len(v_up)
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

                self.id_edge_to_id_face_dict[(id_start, id_end)] = id_patch

        id_point_to_id_up_and_face = []
        for id_start_node in self.graph1.nodes():
            data = []
            for id_end_node in self.graph1[id_start_node]:
                if len(self.graph1[id_start_node][id_end_node]) == 0:
                    continue
                data.append(id_start_node)
                data.append(id_end_node)
                data.append(self.id_edge_to_id_up_dict[(id_start_node, id_end_node)])
                data.append(self.id_edge_to_id_up_dict[(id_end_node, id_start_node)])
                if (id_start_node, id_end_node) not in self.id_edge_to_id_face_dict:
                    data.append(-1)
                else:
                    data.append(self.id_edge_to_id_face_dict[(id_start_node, id_end_node)])
                if (id_end_node, id_start_node) not in self.id_edge_to_id_face_dict:
                    data.append(-1)
                else:
                    data.append(self.id_edge_to_id_face_dict[(id_end_node, id_start_node)])
            if len(data) == 0:
                continue
            id_point_to_id_up_and_face.append(data)

        self.id_point_to_id_up_and_face = np.asarray(
            [np.asarray(item).reshape(-1, 6) for item in id_point_to_id_up_and_face], dtype=object)
        self.id_point_to_id_up_and_face_length = np.asarray([len(item) for item in self.id_point_to_id_up_and_face])

        v_up = torch.stack(v_up, dim=0)
        start_points = torch.stack(start_points, dim=0)
        end_points = torch.stack(end_points, dim=0)

        v_normal = torch.cross(end_points - start_points, v_up)
        self.v_up = nn.Parameter(v_normal, requires_grad=train_up_vectors)
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
                                                        # torch.stack((d1, d2, p1 - p3), dim=1).reshape(-1, 3),
                                                        torch.stack((d1,), dim=1).reshape(-1, 3),
                                                        # torch.stack((p1, p2, p3), dim=1).reshape(-1, 3),
                                                        torch.stack((p1,), dim=1).reshape(-1, 3),
                                                        num_max_sample=num_max_sample)
        # num_edge_points = num_edge_points.reshape(-1, 3).sum(dim=1)
        num_edge_points = num_edge_points.reshape(-1, 1).sum(dim=1)

        if not self.is_regress_normal:
            return num_edge_points, edge_points  # Debug only

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

    def sample_points_2d(self, v_edge_points, v_num_horizontal):
        device = v_edge_points.device
        cur_dir = v_edge_points[:, 1] - v_edge_points[:, 0]
        cur_length = torch.linalg.norm(cur_dir, dim=-1) + 1e-6

        cur_dir_h = torch.cat((cur_dir, torch.zeros_like(cur_dir[:, 0:1])), dim=1)
        z_axis = torch.zeros_like(cur_dir_h)
        z_axis[:, 2] = 1
        edge_up = normalize_tensor(torch.cross(cur_dir_h, z_axis, dim=1)[:, :2]) * 0.00167
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

    def sample_edge(self, num_per_edge_m, cur_dir, start_point, num_max_sample=2000):
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

    def get_up_vector1(self, v_id_edges, start_points, end_points):
        # https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector
        t1 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 0]]
        t2 = self.v_up[self.id_edge_to_id_up[v_id_edges][:, 1]]

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

    def get_up_vector2(self, v_id_edges, start_points, end_points):
        t1 = self.v_up[v_id_edges[:, 0].reshape(-1)]
        t2 = self.v_up[v_id_edges[:, 1].reshape(-1)]

        v_up1 = torch.cross(end_points - start_points, t1)
        v_up2 = torch.cross(start_points - end_points, t2)
        v_up = torch.stack([v_up1, v_up2], dim=0)

        return (v_up / torch.linalg.norm(v_up.detach(), dim=-1, keepdim=True)).reshape(2, -1, 3).permute(1, 0, 2)

    def compute_similarity_wrapper(self, start_rays, end_rays, start_distance, end_distances):
        times = [0 for _ in range(10)]
        cur_time = time.time()
        start_points = start_distance[:, None] * self.seg_distance_normalizer * start_rays
        end_points = end_distances[:, None] * self.seg_distance_normalizer * end_rays
        points_c = torch.stack([start_points, end_points], dim=1).reshape(-1, 3)
        times[0] += time.time() - cur_time
        cur_time = time.time()

        edge_points = (self.intrinsic1 @ points_c.T).T
        edge_points = edge_points[:, :2] / (edge_points[:, 2:3] + 1e-6)
        edge_points = edge_points.reshape(-1, 2, 2)

        # sample step=0.01
        num_horizontal = torch.clamp((torch.linalg.norm(end_points - start_points, dim=-1) / 0.01).to(torch.long), 2,
                                     1000)

        num_per_edge1, points_2d1 = self.sample_points_2d(edge_points, num_horizontal)

        valid_mask1 = torch.logical_and(points_2d1 > 0, points_2d1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        points_2d1 = torch.clamp(points_2d1, 0, 0.999999)

        edge_points = (self.transformation @ to_homogeneous_tensor(points_c).T).T
        edge_points = edge_points[:, :2] / (edge_points[:, 2:3] + 1e-6)
        edge_points = edge_points.reshape(-1, 2, 2)

        num_per_edge2, points_2d2 = self.sample_points_2d(edge_points, num_horizontal)

        valid_mask2 = torch.logical_and(points_2d2 > 0, points_2d2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

        similarity_mask = torch.logical_and(valid_mask1, valid_mask2)

        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs1 = sample_img_prediction(self.img_model1, points_2d1[None, :, :])[0]
        else:
            sample_imgs1 = sample_img(self.o_rgb1, points_2d1[None, :, :])[0]

        # 4. Sample pixel color
        if self.img_method == "model":
            sample_imgs2 = sample_img_prediction(self.img_model2, points_2d2[None, :, :])[0]
        else:
            sample_imgs2 = sample_img(self.o_rgb2, points_2d2[None, :, :])[0]
        times[1] += time.time() - cur_time
        cur_time = time.time()

        # similarity_loss = nn.functional.l1_loss(sample_imgs1, sample_imgs2, reduction="none")
        # times[2] += time.time() - cur_time
        # cur_time = time.time()
        # similarity_loss = scatter_add(similarity_loss, torch.arange(
        #     num_per_edge1.shape[0], device=similarity_loss.device).repeat_interleave(num_per_edge1),dim=0)
        # similarity_loss = (similarity_loss / num_per_edge1[:, None]).mean(dim=1)

        similarity_loss = loss1(sample_imgs1, sample_imgs2, points_2d1, points_2d2, num_per_edge1)

        similarity_mask = scatter_min(similarity_mask.to(torch.long), torch.arange(
            num_per_edge1.shape[0], device=similarity_loss.device).repeat_interleave(num_per_edge1), dim=0)[0]
        times[3] += time.time() - cur_time
        cur_time = time.time()

        is_log = False
        if is_log:
            start_length = 0
            for idx, length in enumerate(num_per_edge1):
                img1 = (sample_imgs1[start_length:start_length + length].reshape(-1,
                                                                                 20).detach().cpu().numpy() * 255).astype(
                    np.uint8).T
                img2 = (sample_imgs2[start_length:start_length + length].reshape(-1,
                                                                                 20).detach().cpu().numpy() * 255).astype(
                    np.uint8).T
                cv2.imwrite(os.path.join(self.log_root, "{}.png".format(idx)),
                            np.concatenate((img1, np.zeros_like(img1[0:1, :]), img2), axis=0))
                start_length += length

        return similarity_loss, similarity_mask.to(torch.bool), similarity_mask.to(torch.bool), [points_2d1, points_2d2]

    def random_search(self, start_rays, end_rays, v_new_distances, v_old_distances,
                      id_length):
        batch_size = 10
        num_point = id_length.shape[0]
        num_edge = id_length.sum()
        num_sampled = v_new_distances.shape[1] # Sample from normal distribution + 1

        losses = []
        masks = []
        for id_batch in range(num_sampled//batch_size+1):
            id_batch_start = min(num_sampled, id_batch * batch_size)
            id_batch_end = min(num_sampled, (id_batch+1) * batch_size)
            if id_batch_start>=id_batch_end:
                continue
            num_batch = id_batch_end - id_batch_start
            similarity_loss, batched_mask, similarity_mask, _ = self.compute_similarity_wrapper(
                start_rays.repeat_interleave(num_batch,dim=0),
                end_rays.repeat_interleave(num_batch,dim=0),
                v_new_distances[:,id_batch_start:id_batch_end].repeat_interleave(id_length, dim=0).reshape(-1),
                v_old_distances.repeat_interleave(num_batch),
            )
            losses.append(similarity_loss.reshape(-1,num_batch).T)
            masks.append(torch.logical_and(batched_mask, similarity_mask).reshape(-1,num_batch).T)
        similarity_loss_ = torch.cat(losses,dim=0).permute(1,0)
        similarity_mask_ = torch.cat(masks,dim=0).permute(1,0)
        id_mask = torch.arange(
            num_point,device=id_length.device
        ).repeat_interleave(id_length, dim=0)
        similarity_loss = scatter_mean(similarity_loss_, id_mask, dim=0)
        similarity_mask = (scatter_min(similarity_mask_.to(torch.long), id_mask, dim=0)[0]).to(torch.bool)
        similarity_loss[~similarity_mask] = torch.inf

        return similarity_loss.argmin(dim=1)

    def forward(self, idxs, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1
        times = [0 for _ in range(10)]
        cur_time = time.time()

        # id_points = torch.from_numpy(np.concatenate([self.id_point_to_id_up_and_face[idx] for idx in idxs], axis=0),
        #                              ).to(device=idxs.device).to(torch.long)
        index = idxs.cpu().numpy()
        id_points = torch.from_numpy(
            np.concatenate(self.id_point_to_id_up_and_face[index], axis=0)
        ).to(device=idxs.device).to(torch.long)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        id_length = torch.from_numpy(
            self.id_point_to_id_up_and_face_length[index]
        ).to(device=idxs.device).to(torch.long)
        times[1] += time.time() - cur_time
        cur_time = time.time()

        # id_points=id_points[18:27]
        # id_length=id_length[6:9]

        id_start_point = id_points[:, 0]
        id_end_point = id_points[:, 1]

        id_up = id_points[:, 2:4]
        id_face = id_points[:, 4:6]

        start_ray = self.ray_c[id_start_point]
        end_ray = self.ray_c[id_end_point]

        start_points = self.seg_distance[id_start_point][:, None] * self.seg_distance_normalizer * start_ray
        end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
        v_up = self.get_up_vector2(id_up, start_points, end_points)

        centroid_ray1 = self.center_ray_c[id_face[:, 0]]
        centroid_ray2 = self.center_ray_c[id_face[:, 1]]

        mask1 = id_face[:, 0] != -1
        mask2 = id_face[:, 1] != -1
        times[2] += time.time() - cur_time
        cur_time = time.time()
        # Random search
        if self.training:
            with torch.no_grad():
                num_sample = 100
                id_start = torch.cumsum(id_length, dim=0)
                id_start = torch.roll(id_start, 1)
                id_start[0] = 0
                scale_factor = 0.16
                new_distance = -torch.ones((id_length.shape[0] * num_sample,),
                                           device=id_length.device, dtype=torch.float32)
                sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                sample_id = id_start_point[id_start].repeat_interleave(num_sample)
                while not torch.all(sample_distance_mask):
                    t_ = new_distance[~sample_distance_mask]
                    a = self.seg_distance[sample_id[~sample_distance_mask]] +\
                        scale_factor * torch.distributions.utils._standard_normal(
                        t_.shape[0],
                        device=self.seg_distance.device,
                        dtype=self.seg_distance.dtype)
                    new_distance[~sample_distance_mask] = a
                    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                new_distance = new_distance.reshape(-1, num_sample)
                new_distance = torch.cat((self.seg_distance[id_start_point[id_start]][:,None], new_distance), dim=1)
                id_best_distance = self.random_search(
                    start_ray, end_ray, new_distance, self.seg_distance[id_end_point],
                    id_length
                )
                self.seg_distance[id_start_point[id_start]] = new_distance[
                    torch.arange(new_distance.shape[0], dtype=torch.long, device=new_distance.device),
                    id_best_distance]
        times[3] += time.time() - cur_time
        cur_time = time.time()
        similarity_loss, batched_mask, similarity_mask, [points_2d1, points_2d2] = self.compute_similarity_wrapper(
            start_ray, end_ray, self.seg_distance[id_start_point], self.seg_distance[id_end_point]
        )
        similarity_loss[~batched_mask] = 0
        similarity_loss[~similarity_mask] = 0
        times[4] += time.time() - cur_time
        cur_time = time.time()
        if is_log:
            with torch.no_grad():
                self.debug_face(self.id_viz_face, v_id_epoch)
                line_thickness = 1
                point_thickness = 2
                point_radius = 1

                polygon_points_2d_1 = points_2d1.detach().cpu().numpy()
                polygon_points_2d_2 = points_2d2.detach().cpu().numpy()

                line_img1 = self.rgb1.copy()
                line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
                shape = line_img1.shape[:2][::-1]

                roi_coor_2d1_numpy = np.clip(polygon_points_2d_1, 0, 0.99999)
                viz_coords = (roi_coor_2d1_numpy * shape).astype(np.int32)
                line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                # Image 2
                line_img2 = self.rgb2.copy()
                line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                shape = line_img2.shape[:2][::-1]

                roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
                viz_coords = (roi_coor_2d2_numpy * shape).astype(np.int32)
                line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)

                # cv2.imwrite(os.path.join(self.log_root, "3d_{:05d}.jpg".format(v_id_epoch)),
                #             np.concatenate((line_img1, line_img2), axis=1))

                start_points_1 = (self.intrinsic1 @ start_points.T).T
                start_points_1 = ((start_points_1[:, :2] / start_points_1[:, 2:3]).detach().cpu().numpy() * shape
                                  ).astype(np.int32)
                end_points_1 = (self.intrinsic1 @ end_points.T).T
                end_points_1 = ((end_points_1[:, :2] / end_points_1[:, 2:3]).detach().cpu().numpy() * shape
                                ).astype(np.int32)
                start_points_2 = (self.transformation @ to_homogeneous_tensor(start_points).T).T
                start_points_2 = ((start_points_2[:, :2] / start_points_2[:, 2:3]).detach().cpu().numpy() * shape
                                  ).astype(np.int32)
                end_points_2 = (self.transformation @ to_homogeneous_tensor(end_points).T).T
                end_points_2 = ((end_points_2[:, :2] / end_points_2[:, 2:3]).detach().cpu().numpy() * shape
                                ).astype(np.int32)

                # line_img1 = self.rgb1.copy()
                # line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
                # line_img2 = self.rgb2.copy()
                # line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)
                # for idx in range(start_points_1.shape[0]):
                #     cv2.line(line_img1, start_points_1[idx], end_points_1[idx],
                #              color=(0, 0, 255), thickness=line_thickness)
                #     cv2.line(line_img2, start_points_2[idx], end_points_2[idx],
                #              color=(0, 0, 255), thickness=line_thickness)
                for idx in range(start_points_1.shape[0]):
                    cv2.circle(line_img1, end_points_1[idx], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.circle(line_img2, end_points_2[idx], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)

                for idx in range(start_points_1.shape[0]):
                    cv2.circle(line_img1, start_points_1[idx], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(line_img2, start_points_2[idx], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "3d_{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=1))

        return torch.mean(similarity_loss), [None, None, None]

    def forwardb(self, id_points, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1

        similarity_losses = []
        for idx in id_points:
            id_start_point = self.id_point_to_id_up_and_face[idx][0, 0]
            id_end_point = self.id_point_to_id_up_and_face[idx][:, 1]

            id_up = self.id_point_to_id_up_and_face[idx][:, 2:4]
            id_face = self.id_point_to_id_up_and_face[idx][:, 4:6]

            start_ray = self.ray_c[id_start_point].repeat(id_end_point.shape[0]).reshape(id_end_point.shape[0], 3)
            end_ray = self.ray_c[id_end_point]

            start_points = self.seg_distance[id_start_point] * self.seg_distance_normalizer * start_ray
            end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
            v_up = self.get_up_vector2(id_up, start_points, end_points)

            centroid_ray1 = self.center_ray_c[id_face[:, 0]]
            centroid_ray2 = self.center_ray_c[id_face[:, 1]]

            mask1 = torch.tensor(id_face[:, 0] != -1, device=centroid_ray1.device)
            mask2 = torch.tensor(id_face[:, 1] != -1, device=centroid_ray1.device)
            # Random search
            if self.training:
                with torch.no_grad():
                    self.seg_distance.data[id_start_point] = self.random_search(
                        start_ray, end_ray, id_start_point, id_end_point, v_up, centroid_ray1, centroid_ray2, mask1,
                        mask2,
                        self.scale[id_start_point]
                    )
            similarity_loss, batched_mask, similarity_mask = self.compute_similarity_wrapper(
                start_ray, end_ray, self.seg_distance[id_start_point], self.seg_distance[id_end_point],
                v_up, centroid_ray1, centroid_ray2, mask1, mask2
            )
            # similarity_loss[~batched_mask] = 0
            # similarity_loss[~similarity_mask] = 0
            similarity_losses.append(similarity_loss.mean())

        return torch.mean(torch.stack(similarity_losses)), [None, None, None]

        if is_log and self.id_viz_edge in id_point:
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
                cv2.circle(line_img1, polygon_2d1[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img1, polygon_2d1[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

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
                cv2.circle(line_img2, polygon_2d2[0], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)
                cv2.circle(line_img2, polygon_2d2[1], radius=point_radius, color=(0, 255, 255),
                           thickness=point_thickness)

                cv2.imwrite(os.path.join(self.log_root, "{:05d}.jpg".format(v_id_epoch)),
                            np.concatenate((line_img1, line_img2), axis=0))
        return total_loss, [None, None, None]

    def debug_save(self, v_index):
        id_epoch = v_index + 1
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        point_pos_c = self.ray_c * seg_distance[:, None]

        id_points = torch.from_numpy(np.concatenate([
            self.id_point_to_id_up_and_face[idx] for idx in np.arange(len(self.id_point_to_id_up_and_face))], axis=0),
        ).to(device=seg_distance.device).to(torch.long)

        id_start_point = id_points[:, 0]
        id_end_point = id_points[:, 1]

        id_up = id_points[:, 2:4]
        id_face = id_points[:, 4:6]

        start_ray = self.ray_c[id_start_point]
        end_ray = self.ray_c[id_end_point]

        start_points = self.seg_distance[id_start_point][:, None] * self.seg_distance_normalizer * start_ray
        end_points = self.seg_distance[id_end_point][:, None] * self.seg_distance_normalizer * end_ray
        v_up = self.get_up_vector2(id_up, start_points, end_points)

        centroid_ray1 = self.center_ray_c[id_face[:, 0]]
        centroid_ray2 = self.center_ray_c[id_face[:, 1]]

        mask1 = id_face[:, 0] != -1
        mask2 = id_face[:, 1] != -1

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

        arrows = get_arrow(torch.stack((start_points, end_points), dim=1), v_up[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(start_points).T).T)[:, :3] \
            .cpu().numpy()
        end_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(end_points).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.stack((
            np.arange(start_point_w.shape[0]), np.arange(start_point_w.shape[0]) + start_point_w.shape[0]
        ), axis=1)
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)),
                      np.concatenate((start_point_w, end_point_w), axis=0), edge_index)
        return

    def debug_save_(self, v_index):
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
        up_c = self.get_up_vector2(np.arange(self.id_viz_edge, self.id_viz_edge + edge_points.shape[0]),
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
        return 0

        # Visualize whole patch
        edge_points = point_pos_c[list(itertools.chain(*self.batched_points_per_patch))].reshape(-1, 4, 3)
        up_c = self.get_up_vector2(np.arange(np.sum([len(item) // 4 for item in self.batched_points_per_patch])),
                                   edge_points[:, 0], edge_points[:, 1])
        arrows = get_arrow(edge_points, up_c[:, 0])
        o3d.io.write_triangle_mesh(os.path.join(self.log_root, "total_{}_arrow.obj".format(id_epoch)), arrows)
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:, :3] \
            .cpu().numpy()
        edge_index = np.asarray(list(self.graph1.edges()))
        get_line_mesh(os.path.join(self.log_root, "total_{}_line.obj".format(id_epoch)), start_point_w, edge_index)
        pass

        return 0

    def debug_face(self, v_id_faces, v_id_epoch):
        line_thickness = 1
        point_thickness = 2
        point_radius = 1

        line_img1 = self.rgb1.copy()
        line_img1 = cv2.cvtColor(line_img1, cv2.COLOR_GRAY2BGR)
        shape = line_img1.shape[:2][::-1]
        line_img2 = self.rgb2.copy()
        line_img2 = cv2.cvtColor(line_img2, cv2.COLOR_GRAY2BGR)

        for id_face in v_id_faces:
            id_points = self.batched_points_per_patch[id_face][::4]
            edges = np.concatenate([item for item in self.id_point_to_id_up_and_face if item[0, 0] in id_points])
            start_points_c = self.seg_distance[edges[:, 0]][:, None] * self.ray_c[edges[:, 0]] \
                             * self.seg_distance_normalizer
            end_points_c = self.seg_distance[edges[:, 1]][:, None] * self.ray_c[edges[:, 1]] \
                           * self.seg_distance_normalizer
            start_points_2d1 = (self.intrinsic1 @ start_points_c.T).T
            start_points_2d1 = ((start_points_2d1[:, :2] / start_points_2d1[:, 2:3]).detach().cpu().numpy()
                                * shape).astype(np.int32)
            end_points_2d1 = (self.intrinsic1 @ end_points_c.T).T
            end_points_2d1 = ((end_points_2d1[:, :2] / end_points_2d1[:, 2:3]).detach().cpu().numpy()
                              * shape).astype(np.int32)
            start_points_2d2 = (self.transformation @ to_homogeneous_tensor(start_points_c).T).T
            start_points_2d2 = ((start_points_2d2[:, :2] / start_points_2d2[:, 2:3]).detach().cpu().numpy()
                                * shape).astype(np.int32)
            end_points_2d2 = (self.transformation @ to_homogeneous_tensor(end_points_c).T).T
            end_points_2d2 = ((end_points_2d2[:, :2] / end_points_2d2[:, 2:3]).detach().cpu().numpy()
                              * shape).astype(np.int32)

            for idx in range(edges.shape[0]):
                cv2.line(line_img1, start_points_2d1[idx], end_points_2d1[idx], (255, 0, 0), thickness=line_thickness)
                cv2.line(line_img2, start_points_2d2[idx], end_points_2d2[idx], (255, 0, 0), thickness=line_thickness)
            for idx in range(edges.shape[0]):
                cv2.circle(line_img1, start_points_2d1[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img1, end_points_2d1[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img2, start_points_2d2[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
                cv2.circle(line_img2, end_points_2d2[idx], radius=point_radius,
                           color=(255, 0, 0), thickness=point_thickness)
            for id_edge in range(edges.shape[0]):
                if edges[id_edge, 0] in id_points and edges[id_edge, 1] in id_points:
                    cv2.circle(line_img1, start_points_2d1[id_edge], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.circle(line_img2, start_points_2d2[id_edge], radius=point_radius,
                               color=(0, 255, 0), thickness=point_thickness)
                    cv2.line(line_img1, start_points_2d1[id_edge], end_points_2d1[id_edge], (0, 0, 255),
                             thickness=line_thickness)
                    cv2.line(line_img2, start_points_2d2[id_edge], end_points_2d2[id_edge], (0, 0, 255),
                             thickness=line_thickness)

        cv2.imwrite(os.path.join(self.log_root, "2d_{:05d}.jpg".format(v_id_epoch)),
                    np.concatenate((line_img1, line_img2), axis=1))

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
        self.model = LModel20(self.data,
                              self.hydra_conf["model"]["regress_normal"],
                              self.hydra_conf["dataset"]["id_viz_face"],
                              self.hydra_conf["dataset"]["id_viz_edge"],
                              self.hydra_conf["trainer"]["loss_weight"],
                              self.hydra_conf["trainer"]["img_model"],
                              self.hydra_conf["trainer"]["output"]
                              )
        # self.model = LModel31(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])
        # self.model = LModel12(self.data, self.hydra_conf["trainer"]["loss_weight"], self.hydra_conf["trainer"]["img_model"])

    def train_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        id_edge = self.hydra_conf["dataset"]["id_viz_edge"]
        self.train_dataset = Singel_node_dataset(
            is_one_target,
            self.model.batched_points_per_patch,
            self.model.id_point_to_id_up_and_face,
            id_face,
            "training",
        )
        # self.train_dataset = Node_dataset(self.model.id_point_to_id_up_and_face, "training")
        # self.train_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "training")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        id_edge = self.hydra_conf["dataset"]["id_viz_edge"]
        self.valid_dataset = Singel_node_dataset(
            is_one_target,
            self.model.batched_points_per_patch,
            self.model.id_point_to_id_up_and_face,
            id_face,
            "validation"
        )
        # self.valid_dataset = Node_dataset(self.model.id_point_to_id_up_and_face, "validation")
        # self.valid_dataset = Single_face_dataset(self.model.batched_points_per_patch, id_face, "validation")
        # self.valid_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "validation")
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

    id_patchs = v_viz_face
    print("Will visualize patch {} during the training".format(id_patchs))
    print("4. Draw input on img1")

    def draw_initial():
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("1", 1600, 900)
        # cv2.moveWindow("1", 5, 5)
        point_img = cv2.cvtColor(rgb1.copy(), cv2.COLOR_GRAY2BGR)
        for point in points_from_sfm_2d:
            cv2.circle(point_img, (point * shape).astype(np.int32), 2, (0, 0, 255), thickness=4)
        print("Draw lines on img1")
        line_img1 = cv2.cvtColor(rgb1.copy(), cv2.COLOR_GRAY2BGR)
        line_img2 = cv2.cvtColor(rgb2.copy(), cv2.COLOR_GRAY2BGR)

        # Draw first img
        for idx, face in enumerate(graph1.graph["faces"]):
            # print(idx)
            vertices = [graph1.nodes[id_node]["pos_2d"] for id_node in face]
            cv2.polylines(line_img1, [(np.asarray(vertices) * shape).astype(np.int32)], True, (0, 0, 255),
                          thickness=1)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()

        # Draw target patch
        if True:
            for id_patch in id_patchs:
                vertices_t = [graph1.nodes[id_node]["pos_2d"] for id_node in graph1.graph["faces"][id_patch]]
                cv2.polylines(line_img1, [(np.asarray(vertices_t) * shape).astype(np.int32)], True, (0, 255, 0),
                              thickness=1)
                for item in vertices_t:
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 255, 255), 2)

        # Draw second img
        for idx, face in enumerate(graph2.graph["faces"]):
            # print(idx)
            vertices = [graph2.nodes[id_node]["pos_2d"] for id_node in face]
            cv2.polylines(line_img2, [(np.asarray(vertices) * shape).astype(np.int32)], True, (0, 0, 255),
                          thickness=1)

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
        for id_patch in id_patchs:
            with open("output/img_field_test/target_patch_{}.obj".format(id_patch), "w") as f:
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
        for id_patch in id_patchs:
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

    model = Phase3(v_cfg, data)
    # offline_optimize(model.model, 1)

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
