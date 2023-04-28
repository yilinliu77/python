import itertools
import sys, os
import time
from typing import List

from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1, loss2

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

class Multi_edge_single_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data, v_is_one_target, v_id_target_face, v_training_mode, mul_number=1):
        super(Multi_edge_single_img_dataset, self).__init__()
        self.img_database: List[Image] = v_data[0]
        self.graphs = v_data[1]
        self.camera_pairs = v_data[2]
        self.target_img = 0
        self.training_edges = self.graphs[self.target_img].graph["training_edges"].copy()
        self.training_mode = v_training_mode
        self.id_training_edges_in_original_set = np.arange(self.training_edges.shape[0])
        if self.training_mode == "validation" or v_is_one_target:
            self.training_edges = self.training_edges[v_id_target_face]
            self.id_training_edges_in_original_set = v_id_target_face
        self.mul_number = mul_number
        pass

    def __len__(self):
        return len(self.training_edges) * self.mul_number

    def __getitem__(self, idx):
        idx = idx % len(self.training_edges)
        id_src_imgs = [int(id_img) for id_img in self.camera_pairs[self.target_img][:, 0]]
        projection2 = np.stack([self.img_database[id_img].projection for id_img in id_src_imgs], axis=0)
        intrinsic = self.img_database[self.target_img].intrinsic
        transformation = projection2 @ np.linalg.inv(self.img_database[self.target_img].extrinsic)
        src_imgs = np.stack([
            cv2.imread(self.img_database[id_img].img_path, cv2.IMREAD_GRAYSCALE) for id_img in id_src_imgs], axis=0)
        ref_img = cv2.imread(self.img_database[self.target_img].img_path, cv2.IMREAD_GRAYSCALE)
        imgs = np.concatenate((ref_img[None, :], src_imgs), axis=0)
        ray_c = np.stack([self.graphs[self.target_img].nodes[id_node]["ray_c"]
                          for id_node in self.training_edges[idx,:2]], axis=0)
        valid_flags = self.training_edges[idx,2]
        # assert valid_flags.shape[0]==3 # Currently we only support 3 edges per point
        return torch.tensor(self.target_img, dtype=torch.long), \
            torch.tensor(self.training_edges[idx]).to(torch.long), \
            torch.from_numpy(transformation).to(torch.float32), \
            torch.from_numpy(intrinsic).to(torch.float32), \
            torch.from_numpy(imgs).to(torch.float32) / 255., \
            torch.from_numpy(ray_c).to(torch.float32), \
            torch.tensor(valid_flags).to(torch.bool), \
            self.id_training_edges_in_original_set[idx]


    @staticmethod
    def collate_fn(items):
        id_cur_imgs = -1
        batched_indexes = []
        batched_transformations = None
        batched_intrinsic = None
        batched_imgs = None
        batched_ray_c = []
        batched_valid_flags = []
        idxs = []
        for item in items:
            id_cur_imgs = item[0]
            batched_indexes.append(item[1])
            batched_transformations = item[2]
            batched_intrinsic = item[3]
            batched_imgs = item[4]
            batched_ray_c.append(item[5])
            batched_valid_flags.append(item[6])
            idxs.append(item[7])
        batched_indexes = torch.stack(batched_indexes, dim=0)
        batched_ray_c = torch.stack(batched_ray_c, dim=0)
        batched_valid_flags = torch.stack(batched_valid_flags, dim=0)
        idxs = torch.tensor(idxs, dtype=torch.long)
        return id_cur_imgs, batched_indexes, batched_transformations, batched_intrinsic, \
            batched_imgs, batched_ray_c, batched_valid_flags, idxs


class LModel22(nn.Module):
    def __init__(self, v_data, v_is_regress_normal, v_viz_patch, v_log_root):
        super(LModel22, self).__init__()
        self.log_root = v_log_root
        self.is_regress_normal = v_is_regress_normal

        self.init_regular_variables(v_data)
        self.distances = nn.ParameterList()
        self.distance_normalizer = 10.
        for id_graph, graph in enumerate(self.graphs):
            distances = np.asarray([graph.nodes[item]["distance"] for item in graph])
            training_edges = graph.graph["training_edges"]
            id_edge_points = training_edges[:,:2]
            distances_ = torch.from_numpy(distances[id_edge_points]).to(torch.float32) / self.distance_normalizer
            self.distances.append(nn.Parameter(distances_))
        # Debug
        self.id_viz_face = v_viz_patch

    # Init-related methods
    def init_regular_variables(self, v_data):
        # Graph related
        self.graphs = v_data[1]
        self.img_database = v_data[0]

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

    def compute_similarity_wrapper(self, start_points, end_points,
                                   imgs, transformations, intrinsic):
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

        num_per_edge1, points_2d1 = self.sample_points_2d(edge_points, num_horizontal)

        valid_mask1 = torch.logical_and(points_2d1 > 0, points_2d1 < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        points_2d1 = torch.clamp(points_2d1, 0, 0.999999)

        edge_points = (transformations @ to_homogeneous_tensor(points_c).T).transpose(1, 2)
        edge_points = edge_points[:, :, :2] / (edge_points[:, :, 2:3] + 1e-6)
        edge_points = edge_points.reshape(num_src_imgs, -1, 2, 2)

        num_per_edge2, points_2d2 = self.sample_points_2d(edge_points.reshape(-1, 2, 2),
                                                          num_horizontal.tile(num_src_imgs))
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

        similarity_loss = loss2(sample_imgs1, sample_imgs2, num_per_edge1)
        similarity_mask = torch.logical_and(valid_mask1[None, :].tile([valid_mask2.shape[0], 1]), valid_mask2)
        similarity_mask = scatter_min(similarity_mask.to(torch.long), torch.arange(
            num_per_edge1.shape[0], device=similarity_loss.device).repeat_interleave(num_per_edge1), dim=1)[0]
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

        return similarity_loss, similarity_mask.to(torch.bool), [points_2d1, points_2d2]

    #
    # start_rays: (B, E, 3)
    # end_points_c: (B, E, 3)
    # v_new_distances: (B, S)
    # valid_flags: (B, S)
    #
    def random_search(self, start_rays, end_rays, v_new_distances,
                      imgs, transformations, intrinsic, valid_flags):
        batch_size = 10
        num_edges = v_new_distances.shape[0]
        num_sampled = v_new_distances.shape[2]  # Sample from normal distribution + 1
        num_imgs = imgs.shape[0] - 1

        repeated_start_points_c = start_rays[:, None, :] * v_new_distances[:, 0][:,:,None] * self.distance_normalizer
        repeated_end_points_c = end_rays[:, None, :] * v_new_distances[:, 1][:,:,None] * self.distance_normalizer

        losses = []
        masks = []
        for id_batch in range(num_sampled // batch_size + 1):
            id_batch_start = min(num_sampled, id_batch * batch_size)
            id_batch_end = min(num_sampled, (id_batch + 1) * batch_size)
            if id_batch_start >= id_batch_end:
                continue
            num_batch = id_batch_end - id_batch_start

            similarity_loss, similarity_mask, _ = self.compute_similarity_wrapper(
                #
                repeated_start_points_c[:, id_batch_start:id_batch_end].reshape(-1,3),
                repeated_end_points_c[:, id_batch_start:id_batch_end].reshape(-1,3),
                imgs, transformations, intrinsic
            )
            similarity_loss = similarity_loss.reshape(num_imgs, num_edges, num_batch)
            similarity_mask = similarity_mask.reshape(num_imgs, num_edges, num_batch)

            losses.append(similarity_loss)
            masks.append(similarity_mask)

        penalized_loss = torch.inf
        similarity_loss_ = torch.cat(losses, dim=2)
        similarity_mask_ = torch.cat(masks, dim=2)
        # Some vertices are project outside to the images
        similarity_loss_[~similarity_mask_] = penalized_loss
        # (num_img, num_edge, num_sample)
        # -> (num_edge, num_sample, num_img)
        similarity_loss_ = similarity_loss_.permute(1, 2, 0)
        # Some vertices are along the border, discard them
        similarity_loss_[~valid_flags] = penalized_loss
        similarity_loss_avg = torch.mean(similarity_loss_,dim=-1) # Average about images

        id_best = similarity_loss_avg.argmin(dim=1)
        # id_best[similarity_loss_avg[
        #             torch.arange(similarity_loss_avg.shape[0], device=start_rays.device), id_best] > 2] = 0
        return id_best

    def forward(self, idxs, v_id_epoch, is_log):
        # 0: Unpack data
        v_id_epoch += 1
        # (1,)
        id_cur_imgs = idxs[0]
        # (B, 5)
        batched_ids = idxs[1]
        # (N, 4, 4)
        transformations = idxs[2]
        # (3, 3)
        intrinsic = idxs[3]
        # (N+1, h, w)
        imgs = idxs[4]
        # (B, 2, 3)
        ray_c = idxs[5]
        # (B,)
        valid_flags = idxs[6]
        id_sample = idxs[7]
        batch_size = batched_ids.shape[0]
        num_edges = batch_size
        device = id_cur_imgs.device
        times = [0 for _ in range(10)]
        cur_time = time.time()

        # (B)
        id_start_point = batched_ids[:, 0]
        # (B)
        id_end_point = batched_ids[:, 1]
        # (B, 3)
        start_ray_c = ray_c[:, 0]
        # (B, 3)
        end_ray_c = ray_c[:, 1]
        # (B, 2)
        vertices_distances = self.distances[id_cur_imgs][id_sample]

        times[1] += time.time() - cur_time
        cur_time = time.time()

        # Random search
        if self.training or False:
            with torch.no_grad():
                num_sample = 100
                scale_factor = 0.16
                # (B * S,)
                new_distance = -torch.ones((num_edges, 2, num_sample), device=device, dtype=torch.float32)
                sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                # (B * S)
                repeated_vertices_distances = vertices_distances[:,:,None].tile((1,1,num_sample))
                while not torch.all(sample_distance_mask):
                    t_ = new_distance[~sample_distance_mask]
                    a = repeated_vertices_distances[~sample_distance_mask] + \
                        scale_factor * torch.distributions.utils._standard_normal(
                        t_.shape[0],
                        device=device,
                        dtype=t_.dtype)
                    new_distance[~sample_distance_mask] = a
                    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                # (B, (S + 1))
                new_distance = torch.cat((vertices_distances[:, :, None], new_distance), dim=2)
                id_best_distance = self.random_search(
                    start_ray_c, end_ray_c, new_distance,
                    imgs, transformations, intrinsic, valid_flags
                )
                self.distances[id_cur_imgs][id_sample] = torch.gather(
                    new_distance,2,(id_best_distance[None,None,:]).expand(-1,2,-1)).permute(2,1,0)[:,:,0]
        times[3] += time.time() - cur_time
        cur_time = time.time()

        vertices_distances = self.distances[id_cur_imgs][id_sample]
        start_points_c = start_ray_c * vertices_distances[:, 0:1] * self.distance_normalizer
        end_points_c = end_ray_c * vertices_distances[:, 1:2] * self.distance_normalizer

        # similarity_loss: (N, B)
        similarity_loss, similarity_mask, [points_2d1, points_2d2] = self.compute_similarity_wrapper(
            start_points_c,
            end_points_c,
            imgs, transformations, intrinsic
        )

        penalized_loss = 10.
        similarity_loss[~similarity_mask] = penalized_loss
        similarity_loss[:, ~valid_flags] = penalized_loss

        times[4] += time.time() - cur_time
        cur_time = time.time()
        if is_log:
            with torch.no_grad():
                rgb1 = cv2.cvtColor((imgs[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                shape1 = rgb1.shape[:2][::-1]

                start_points_c = start_points_c.reshape(-1, 3)
                end_points_c = end_points_c.reshape(-1, 3)
                start_points_c = start_points_c[(start_points_c != 0).all(axis=1)]
                end_points_c = end_points_c[(end_points_c != 0).all(axis=1)]
                start_points_2d1 = (intrinsic @ start_points_c.T).T
                start_points_2d1 = (start_points_2d1[:, :2] / start_points_2d1[:, 2:3]).cpu().numpy()
                start_points_2d1 = (np.clip(start_points_2d1, 0, 0.99999) * shape1).astype(int)
                end_points_2d1 = (intrinsic @ end_points_c.T).T
                end_points_2d1 = (end_points_2d1[:, :2] / end_points_2d1[:, 2:3]).cpu().numpy()
                end_points_2d1 = (np.clip(end_points_2d1, 0, 0.99999) * shape1).astype(int)

                line_img1 = rgb1.copy()

                line_thickness = 1
                point_thickness = 2
                point_radius = 1

                for id_ver, _ in enumerate(end_points_2d1):
                    cv2.line(line_img1, start_points_2d1[id_ver], end_points_2d1[id_ver], (0, 0, 255),
                             thickness=line_thickness)
                for id_ver, _ in enumerate(end_points_2d1):
                    cv2.circle(line_img1, start_points_2d1[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)
                    cv2.circle(line_img1, end_points_2d1[id_ver], radius=point_radius,
                               color=(0, 255, 255), thickness=point_thickness)

                line_imgs2 = []
                for i_img in range(imgs[1:].shape[0]):
                    rgb2 = cv2.cvtColor((imgs[1 + i_img].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    line_img2 = rgb2.copy()
                    shape2 = rgb2.shape[:2][::-1]
                    start_points_2d2 = (transformations[i_img] @ to_homogeneous_tensor(start_points_c).T).T
                    start_points_2d2 = (start_points_2d2[:, :2] / start_points_2d2[:, 2:3]).cpu().numpy()
                    start_points_2d2 = (np.clip(start_points_2d2, 0, 0.99999) * shape2).astype(int)
                    end_points_2d2 = (transformations[i_img] @ to_homogeneous_tensor(end_points_c).T).T
                    end_points_2d2 = (end_points_2d2[:, :2] / end_points_2d2[:, 2:3]).cpu().numpy()
                    end_points_2d2 = (np.clip(end_points_2d2, 0, 0.99999) * shape2).astype(int)

                    for id_ver, _ in enumerate(end_points_2d2):
                        cv2.line(line_img2, start_points_2d2[id_ver], end_points_2d2[id_ver], (0, 0, 255),
                                 thickness=line_thickness)
                    for id_ver, _ in enumerate(end_points_2d2):
                        cv2.circle(line_img2, start_points_2d2[id_ver], radius=point_radius,
                                   color=(0, 255, 255), thickness=point_thickness)
                        cv2.circle(line_img2, end_points_2d2[id_ver], radius=point_radius,
                                   color=(0, 255, 255), thickness=point_thickness)
                    line_imgs2.append(line_img2)

                big_imgs = np.concatenate(
                    (np.concatenate(
                        (line_img1, line_imgs2[0], line_imgs2[1], line_imgs2[2]), axis=1),
                     np.concatenate(
                         (line_imgs2[3], line_imgs2[4], line_imgs2[5], line_imgs2[6]), axis=1),
                     np.concatenate(
                         (line_imgs2[7], line_imgs2[8], line_imgs2[9], line_imgs2[9]), axis=1),
                    )
                    , axis=0)

                cv2.imwrite(os.path.join(self.log_root, "2d_{:05d}.jpg".format(v_id_epoch)),
                            big_imgs)

                # polygon_points_2d_1 = points_2d1.detach().cpu().numpy()
                # polygon_points_2d_2 = points_2d2[0].detach().cpu().numpy()
                #
                # line_img1 = rgb1.copy()
                # line_img2 = cv2.cvtColor((imgs[1].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                #
                # roi_coor_2d1_numpy = np.clip(polygon_points_2d_1, 0, 0.99999)
                # viz_coords = (roi_coor_2d1_numpy * shape1).astype(np.int32)
                # line_img1[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                #
                # # Image 2
                # roi_coor_2d2_numpy = np.clip(polygon_points_2d_2, 0, 0.99999)
                # viz_coords = (roi_coor_2d2_numpy * shape2).astype(np.int32)
                # line_img2[viz_coords[:, 1], viz_coords[:, 0]] = (0, 0, 255)
                # cv2.imwrite(os.path.join(self.log_root, "3d_{:05d}.jpg".format(v_id_epoch)),
                #             np.concatenate((line_img1, line_img2), axis=1))

        return torch.mean(similarity_loss), [None, None, None]

    def len(self):
        return len(self.graph1.graph["faces"])


class Phase5(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Phase5, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        if not os.path.exists(self.hydra_conf["trainer"]["output"]):
            os.makedirs(self.hydra_conf["trainer"]["output"])

        self.data = v_data
        self.model = LModel22(self.data,
                              self.hydra_conf["model"]["regress_normal"],
                              self.hydra_conf["dataset"]["id_viz_face"],
                              self.hydra_conf["trainer"]["output"]
                              )

    def train_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        self.train_dataset = Multi_edge_single_img_dataset(
            self.data,
            is_one_target,
            id_face,
            "training",
        )
        # self.train_dataset = Node_dataset(self.model.id_point_to_id_up_and_face, "training")
        # self.train_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "training")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Multi_edge_single_img_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        self.valid_dataset = Multi_edge_single_img_dataset(
            self.data,
            is_one_target,
            id_face,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=64,
                          collate_fn=Multi_edge_single_img_dataset.collate_fn,
                          num_workers=0)

    def configure_optimizers(self):
        # grouped_parameters = [
        #     {"params": [self.model.seg_distance], 'lr': self.learning_rate},
        #     {"params": [self.model.v_up], 'lr': 1e-2},
        # ]

        optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )
        # optimizer = SGD(grouped_parameters, lr=self.learning_rate, )

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

        # self.model.debug_save(self.current_epoch if not self.trainer.sanity_checking else -1)

        if self.trainer.sanity_checking:
            return


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
        for id_point in range(len(face)):
            id_start = id_point
            id_end = (id_start + 1) % len(face)
            if "id_face" not in graph[face[id_start]][face[id_end]]:
                graph[face[id_start]][face[id_end]]["id_face"] = []
            graph[face[id_start]][face[id_end]]["id_face"].append(id_face)

    graph.graph["face_flags"] = face_flags

    training_edges = []
    for id_start_node, id_end_node in graph.edges():
        training_edges.append(np.asarray((
            id_start_node, id_end_node,
            graph[id_start_node][id_end_node]["valid_flag"],
            graph[id_start_node][id_end_node]["id_face"][0],
            graph[id_start_node][id_end_node]["id_face"][1] if len(
                graph[id_start_node][id_end_node]["id_face"]) > 0 else -1,)
        ))
    graph.graph["training_edges"] = np.stack(training_edges,axis=0).astype(np.int64)
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
        for idx, (id_start,id_end) in enumerate(v_graph.edges()):
            # print(idx)
            start = (v_graph.nodes[id_start]["pos_2d"] * img.img_size).astype(np.int32)
            end = (v_graph.nodes[id_end]["pos_2d"] * img.img_size).astype(np.int32)
            cv2.line(line_img1, start, end, (0, 0, 255), thickness=1)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()

        # Draw target patch
        for id_edge in v_viz_face:
            (id_start,id_end) = np.asarray(v_graph.edges)[id_edge]
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

    return img_database, graphs, camera_pair_data


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
