import itertools
import sys, os
import time
from typing import List

from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1, loss2, loss4

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
from src.neural_recon.bak.phase1 import NGPModel

class Multi_node_single_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data, v_is_one_target, v_id_target_face, v_training_mode, mul_number=1):
        super(Multi_node_single_img_dataset, self).__init__()
        self.img_database: List[Image] = v_data[0]
        self.graphs = v_data[1]
        self.camera_pairs = v_data[2]
        self.target_img = 0
        self.training_vertices = self.graphs[self.target_img].graph["training_vertices"].copy()
        self.training_mode = v_training_mode
        if self.training_mode == "validation" or v_is_one_target:
            id_nodes = np.unique(list(itertools.chain(*[
                self.graphs[self.target_img].graph["faces"][id_face] for id_face in v_id_target_face])))
            self.training_vertices = [item for item in self.training_vertices if item[0][0] in id_nodes]
        self.mul_number = mul_number
        pass

    def __len__(self):
        return len(self.training_vertices) * self.mul_number

    def __getitem__(self, idx):
        idx = idx % len(self.training_vertices)
        id_src_imgs = [int(id_img) for id_img in self.camera_pairs[self.target_img][:, 0]]
        projection2 = np.stack([self.img_database[id_img].projection for id_img in id_src_imgs], axis=0)
        intrinsic = self.img_database[self.target_img].intrinsic
        transformation = projection2 @ np.linalg.inv(self.img_database[self.target_img].extrinsic)
        src_imgs = np.stack([
            cv2.imread(self.img_database[id_img].img_path, cv2.IMREAD_GRAYSCALE) for id_img in id_src_imgs], axis=0)
        ref_img = cv2.imread(self.img_database[self.target_img].img_path, cv2.IMREAD_GRAYSCALE)
        imgs = np.concatenate((ref_img[None, :], src_imgs), axis=0)
        ray_c = np.stack([self.graphs[self.target_img].nodes[id_node]["ray_c"]
                          for id_node in self.training_vertices[idx][:, :2].reshape(-1)], axis=0)
        valid_flags = np.stack([self.graphs[self.target_img].edges[edge]["valid_flag"]
                                for edge in self.training_vertices[idx][:, :2]], axis=0)
        # assert valid_flags.shape[0]==3 # Currently we only support 3 edges per point
        return torch.tensor(self.target_img, dtype=torch.long), \
            torch.tensor(self.training_vertices[idx]).to(torch.long), \
            torch.from_numpy(transformation).to(torch.float32), \
            torch.from_numpy(intrinsic).to(torch.float32), \
            torch.from_numpy(imgs).to(torch.float32) / 255., \
            torch.from_numpy(ray_c).to(torch.float32), \
            torch.from_numpy(valid_flags).to(torch.bool), \
            idx


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
        batched_indexes = pad_sequence(batched_indexes, batch_first=True)
        batched_ray_c = pad_sequence(batched_ray_c, batch_first=True)
        batched_valid_flags = pad_sequence(batched_valid_flags, batch_first=True)
        idxs = torch.tensor(idxs, dtype=torch.long)
        return id_cur_imgs, batched_indexes, batched_transformations, batched_intrinsic, \
            batched_imgs, batched_ray_c, batched_valid_flags, idxs


class LModel21(nn.Module):
    def __init__(self, v_data, v_is_regress_normal, v_viz_patch, v_log_root):
        super(LModel21, self).__init__()
        self.log_root = v_log_root
        self.is_regress_normal = v_is_regress_normal

        self.init_regular_variables(v_data)
        self.distances = nn.ParameterList()
        self.distance_normalizer = 10.
        max_num_edge_per_vertices = max([max([len(item) for item in graph.graph["training_vertices"]]) for graph in self.graphs])
        indicator = []
        for id_graph, graph in enumerate(self.graphs):
            distances = np.asarray([graph.nodes[item]["distance"] for item in graph])
            distances = torch.from_numpy(distances).to(torch.float32) / self.distance_normalizer
            self.distances.append(nn.Parameter(torch.stack((distances,distances),dim=1)))
            training_vertices = graph.graph["training_vertices"]
            local_indicator = -torch.ones((len(training_vertices), max_num_edge_per_vertices), dtype=torch.long)
            lengths = torch.tensor([len(item) for item in training_vertices])
            mask = torch.arange(max_num_edge_per_vertices)[None,:].tile((len(training_vertices),1))<lengths[:,None]
            local_indicator[mask] = 0
            indicator.append(local_indicator)
        self.register_buffer("indicator", pad_sequence(indicator,batch_first=True,padding_value=-1))
        self.register_buffer("id_choose_distance1", torch.tensor((0, 0, 1, 1), dtype=torch.long))
        self.register_buffer("id_choose_distance2", torch.tensor((0, 1, 0, 1), dtype=torch.long))
        # Debug
        self.id_viz_face = v_viz_patch

    # Init-related methods
    def init_regular_variables(self, v_data):
        # Graph related
        self.graphs = v_data[1]
        self.img_database = v_data[0]

    # Pls change the v_img_width!!!!!!!
    def sample_points_2d(self, v_edge_points, v_num_horizontal, v_img_width=800):
        device = v_edge_points.device
        cur_dir = v_edge_points[:, 1] - v_edge_points[:, 0]
        cur_length = torch.linalg.norm(cur_dir, dim=-1) + 1e-6

        cur_dir_h = torch.cat((cur_dir, torch.zeros_like(cur_dir[:, 0:1])), dim=1)
        z_axis = torch.zeros_like(cur_dir_h)
        z_axis[:, 2] = 1
        edge_up = normalize_tensor(torch.cross(cur_dir_h, z_axis, dim=1)[:, :2]) * 10 / v_img_width
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

        similarity_loss, black_area_in_img1 = loss4(sample_imgs1, sample_imgs2, num_per_edge1)
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

        return similarity_loss, similarity_mask.to(torch.bool), black_area_in_img1, [points_2d1, points_2d2]

    #
    # start_rays: (B, E, 3)
    # end_points_c: (B, E, 3)
    # v_new_distances: (B, S)
    # valid_flags: (B, S)
    #
    def random_search(self, start_rays, end_points_c, v_new_distances,
                      imgs, transformations, intrinsic, valid_flags, is_single_img):
        batch_size = 4
        num_points = v_new_distances.shape[0]
        num_sampled = v_new_distances.shape[1]  # Sample from normal distribution + 1
        num_imgs = imgs.shape[0] - 1
        num_max_edges = start_rays.shape[1]

        # (B, E, S * 2, 3)
        repeated_start_points_c = start_rays[:,:,None,:] * v_new_distances[:,None,:,None] \
                                  * self.distance_normalizer
        repeated_end_points_c = end_points_c[:, :, None].tile([1, 1, num_sampled, 1, 1])

        losses = []
        masks = []
        black_area_in_img1s = []
        for id_batch in range(num_sampled // batch_size + 1):
            id_batch_start = min(num_sampled, id_batch * batch_size)
            id_batch_end = min(num_sampled, (id_batch + 1) * batch_size)
            if id_batch_start >= id_batch_end:
                continue
            num_batch = id_batch_end - id_batch_start

            # (B, 2, E, S', 3)
            start_points_c_multiple = torch.stack((
                repeated_start_points_c[:, :, id_batch_start:id_batch_end],
                repeated_start_points_c[:, :, id_batch_start:id_batch_end],
            ), dim=1)
            end_points_c_multiple = torch.stack((
                repeated_end_points_c[:, :, id_batch_start:id_batch_end, 0],
                repeated_end_points_c[:, :, id_batch_start:id_batch_end, 1],
            ), dim=1)

            similarity_loss, similarity_mask, black_area_in_img1, _ = self.compute_similarity_wrapper(
                # The layout is like p0_s0, p0_s1, p0_s2, ..., p1_s0, p1_s1
                start_points_c_multiple.reshape(-1, 3),
                end_points_c_multiple.reshape(-1, 3),
                imgs, transformations, intrinsic
            )
            # (N, B, 2, E, S')
            similarity_loss = similarity_loss.reshape(num_imgs, num_points, 2, num_max_edges, num_batch)
            similarity_mask = similarity_mask.reshape(num_imgs, num_points, 2, num_max_edges, num_batch)
            black_area_in_img1 = black_area_in_img1.reshape(num_points, 2, num_max_edges, num_batch)

            losses.append(similarity_loss)
            masks.append(similarity_mask)
            black_area_in_img1s.append(black_area_in_img1)

        # (N, B, 2, E, S)
        similarity_loss_ = torch.cat(losses, dim=-1)
        similarity_mask_ = torch.cat(masks, dim=-1)
        black_area_in_img1s = torch.cat(black_area_in_img1s, dim=-1)[:, 0, :, 0]
        valid_flags = torch.logical_and(~black_area_in_img1s, valid_flags)

        penalized_loss = 10.
        penalized_loss = torch.inf
        # Some vertices are project outside to the images
        similarity_loss_[~similarity_mask_] = penalized_loss
        # (num_img, num_vertex, 2, num_edge, num_sample)
        # -> (num_vertex, num_edge, 2, num_sample, num_img)
        similarity_loss_ = similarity_loss_.permute(1, 3, 2, 4, 0)
        # Some vertices are along the border, discard them
        similarity_loss_[~valid_flags] = penalized_loss

        if is_single_img:
            similarity_loss_avg = similarity_loss_[...,1]  # select the first loss
        else:
            similarity_loss_avg = torch.mean(similarity_loss_,dim=-1) # Average about images

        # Find best combinations
        similarity_loss_avg_ = similarity_loss_avg.min(dim=2)[0]
        if False:
            N, E, S = similarity_loss_avg_.shape
            K = 3
            device = similarity_loss_avg.device

            # Get the top k persons for each role
            top_k_values, top_k_indices = torch.topk(similarity_loss_avg_, K, dim=2, largest=False)
            top_k_indices_flatten = top_k_indices.reshape((N,-1))
            # Create index pairs for all possible person combinations (including single person)
            s1_indices, s2_indices = torch.meshgrid(torch.arange(E*K,device=device), torch.arange(E*K,device=device),indexing="ij")
            num_pairs = s1_indices.shape[0] * s1_indices.shape[1]
            index_coordinates = torch.stack((s1_indices,s2_indices),dim=-1).reshape(num_pairs,2)

            # Index the two distance candidate from all pairs
            selected_index = top_k_indices_flatten[:,index_coordinates]
            # Get the value of the two distance candidate for each edge, and choose the smaller one
            selected_values_ = torch.gather(similarity_loss_avg_[:,:,None,:].expand(-1,-1,num_pairs,-1),3,selected_index[:,None,:,:].expand(-1,E,-1,-1))
            selected_values = torch.min(selected_values_, dim=3)[0]
            # Calculate the sums for each person pair
            is_inf = torch.isinf(selected_values)
            selected_values[is_inf]=10.
            summed_value = torch.sum(selected_values, dim=1)
            min_index = torch.min(summed_value, dim=1)[1]
            best_index = torch.gather(selected_index,1,min_index[:,None,None].expand(-1,-1,2))[:,0]
        else:
            similarity_loss_avg_[torch.isinf(similarity_loss_avg_)] = 10.
            best_index = similarity_loss_avg_.sum(dim=1).argmin(dim=1,keepdims=True)
        # id_best[similarity_loss_avg[
        #             torch.arange(similarity_loss_avg.shape[0], device=start_rays.device), id_best] == 5] = 0
        return best_index

    def forward(self, idxs, v_id_epoch, is_log):
        is_single_img=True
        # 0: Unpack data
        v_id_epoch += 1
        # (1,)
        id_cur_imgs = idxs[0]
        # (B, E, 4)
        batched_ids = idxs[1]
        # (N, 4, 4)
        transformations = idxs[2]
        # (3, 3)
        intrinsic = idxs[3]
        # (N+1, h, w)
        imgs = idxs[4]
        # (B, E * 2, 3)
        ray_c = idxs[5]
        # (B, E * 2)
        valid_flags = idxs[6]
        id_vertices = idxs[7]
        batch_size = batched_ids.shape[0]
        num_vertices = batch_size
        num_max_edges_per_vertice = batched_ids.shape[1]
        device = id_cur_imgs.device
        times = [0 for _ in range(10)]
        cur_time = time.time()

        # (B * E)
        id_start_point = batched_ids[:, :, 0]
        id_start_point_unique = batched_ids[:, 0, 0]
        # (B, E)
        id_end_point = batched_ids[:, :, 1]
        # (B, E, 3)
        start_ray_c = ray_c[:, ::2]
        # (B, E, 3)
        end_ray_c = ray_c[:, 1::2]
        # (B, 2)
        vertices_distances = self.distances[id_cur_imgs][id_start_point_unique]
        # (B, E, 2)
        end_point_distances = self.distances[id_cur_imgs][id_end_point]
        id_distance_indicator = self.indicator[id_cur_imgs][id_start_point_unique]
        id_choose_distance = self.id_choose_distance2[id_distance_indicator][:,:num_max_edges_per_vertice]
        # (B, E, 1)
        # end_point_distances = torch.gather(end_point_distances, 2, id_choose_distance[:,:,None])

        # (B, E, 2, 3)
        end_points_c = end_ray_c[:,:,None,:] * end_point_distances[:,:,:,None]\
                       * self.distance_normalizer

        times[1] += time.time() - cur_time
        cur_time = time.time()

        # Random search
        if self.training or False:
            with torch.no_grad():
                num_sample = 100
                scale_factor = 0.16
                # (B, S, 2)
                new_distance = -torch.ones((num_vertices, num_sample, 2), device=device, dtype=torch.float32)
                sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                # (B, S, 2)
                repeated_vertices_distances = vertices_distances[:,None,:].tile((1,num_sample,1))
                while not torch.all(sample_distance_mask):
                    t_ = new_distance[~sample_distance_mask]
                    a = repeated_vertices_distances[~sample_distance_mask] + \
                        scale_factor * torch.distributions.utils._standard_normal(
                        t_.shape[0],
                        device=device,
                        dtype=t_.dtype)
                    new_distance[~sample_distance_mask] = a
                    sample_distance_mask = torch.logical_and(new_distance > 0, new_distance < 1)
                # (B, (S + 1), 2)
                new_distance = torch.cat((vertices_distances[:, None], new_distance), dim=1)
                # (B, 2 * (S + 1))
                new_distance = new_distance.reshape(num_vertices, -1)
                id_best_distance = self.random_search(
                    start_ray_c, end_points_c, new_distance,
                    imgs, transformations, intrinsic, valid_flags, is_single_img
                )
                self.distances[id_cur_imgs][id_start_point_unique] = torch.gather(new_distance,1,id_best_distance)
        times[3] += time.time() - cur_time
        cur_time = time.time()

        start_points_c = start_ray_c[:,:,None,:] * self.distances[id_cur_imgs][id_start_point][:,:,:,None] \
                         * self.distance_normalizer
        end_points_c = end_ray_c[:,:,None,:] * self.distances[id_cur_imgs][id_end_point][:,:,:,None] \
                         * self.distance_normalizer

        # Make different combinations of the edges
        num_imgs = transformations.shape[0]
        num_points = start_points_c.shape[0]
        num_max_edges = start_points_c.shape[1]
        start_points_c_multiple=torch.cat((
            start_points_c[:, :, 0],
            start_points_c[:, :, 0],
            start_points_c[:, :, 1],
            start_points_c[:, :, 1],
        ),dim=1)
        end_points_c_multiple = torch.cat((
            end_points_c[:, :, 0],
            end_points_c[:, :, 1],
            end_points_c[:, :, 0],
            end_points_c[:, :, 1],
        ), dim=1)
        similarity_loss, similarity_mask, black_area_in_img1s, [points_2d1, points_2d2] = self.compute_similarity_wrapper(
            start_points_c_multiple.reshape(-1, 3),
            end_points_c_multiple.reshape(-1, 3),
            imgs, transformations, intrinsic
        )

        # Unpack the combination
        similarity_loss = similarity_loss.reshape((num_imgs, num_points, 4, num_max_edges)) # 4 combinations
        similarity_mask = similarity_mask.reshape((num_imgs, num_points, 4, num_max_edges))
        black_area_in_img1s = black_area_in_img1s.reshape((num_points, 4, num_max_edges))[:,0]
        valid_flags = torch.logical_and(~black_area_in_img1s, valid_flags)
        penalized_loss = 10.
        # penalized_loss = torch.inf
        similarity_loss[~similarity_mask] = penalized_loss
        similarity_loss = similarity_loss.permute(1, 3, 2, 0)
        similarity_loss[~valid_flags] = penalized_loss
        if is_single_img:
            similarity_loss_avg = similarity_loss[...,1]
        else:
            similarity_loss_avg = torch.mean(similarity_loss, dim=3)
        id_min = similarity_loss_avg.detach().argmin(dim=2)
        similarity_loss_final = torch.gather(similarity_loss_avg, 2, id_min.unsqueeze(2)).squeeze(2)
        self.indicator[id_cur_imgs][id_vertices,:num_max_edges_per_vertice]=id_min
        self.indicator[id_cur_imgs][id_vertices,:num_max_edges_per_vertice][~valid_flags]=-1

        times[4] += time.time() - cur_time
        cur_time = time.time()
        if is_log:
            with torch.no_grad():
                rgb1 = cv2.cvtColor((imgs[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                shape1 = rgb1.shape[:2][::-1]
                start_points_c = torch.gather(start_points_c,2,
                                              self.id_choose_distance1[id_min][:,:,None,None].expand((-1,-1,-1,3))
                                              ).squeeze(2)
                end_points_c = torch.gather(end_points_c,2,
                                              self.id_choose_distance2[id_min][:,:,None,None].expand((-1,-1,-1,3))
                                              ).squeeze(2)

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

        return torch.mean(similarity_loss_final), [None, None, None]

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
        self.model = LModel21(self.data,
                              self.hydra_conf["model"]["regress_normal"],
                              self.hydra_conf["dataset"]["id_viz_face"],
                              self.hydra_conf["trainer"]["output"]
                              )

    def train_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        self.train_dataset = Multi_node_single_img_dataset(
            self.data,
            is_one_target,
            id_face,
            "training",
        )
        # self.train_dataset = Node_dataset(self.model.id_point_to_id_up_and_face, "training")
        # self.train_dataset = Edge_dataset(self.model.batched_points_per_patch, is_one_target, id_edge, "training")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Multi_node_single_img_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        is_one_target = self.hydra_conf["dataset"]["only_train_target"]
        id_face = self.hydra_conf["dataset"]["id_viz_face"]
        self.valid_dataset = Multi_node_single_img_dataset(
            self.data,
            is_one_target,
            id_face,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=64,
                          collate_fn=Multi_node_single_img_dataset.collate_fn,
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

    training_vertices = []
    for id_start_node in graph.nodes():
        training_vertices.append(np.asarray([(
            id_start_node, id_end_node,
            graph[id_start_node][id_end_node]["valid_flag"],
            graph[id_start_node][id_end_node]["id_face"][0],
            graph[id_start_node][id_end_node]["id_face"][1] if len(
                graph[id_start_node][id_end_node]["id_face"]) > 0 else -1,
        ) for id_end_node in graph[id_start_node]], dtype=np.int32))
    # Trim the last 4 vertices. They are boundaries
    training_vertices = training_vertices[:-4]
    graph.graph["training_vertices"] = training_vertices
    graph.graph["id_distance"] = [np.zeros(item.shape[0], dtype=np.int64) for item in training_vertices]

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
        # for i_img in tqdm(range(len(img_database))):
        #     data = [item for item in open(
        #         os.path.join(v_colmap_dir, "wireframe/{}.obj".format(img_database[i_img].img_name))).readlines()]
        #     vertices = [item.strip().split(" ")[1:-1] for item in data if item[0] == "v"]
        #     vertices = np.asarray(vertices).astype(np.float32) / img_database[i_img].img_size
        #     faces = [item.strip().split(" ")[1:] for item in data if item[0] == "f"]
        #     graph = nx.Graph()
        #     graph.add_nodes_from([(idx, {"pos_2d": item}) for idx, item in enumerate(vertices)])
        #     new_faces = []  # equal faces - 1 because of the obj format
        #
        #     for id_face, id_edge_per_face in enumerate(faces):
        #         id_edge_per_face = (np.asarray(id_edge_per_face).astype(np.int32) - 1).tolist()
        #         new_faces.append(id_edge_per_face)
        #         id_edge_per_face = [(id_edge_per_face[idx], id_edge_per_face[idx + 1]) for idx in
        #                             range(len(id_edge_per_face) - 1)] + [(id_edge_per_face[-1], id_edge_per_face[0])]
        #         graph.add_edges_from(id_edge_per_face)
        #
        #     graph.graph["faces"] = new_faces
        #
        #     # Mark boundary nodes, lines and faces
        #     for node in graph.nodes():
        #         graph.nodes[node]["valid_flag"] = graph.nodes[node]["pos_2d"][0] != 0 and \
        #                                           graph.nodes[node]["pos_2d"][1] != 0 and \
        #                                           graph.nodes[node]["pos_2d"][0] != 1 and \
        #                                           graph.nodes[node]["pos_2d"][1] != 1
        #     for node1, node2 in graph.edges():
        #         graph.edges[(node1, node2)]["valid_flag"] = graph.nodes[node1]["valid_flag"] and \
        #                                                     graph.nodes[node1]["valid_flag"]
        #     face_flags = []
        #     for id_face, face in enumerate(graph.graph["faces"]):
        #         face_flags.append(min([graph.nodes[point]["valid_flag"] for point in face]))
        #         for id_point in range(len(face)):
        #             id_start = id_point
        #             id_end = (id_start + 1) % len(face)
        #             graph[face[id_start]][face[id_end]]["id_face"] = id_face
        #
        #     graph.graph["face_flags"] = face_flags
        #     # print("Read {}/{} vertices".format(vertices.shape[0], len(graph.nodes)))
        #     # print("Read {} faces".format(len(faces)))
        #     graphs.append(graph)
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
        for idx, face in enumerate(v_graph.graph["faces"]):
            # print(idx)
            vertices = [v_graph.nodes[id_node]["pos_2d"] for id_node in face]
            cv2.polylines(line_img1, [(np.asarray(vertices) * img.img_size).astype(np.int32)], True, (0, 0, 255),
                          thickness=1)
            # cv2.imshow("1", line_img1)
            # cv2.waitKey()

        # Draw target patch
        for id_patch in v_viz_face:
            vertices_t = [v_graph.nodes[id_node]["pos_2d"] for id_node in v_graph.graph["faces"][id_patch]]
            cv2.polylines(line_img1, [(np.asarray(vertices_t) * img.img_size).astype(np.int32)], True, (0, 255, 0),
                          thickness=1)
            for item in vertices_t:
                cv2.circle(line_img1, (item * img.img_size).astype(np.int32), 1, (0, 255, 255), 2)
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


@hydra.main(config_name="phase5.yaml", config_path="../../../configs/neural_recon/", version_base="1.1")
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
