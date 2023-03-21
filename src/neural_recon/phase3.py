import itertools
import sys, os
import time

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
    pad_and_enlarge_along_y, refresh_timer

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
from src.neural_recon.phase1 import NGPModel


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_length):
        super(Dummy_dataset, self).__init__()
        self.length = v_length
        pass

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.long)

    def __len__(self):
        return self.length


class LModel(nn.Module):
    def __init__(self, v_data):
        super(LModel, self).__init__()
        self.seg_distance_normalizer = 300
        self.graph1 = v_data["graph1"]
        self.graph2 = v_data["graph2"]
        self.register_buffer("ray_c", torch.tensor(
            [self.graph1.nodes[id_node]["ray_c"].tolist() for id_node in self.graph1.nodes()],
            dtype=torch.float32))  # (M, 2)
        self.register_buffer("seg_distance",
                             torch.tensor([self.graph1.nodes[id_node]["distance"] for id_node in self.graph1.nodes()],
                                          dtype=torch.float32))  # (M, 2)
        self.seg_distance /= self.seg_distance_normalizer

        v_up_c = []
        # self.v_up_dict = {item: {} for item in range(len(self.graph1.graph["faces"]))}
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                # self.v_up_dict[id_patch][id_segment] = len(v_up_c)
                id_end = face_ids[(id_segment + 1) % len(face_ids)]
                up_c = self.graph1.edges[(face_ids[id_segment], id_end)]["up_c"][id_patch]
                v_up_c.append(up_c.tolist())

        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up = torch.tensor(v_up_c).float()
        self.v_up = vector_to_sphere_coordinate(self.v_up)
        self.v_up[:, 0] = self.v_up[:, 0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up[:, 1] = self.v_up[:, 1] / self.theta_normalizer  # [0,pi] -> [0,1]

        self.length_normalizer = 5
        self.vertical_length1 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length2 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length1 /= self.length_normalizer
        self.vertical_length2 /= self.length_normalizer

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=True)
        self.vertical_length1 = nn.Parameter(self.vertical_length1, requires_grad=False)
        self.vertical_length2 = nn.Parameter(self.vertical_length2, requires_grad=False)
        self.v_up = nn.Parameter(self.v_up, requires_grad=True)

        self.register_buffer("intrinsic1", torch.as_tensor(v_data["intrinsic1"]).float())
        self.register_buffer("intrinsic2", torch.as_tensor(v_data["intrinsic2"]).float())
        self.register_buffer("extrinsic1", torch.as_tensor(v_data["extrinsic1"]).float())
        self.register_buffer("extrinsic2", torch.as_tensor(v_data["extrinsic2"]).float())

        self.img_model1 = v_data["img_model1"]
        self.img_model2 = v_data["img_model2"]
        for p in self.img_model1.parameters():
            p.requires_grad = False
        for p in self.img_model2.parameters():
            p.requires_grad = False

        # Visualization
        viz_shape = (6000, 4000)

        self.register_buffer("o_rgb1",
                             torch.asarray(v_data["rgb1"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.register_buffer("o_rgb2",
                             torch.asarray(v_data["rgb2"].copy().astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(
                                 0))
        self.rgb1 = cv2.resize(v_data["rgb1"], viz_shape, cv2.INTER_AREA)
        self.rgb2 = cv2.resize(v_data["rgb2"], viz_shape, cv2.INTER_AREA)

        # Debug
        self.id_viz_patch = v_data["id_patch"]

        # Batch index
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

    def __init__1(self, v_data):
        super(LModel, self).__init__()
        self.seg_distance_normalizer = 300
        self.seg_distance = torch.tensor(v_data["seg_distances"]).float()
        self.seg_distance /= self.seg_distance_normalizer

        self.length_normalizer = 5
        self.vertical_length1 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length2 = torch.tensor(0.5, dtype=torch.float)
        self.vertical_length1 /= self.length_normalizer
        self.vertical_length2 /= self.length_normalizer

        self.phi_normalizer = 2 * math.pi
        self.theta_normalizer = math.pi
        self.v_up1 = torch.tensor(v_data["v_up_c"]).float()
        self.v_up2 = -torch.tensor(v_data["v_up_c"]).float()
        self.v_up1 = vector_to_sphere_coordinate(self.v_up1)
        self.v_up2 = vector_to_sphere_coordinate(self.v_up2)
        self.v_up1[0] = self.v_up1[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up2[0] = self.v_up2[0] / self.phi_normalizer + 0.5  # [-pi,pi] -> [0,1]
        self.v_up1[1] = self.v_up1[1] / self.theta_normalizer  # [0,pi] -> [0,1]
        self.v_up2[1] = self.v_up2[1] / self.theta_normalizer  # [0,pi] -> [0,1]

        self.seg_distance = nn.Parameter(self.seg_distance, requires_grad=False)
        self.v_up1 = nn.Parameter(self.v_up1, requires_grad=True)
        self.v_up2 = nn.Parameter(self.v_up2, requires_grad=True)
        self.vertical_length1 = nn.Parameter(self.vertical_length1, requires_grad=False)
        self.vertical_length2 = nn.Parameter(self.vertical_length2, requires_grad=False)

        self.seg_2d = torch.tensor(v_data["seg2d"]).float().cuda()
        self.ray_c = torch.tensor(v_data["ray_c"]).float().cuda()
        self.intrinsic1 = torch.as_tensor(v_data["intrinsic1"]).float().cuda()
        self.intrinsic2 = torch.as_tensor(v_data["intrinsic2"]).float().cuda()
        self.extrinsic1 = torch.as_tensor(v_data["extrinsic1"]).float().cuda()
        self.extrinsic2 = torch.as_tensor(v_data["extrinsic2"]).float().cuda()

        self.img_model1 = v_data["img_model1"]
        self.img_model1.freeze()
        self.img_model2 = v_data["img_model2"]
        self.img_model2.freeze()

        # Visualization
        self.rgb1 = v_data["rgb1"]
        self.rgb2 = v_data["rgb2"]

    def denormalize(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer
        v_up11 = (self.v_up[:, 0] - 0.5) * self.phi_normalizer
        v_up12 = self.v_up[:, 1] * self.theta_normalizer
        v_up = sphere_coordinate_to_vector(v_up11, v_up12)

        vertical_length1 = self.vertical_length1 * self.length_normalizer
        vertical_length2 = self.vertical_length2 * self.length_normalizer
        return seg_distance, v_up, vertical_length1, vertical_length2

    def forward(self, v_index, v_id_epoch, is_log):
        v_is_debug = False
        time_profile = [0 for _ in range(10)]
        timer = time.time()

        # 0: Unpack data
        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        point_index = list(itertools.chain(*[self.edge_point_index[item] for item in v_index]))
        edge_points = point_pos_c[point_index].reshape((-1, 4, 3))
        start_point = edge_points[:, 0]
        end_point = edge_points[:, 1]
        next_point = edge_points[:, 2]
        prev_point = edge_points[:, 3]
        up_cur = v_up[point_index].reshape((-1, 4, 3))[:, 0]
        up_next = v_up[point_index].reshape((-1, 4, 3))[:, 3]
        up_prev = v_up[point_index].reshape((-1, 4, 3))[:, 2]

        cur_dir = end_point - start_point
        cur_length = torch.linalg.norm(cur_dir, dim=1)
        cur_dir = cur_dir / cur_length[:, None]
        center_point = (end_point + start_point) / 2
        time_profile[0], timer = refresh_timer(timer)

        # 1-7: compute_roi
        half_window_size_meter_horizontal = cur_length  # m
        half_window_size_meter_vertical = vertical_length1  # m
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
        dx = torch.arange(num_horizontal.sum()).to(begin_idxes.device) - begin_idxes.repeat_interleave(
            num_horizontal)  # (total_num_x_coords,)
        dx = dx / (num_horizontal - 1).repeat_interleave(
            num_horizontal) * half_window_size_meter_horizontal.repeat_interleave(
            num_horizontal)  # (total_num_x_coords,)
        dy = torch.arange(num_vertical).to(begin_idxes.device) / (num_vertical - 1) * half_window_size_meter_vertical
        time_profile[1], timer = refresh_timer(timer)

        # Meshgrid
        total_num_coords = total_num_x_coords * dy.shape[0]
        coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * num_vertical)  # (total_num_coords,)
        coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
        coords = torch.stack((coords_x, coords_y), dim=1)
        time_profile[2], timer = refresh_timer(timer)

        # Local -> Global image coordinates
        interpolated_coordinates_camera = cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] \
                                          + up_cur.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:,
                                                                                                        None] \
                                          + start_point.repeat_interleave(num_coordinates_per_edge, dim=0)
        time_profile[3], timer = refresh_timer(timer)

        roi_coor_2d = (self.intrinsic1 @ interpolated_coordinates_camera.T).T
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        valid_mask1 = torch.logical_and(roi_coor_2d > 0, roi_coor_2d < 1)
        valid_mask1 = torch.logical_and(valid_mask1[:, 0], valid_mask1[:, 1])
        roi_coor_2d = torch.clamp(roi_coor_2d, 0, 1)
        time_profile[4], timer = refresh_timer(timer)
        sample_imgs1 = sample_img_prediction(self.img_model1, roi_coor_2d[None, :, :])[0]
        time_profile[5], timer = refresh_timer(timer)

        # Second img
        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        roi_coor_2d_img2 = torch.transpose(
            transformation @ torch.transpose(to_homogeneous_tensor(interpolated_coordinates_camera), 0, 1), 0, 1)
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        valid_mask2 = torch.logical_and(roi_coor_2d_img2 > 0, roi_coor_2d_img2 < 1)
        valid_mask2 = torch.logical_and(valid_mask2[:, 0], valid_mask2[:, 1])
        roi_coor_2d_img2 = torch.clamp(roi_coor_2d_img2, 0, 1)
        sample_imgs2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2[None, :, :])[0]
        time_profile[6], timer = refresh_timer(timer)

        similarity_loss = nn.functional.mse_loss(sample_imgs1, sample_imgs2)
        time_profile[7], timer = refresh_timer(timer)

        # 8: Normal loss
        v_cur_normal = normalize_tensor(torch.cross(cur_dir, up_cur))
        next_dir = normalize_tensor(next_point - end_point)
        v_next_normal = normalize_tensor(torch.cross(next_dir, up_next))
        prev_dir = normalize_tensor(start_point - prev_point)
        v_prev_normal = normalize_tensor(torch.cross(prev_dir, up_prev))
        normal_loss_next = (1 - (v_next_normal * v_cur_normal).sum(dim=1).mean()) / 2  # [0, 2] -> [0, 1]
        normal_loss_prev = (1 - (v_prev_normal * v_cur_normal).sum(dim=1).mean()) / 2  # [0, 2] -> [0, 1]
        normal_loss = (normal_loss_next + normal_loss_prev) / 2
        time_profile[8], timer = refresh_timer(timer)

        # total_loss = normal_loss * 0.5 + similarity_loss * 0.5
        total_loss = normal_loss * 0.25 + similarity_loss * 0.75

        # 9: Viz
        if self.id_viz_patch in v_index and is_log:
            id_pos = torch.where(v_index == self.id_viz_patch)[0]

            line_img1_base = self.rgb1.copy()
            shape = line_img1_base.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1_base, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            for idx, _ in enumerate(self.graph1.graph["faces"][self.id_viz_patch]):
                id_start = self.graph1.graph["faces"][self.id_viz_patch][idx]
                id_end = self.graph1.graph["faces"][self.id_viz_patch][
                    (idx + 1) % len(self.graph1.graph["faces"][self.id_viz_patch])]
                line_img1 = line_img1_base.copy()
                # Selected segment
                cv2.line(line_img1,
                         (self.graph1.nodes[id_start]["pos_2d"] * shape).astype(np.int32),
                         (self.graph1.nodes[id_end]["pos_2d"] * shape).astype(np.int32),
                         (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

                id_magic = idx + len(
                    list(itertools.chain(*[self.edge_point_index[item] for item in v_index[:id_pos]]))) // 4

                # RoI
                # Compute extreme point
                point1 = start_point[id_magic]
                point2 = start_point[id_magic] + up_cur[id_magic] * half_window_size_meter_vertical
                point3 = end_point[id_magic] + up_cur[id_magic] * half_window_size_meter_vertical
                point4 = end_point[id_magic]
                roi = torch.stack((point1, point2, point3, point4), dim=0)
                roi_2d1 = torch.transpose(self.intrinsic1 @ torch.transpose(roi, 0, 1), 0, 1)
                roi_2d1 = roi_2d1[:, :2] / roi_2d1[:, 2:3]

                roi_2d_numpy = roi_2d1.detach().cpu().numpy()
                line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (0, 0, 255),
                                          thickness=2, lineType=cv2.LINE_AA)

                roi_2d2 = (transformation @ to_homogeneous_tensor(roi).T).T
                roi_2d2 = roi_2d2[:, :2] / roi_2d2[:, 2:3]
                line_img2 = self.rgb2.copy()
                roi_2d_numpy = roi_2d2.detach().cpu().numpy()
                line_img2 = cv2.polylines(line_img2, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (0, 0, 255),
                                          thickness=2, lineType=cv2.LINE_AA)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((line_img1, line_img2), axis=0))
                if v_is_debug:
                    print("Visualize the calculated roi")
                    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("1", 1600, 900)
                    cv2.moveWindow("1", 5, 5)
                    cv2.imshow("1", line_img1)
                    cv2.waitKey()

                # Visualize query points
                # For pixel coordinate
                id_pixel_start = torch.sum(num_coordinates_per_edge[:id_magic])
                w = num_horizontal[id_magic]
                h = num_vertical

                line_img1 = self.rgb1.copy()
                shape = line_img1.shape[:2][::-1]

                roi_coor_2d1_numpy = roi_coor_2d[id_pixel_start:id_pixel_start + h * w].detach().cpu().numpy()
                sampled_img1_numpy = normalized_torch_img_to_numpy(
                    sample_imgs1[id_pixel_start:id_pixel_start + h * w].reshape((h, w, 3)).permute(2, 0, 1))
                roi_coor_2d2_numpy = roi_coor_2d_img2[id_pixel_start:id_pixel_start + h * w].detach().cpu().numpy()
                sampled_img2_numpy = normalized_torch_img_to_numpy(
                    sample_imgs2[id_pixel_start:id_pixel_start + h * w].reshape((h, w, 3)).permute(2, 0, 1))

                for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                    cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                img1 = pad_and_enlarge_along_y(sampled_img1_numpy, line_img1)
                line_img2 = self.rgb2.copy()
                # line_img2 = cv2.resize(line_img2, (600, 400))
                shape = line_img2.shape[:2][::-1]
                for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                    cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
                img2 = pad_and_enlarge_along_y(sampled_img2_numpy, line_img2)

                cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(idx, v_id_epoch),
                            np.concatenate((img1, img2), axis=0))
                if v_is_debug:
                    print("Visualize the extracted region")
                    cv2.imshow("1", np.concatenate((img1, img2), axis=0))
                    cv2.waitKey()
        time_profile[9], timer = refresh_timer(timer)

        return total_loss, normal_loss, similarity_loss

    def debug_save(self, v_index):
        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()
        point_pos_c = self.ray_c * seg_distance[:, None]

        def get_mesh(v_edge_point_index):
            total_edge_points = point_pos_c[v_edge_point_index].reshape((-1, 4, 3))
            total_up = v_up[v_edge_point_index].reshape((-1, 4, 3))

            center_point_c = (total_edge_points[:, 0] + total_edge_points[:, 1]) / 2
            up_point = center_point_c + total_up[:, 0]

            center_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(center_point_c).T).T)[:,
                             :3].cpu().numpy()
            up_vector_w = normalize_vector(((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(up_point).T).T)[:,
                                           :3].cpu().numpy() - center_point_w)

            arrows = o3d.geometry.TriangleMesh()
            for i in range(center_point_w.shape[0]):
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00005, cone_radius=0.00007,
                                                               cylinder_height=0.00025, cone_height=0.00025,
                                                               resolution=3, cylinder_split=1)
                arrow.rotate(caculate_align_mat(up_vector_w[i]), center=(0, 0, 0))
                arrow.translate(center_point_w[i])
                arrows += arrow

            start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c).T).T)[:,
                            :3].cpu().numpy()
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(start_point_w)
            lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
            edges = np.asarray(list(self.graph1.edges()))
            lineset.lines = o3d.utility.Vector2iVector(edges)
            return lineset, arrows

        # Visualize target patch
        _, arrows = get_mesh(self.edge_point_index[self.id_viz_patch])
        id_points = np.asarray(self.edge_point_index[self.id_viz_patch]).reshape(-1, 4)[:, 0]
        start_point_w = ((torch.inverse(self.extrinsic1) @ to_homogeneous_tensor(point_pos_c[id_points]).T).T)[:,
                        :3].cpu().numpy()
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(start_point_w)
        lineset.colors = o3d.utility.Vector3dVector(np.ones_like(start_point_w))
        lineset.lines = o3d.utility.Vector2iVector(np.stack((
            np.arange(start_point_w.shape[0]), (np.arange(start_point_w.shape[0]) + 1) % start_point_w.shape[0]
        ), axis=1))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/target_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/target_{}_line.ply".format(v_index), lineset)
        # Visualize total
        lineset, arrows = get_mesh(list(itertools.chain(*self.edge_point_index)))
        o3d.io.write_triangle_mesh(r"output/img_field_test/imgs_log/total_{}_arrow.obj".format(v_index), arrows)
        o3d.io.write_line_set(r"output/img_field_test/imgs_log/total_{}_line.ply".format(v_index), lineset)

    def len(self):
        return len(self.graph1.graph["faces"])

    # Out of date
    def denormalize2(self):
        seg_distance = self.seg_distance * self.seg_distance_normalizer

        vertical_length1 = self.vertical_length1 * self.length_normalizer
        vertical_length2 = self.vertical_length2 * self.length_normalizer
        return seg_distance, vertical_length1, vertical_length2

    def find_up_vector(self, id_patch, id_start, id_end, v_up):
        id_up = -1
        if (id_start, id_end) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_start, id_end)]
        elif (id_end, id_start) in self.v_up_dict[id_patch]:
            id_up = self.v_up_dict[id_patch][(id_end, id_start)]
        else:
            raise
        return v_up[id_up]

    def compute_similarity_loss(self,
                                start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                v_is_debug, v_is_log, v_log_frequency, v_index, id_segment
                                ):
        roi_bound1, roi_coor1, roi_2d1 = compute_roi(
            start_c, end_c, (start_c + end_c) / 2,
            cur_dir, v_up_c, vertical_length1,
            self.intrinsic1
        )

        # Visualize region
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            shape = line_img1.shape[:2][::-1]

            # Original 2D polygon
            polygon = [self.graph1.nodes[id_point]["pos_2d"] for id_point in
                       self.graph1.graph["faces"][self.id_viz_patch]]
            polygon = (np.asarray(polygon) * shape).astype(np.int32)
            cv2.polylines(line_img1, [polygon], True,
                          (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            # Selected segment
            cv2.line(line_img1,
                     (self.graph1.nodes[id_start]["pos_2d"] * shape).astype(np.int32),
                     (self.graph1.nodes[id_end]["pos_2d"] * shape).astype(np.int32),
                     (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            # RoI
            roi_2d_numpy = roi_2d1.detach().cpu().numpy()
            line_img1 = cv2.polylines(line_img1, [(roi_2d_numpy * shape).astype(np.int32).reshape(-1, 1, 2)], True,
                                      (255, 0, 0),
                                      thickness=1, lineType=cv2.LINE_AA)

            # cv2.circle(line_img1, (roi_2d_numpy[0] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[1] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[2] * shape).astype(np.int32), 10, (0, 0, 255), 10)
            # cv2.circle(line_img1, (roi_2d_numpy[3] * shape).astype(np.int32), 10, (0, 0, 255), 10)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\2d_{}_{:05d}.jpg".format(id_segment, v_index),
                        line_img1)
            if v_is_debug:
                print("Visualize the calculated roi")
                cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1", 1600, 900)
                cv2.moveWindow("1", 5, 5)
                cv2.imshow("1", line_img1)
                cv2.waitKey()

        roi_coor_shape = roi_coor1.shape
        roi_coor_2d = torch.transpose(self.intrinsic1 @ torch.transpose(roi_coor1.reshape((-1, 3)), 0, 1), 0, 1)
        roi_coor_2d = roi_coor_2d[:, :2] / roi_coor_2d[:, 2:3]
        roi_coor_2d = roi_coor_2d.reshape(roi_coor_shape[:2] + (2,))

        sampled_img = sample_img_prediction(self.img_model1, roi_coor_2d)
        # sampled_img = sample_img(self.o_rgb1, roi_coor_2d)

        transformation = to_homogeneous_mat_tensor(self.intrinsic2) @ self.extrinsic2 @ torch.inverse(self.extrinsic1)
        roi_coor_2d_img2 = torch.transpose(
            transformation @ torch.transpose(to_homogeneous_tensor(roi_coor1.reshape((-1, 3))), 0, 1), 0, 1)
        roi_coor_2d_img2 = roi_coor_2d_img2[:, :2] / roi_coor_2d_img2[:, 2:3]
        roi_coor_2d_img2 = roi_coor_2d_img2.reshape(roi_coor_shape[:2] + (2,))
        viz_sampled_img2 = sample_img_prediction(self.img_model2, roi_coor_2d_img2)
        # viz_sampled_img2 = sample_img(self.o_rgb2, roi_coor_2d_img2)

        # Visualize query points
        if v_is_debug or (v_is_log and v_index % v_log_frequency == 0):
            line_img1 = self.rgb1.copy()
            # line_img1 = cv2.resize(line_img1, (600, 400))
            shape = line_img1.shape[:2][::-1]

            roi_coor_2d1_numpy = roi_coor_2d.detach().cpu().numpy()
            sampled_img1_numpy = normalized_torch_img_to_numpy(sampled_img.permute(2, 0, 1))
            roi_coor_2d2_numpy = roi_coor_2d_img2.detach().cpu().numpy()
            sampled_img2_numpy = normalized_torch_img_to_numpy(viz_sampled_img2.permute(2, 0, 1))

            for item in roi_coor_2d1_numpy.reshape((-1, 2)):
                cv2.circle(line_img1, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img1 = pad_and_enlarge_along_y(sampled_img1_numpy, line_img1)
            line_img2 = self.rgb2.copy()
            # line_img2 = cv2.resize(line_img2, (600, 400))
            shape = line_img2.shape[:2][::-1]
            for item in roi_coor_2d2_numpy.reshape((-1, 2)):
                cv2.circle(line_img2, (item * shape).astype(np.int32), 1, (0, 0, 255), 1)
            img2 = pad_and_enlarge_along_y(sampled_img2_numpy, line_img2)

            cv2.imwrite(r"D:\repo\python\output\img_field_test\imgs_log\3d_{}_{:05d}.jpg".format(id_segment, v_index),
                        np.concatenate((img1, img2), axis=0))
            if v_is_debug:
                print("Visualize the extracted region")
                cv2.imshow("1", np.concatenate((img1, img2), axis=0))
                cv2.waitKey()

        loss = torch.nn.functional.mse_loss(sampled_img, viz_sampled_img2)
        return loss

    def compute_normal_consistency(self,
                                   point_pos_c, id1, id2, v_ref_normal, v_target_normal
                                   ):
        point1 = point_pos_c[id1]
        point2 = point_pos_c[id2]
        cur_dir = normalize_tensor(point2 - point1)

        v_ref_normal = normalize_tensor(v_ref_normal)
        v_next_normal = normalize_tensor(v_target_normal)

        loss = (1 - v_next_normal.dot(v_ref_normal)) / 2  # [0, 2] -> [0, 1]
        return loss

    def forward_(self, v_index):
        v_is_debug = False
        v_is_log = True
        v_log_frequency = 10000

        seg_distance, v_up, vertical_length1, vertical_length2 = self.denormalize()

        point_pos_c = self.ray_c * seg_distance[:, None]
        normal_losses = []
        similarity_losses = []
        time_similarity = 0
        time_consistency = 0
        # for face_ids in self.graph1.graph["faces"][self.id_patch,self.id_patch+1]:
        for id_patch, face_ids in enumerate(self.graph1.graph["faces"]):
            for id_segment in range(len(face_ids)):
                id_start = face_ids[id_segment]
                id_end = face_ids[(id_segment + 1) % len(face_ids)]

                start_c = point_pos_c[id_start]
                end_c = point_pos_c[id_end]

                cur_dir = normalize_tensor(end_c - start_c)

                v_up_c = self.find_up_vector(id_patch, id_start, id_end, v_up)

                a = time.time()
                loss = self.compute_similarity_loss(start_c, end_c, cur_dir, v_up_c, vertical_length1, id_start, id_end,
                                                    v_is_debug and id_patch == self.id_viz_patch,
                                                    v_is_log and id_patch == self.id_viz_patch, v_log_frequency,
                                                    v_index,
                                                    id_segment)
                time_similarity += time.time() - a
                a = time.time()
                # Normal loss
                v_cur_normal = torch.cross(cur_dir, v_up_c)
                id_prev = face_ids[(id_segment - 1) % len(face_ids)]
                id_next = face_ids[(id_segment + 2) % len(face_ids)]

                v_next_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_end, id_next, v_up))
                v_prev_normal = torch.cross(cur_dir, self.find_up_vector(id_patch, id_prev, id_start, v_up))
                normal_loss1 = self.compute_normal_consistency(point_pos_c, id_prev, id_start, v_cur_normal,
                                                               v_prev_normal)
                normal_loss2 = self.compute_normal_consistency(point_pos_c, id_end, id_next, v_cur_normal,
                                                               v_next_normal)
                normal_loss = (normal_loss1 + normal_loss2) / 2
                time_consistency += time.time() - a
                normal_losses.append(normal_loss)
                similarity_losses.append(loss)

        normal_losses = torch.stack(normal_losses).mean()
        similarity_losses = torch.stack(similarity_losses).mean()
        total_loss = normal_losses * 0.5 + similarity_losses * 0.5
        return total_loss, normal_losses, similarity_losses


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
        self.model = LModel(self.data)

    def train_dataloader(self):
        self.train_dataset = Dummy_dataset(self.model.len())
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        self.valid_dataset = Dummy_dataset(self.model.len())
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        total_loss, normal_losses, similarity_losses = self.model(batch, self.current_epoch, False)

        self.log("Training_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)
        self.log("Training_Normal_Loss", normal_losses.detach(), prog_bar=True, logger=True, on_step=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=1)
        self.log("Training_Similarity_Loss", similarity_losses.detach(), prog_bar=True, logger=True, on_step=True,
                 sync_dist=True,
                 on_epoch=True,
                 batch_size=1)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, normal_losses, similarity_losses = self.model(batch, self.current_epoch, True)

        self.log("Validation_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)

        return total_loss

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        self.model.debug_save(self.current_epoch)

        if self.trainer.sanity_checking:
            return


def prepare_dataset_and_model(v_colmap_dir):
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

    id_patch = 1522
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
        ray_c = ray_c / np.linalg.norm(ray_c, axis=1, keepdims=True)  # Normalize the points
        nearest_candidates = points_from_sfm_camera[index_shortest_distance]  # (M, K, 3)
        # Compute the shortest distance from the candidate point to the ray for each query point
        distance_of_projection = nearest_candidates @ ray_c[:, :,
                                                      np.newaxis]  # (M, K, 1): K projected distance of the candidate point along each ray
        projected_points_on_ray = distance_of_projection * ray_c[:, np.newaxis,
                                                           :]  # (M, K, 3): K projected points along the ray
        distance_from_candidate_points_to_ray = np.linalg.norm(nearest_candidates - projected_points_on_ray,
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

    data = prepare_dataset_and_model(v_cfg["dataset"]["colmap_dir"])

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e6),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        # precision=16,
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
    main()
