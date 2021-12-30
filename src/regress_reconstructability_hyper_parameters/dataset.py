import math
import os
import pickle

import cv2
import hydra
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import List, Tuple

from plyfile import PlyData, PlyElement
from torchvision.transforms import transforms
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats
import multiprocessing as mp
import ctypes
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import farthest_point_sample, \
    index_points, square_distance


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    gussian_distance = torch.exp(-0.5 * (sqrdists * 2) ** 2) / 1.2529964086141667  # (0.5*math.sqrt(6.28))
    gussian_distance[sqrdists > radius ** 2] = 0
    group_idxs = []
    for id_batch in range(B):
        group_idx = torch.multinomial(gussian_distance[id_batch], nsample, replacement=True)
        group_idxs.append(group_idx)
    return torch.stack(group_idxs, dim=0)


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class Regress_hyper_parameters_dataset(torch.utils.data.Dataset):
    def __init__(self, v_params, v_mode):
        super(Regress_hyper_parameters_dataset, self).__init__()
        self.is_training = v_mode == "training"
        self.params = v_params
        self.views = np.load(os.path.join(v_params["model"].preprocess_path, "views.npz"))["arr_0"]
        self.view_pairs = np.load(os.path.join(v_params["model"].preprocess_path, "view_pairs.npz"))["arr_0"]
        self.point_attribute = np.load(os.path.join(v_params["model"].preprocess_path, "point_attribute.npz"))["arr_0"]
        self.view_paths = np.load(os.path.join(v_params["model"].preprocess_path, "view_paths.npz"), allow_pickle=True)[
            "arr_0"]
        # mask = self.target_data[:, 0, -1] < 10
        mask = self.point_attribute[:, 0] < 9999999
        stats.spearmanr(self.point_attribute[:, 0][mask], self.point_attribute[:, 1][mask])

        batch_size = self.params["trainer"]["batch_size"]
        self.num_item = self.point_attribute.shape[0] // batch_size * 5

        whole_index = np.arange(self.point_attribute.shape[0])
        np.random.shuffle(whole_index)
        self.train_index = whole_index[:whole_index.shape[0] // 4 * 3]
        self.test_index = whole_index[whole_index.shape[0] // 4 * 3:]

        pass

    def __getitem__(self, index):
        if self.is_training:
            batch_index = np.random.choice(self.train_index, self.params["trainer"]["batch_size"],
                                           False)
        else:
            # batch_index = np.arange(self.views.shape[0])
            batch_index = self.test_index[[index]]
        output_dict = {
            "views": torch.tensor(self.views[batch_index], dtype=torch.float32),
            "view_pairs": torch.tensor(self.view_pairs[batch_index], dtype=torch.float32),
            "point_attribute": torch.tensor(self.point_attribute[batch_index], dtype=torch.float32),
        }
        return output_dict

    def __len__(self):
        return self.num_item if self.is_training else self.test_index.shape[0]

    @staticmethod
    def collate_fn(batch):
        views = [item["views"] for item in batch]
        view_pairs = [item["view_pairs"] for item in batch]
        point_attribute = [item["point_attribute"] for item in batch]

        return {
            'views': torch.cat(views),
            'view_pairs': torch.cat(view_pairs),
            'point_attribute': torch.cat(point_attribute),
        }


class Regress_hyper_parameters_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_path, v_points_feature_path):
        super(Regress_hyper_parameters_img_dataset, self).__init__()
        self.data_root = v_path
        self.views = np.load(os.path.join(v_path, "../views.npz"))["arr_0"]
        self.num_point_per_patch = v_points_feature_path.shape[0]
        self.points_feature_path = v_points_feature_path

    def __getitem__(self, point_indexes):
        img_features_on_point_list = []
        num_max_points = 0
        for point_index in point_indexes:
            point_path = self.points_feature_path[point_index]
            data = point_path.split("\\")
            data = point_path.split("/") if len(data)==1 else data
            point_path = os.path.join(self.data_root,data[-1])

            img_features_on_point_list.append(torch.tensor(np.load(point_path)["arr_0"], dtype=torch.float32))
            num_features = img_features_on_point_list[-1].shape[1]

            num_max_points = max(num_max_points, img_features_on_point_list[-1].shape[0])

        # Align the features
        point_features = torch.zeros((point_indexes.shape[0], num_max_points, num_features), dtype=torch.float32)
        point_features_mask = torch.ones((point_indexes.shape[0], num_max_points), dtype=torch.bool)
        for id_item, item in enumerate(img_features_on_point_list):
            point_features[id_item, :item.shape[0]] = item
            point_features_mask[id_item, :item.shape[0]] = False

        return point_features, point_features_mask, self.views[point_indexes]

"""
    views: num of patches * max views * 9 (valid_flag, dx, dy, dz, distance, angle to normal, angle to direction, px, py)
    points: num of patches * num point per patch * 7 (x, y, z, index, id_centre)
    point_attribute: baseline recon, max error, avg error, x, y, z, is inconsistent point
"""
class Regress_hyper_parameters_dataset_with_imgs(torch.utils.data.Dataset):
    def __init__(self, v_path, v_params, v_mode):
        super(Regress_hyper_parameters_dataset_with_imgs, self).__init__()
        self.trainer_mode = v_mode
        self.params = v_params
        self.data_root = v_path
        self.views = np.load(os.path.join(v_path, "views.npz"))["arr_0"]
        img_dataset_path = open(os.path.join(v_path, "../img_dataset_path.txt")).readline().strip()

        self.is_involve_img = self.params["model"]["involve_img"]

        self.point_attribute = np.load(os.path.join(v_path, "point_attribute.npz"))["arr_0"]
        self.view_paths = np.load(os.path.join(v_path, "view_paths.npz"), allow_pickle=True)[
            "arr_0"]
        self.original_points = self.point_attribute[:, 3:6]
        self.num_seeds = 4096 + 1024
        self.sample_points_to_different_patches()
        self.img_dataset = Regress_hyper_parameters_img_dataset(img_dataset_path, self.view_paths)

        pass

    """
    Points: num of patches * num point per patch * 7 (x, y, z, index, id_centre)
    """
    def sample_points_to_different_patches(self):
        print("KNN Sample")
        # seed points, radius, max local points,
        accept_sample = False
        self.num_seeds -= 1024
        while not accept_sample:
            new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(
                self.num_seeds, 0.75, int(self.params["model"]["num_points_per_batch"]),
                torch.tensor(self.original_points, dtype=torch.float32).unsqueeze(0),
                torch.arange(self.original_points.shape[0]).unsqueeze(0).unsqueeze(-1),
                True)
            unique_result = np.unique(new_points[0][:, :, 3].reshape(-1).numpy(), return_counts=True)
            if unique_result[0].shape[0] != self.point_attribute.shape[0]:
                print("Uneven sampling, only sample {}/{}".format(unique_result[0].shape[0],
                                                                  self.point_attribute.shape[0]))
                print("Uneven sampling, sampling density: {}".format(np.mean(unique_result[1])))
                # break
            else:
                accept_sample = True
            self.num_seeds += 1024
        if True:
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(new_xyz.numpy()[0, :, :3])
            o3d.io.write_point_cloud("temp/seed_point.ply", pcl)
            # for id_item, item in enumerate(new_points[0]):
            #     pcl = o3d.geometry.PointCloud()
            #     pcl.points = o3d.utility.Vector3dVector(item.numpy()[:, :3])
            #     o3d.io.write_point_cloud("{}.ply".format(id_item), pcl)
            pcl.points = o3d.utility.Vector3dVector(new_points.numpy()[0, :, :, :3].reshape([-1, 3]))
            o3d.io.write_point_cloud("temp/total_point.ply", pcl)

        self.points = torch.cat([new_points[0],(fps_idx[0].unsqueeze(1).repeat(1,256)).unsqueeze(2)],dim=2)
        mask = self.point_attribute[:, 0] < 9999999

        stats.spearmanr(self.point_attribute[:, 0][mask], self.point_attribute[:, 2][mask])

        self.num_item = self.points.shape[0]

        self.whole_index = np.arange(self.num_item)
        np.random.shuffle(self.whole_index)
        self.train_index = self.whole_index[:self.whole_index.shape[0] // 4 * 3]
        self.validation_index = self.whole_index[self.whole_index.shape[0] // 4 * 3:]

    def __getitem__(self, index):
        if self.trainer_mode == "training":
            used_index = self.train_index
        elif self.trainer_mode == "validation":
            used_index = self.validation_index
        else:
            used_index = self.whole_index
        point_indexes = self.points[used_index[index], :, 3].int()

        point_features,point_features_mask,img_pose = None,None,None
        if self.is_involve_img:
            point_features, point_features_mask, img_pose = self.img_dataset[point_indexes]

        output_dict = {
            # "views": torch.tensor(np.load(self.views_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            # "view_pairs": torch.tensor(np.load(self.view_pairs_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            "views": torch.tensor(self.views[point_indexes], dtype=torch.float32),
            # "view_pairs": torch.tensor(self.view_pairs[point_indexes], dtype=torch.float32),

            "point_attribute": torch.tensor(self.point_attribute[point_indexes], dtype=torch.float32),
            "points": self.points[used_index[index]],
            "point_features": point_features,
            "point_features_mask": point_features_mask,
            "img_pose": torch.tensor(img_pose, dtype=torch.float32) if img_pose is not None else torch.tensor(self.views[point_indexes], dtype=torch.float32),
        }
        return output_dict

    def __len__(self):
        if self.trainer_mode == "training":
            return self.train_index.shape[0]
        elif self.trainer_mode == "validation":
            return self.validation_index.shape[0]
        else:
            return self.whole_index.shape[0]
            # return 32

    @staticmethod
    def collate_fn(batch):
        views = [torch.transpose(item["views"], 0, 1) for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        views_pad = torch.transpose(torch.transpose(pad_sequence(views), 0, 1), 1, 2)
        # view_pairs = [torch.transpose(item["view_pairs"],0,1) for item in batch]
        # view_pairs_pad = torch.transpose(torch.transpose(pad_sequence(view_pairs),0,1),1,2)
        point_attribute = [item["point_attribute"] for item in batch]
        point_features = [item["point_features"] for item in batch]
        points = [item["points"] for item in batch]
        point_features_mask = [item["point_features_mask"] for item in batch]
        img_pose = [torch.transpose(item["img_pose"],0,1) for item in batch]
        img_pose_pad = torch.transpose(torch.transpose(pad_sequence(img_pose), 0, 1), 1, 2)
        return {
            'views': views_pad,
            # 'view_pairs': view_pairs_pad,
            'point_attribute': torch.stack(point_attribute, dim=0),
            'point_features': point_features,
            'point_features_mask': point_features_mask,
            'points': torch.stack(points, dim=0),
            'img_pose': img_pose_pad,
        }
