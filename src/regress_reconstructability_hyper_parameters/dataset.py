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
    def __init__(self, v_path, v_params, v_mode):
        super(Regress_hyper_parameters_dataset, self).__init__()
        self.trainer_mode = v_mode
        self.params = v_params
        self.data_root = v_path
        self.views = np.load(os.path.join(v_path, "views.npz"))["arr_0"]
        self.point_attribute = np.load(os.path.join(v_path, "point_attribute.npz"))["arr_0"]
        self.points = np.concatenate(
            [self.point_attribute[:, 3:6], np.arange(self.point_attribute.shape[0])[:, np.newaxis]], axis=1)
        self.num_item = self.point_attribute.shape[0]

        self.whole_index = np.arange(self.num_item)
        np.random.shuffle(self.whole_index)
        self.train_index = self.whole_index[:self.whole_index.shape[0] // 4 * 3]
        self.validation_index = self.whole_index[self.whole_index.shape[0] // 4 * 3:]

        pass

    def sample_points_to_different_patches(self):
        pass

    def __getitem__(self, index):
        if self.trainer_mode == "training":
            used_index = self.train_index
        elif self.trainer_mode == "validation":
            used_index = self.validation_index
        else:
            used_index = self.whole_index
        output_dict = {
            "views": torch.tensor(self.views[used_index[index]], dtype=torch.float32),
            "point_attribute": torch.tensor(self.point_attribute[used_index[index]], dtype=torch.float32),
            "points": torch.tensor(self.points[used_index[index]], dtype=torch.float32),
        }
        return output_dict

    def __len__(self):
        if self.trainer_mode == "training":
            return self.train_index.shape[0]
        elif self.trainer_mode == "validation":
            return self.validation_index.shape[0]
        else:
            return self.whole_index.shape[0]

    @staticmethod
    def collate_fn(batch):
        views = [item["views"] for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        views_pad = torch.transpose(pad_sequence(views, padding_value=0.0), 0, 1)
        # view_pairs = [torch.transpose(item["view_pairs"],0,1) for item in batch]
        # view_pairs_pad = torch.transpose(torch.transpose(pad_sequence(view_pairs),0,1),1,2)
        point_attribute = [item["point_attribute"] for item in batch]
        points = [item["points"] for item in batch]
        return {
            'views': views_pad.unsqueeze(1),
            'point_attribute': torch.stack(point_attribute, dim=0).unsqueeze(1),
            'points': torch.stack(points, dim=0).unsqueeze(1),
        }


class Regress_hyper_parameters_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_path, v_paths):
        super(Regress_hyper_parameters_img_dataset, self).__init__()
        self.data_root = v_path
        self.data_dirs = os.listdir(self.data_root)
        self.points_feature_path = v_paths

    def __getitem__(self, point_indexes):
        img_features_on_point_list = []
        num_max_points = 0
        real_indexes = []
        for point_index in point_indexes:
            point_path = self.points_feature_path[point_index]
            if point_path=="":
                img_features_on_point_list.append(torch.zeros((1, 32), dtype=torch.float32))
            else:
                data = point_path.split("\\")
                data = point_path.split("/") if len(data) == 1 else data
                point_path = os.path.join(self.data_root, data[0].strip(), "point_features", data[1].strip())
                # point_path[-1]="y"
                real_indexes.append(int(data[-1].split(".")[0]))
                if not os.path.exists(point_path):
                    img_features_on_point_list.append(torch.zeros((1, 32), dtype=torch.float32))
                else:
                    img_features_on_point_list.append(torch.tensor(np.load(point_path)["arr_0"], dtype=torch.float32))
            num_features = img_features_on_point_list[-1].shape[1]

            num_max_points = max(num_max_points, img_features_on_point_list[-1].shape[0])

        # Align the features
        point_features = torch.zeros((point_indexes.shape[0], num_max_points, num_features), dtype=torch.float32)
        point_features_mask = torch.ones((point_indexes.shape[0], num_max_points), dtype=torch.bool)
        for id_item, item in enumerate(img_features_on_point_list):
            point_features[id_item, :item.shape[0]] = item
            point_features_mask[id_item, :item.shape[0]] = False

        return point_features, point_features_mask


"""
    views: num of patches * max views * 8 (valid_flag, delta_theta, delta_phi, distance, angle to normal, angle to direction, px, py)
    points: num of patches * num point per patch * 7 (x, y, z, index, id_centre)
    point_attribute: baseline recon, avg recon error, avg gt error, x, y, z, is inconsistent point, nx, ny, nz
"""
class Regress_hyper_parameters_dataset_with_imgs(torch.utils.data.Dataset):
    def __init__(self, v_path, v_params, v_mode):
        super(Regress_hyper_parameters_dataset_with_imgs, self).__init__()
        self.trainer_mode = v_mode
        self.params = v_params
        self.data_root = v_path
        self.views = np.load(os.path.join(v_path, "views.npy"), mmap_mode="r+")
        self.point_attribute = np.load(os.path.join(v_path, "point_attribute.npy"), mmap_mode="r+")
        self.view_paths = np.load(os.path.join(v_path, "view_paths.npy"), mmap_mode="r+", allow_pickle=True)
        img_dataset_path = os.path.join(v_path, "../")
        assert os.path.exists(img_dataset_path)
        self.is_involve_img = self.params["model"]["involve_img"]

        self.scene_name = self.data_root.split("\\")
        self.scene_name = self.scene_name if len(self.scene_name) != 1 else self.scene_name[0].split("/")
        self.scene_name = self.scene_name[-1] if self.scene_name[-1] != "" else self.scene_name[-2]

        self.view_mean_std = np.array(self.params["model"]["view_mean_std"])
        self.error_mean_std = np.array(self.params["model"]["error_mean_std"])

        assert self.point_attribute.shape[1] == 10
        # self.original_points = self.point_attribute[:, 3:6]
        self.num_seeds = 4096 + 1024
        # self.num_seeds = 20
        self.sample_points_to_different_patches()
        if self.is_involve_img:
            self.img_dataset = Regress_hyper_parameters_img_dataset(img_dataset_path, self.view_paths)

    def sample_points_to_different_patches(self):
        # if True:
        #     new_xyz = torch.tensor(self.original_points, dtype=torch.float32).unsqueeze(0)
        #     new_points = torch.zeros_like(new_xyz).unsqueeze(2)
        #     fps_idx = torch.arange(self.original_points.shape[0]).unsqueeze(0)
        # else:
        #     print("KNN Sample")
        #     # seed points, radius, max local points,
        #     accept_sample = False
        #     self.num_seeds -= 1024
        #     while not accept_sample:
        #         new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(
        #             self.num_seeds, 0.75, int(self.params["model"]["num_points_per_batch"]),
        #             torch.tensor(self.original_points, dtype=torch.float32).unsqueeze(0),
        #             torch.arange(self.original_points.shape[0]).unsqueeze(0).unsqueeze(-1),
        #             True)
        #         unique_result = np.unique(new_points[0][:, :, 3].reshape(-1).numpy(), return_counts=True)
        #         if unique_result[0].shape[0] != self.point_attribute.shape[0]:
        #             print("Uneven sampling, only sample {}/{}".format(unique_result[0].shape[0],
        #                                                               self.point_attribute.shape[0]))
        #             print("Uneven sampling, sampling density: {}".format(np.mean(unique_result[1])))
        #             # break
        #         else:
        #             accept_sample = True
        #         self.num_seeds += 1024
        #         # self.num_seeds += 5
        # if False:
        #     pcl = o3d.geometry.PointCloud()
        #     pcl.points = o3d.utility.Vector3dVector(new_xyz.numpy()[0, :, :3])
        #     o3d.io.write_point_cloud("temp/seed_point.ply", pcl)
        #     # for id_item, item in enumerate(new_points[0]):
        #     #     pcl = o3d.geometry.PointCloud()
        #     #     pcl.points = o3d.utility.Vector3dVector(item.numpy()[:, :3])
        #     #     o3d.io.write_point_cloud("{}.ply".format(id_item), pcl)
        #     pcl.points = o3d.utility.Vector3dVector(new_points.numpy()[0, :, :, :3].reshape([-1, 3]))
        #     o3d.io.write_point_cloud("temp/total_point.ply", pcl)
        #
        # # self.points = torch.cat([new_points[0], (fps_idx[0].unsqueeze(1).repeat(1, new_points.shape[2])).unsqueeze(2)],
        # #                         dim=2)
        #
        # # mask = self.point_attribute[:, 0] < 9999999
        # # stats.spearmanr(self.point_attribute[:, 0][mask], self.point_attribute[:, 1][mask])
        self.num_item = self.views.shape[0]
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
        point_indexes = used_index[index]

        views_item = self.views[point_indexes].copy()
        valid_mask = views_item[:, 0] == 1
        views_item[valid_mask, 1:6] = (views_item[valid_mask, 1:6] - self.view_mean_std[:5]) / self.view_mean_std[5:]

        points_item = self.point_attribute[point_indexes].copy()
        if points_item[1] != -1:
            points_item[1] = (points_item[1] - self.error_mean_std[0]) / self.error_mean_std[2]

        if points_item[2] != -1:
            points_item[2] = (points_item[2] - self.error_mean_std[1]) / self.error_mean_std[3]

        point_features, point_features_mask = None, None
        if self.is_involve_img:
            point_features, point_features_mask = self.img_dataset[torch.tensor([point_indexes]).int()]

        output_dict = {
            # "views": torch.tensor(np.load(self.views_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            # "view_pairs": torch.tensor(np.load(self.view_pairs_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            "views": torch.tensor(views_item, dtype=torch.float32).reshape(
                [-1, self.views.shape[1], self.views.shape[2]]),
            "point_attribute": torch.tensor(
                points_item.reshape([-1, self.point_attribute.shape[1]]), dtype=torch.float32),
            "point_features": point_features,
            "point_features_mask": point_features_mask,
            "scene_name": [self.scene_name] * 1,
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
        scene_name = [item["scene_name"] for item in batch]
        views = [torch.transpose(item["views"], 0, 1) for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        views_pad = pad_sequence(views, batch_first=True)
        views_pad = torch.transpose(views_pad, 1, 2)

        # view_pairs = [torch.transpose(item["view_pairs"],0,1) for item in batch]
        # view_pairs_pad = torch.transpose(torch.transpose(pad_sequence(view_pairs),0,1),1,2)
        point_attribute = [item["point_attribute"] for item in batch]

        point_features_pad, point_features_mask_pad = None, None
        if batch[0]["point_features"] is not None:
            point_features = [torch.swapaxes(item["point_features"],0,1) for item in batch]
            point_features_pad = torch.swapaxes(pad_sequence(point_features, batch_first=True),1,2)
            point_features_mask = [torch.swapaxes(item["point_features_mask"],0,1) for item in batch]
            point_features_mask_pad = torch.swapaxes(pad_sequence(point_features_mask, batch_first=True, padding_value=True),1,2)
        return {
            'views': views_pad,
            'point_attribute': torch.stack(point_attribute, dim=0),
            'point_features': point_features_pad,
            'point_features_mask': point_features_mask_pad,
            'scene_name': scene_name,
        }


class Recon_dataset_imgs_and_batch_points(Regress_hyper_parameters_dataset_with_imgs):
    def __init__(self, v_path, v_params, v_mode):
        super(Recon_dataset_imgs_and_batch_points, self).__init__(v_path, v_params, v_mode)

        pass

    def __getitem__(self, index):
        if self.trainer_mode == "training":
            used_index = self.train_index
        elif self.trainer_mode == "validation":
            used_index = self.validation_index
        else:
            used_index = self.whole_index
        num_points_per_batch = self.params["model"]["num_points_per_batch"]
        index = index * num_points_per_batch
        point_indexes = used_index[index:min(self.num_item,index+num_points_per_batch)]

        point_features, point_features_mask = None, None
        if self.is_involve_img:
            point_features, point_features_mask = self.img_dataset[point_indexes]

        views_item = self.views[point_indexes]
        valid_mask = views_item[:, :, 0] == 1
        views_item[valid_mask, 1:6] = (views_item[valid_mask, 1:6] - self.view_mean_std[:5]) / self.view_mean_std[5:]

        points_item = self.point_attribute[point_indexes]
        acc_point_mask = points_item[:, 1] != -1
        com_point_mask = points_item[:, 2] != -1

        points_item[acc_point_mask, 1] = (points_item[acc_point_mask, 1] - self.error_mean_std[0]) / \
                                                  self.error_mean_std[2]
        points_item[com_point_mask, 2] = (points_item[com_point_mask, 2] - self.error_mean_std[1]) / \
                                                  self.error_mean_std[3]

        output_dict = {
            "views": torch.tensor(views_item, dtype=torch.float32).reshape(
                [-1, self.views.shape[1], self.views.shape[2]]),
            "point_attribute": torch.tensor(points_item.reshape([-1, self.point_attribute.shape[1]]), dtype=torch.float32),
            # "points": self.points[used_index[index]],
            "point_features": point_features,
            "point_features_mask": point_features_mask,
            "scene_name": [self.scene_name] * num_points_per_batch,
        }
        return output_dict

    def __len__(self):
        if self.trainer_mode == "training":
            used_index = self.train_index
        elif self.trainer_mode == "validation":
            used_index = self.validation_index
        else:
            used_index = self.whole_index
        return used_index.shape[0] // self.params["model"]["num_points_per_batch"]

    @staticmethod
    def collate_fn(batch):
        scene_name = [item["scene_name"] for item in batch]
        views = [torch.transpose(item["views"], 0, 1) for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        views_pad = pad_sequence(views, batch_first=True)
        views_pad = torch.transpose(views_pad, 1, 2)

        # view_pairs = [torch.transpose(item["view_pairs"],0,1) for item in batch]
        # view_pairs_pad = torch.transpose(torch.transpose(pad_sequence(view_pairs),0,1),1,2)
        point_attribute = [item["point_attribute"] for item in batch]

        point_features_pad,point_features_mask_pad = None, None
        if batch[0]["point_features"] is not None:
            point_features = [item["point_features"] for item in batch]
            point_features_pad = pad_sequence(point_features, batch_first=True)
            point_features_mask = [item["point_features_mask"] for item in batch]
            point_features_mask_pad = pad_sequence(point_features_mask, batch_first=True, padding_value=True)

        return {
            'views': views_pad,
            # 'view_pairs': view_pairs_pad,
            'point_attribute': torch.stack(point_attribute, dim=0),
            'point_features': point_features_pad,
            'point_features_mask': point_features_mask_pad,
            'scene_name': scene_name,
        }


class Regress_hyper_parameters_dataset_with_imgs_with_truncated_error(torch.utils.data.Dataset):
    def __init__(self, v_path, v_params, v_mode):
        super(Regress_hyper_parameters_dataset_with_imgs_with_truncated_error, self).__init__()
        self.trainer_mode = v_mode
        self.params = v_params
        self.data_root = v_path
        self.views = np.load(os.path.join(v_path, "views.npz"))["arr_0"]
        img_dataset_path = open(os.path.join(v_path, "../img_dataset_path.txt")).readline().strip()
        self.truncated_error = float(open(os.path.join(v_path, "../suggest_error.txt")).readline().strip())

        self.is_involve_img = self.params["model"]["involve_img"]

        self.point_attribute = np.load(os.path.join(v_path, "point_attribute.npz"))["arr_0"]
        self.view_paths = np.load(os.path.join(v_path, "view_paths.npz"), allow_pickle=True)[
            "arr_0"]
        self.original_points = self.point_attribute[:, 3:6]
        # self.num_seeds = 4096 + 1024
        self.num_seeds = 20
        self.sample_points_to_different_patches()
        if self.is_involve_img:
            self.img_dataset = Regress_hyper_parameters_img_dataset(os.path.join(v_path, img_dataset_path),
                                                                    self.view_paths)

        pass

    """
    Points: num of patches * num point per patch * 7 (x, y, z, index, id_centre)
    """

    def sample_points_to_different_patches(self):
        print("KNN Sample")
        # seed points, radius, max local points,
        accept_sample = False
        # self.num_seeds -= 1024
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
            # self.num_seeds += 1024
            self.num_seeds += 5
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

        self.points = torch.cat([new_points[0], (fps_idx[0].unsqueeze(1).repeat(1, 256)).unsqueeze(2)], dim=2)
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

        point_features, point_features_mask, img_pose = None, None, None
        if self.is_involve_img:
            point_features, point_features_mask, img_pose = self.img_dataset[point_indexes]

        truncated_error = self.point_attribute[point_indexes].copy()
        truncated_error[:, 2:3] = truncated_error[:, 2:3] > self.truncated_error
        output_dict = {
            # "views": torch.tensor(np.load(self.views_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            # "view_pairs": torch.tensor(np.load(self.view_pairs_path,mmap_mode="r")["arr_0"][point_indexes], dtype=torch.float32),
            "views": torch.tensor(self.views[point_indexes], dtype=torch.float32),
            # "view_pairs": torch.tensor(self.view_pairs[point_indexes], dtype=torch.float32),

            "point_attribute": torch.tensor(truncated_error, dtype=torch.float32),
            "points": self.points[used_index[index]],
            "point_features": point_features,
            "point_features_mask": point_features_mask,
            "img_pose": torch.tensor(img_pose, dtype=torch.float32) if img_pose is not None else torch.tensor(
                self.views[point_indexes], dtype=torch.float32),
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
        img_pose = [torch.transpose(item["img_pose"], 0, 1) for item in batch]
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
