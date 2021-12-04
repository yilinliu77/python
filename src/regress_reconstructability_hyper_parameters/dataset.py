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

from src.regress_reconstructability_hyper_parameters.preprocess_data import pre_compute_img_features
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import farthest_point_sample, sample_and_group


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


class Regress_hyper_parameters_dataset_with_imgs(torch.utils.data.Dataset):
    def __init__(self,v_path, v_params, v_mode):
        super(Regress_hyper_parameters_dataset_with_imgs, self).__init__()
        self.trainer_mode = v_mode
        self.params = v_params
        self.data_root = v_path
        # self.views = np.load(os.path.join(v_path, "views.npz"))["arr_0"]
        # self.view_pairs = np.load(os.path.join(v_path, "view_pairs.npz"))["arr_0"]
        self.views_path=os.path.join(v_path, "views.npz")
        self.view_pairs_path=os.path.join(v_path, "view_pairs.npz")

        self.point_attribute = np.load(os.path.join(v_path, "point_attribute.npz"))["arr_0"]
        self.view_paths = np.load(os.path.join(v_path, "view_paths.npz"), allow_pickle=True)[
            "arr_0"]
        self.original_points = self.point_attribute[:, 3:]
        self.sample_points_to_different_patches()
        pass

    def sample_points_to_different_patches(self):
        print("KNN Sample")
        # seed points, radius, max local points,
        new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(
            4096, 0.1, int(self.params["model"]["num_points_per_batch"]), torch.tensor(self.original_points, dtype=torch.float32).unsqueeze(0),
            torch.arange(self.original_points.shape[0]).unsqueeze(0).unsqueeze(-1), True)
        self.points = new_points[0]
        mask = self.point_attribute[:, 0] < 9999999

        stats.spearmanr(self.point_attribute[:, 0][mask], self.point_attribute[:, 2][mask])

        self.num_item = self.points.shape[0]

        self.whole_index = np.arange(self.num_item)
        np.random.shuffle(self.whole_index)
        self.train_index = self.whole_index[:self.whole_index.shape[0] // 4 * 3]
        self.validation_index = self.whole_index[self.whole_index.shape[0] // 4 * 3:]

    def __getitem__(self, index):
        num_point_per_patch = self.points.shape[1]
        num_features=None
        num_max_points = 0

        if self.trainer_mode=="training":
            used_index = self.train_index
        elif self.trainer_mode=="validation":
            used_index = self.validation_index
        else:
            used_index = self.whole_index
        point_indexes = self.points[used_index[index], :, 3].int()

        img_features_on_point_list=[]
        for point_index in point_indexes:
            point_path = os.path.join(self.data_root, "point_features",
                                      str(point_index.item()) + ".npz")

            img_features_on_point_list.append(torch.tensor(np.load(point_path)["arr_0"],dtype=torch.float32))
            num_features = img_features_on_point_list[-1].shape[1]

            num_max_points=max(num_max_points,img_features_on_point_list[-1].shape[0])

        # Align the features
        point_features = torch.zeros((num_point_per_patch,num_max_points,num_features),dtype=torch.float32)
        point_features_mask = torch.ones((num_point_per_patch,num_max_points),dtype=torch.bool)
        for id_item,item in enumerate(img_features_on_point_list):
            point_features[id_item,:item.shape[0]]=item
            point_features_mask[id_item,:item.shape[0]]=False

        output_dict = {
            "views": torch.tensor(np.load(self.views_path)["arr_0"][point_indexes], dtype=torch.float32),
            "view_pairs": torch.tensor(np.load(self.view_pairs_path)["arr_0"][point_indexes], dtype=torch.float32),
            "point_attribute": torch.tensor(self.point_attribute[point_indexes], dtype=torch.float32),
            "points": self.points[used_index[index]],
            "point_features": point_features,
            "point_features_mask": point_features_mask,
        }
        return output_dict

    def __len__(self):
        if self.trainer_mode=="training":
            return self.train_index.shape[0]
        elif self.trainer_mode=="validation":
            return self.validation_index.shape[0]
        else:
            return self.whole_index.shape[0]

    @staticmethod
    def collate_fn(batch):
        views = [torch.transpose(item["views"],0,1) for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        views_pad = torch.transpose(torch.transpose(pad_sequence(views),0,1),1,2)
        view_pairs = [torch.transpose(item["view_pairs"],0,1) for item in batch]
        view_pairs_pad = torch.transpose(torch.transpose(pad_sequence(view_pairs),0,1),1,2)
        point_attribute = [item["point_attribute"] for item in batch]
        point_features = [item["point_features"] for item in batch]
        points = [item["points"] for item in batch]
        point_features_mask = [item["point_features_mask"] for item in batch]

        return {
            'views': views_pad,
            'view_pairs': view_pairs_pad,
            'point_attribute': torch.stack(point_attribute,dim=0),
            'point_features': point_features,
            'points': torch.stack(points,dim=0),
            'point_features_mask': point_features_mask,
        }
