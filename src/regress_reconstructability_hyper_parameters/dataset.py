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
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from scipy import stats


def preprocess_data(v_root: str, v_error_point_cloud: str) -> (np.ndarray, np.ndarray):
    MAX_VIEW = 150
    print("Read attribute with max view: ", MAX_VIEW)
    files = [os.path.join(v_root, item) for item in os.listdir(v_root)]
    files = list(filter(lambda item: ".txt" in item, files))
    files = sorted(files, key=lambda item: int(item.split("\\")[-1][:-4]))
    views = np.zeros((len(files),MAX_VIEW, 7), dtype=np.float16)
    views_pair = np.zeros((len(files), MAX_VIEW, MAX_VIEW, 3), dtype=np.float16)
    point_index_has_nonzero_reconstructability = []
    point_index_has_short_index = []
    reconstructabilities = []
    for id_file, file in tqdm(enumerate(files)):
        real_index = int(file.split("\\")[-1][:-4])

        raw_data = [item.strip() for item in open(file).readlines()]
        num_views = int(raw_data[0])
        if num_views>MAX_VIEW-1:
            raise
        reconstructability = float(raw_data[1])
        if reconstructability==0:
            continue
        point_index_has_short_index.append(id_file)
        reconstructabilities.append(reconstructability)
        # Read views
        for i_view in range(num_views):
            view_data = raw_data[2 + i_view].split(",")
            view_to_point = np.array(view_data[:3], dtype=np.float16)
            distance_ratio = float(view_data[3])
            angle_to_normal_ratio = float(view_data[4])
            angle_to_direction_ratio = float(view_data[5])

            views[id_file][i_view][0] = 1
            views[id_file][i_view][1] = view_to_point[0]
            views[id_file][i_view][2] = view_to_point[1]
            views[id_file][i_view][3] = view_to_point[2]
            views[id_file][i_view][4] = distance_ratio
            views[id_file][i_view][5] = angle_to_normal_ratio
            views[id_file][i_view][6] = angle_to_direction_ratio


        # Read view pair
        cur_iter=0
        for i_view1 in range(MAX_VIEW):
            for i_view2 in range(i_view1 + 1, MAX_VIEW):
                if i_view1>=num_views or i_view2>=num_views:
                    continue
                view_pair_data = raw_data[2 + num_views + cur_iter].split(",")
                alpha_ratio = float(view_pair_data[0])
                relative_distance_ratio = float(view_pair_data[1])
                views_pair[id_file][i_view1][i_view2][0] = 1
                views_pair[id_file][i_view1][i_view2][1] = alpha_ratio
                views_pair[id_file][i_view1][i_view2][2] = relative_distance_ratio
                cur_iter+=1

        point_index_has_nonzero_reconstructability.append(real_index)
    views=views[point_index_has_short_index]
    views_pair=views_pair[point_index_has_short_index]
    usable_indices = np.triu_indices(MAX_VIEW, 1)
    views_pair = views_pair[:,usable_indices[0],usable_indices[1]]

    print("Read point cloud")
    with open(v_error_point_cloud, "rb") as f:
        plydata = PlyData.read(f)
        error_list = plydata['vertex']['error'].copy()
    error_list = error_list[point_index_has_nonzero_reconstructability]
    point_attribute = np.stack([reconstructabilities,error_list],axis=1).astype(np.float16)
    return views, views_pair, point_attribute


class Regress_hyper_parameters_dataset(torch.utils.data.Dataset):
    def __init__(self, v_params, v_mode):
        super(Regress_hyper_parameters_dataset, self).__init__()
        self.is_training = v_mode == "training"
        self.params = v_params
        self.views = np.load(os.path.join(v_params["model"].preprocess_path, "views.npz"))["arr_0"]
        self.view_pairs = np.load(os.path.join(v_params["model"].preprocess_path, "view_pairs.npz"))["arr_0"]
        self.point_attribute = np.load(os.path.join(v_params["model"].preprocess_path, "point_attribute.npz"))["arr_0"]
        # mask = self.target_data[:, 0, -1] < 10
        mask = self.point_attribute[:, 0] < 9999999
        stats.spearmanr(self.point_attribute[:, 0][mask], self.point_attribute[:, 1][mask])

        batch_size = self.params["trainer"]["batch_size"]
        self.num_item = self.point_attribute.shape[0] // batch_size * 5

        whole_index = np.arange(self.point_attribute.shape[0])
        np.random.shuffle(whole_index)
        self.train_index = whole_index[:whole_index.shape[0]//4*3]
        self.test_index = whole_index[whole_index.shape[0]//4*3:]

        pass

    def __getitem__(self, index):
        if self.is_training:
            batch_index = np.random.choice(self.train_index, self.params["trainer"]["batch_size"],
                                           False)
        else:
            # batch_index = np.arange(self.views.shape[0])
            batch_index = self.test_index[[index]]
        output_dict = {
            "views": torch.tensor(self.views[batch_index],dtype=torch.float32),
            "view_pairs": torch.tensor(self.view_pairs[batch_index],dtype=torch.float32),
            "point_attribute": torch.tensor(self.point_attribute[batch_index],dtype=torch.float32),
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
