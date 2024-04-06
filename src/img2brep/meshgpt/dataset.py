import math
import os.path
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from shared.common_utils import export_point_cloud, check_dir
from typing import Final


class Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = v_conf['root']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_id = -1
        self.vertices_all, self.faces_all = self.read_data_and_pool()
        self._vertices = None
        self._faces = None

        self.vertices_all = self.vertices_all[0:2]
        self.faces_all = self.faces_all[0:2]

        self.sum_num = self.vertices_all.shape[0]

        print(f"sum_num: {self.sum_num}")

        # self.training_range = int(0.8 * self.sum_num)
        # self.validation_range = int(0.9 * self.sum_num)

        self.training_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]
        self.validation_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]

        if self.mode == "training":
            self._vertices = self.vertices_all[self.training_range[0]:self.training_range[1]]
            self._faces = self.faces_all[self.training_range[0]:self.training_range[1]]
        elif self.mode == "validation":
            self._vertices = self.vertices_all[self.validation_range[0]:self.validation_range[1]]
            self._faces = self.faces_all[self.validation_range[0]:self.validation_range[1]]
        elif self.mode == "testing":
            self._vertices = self.vertices_all[self.validation_range[0]:self.validation_range[1]]
            self._faces = self.faces_all[self.validation_range[0]:self.validation_range[1]]
        else:
            raise ""

        self._vertices = self._vertices.clone()
        self._faces = self._faces.clone()

    @property
    def vertices(self) -> torch.Tensor:
        return self._vertices

    @property
    def faces(self) -> torch.Tensor:
        return self._faces

    @vertices.setter
    def vertices(self, new_value):
        if new_value != self._vertices:
            print(f"Value is changing from {self._vertices} to {new_value}")
            import pdb
            pdb.set_trace()

    def read_data_and_pool(self):
        data_items = os.listdir(self.dataset_path)
        data_items.sort()
        max_vertices_num = 0
        max_faces_num = 0
        vertices_tensor_list = []
        faces_tensor_list = []
        for item in data_items:
            item_path = os.path.join(self.dataset_path, item)
            tri_mesh_path = os.path.join(item_path, "triangulation.ply")
            tri_mesh = o3d.io.read_triangle_mesh(tri_mesh_path)

            vertices = np.array(tri_mesh.vertices)
            triangle_vertices_idx = np.array(tri_mesh.triangles)
            triangle = vertices[triangle_vertices_idx]

            if triangle.shape[0] > 100:
                continue

            vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(self.device)
            faces_tensor = torch.tensor(triangle_vertices_idx, dtype=torch.long).to(self.device)
            vertices_tensor_list.append(vertices_tensor)
            faces_tensor_list.append(faces_tensor)
            max_vertices_num = vertices.shape[0] if vertices.shape[0] > max_vertices_num else max_vertices_num
            max_faces_num = triangle_vertices_idx.shape[0] if triangle_vertices_idx.shape[0] > max_faces_num \
                else max_faces_num

        # pool
        for i in range(len(vertices_tensor_list)):
            vertices = vertices_tensor_list[i]
            faces = faces_tensor_list[i]
            vertices_num = vertices.shape[0]
            faces_num = faces.shape[0]
            if vertices_num < max_vertices_num:
                vertices_tensor_list[i] = torch.cat(
                        [vertices, torch.zeros(max_vertices_num - vertices_num, 3).to(self.device)])
            if faces_num < max_faces_num:
                faces_tensor_list[i] = torch.cat(
                        [faces, self.pad_id * torch.ones(max_faces_num - faces_num, 3).to(torch.long).to(self.device)])

        return torch.stack(vertices_tensor_list), torch.stack(faces_tensor_list)

    def __len__(self):
        return self.vertices.shape[0]

    def __getitem__(self, idx):
        return self.vertices[idx], self.faces[idx]


class Single_obj_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Single_obj_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = r"H:\Data\SIGA23\Baseline\data\0planar_shapes"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vertices_tensor_list = []
        self.faces_tensor_list = []
        self.pre_data()

        self.vertices_tensor = self.vertices_tensor_list[0].unsqueeze(0)
        self.faces_tensor = self.faces_tensor_list[0].unsqueeze(0)

        if self.mode == "training":
            pass
        elif self.mode == "validation":
            pass
        elif self.mode == "testing":
            pass
        else:
            raise ""

    def pre_data(self):
        data_items = os.listdir(self.dataset_path)
        data_items.sort()
        vertices_tensor_list = []
        faces_tensor_list = []
        for item in data_items:
            item_path = os.path.join(self.dataset_path, item)
            tri_mesh_path = os.path.join(item_path, "triangulation.ply")
            tri_mesh = o3d.io.read_triangle_mesh(tri_mesh_path)

            vertices = np.array(tri_mesh.vertices)
            triangle_vertices_idx = np.array(tri_mesh.triangles)
            triangle = vertices[triangle_vertices_idx]

            vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(self.device)
            faces_tensor = torch.tensor(triangle_vertices_idx, dtype=torch.long).to(self.device)
            self.vertices_tensor_list.append(vertices_tensor)
            self.faces_tensor_list.append(faces_tensor)

    def __len__(self):
        if self.mode == "training":
            return 100
        else:
            return 1

    def __getitem__(self, idx):
        return self.vertices_tensor[0], self.faces_tensor[0]
