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
from meshgpt_pytorch.data import derive_face_edges_from_faces

from typing import Final
from einops import rearrange

from meshgpt_pytorch import MeshAutoencoder
from torch.nn.utils.rnn import pad_sequence


class Auotoencoder_Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Auotoencoder_Dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = v_conf['root']
        self.device = torch.device("cpu")

        self.pad_id = -1
        self.vertices_all, self.faces_all, self.face_edges_all = self.read_data_and_pool()

        repeat_Num = 1
        self.vertices_all = self.vertices_all.tile(repeat_Num, 1, 1)
        self.faces_all = self.faces_all.tile(repeat_Num, 1, 1)
        self.face_edges_all = self.face_edges_all.tile(repeat_Num, 1, 1)

        self.sum_num = self.vertices_all.shape[0]

        print("\nDataset INFO")
        print("Vertices:", self.vertices_all.shape)
        print("Faces:", self.faces_all.shape)

        # self.training_range = int(0.8 * self.sum_num)
        # self.validation_range = int(0.9 * self.sum_num)

        self.training_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]
        self.validation_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]

        if self.mode == "training":
            self.vertices_all = self.vertices_all[self.training_range[0]:self.training_range[1]]
            self.faces_all = self.faces_all[self.training_range[0]:self.training_range[1]]
            self.face_edges_all = self.face_edges_all[self.training_range[0]:self.training_range[1]]

        elif self.mode == "validation":
            self.vertices_all = self.vertices_all[self.validation_range[0]:self.validation_range[1]]
            self.faces_all = self.faces_all[self.validation_range[0]:self.validation_range[1]]
            self.face_edges_all = self.face_edges_all[self.validation_range[0]:self.validation_range[1]]

        elif self.mode == "testing":
            self.vertices_all = self.vertices_all[self.validation_range[0]:self.validation_range[1]]
            self.faces_all = self.faces_all[self.validation_range[0]:self.validation_range[1]]
            self.face_edges_all = self.face_edges_all[self.validation_range[0]:self.validation_range[1]]

        else:
            raise ""

    @property
    def vertices(self) -> torch.Tensor:
        return self.vertices_all

    @property
    def faces(self) -> torch.Tensor:
        return self.faces_all

    @property
    def face_edges(self) -> torch.Tensor:
        return self.face_edges_all

    def read_data_and_pool(self):
        data_folders = os.listdir(self.dataset_path)
        data_folders.sort()

        vertices_tensor_list = []
        faces_tensor_list = []
        face_edges_list = []

        for folder in data_folders:
            # if folder not in ["00000325", "00000797"]:
            #     continue
            folder_path = os.path.join(self.dataset_path, folder)
            tri_mesh_path = os.path.join(folder_path, "triangulation.ply")

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

            face_edges = derive_face_edges_from_faces(faces_tensor, pad_id=self.pad_id)
            face_edges_list.append(face_edges)

        vertices_tensor = pad_sequence(vertices_tensor_list, batch_first=True, padding_value=0)
        faces_tensor = pad_sequence(faces_tensor_list, batch_first=True, padding_value=self.pad_id)
        face_edges = pad_sequence(face_edges_list, batch_first=True, padding_value=self.pad_id)

        return vertices_tensor, faces_tensor, face_edges

    def __len__(self):
        return self.vertices.shape[0]

    def __getitem__(self, idx):
        return {"vertices": self.vertices[idx], "faces": self.faces[idx], "face_edges": self.face_edges[idx]}


class Transformer_Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf, autoencoder):
        super(Transformer_Dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = v_conf['root']
        self.tokenized_batch_size = v_conf['tokenized_batch_size']
        self.device = torch.device("cpu")

        self.autoencoder: MeshAutoencoder = autoencoder.cuda()

        self.pad_id = -1
        self.codes_all, self.img_embed_all = self.read_data_and_pool()

        self.codes_all = self.codes_all.tile(10, 1, 1)
        self.img_embed_all = self.img_embed_all.tile(10, 1, 1)

        self.sum_num = self.codes_all.shape[0]

        print("\nDataset INFO")
        print("Codes:", self.codes_all.shape)
        print("ImgEmbed:", self.img_embed_all.shape)

        # self.training_range = int(0.8 * self.sum_num)
        # self.validation_range = int(0.9 * self.sum_num)

        self.training_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]
        self.validation_range = [int(0 * self.sum_num), int(1.0 * self.sum_num)]

        if self.mode == "training":
            self.codes_all = self.codes_all[self.training_range[0]:self.training_range[1]]
            self.img_embed_all = self.img_embed_all[self.training_range[0]:self.training_range[1]]

        elif self.mode == "validation":
            self.codes_all = self.codes_all[self.validation_range[0]:self.validation_range[1]]
            self.img_embed_all = self.img_embed_all[self.validation_range[0]:self.validation_range[1]]

        elif self.mode == "testing":
            self.codes_all = self.codes_all[self.validation_range[0]:self.validation_range[1]]
            self.img_embed_all = self.img_embed_all[self.validation_range[0]:self.validation_range[1]]

        else:
            raise ""

    @property
    def codes(self) -> torch.Tensor:
        return self.codes_all

    @property
    def img_embed(self) -> torch.Tensor:
        return self.img_embed_all

    def read_data_and_pool(self):
        data_folders = os.listdir(self.dataset_path)
        data_folders.sort()

        vertices_tensor_list = []
        faces_tensor_list = []
        img_embed_list = []
        face_edges_list = []
        codes_list = []

        for folder in data_folders:
            # if folder not in ["00000325", "00000797"]:
            #     continue
            folder_path = os.path.join(self.dataset_path, folder)
            tri_mesh_path = os.path.join(folder_path, "triangulation.ply")

            tri_mesh = o3d.io.read_triangle_mesh(tri_mesh_path)

            vertices = np.array(tri_mesh.vertices)
            triangle_vertices_idx = np.array(tri_mesh.triangles)

            triangle = vertices[triangle_vertices_idx]

            if triangle.shape[0] > 100:
                continue

            vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(self.device)
            faces_tensor = torch.tensor(triangle_vertices_idx, dtype=torch.long).to(self.device)
            faces_tensor, _ = torch.sort(faces_tensor, dim=1)

            vertices_tensor_list.append(vertices_tensor)
            faces_tensor_list.append(faces_tensor)

            face_edges = derive_face_edges_from_faces(faces_tensor, pad_id=self.pad_id)
            face_edges_list.append(face_edges)

            img_embed = np.load(os.path.join(folder_path, "train_embed_vb16.npy"))
            img_embed = torch.tensor(img_embed, dtype=torch.float32).to(self.device)
            img_embed = rearrange(img_embed, 'img_num patch_num embed_dim -> (img_num patch_num) embed_dim')

            img_embed_list.append(img_embed)

        img_embed = torch.stack(img_embed_list)

        vertices_tensor = pad_sequence(vertices_tensor_list, batch_first=True, padding_value=0)
        faces_tensor = pad_sequence(faces_tensor_list, batch_first=True, padding_value=self.pad_id)
        face_edges = pad_sequence(face_edges_list, batch_first=True, padding_value=self.pad_id)

        for i in range(0, vertices_tensor.shape[0], self.tokenized_batch_size):
            vertices_tensor_batch = vertices_tensor[i:i + self.tokenized_batch_size]
            faces_tensor_batch = faces_tensor[i:i + self.tokenized_batch_size]
            face_edges_batch = face_edges[i:i + self.tokenized_batch_size]

            codes = self.autoencoder.tokenize(
                    vertices=vertices_tensor_batch.cuda(),
                    faces=faces_tensor_batch.cuda(),
                    face_edges=face_edges_batch.cuda(),
                    )
            codes_list.append(codes.cpu())

        codes = torch.cat(codes_list, dim=0)

        return codes, img_embed

    def __len__(self):
        return self.codes.shape[0]

    def __getitem__(self, idx):
        return {'codes': self.codes[idx], 'img_embed': self.img_embed[idx]}
