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

import torch.nn.functional as F


class Auotoencoder_Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Auotoencoder_Dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = v_conf['root']
        self.device = torch.device("cpu")
        self.max_face_num = -1
        self.max_edge_num = -1
        self.max_edge_adj_num = -1
        self.face_edges_idx_shape1 = -1

        self.pad_id = -1

        self.data_folders = [os.path.join(self.dataset_path, folder) for folder in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, folder))]

        # with open(os.path.join(self.dataset_path, "single_loop.txt"), "r") as f:
        #     self.data_folders = [os.path.join(self.dataset_path, line.strip()) for line in f.readlines()]

        self.data_sum = len(self.data_folders)

        print("\nDataset INFO")
        print("data_folders:", len(self.data_folders))

        # self.training_range = int(0.8 * self.sum_num)
        # self.validation_range = int(0.9 * self.sum_num)

        self.training_range = [int(0 * self.data_sum), int(1.0 * self.data_sum)]
        self.validation_range = [int(0 * self.data_sum), int(1.0 * self.data_sum)]

        if self.mode == "training":
            self.data_folders = self.data_folders[self.training_range[0]:self.training_range[1]]

        elif self.mode == "validation":
            self.data_folders = self.data_folders[self.validation_range[0]:self.validation_range[1]]

        elif self.mode == "testing":
            self.data_folders = self.data_folders[self.validation_range[0]:self.validation_range[1]]

        else:
            raise

    def __len__(self):
        return self.data_sum

    def __getitem__(self, idx):
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # (num_faces, max_num_edges_this_face)
        face_edge_idx = torch.from_numpy(data_npz['face_edge_idx']).to(torch.int64).to(self.device)

        # (num_faces, 400, 3)
        sample_points_faces = torch.from_numpy(data_npz['sample_points_faces']).to(torch.float32).to(self.device)
        # sample_points_faces = rearrange(sample_points_faces, 'f (h w) dim -> f h w dim', h=20)

        # (num_lines, 20, 3)
        sample_points_lines = torch.from_numpy(data_npz['sample_points_lines']).to(torch.float32).to(self.device)

        # (num_faces, max_num_edges, 2)
        edge_adj = torch.from_numpy(data_npz['edge_adj']).to(torch.int64).to(self.device)
        edge_adj = edge_adj[(edge_adj != -1).all(dim=-1)]  # (sum_edges,2) # TODO fix it in data_preprocess.py

        # (num_faces, num_faces)
        face_adj = torch.from_numpy(data_npz['face_adj']).to(torch.int64).to(self.device)

        # (num_edges, 2)
        edge_face_idx = torch.from_numpy(data_npz['edge_face_idx']).to(torch.int64).to(self.device)

        return sample_points_faces, sample_points_lines, face_edge_idx, edge_adj, face_adj, edge_face_idx

    @staticmethod
    def collate_fn(batch):
        sample_points_faces, sample_points_lines, face_edge_idx, edge_adj, face_adj, edge_face_idx = zip(*batch)

        sample_points_faces = pad_sequence(sample_points_faces, batch_first=True, padding_value=-1)
        sample_points_lines = pad_sequence(sample_points_lines, batch_first=True, padding_value=-1)

        # 2D pad
        face_edges_idx_max_shape0 = max([item.shape[0] for item in face_edge_idx])
        face_edges_idx_max_shape1 = max([item.shape[1] for item in face_edge_idx])
        face_edge_idx = list(face_edge_idx)
        for idx in range(len(face_edge_idx)):
            face_edge_idx[idx] = F.pad(face_edge_idx[idx], (0, face_edges_idx_max_shape1 - face_edge_idx[idx].shape[1],
                                                            0, face_edges_idx_max_shape0 - face_edge_idx[idx].shape[0]),
                                       'constant', -1)
        face_edge_idx = torch.stack(face_edge_idx)

        edge_adj = pad_sequence(edge_adj, batch_first=True, padding_value=-1)

        face_adj_max_shape0 = max([item.shape[0] for item in face_adj])
        face_adj__max_shape1 = max([item.shape[1] for item in face_adj])
        face_adj = list(face_adj)
        for idx in range(len(face_adj)):
            face_adj[idx] = F.pad(face_adj[idx], (0, face_adj__max_shape1 - face_adj[idx].shape[1],
                                                  0, face_adj_max_shape0 - face_adj[idx].shape[0]), 'constant', -1)
        face_adj = torch.stack(face_adj)

        edge_face_idx = pad_sequence(edge_face_idx, batch_first=True, padding_value=-1)

        return {
            "sample_points_faces": sample_points_faces,
            "sample_points_lines": sample_points_lines,
            "face_edge_idx"      : face_edge_idx,
            "edge_adj"           : edge_adj,
            "face_adj"           : face_adj,
            "edge_face_idx"      : edge_face_idx,
            }


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
