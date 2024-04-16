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
from einops import rearrange

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F


class Auotoencoder_Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Auotoencoder_Dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.dataset_path = v_conf['root']
        self.is_overfit = v_conf['overfit']
        self.bd = v_conf['bbox_discrete_dim'] // 2 - 1 # discrete_dim
        self.cd = v_conf['coor_discrete_dim'] // 2 - 1 # discrete_dim

        self.data_folders = [os.path.join(self.dataset_path, folder) for folder in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, folder))]
        self.data_folders.sort()

        self.data_folders = self.data_folders

        self.src_data_sum = len(self.data_folders)

        self.check_data()

        self.data_sum = len(self.data_folders)

        print("\nDataset INFO")
        print("Src data_folders:", self.src_data_sum)
        print("After removing:", self.data_sum)

        self.training_range = [int(0 * self.data_sum), int(0.9 * self.data_sum)]
        self.validation_range = [int(0.9 * self.data_sum), int(1.0 * self.data_sum)]
        self.test_range = [int(0 * self.data_sum), int(1.0 * self.data_sum)]

        if self.is_overfit or self.mode == "testing":
            self.data_folders = self.data_folders[self.test_range[0]:self.test_range[1]]
        elif self.mode == "training":
            self.data_folders = self.data_folders[self.training_range[0]:self.training_range[1]]
        elif self.mode == "validation":
            self.data_folders = self.data_folders[self.validation_range[0]:self.validation_range[1]]
        else:
            raise

    def __len__(self):
        return len(self.data_folders)

    def check_data(self):
        miss = []
        for folder_path in self.data_folders:
            if not os.path.exists(os.path.join(folder_path, "data.npz")):
                miss.append(folder_path)
                # print("Missing data.npz in", folder_path)

        if len(miss) > 0:
            for folder_path in miss:
                self.data_folders.remove(folder_path)
            print("Remove missing data.npz folders:", len(miss))

    def __getitem__(self, idx):
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*20*20*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])
        vertex_points = torch.from_numpy(data_npz['sample_points_vertices'])

        # Compute bounding box and discrete coordinates for faces
        min_face = face_points.min(dim=1).values.min(dim=1).values
        max_face = face_points.max(dim=1).values.max(dim=1).values
        length_face = max_face - min_face
        center_face = (max_face + min_face) / 2
        sample_points_faces_normalized = ((face_points - center_face[:, None, None]) /
                                          (length_face[:, None, None] + 1e-8)) * 2
        discrete_face_points = torch.round(
            sample_points_faces_normalized * self.cd).long().clamp(-self.cd, self.cd)
        discrete_face_bboxes = torch.round((torch.cat([
            min_face, max_face], dim=-1) * self.bd)).long().clamp(-self.bd, self.bd)
        discrete_face_points += self.cd
        discrete_face_bboxes += self.bd

        # Compute bounding box and discrete coordinates for edges
        min_edge = line_points.min(dim=1).values
        max_edge = line_points.max(dim=1).values
        length_edge = max_edge - min_edge
        center_edge = (max_edge + min_edge) / 2
        sample_points_edges_normalized = ((line_points - center_edge[:, None]) /
                                          (length_edge[:, None] + 1e-8)) * 2
        discrete_edge_points = torch.round(
            sample_points_edges_normalized * self.cd).long().clamp(-self.cd, self.cd)
        discrete_edge_bboxes = torch.round((torch.cat([
            min_edge, max_edge], dim=-1) * self.bd)).long().clamp(-self.bd, self.bd)
        discrete_edge_points += self.cd
        discrete_edge_bboxes += self.bd

        # Loops along each face, -2 means start token and -1 means padding (num_faces, max_num_edges_this_face)
        face_edge_loop = torch.from_numpy(data_npz['face_edge_loop'])

        # Adjacency matrix for face (num_faces, num_faces)
        face_adj = torch.from_numpy(data_npz['face_adj'])

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])
        return (face_points, line_points, vertex_points,
                discrete_face_points, discrete_face_bboxes, discrete_edge_points, discrete_edge_bboxes,
                face_edge_loop, face_adj, edge_face_connectivity, vertex_edge_connectivity)

    @staticmethod
    def collate_fn(batch):
        (face_points, line_points, vertex_points,
         discrete_face_points, discrete_face_bboxes, discrete_edge_points, discrete_edge_bboxes,
         face_edge_loop, face_adj, edge_face_connectivity, vertex_edge_connectivity) = zip(*batch)

        face_points = pad_sequence(face_points, batch_first=True, padding_value=-1)
        discrete_face_points = pad_sequence(discrete_face_points, batch_first=True, padding_value=-1)
        discrete_face_bboxes = pad_sequence(discrete_face_bboxes, batch_first=True, padding_value=-1)
        discrete_edge_points = pad_sequence(discrete_edge_points, batch_first=True, padding_value=-1)
        discrete_edge_bboxes = pad_sequence(discrete_edge_bboxes, batch_first=True, padding_value=-1)
        line_points = pad_sequence(line_points, batch_first=True, padding_value=-1)
        vertex_points = pad_sequence(vertex_points, batch_first=True, padding_value=-1)

        edge_face_connectivity = pad_sequence(edge_face_connectivity, batch_first=True, padding_value=-1)
        vertex_edge_connectivity = pad_sequence(vertex_edge_connectivity, batch_first=True, padding_value=-1)

        # 2D pad
        max_shape0 = max([item.shape[0] for item in face_edge_loop])
        max_shape1 = max([item.shape[1] for item in face_edge_loop])
        face_edge_loop = list(face_edge_loop)
        for idx in range(len(face_edge_loop)):
            face_edge_loop[idx] = F.pad(face_edge_loop[idx],
                                        (0, max_shape1 - face_edge_loop[idx].shape[1],
                                         0, max_shape0 - face_edge_loop[idx].shape[0]),
                                        'constant', -1)
        face_edge_loop = torch.stack(face_edge_loop)

        max_shape0 = max([item.shape[0] for item in face_adj])
        max_shape1 = max([item.shape[1] for item in face_adj])
        face_adj = list(face_adj)
        for idx in range(len(face_adj)):
            face_adj[idx] = F.pad(face_adj[idx], (0, max_shape1 - face_adj[idx].shape[1],
                                                  0, max_shape0 - face_adj[idx].shape[0]), 'constant', -1)
        face_adj = torch.stack(face_adj)

        return {
            "vertex_points": vertex_points,
            "edge_points": line_points,
            "face_points": face_points,
            "discrete_face_points": discrete_face_points,
            "discrete_face_bboxes": discrete_face_bboxes,
            "discrete_edge_points": discrete_edge_points,
            "discrete_edge_bboxes": discrete_edge_bboxes,

            "face_edge_loop": face_edge_loop,
            "face_adj": face_adj,
            "edge_face_connectivity": edge_face_connectivity,
            "vertex_edge_connectivity": vertex_edge_connectivity,
        }
