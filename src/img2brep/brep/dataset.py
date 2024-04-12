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

        self.data_folders = [os.path.join(self.dataset_path, folder) for folder in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, folder))]
        self.data_folders.sort()

        self.data_folders = self.data_folders[0:100]

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
        sample_points_faces = torch.from_numpy(data_npz['sample_points_faces'])
        sample_points_lines = torch.from_numpy(data_npz['sample_points_lines'])
        sample_points_vertices = torch.from_numpy(data_npz['sample_points_vertices'])

        # Loops along each face, -2 means start token and -1 means padding (num_faces, max_num_edges_this_face)
        face_edge_loop = torch.from_numpy(data_npz['face_edge_loop'])

        # Adjacency matrix for face (num_faces, num_faces)
        face_adj = torch.from_numpy(data_npz['face_adj'])

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])

        return (sample_points_faces, sample_points_lines, sample_points_vertices,
                face_edge_loop, face_adj, edge_face_connectivity, vertex_edge_connectivity)

    @staticmethod
    def collate_fn(batch):
        (sample_points_faces, sample_points_lines, sample_points_vertices,
         face_edge_loop, face_adj, edge_face_connectivity, vertex_edge_connectivity) = zip(*batch)

        sample_points_faces = pad_sequence(sample_points_faces, batch_first=True, padding_value=-1)
        sample_points_lines = pad_sequence(sample_points_lines, batch_first=True, padding_value=-1)
        sample_points_vertices = pad_sequence(sample_points_vertices, batch_first=True, padding_value=-1)

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
            "sample_points_vertices"  : sample_points_vertices,
            "sample_points_lines"     : sample_points_lines,
            "sample_points_faces"     : sample_points_faces,
            "face_edge_loop"          : face_edge_loop,
            "face_adj"                : face_adj,
            "edge_face_connectivity"  : edge_face_connectivity,
            "vertex_edge_connectivity": vertex_edge_connectivity,
            }
