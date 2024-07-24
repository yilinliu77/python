import math
import os.path
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import plyfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from shared.common_utils import export_point_cloud, check_dir

from typing import Final
from einops import rearrange

import networkx as nx

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F


def get_face_idx_sequence(edge_face_connectivity, face_points):
    G = nx.Graph()

    for idx, (edge_id, face1, face2) in enumerate(edge_face_connectivity):
        G.add_edge(int(face1), int(face2))

    if G.number_of_nodes() != face_points.shape[0]:
        raise ValueError("Number of nodes is not equal to number of faces")

    face_idx_sequence = list(nx.bfs_tree(G, 0))

    return torch.tensor(face_idx_sequence, dtype=torch.long, device=edge_face_connectivity.device)


class AutoEncoder_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf

        if v_training_mode == "testing":
            self.dataset_path = v_conf['test_dataset']
        elif v_training_mode == "training":
            self.dataset_path = v_conf['train_dataset']
        elif v_training_mode == "validation":
            self.dataset_path = v_conf['val_dataset']
        else:
            raise

        self.data_folders = [os.path.join(self.dataset_path, folder) for folder in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, folder))]
        self.data_folders.sort()

        self.data_folders = self.data_folders

        self.src_data_sum = len(self.data_folders)

        self.check_data(self.dataset_path, v_training_mode)

        self.data_sum = len(self.data_folders)

        print("Dataset INFO")
        print("Src data_folders:", self.src_data_sum)
        print("After removing:", self.data_sum)
        print("Remove invalid data.npz folders:", self.src_data_sum - self.data_sum)

    def __len__(self):
        return len(self.data_folders)

    def check_data(self, v_path, v_training_mode):
        if v_training_mode == "training":
            filepath = os.path.join(v_path, r"id_larger_than_128_faces.txt")
        elif v_training_mode == "validation":
            filepath = os.path.join(v_path, r"id_larger_than_128_faces.txt")
        else:
            filepath = os.path.join(v_path, r"id_larger_than_128_faces.txt")
        if os.path.exists(filepath):
            ignore_ids = [item.strip() for item in open(filepath).readlines()]
        else:
            ignore_ids = []
        miss = []
        for folder_path in self.data_folders:
            if not os.path.exists(os.path.join(folder_path, "data.npz")) or folder_path[-8:] in ignore_ids:
                miss.append(folder_path)

        for folder_path in miss:
            self.data_folders.remove(folder_path)

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*20*20*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])
        vertex_points = torch.from_numpy(data_npz['sample_points_vertices'])

        # Loops along each face, -2 means start token and -1 means padding (num_faces, max_num_edges_this_face)
        face_edge_loop = torch.from_numpy(data_npz['face_edge_loop'])

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])

        face_adj = torch.zeros(face_points.shape[0], face_points.shape[0], dtype=torch.bool)
        face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True
        face_adj = torch.logical_or(face_adj, face_adj.T)
        face_adj.diagonal().fill_(True)

        edge_adj = torch.zeros(line_points.shape[0], line_points.shape[0], dtype=torch.bool)
        edge_adj[vertex_edge_connectivity[:, 1], vertex_edge_connectivity[:, 2]] = True
        edge_adj = torch.logical_or(edge_adj, edge_adj.T)
        edge_adj.diagonal().fill_(True)

        face_idx_sequence_c = get_face_idx_sequence(edge_face_connectivity, face_points)

        return (
            Path(folder_path).stem,
            face_points, line_points, vertex_points,
            face_edge_loop, face_adj, edge_adj,
            edge_face_connectivity, vertex_edge_connectivity,
            face_idx_sequence_c,
        )

    @staticmethod
    def collate_fn(batch):
        (v_prefix, face_points, edge_points, vertex_points,
         face_edge_loop, face_adj, edge_adj,
         edge_face_connectivity, vertex_edge_connectivity,
         face_idx_sequence_c,
         ) = zip(*batch)

        face_points = pad_sequence(face_points, batch_first=True, padding_value=-1)
        edge_points = pad_sequence(edge_points, batch_first=True, padding_value=-1)
        vertex_points = pad_sequence(vertex_points, batch_first=True, padding_value=-1)

        edge_face_connectivity = pad_sequence(edge_face_connectivity, batch_first=True, padding_value=-1)
        vertex_edge_connectivity = pad_sequence(vertex_edge_connectivity, batch_first=True, padding_value=-1)
        face_idx_sequence = pad_sequence(face_idx_sequence_c, batch_first=True, padding_value=-1)

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

        max_shape0 = max([item.shape[0] for item in edge_adj])
        max_shape1 = max([item.shape[1] for item in edge_adj])
        edge_adj = list(edge_adj)
        for idx in range(len(edge_adj)):
            edge_adj[idx] = F.pad(edge_adj[idx], (0, max_shape1 - edge_adj[idx].shape[1],
                                                  0, max_shape0 - edge_adj[idx].shape[0]), 'constant', -1)
        edge_adj = torch.stack(edge_adj)

        return {
            "v_prefix": v_prefix,
            "vertex_points": vertex_points,
            "edge_points": edge_points,
            "face_points": face_points,

            "face_edge_loop": face_edge_loop,
            "face_adj": face_adj,
            "edge_adj": edge_adj,
            "edge_face_connectivity": edge_face_connectivity,
            "vertex_edge_connectivity": vertex_edge_connectivity,

            "face_idx_sequence": face_idx_sequence,
        }

