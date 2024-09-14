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
        self.max_intersection = 500
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
        filepath = os.path.join(v_path, r"id_larger_than_64_faces.txt")
        ignore_ids=[]
        if os.path.exists(filepath):
            ignore_ids = [item.strip() for item in open(filepath).readlines()]
        else:
            for folder_path in self.data_folders:
                if not os.path.exists(os.path.join(folder_path, "data.npz")):
                    ignore_ids.append(folder_path)
                    continue
                data_npz = np.load(os.path.join(folder_path, "data.npz"))
                if data_npz['sample_points_faces'].shape[0] > 64:
                    ignore_ids.append(folder_path)
            with open(filepath, "w") as f:
                for item in ignore_ids:
                    f.write(item + "\n")

        for folder_path in ignore_ids:
            self.data_folders.remove(folder_path)

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*32*32*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        # Ignore self intersection
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        # vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])

        face_adj = torch.from_numpy(data_npz['face_adj'])
        zero_positions = torch.from_numpy(data_npz['zero_positions'])
        if zero_positions.shape[0] > face_adj.shape[0] * 2:
            index = np.random.choice(zero_positions.shape[0], face_adj.shape[0] * 2, replace=False)
            zero_positions = zero_positions[index]
        # Assume the number of true intersection is less than self.max_intersection

        return (
            Path(folder_path).stem,
            face_points, line_points,
            face_adj, zero_positions,
            edge_face_connectivity
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, face_points, edge_points,
            face_adj, zero_positions,
            edge_face_connectivity
        ) = zip(*batch)
        bs = len(v_prefix)
        flat_face_points = []
        flat_edge_points = []
        flat_edge_face_connectivity = []
        flat_zero_positions = []
        num_face_record = []
        
        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            flat_face_points.append(face_points[i])
            flat_edge_points.append(edge_points[i])
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_edge_face_connectivity.append(edge_face_connectivity[i])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces+=face_points[i].shape[0]
            num_edges+=edge_points[i].shape[0]
            num_face_record.append(face_points[i].shape[0])
        num_face_record = torch.tensor(num_face_record, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur+edge_conn_num[i], id_cur:id_cur+edge_conn_num[i]] = False
            id_cur+=edge_conn_num[i]

        num_max_faces = num_face_record.max()
        valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
        for i in range(bs):
            valid_mask[i, :num_face_record[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur+face_points[i].shape[0], id_cur : id_cur+face_points[i].shape[0]] = False
            id_cur+=face_points[i].shape[0]
            
        flat_face_points = torch.cat(flat_face_points, dim=0)
        flat_edge_points = torch.cat(flat_edge_points, dim=0)
        flat_edge_face_connectivity = torch.cat(flat_edge_face_connectivity, dim=0)
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        return {
            "v_prefix": v_prefix,
            "edge_points": flat_edge_points,
            "face_points": flat_face_points,

            "edge_face_connectivity": flat_edge_face_connectivity,
            "zero_positions": flat_zero_positions,
            "attn_mask": attn_mask,
            "edge_attn_mask": edge_attn_mask,

            "num_face_record": num_face_record,
            "valid_mask": valid_mask,
        }


class Diffusion_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Diffusion_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.max_intersection = 500
        if v_training_mode == "testing":
            self.dataset_path = Path(v_conf['test_dataset'])
        elif v_training_mode == "training":
            self.dataset_path = Path(v_conf['train_dataset'])
        elif v_training_mode == "validation":
            self.dataset_path = Path(v_conf['val_dataset'])
        else:
            raise
        self.max_faces = 64
        self.data_folders = [self.dataset_path/item for item in os.listdir(self.dataset_path) if
                            item.endswith("feature.npz")]
        self.data_folders.sort()
        self.data_folders = self.data_folders
        self.check_data(self.dataset_path, self.max_faces)
        return

    def __len__(self):
        return len(self.data_folders)

    def check_data(self, v_path, v_max_num=64):
        print("Original data num:", len(self.data_folders))
        filepath = os.path.join(v_path, "id_larger_than_{}_faces.txt".format(v_max_num))
        if os.path.exists(filepath):
            print("Found ignore file")
            ignore_ids = [item.strip() for item in open(filepath).readlines()]
        else:
            print("Create ignore file")
            ignore_ids = []
            for item in self.data_folders:
                shape_num = np.load(item)["face_features"].shape[0]
                if shape_num > v_max_num:
                    ignore_ids.append(item.stem)
            with open(filepath, "w") as f:
                for item in ignore_ids:
                    f.write(item + "\n")

        self.data_folders = [item for item in self.data_folders if item.stem not in ignore_ids]
        print("Final data num:", len(self.data_folders))

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        data_npz = np.load(str(folder_path))

        face_features = torch.from_numpy(data_npz['face_features'])
        padded_face_features = torch.zeros((self.max_faces, *face_features.shape[1:]), dtype=torch.float32)
        padded_face_features[:face_features.shape[0]] = face_features
        return (
            Path(folder_path).stem,
            face_features
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, v_face_features
        ) = zip(*batch)
        bs = len(v_prefix)
        
        # Pad with 0
        face_features = pad_sequence(v_face_features, batch_first=True, padding_value=0)

        return {
            "v_prefix": v_prefix,
            "face_features": face_features,
        }

