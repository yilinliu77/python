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

from src.img2brep.brep.model import AutoEncoder


def get_face_idx_sequence(edge_face_connectivity, face_points):
    G = nx.Graph()

    for idx, (edge_id, face1, face2) in enumerate(edge_face_connectivity):
        G.add_edge(int(face1), int(face2))

    if G.number_of_nodes() != face_points.shape[0]:
        raise ValueError("Number of nodes is not equal to number of faces")

    face_idx_sequence = list(nx.bfs_tree(G, 0))

    return torch.tensor(face_idx_sequence, dtype=torch.long, device=edge_face_connectivity.device)


class Autoencoder_Dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Autoencoder_Dataset, self).__init__()
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
        self.bd = v_conf['bbox_discrete_dim'] // 2 - 1  # discrete_dim
        self.cd = v_conf['coor_discrete_dim'] // 2 - 1  # discrete_dim

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

        self.data_folders = self.data_folders

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

    def discrete_coordinates(self, face_points, line_points):
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
        return discrete_face_points, discrete_face_bboxes, discrete_edge_points, discrete_edge_bboxes

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*20*20*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])
        vertex_points = torch.from_numpy(data_npz['sample_points_vertices'])

        discrete_face_points, discrete_face_bboxes, discrete_edge_points, discrete_edge_bboxes = (
            self.discrete_coordinates(face_points, line_points))
        # Vertex
        discrete_vertex_points = torch.round(
            vertex_points * self.cd).long().clamp(-self.cd, self.cd)
        discrete_vertex_points += self.cd

        # Loops along each face, -2 means start token and -1 means padding (num_faces, max_num_edges_this_face)
        face_edge_loop = torch.from_numpy(data_npz['face_edge_loop'])

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])

        if True:
            face_adj = torch.zeros(face_points.shape[0], face_points.shape[0], dtype=torch.bool)
            face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True
            face_adj = torch.logical_or(face_adj, face_adj.T)
            face_adj.diagonal().fill_(True)
        else:
            face_adj = torch.from_numpy(data_npz['face_adj'])

        edge_adj = torch.zeros(line_points.shape[0], line_points.shape[0], dtype=torch.bool)
        edge_adj[vertex_edge_connectivity[:, 1], vertex_edge_connectivity[:, 2]] = True
        edge_adj = torch.logical_or(edge_adj, edge_adj.T)
        edge_adj.diagonal().fill_(True)

        face_idx_sequence_c = get_face_idx_sequence(edge_face_connectivity, face_points)

        # Write gt data to file for testing
        # indices = torch.arange(face_points.shape[0]).repeat_interleave(400)
        # data = np.array(
        #     [(face_points.reshape(-1,3)[i,0], face_points.reshape(-1,3)[i,1], face_points.reshape(-1,3)[i,2],indices[i]) for i in range(face_points.shape[0]*400)],
        #     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('primitive_index', 'i4')]
        # )
        # plyfile.PlyData([
        #     plyfile.PlyElement.describe(data, 'vertex')
        # ], text=True).write("face_points.ply")
        #
        # indices = torch.arange(line_points.shape[0]).repeat_interleave(20)
        # data = np.array(
        #     [(line_points.reshape(-1,3)[i,0], line_points.reshape(-1,3)[i,1], line_points.reshape(-1,3)[i,2],indices[i]) for i in range(line_points.shape[0]*20)],
        #     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('primitive_index', 'i4')]
        # )
        # plyfile.PlyData([
        #     plyfile.PlyElement.describe(data, 'vertex')
        # ], text=True).write("edge_points.ply")
        return (
            Path(folder_path).stem,
            face_points, line_points, vertex_points,
            discrete_face_points, discrete_face_bboxes,
            discrete_edge_points, discrete_edge_bboxes,
            discrete_vertex_points,
            face_edge_loop, face_adj, edge_adj,
            edge_face_connectivity, vertex_edge_connectivity,
            face_idx_sequence_c,
        )

    @staticmethod
    def collate_fn(batch):
        (v_prefix, face_points, edge_points, vertex_points,
         discrete_face_points, discrete_face_bboxes,
         discrete_edge_points, discrete_edge_bboxes,
         discrete_vertex_points,
         face_edge_loop, face_adj, edge_adj,
         edge_face_connectivity, vertex_edge_connectivity,
         face_idx_sequence_c,
         ) = zip(*batch)

        face_points = pad_sequence(face_points, batch_first=True, padding_value=-1)
        edge_points = pad_sequence(edge_points, batch_first=True, padding_value=-1)
        vertex_points = pad_sequence(vertex_points, batch_first=True, padding_value=-1)
        discrete_face_points = pad_sequence(discrete_face_points, batch_first=True, padding_value=-1)
        discrete_face_bboxes = pad_sequence(discrete_face_bboxes, batch_first=True, padding_value=-1)
        discrete_edge_points = pad_sequence(discrete_edge_points, batch_first=True, padding_value=-1)
        discrete_edge_bboxes = pad_sequence(discrete_edge_bboxes, batch_first=True, padding_value=-1)
        discrete_vertex_points = pad_sequence(discrete_vertex_points, batch_first=True, padding_value=-1)

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
            "discrete_face_points": discrete_face_points,
            "discrete_face_bboxes": discrete_face_bboxes,
            "discrete_edge_points": discrete_edge_points,
            "discrete_edge_bboxes": discrete_edge_bboxes,
            "discrete_vertex_points": discrete_vertex_points,

            "face_edge_loop": face_edge_loop,
            "face_adj": face_adj,
            "edge_adj": edge_adj,
            "edge_face_connectivity": edge_face_connectivity,
            "vertex_edge_connectivity": vertex_edge_connectivity,

            "face_idx_sequence": face_idx_sequence,
        }


class Face_feature_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Face_feature_dataset, self).__init__()
        self.conf = v_conf
        if v_training_mode == "testing":
            self.root = Path(v_conf['test_dataset'])
        elif v_training_mode == "training":
            self.root = Path(v_conf['train_dataset'])
        elif v_training_mode == "validation":
            self.root = Path(v_conf['val_dataset'])
        folders = [item.strip() for item in os.listdir(self.root)]
        folders.sort()
        num_max_faces = v_conf['num_max_faces']
        self.length_scaling_factors = int(v_conf['length_scaling_factors'])
        self.folders = [f for f in folders if np.load(self.root/f).shape[0] < num_max_faces]
        print("Filter out {} folders with faces > {}".format(len(folders) - len(self.folders), num_max_faces))

    def __len__(self):
        return len(self.folders) * self.length_scaling_factors

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.root/self.folders[idx % len(self.folders)]))

    @staticmethod
    def collate_fn(batch):
        data = pad_sequence(batch, batch_first=True, padding_value=0)
        return data

