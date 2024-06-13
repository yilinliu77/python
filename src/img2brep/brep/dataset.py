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
            # self.feature_dataset_path = v_conf['test_feature_dataset']
        elif v_training_mode == "training":
            self.dataset_path = v_conf['train_dataset']
            # self.feature_dataset_path = v_conf['train_feature_dataset']
        elif v_training_mode == "validation":
            self.dataset_path = v_conf['val_dataset']
            # self.feature_dataset_path = v_conf['val_feature_dataset']
        else:
            raise
        self.bd = v_conf['bbox_discrete_dim'] // 2 - 1  # discrete_dim
        self.cd = v_conf['coor_discrete_dim'] // 2 - 1  # discrete_dim

        self.data_folders = [os.path.join(self.dataset_path, folder) for folder in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, folder))]
        self.data_folders.sort()

        self.data_folders = self.data_folders

        self.src_data_sum = len(self.data_folders)

        self.check_data(self.dataset_path, v_training_mode)

        self.data_sum = len(self.data_folders)

        print("\nDataset INFO")
        print("Src data_folders:", self.src_data_sum)
        print("After removing:", self.data_sum)

        self.data_folders = self.data_folders

        if False and self.feature_dataset_path is not None:
            self.num_max_faces = v_conf['num_max_faces']
            folders = [item.strip() for item in os.listdir(self.feature_dataset_path)]
            self.folders = [f for f in folders if
                            np.load(os.path.join(self.feature_dataset_path, f)).shape[0] < self.num_max_faces]
            print("Pre-load data into memory".format(len(folders) - len(self.folders), self.num_max_faces))
            self.data = []
            for folder in self.folders:
                data = torch.from_numpy(np.load(self.root / folder))
                self.data.append(data)
            self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
            self.length = self.data.shape[0]

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
        # length = []
        # for folder_path in tqdm(self.data_folders):
        #     length.append(np.load(os.path.join(folder_path, "data.npz"))["sample_points_faces"].shape[0])

        for folder_path in miss:
            self.data_folders.remove(folder_path)
        print("Remove invalid data.npz folders:", len(miss))

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

class AutoEncoder_direct_discrete_dataset(AutoEncoder_dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_direct_discrete_dataset, self).__init__(v_training_mode, v_conf)

    def discrete_coordinates(self, face_points, line_points):
        discrete_face_points = torch.round(
            face_points * self.cd).long().clamp(-self.cd, self.cd)
        discrete_face_points += self.cd

        discrete_edge_points = torch.round(
            line_points * self.cd).long().clamp(-self.cd, self.cd)
        discrete_edge_points += self.cd
        return discrete_face_points, discrete_edge_points

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*20*20*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])
        vertex_points = torch.from_numpy(data_npz['sample_points_vertices'])

        discrete_face_points, discrete_edge_points = (
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
            discrete_face_points,
            discrete_edge_points,
            discrete_vertex_points,
            face_edge_loop, face_adj, edge_adj,
            edge_face_connectivity, vertex_edge_connectivity,
            face_idx_sequence_c,
        )

    @staticmethod
    def collate_fn(batch):
        (v_prefix, face_points, edge_points, vertex_points,
         discrete_face_points,
         discrete_edge_points,
         discrete_vertex_points,
         face_edge_loop, face_adj, edge_adj,
         edge_face_connectivity, vertex_edge_connectivity,
         face_idx_sequence_c,
         ) = zip(*batch)

        face_points = pad_sequence(face_points, batch_first=True, padding_value=-1)
        edge_points = pad_sequence(edge_points, batch_first=True, padding_value=-1)
        vertex_points = pad_sequence(vertex_points, batch_first=True, padding_value=-1)
        discrete_face_points = pad_sequence(discrete_face_points, batch_first=True, padding_value=-1)
        discrete_edge_points = pad_sequence(discrete_edge_points, batch_first=True, padding_value=-1)
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
            "discrete_edge_points": discrete_edge_points,
            "discrete_vertex_points": discrete_vertex_points,

            "face_edge_loop": face_edge_loop,
            "face_adj": face_adj,
            "edge_adj": edge_adj,
            "edge_face_connectivity": edge_face_connectivity,
            "vertex_edge_connectivity": vertex_edge_connectivity,

            "face_idx_sequence": face_idx_sequence,
        }

class Face_dataset(AutoEncoder_dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Face_dataset, self).__init__(v_training_mode, v_conf)

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]

        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*20*20*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        discrete_face_points, discrete_face_bboxes, discrete_edge_points, discrete_edge_bboxes = (
            self.discrete_coordinates(face_points, line_points))

        min_face = face_points.min(dim=1).values.min(dim=1).values
        max_face = face_points.max(dim=1).values.max(dim=1).values
        discrete_face_points_unnormalized = (
            face_points * self.cd).long().clamp(-self.cd, self.cd)
        continuous_face_bboxes = torch.cat([min_face, max_face], dim=-1)
        discrete_face_points_unnormalized += self.cd

        return (
            Path(folder_path).stem,
            face_points,
            discrete_face_points, discrete_face_bboxes,
            continuous_face_bboxes, discrete_face_points_unnormalized
        )

    @staticmethod
    def collate_fn(batch):
        (v_prefix, face_points,
         discrete_face_points, discrete_face_bboxes,
            continuous_face_bboxes, discrete_face_points_unnormalized
         ) = zip(*batch)

        face_points = pad_sequence(face_points, batch_first=True, padding_value=-1)
        discrete_face_points_unnormalized = pad_sequence(discrete_face_points_unnormalized, batch_first=True, padding_value=-1)
        continuous_face_bboxes = pad_sequence(continuous_face_bboxes, batch_first=True, padding_value=-1)
        discrete_face_points = pad_sequence(discrete_face_points, batch_first=True, padding_value=-1)
        discrete_face_bboxes = pad_sequence(discrete_face_bboxes, batch_first=True, padding_value=-1)

        return {
            "v_prefix": v_prefix,
            "face_points": face_points,
            "discrete_face_points": discrete_face_points,
            "discrete_face_bboxes": discrete_face_bboxes,
            "continuous_face_bboxes": continuous_face_bboxes,
            "discrete_face_points_unnormalized": discrete_face_points_unnormalized,
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
        self.num_max_faces = v_conf['num_max_faces']
        self.is_map = v_conf['is_map']
        self.face_embedding_batch_size = v_conf['face_embedding_batch_size']
        self.length_scaling_factors = int(v_conf['length_scaling_factors']) if v_training_mode == "training" else 1
        self.folders = [f for f in folders if np.load(self.root / f).shape[0] < self.num_max_faces]
        print("Filter out {} folders with faces > {}".format(len(folders) - len(self.folders), self.num_max_faces))

        if self.is_map:
            print("Pre-load data into memory".format(len(folders) - len(self.folders), self.num_max_faces))
            self.data = []
            for folder in self.folders:
                data = torch.from_numpy(np.load(self.root / folder))
                self.data.append(data)
            self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
            self.length = max(1, self.data.shape[0] // self.face_embedding_batch_size + 1)
        else:
            self.length = len(self.folders)

    def __len__(self):
        return int(self.length_scaling_factors * self.length)

    def __getitem__(self, idx):
        # idx = 0
        if not self.is_map:
            raise NotImplementedError
            data = torch.from_numpy(np.load(self.root / self.folders[idx % self.length]))
        else:
            idx = torch.arange(self.face_embedding_batch_size * idx, self.face_embedding_batch_size * (idx + 1))
            idx = idx % self.data.shape[0]
            data = self.data[idx]
        return data

    @staticmethod
    def collate_fn(batch):
        return batch[0]
