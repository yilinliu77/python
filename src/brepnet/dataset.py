import math
import os.path
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import plyfile
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
from shared.common_utils import export_point_cloud, check_dir

from typing import Final
from einops import rearrange
import torchvision.transforms as T
import networkx as nx

from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.transform import Rotation

import torch.nn.functional as F


def get_face_idx_sequence(edge_face_connectivity, face_points):
    G = nx.Graph()

    for idx, (edge_id, face1, face2) in enumerate(edge_face_connectivity):
        G.add_edge(int(face1), int(face2))

    if G.number_of_nodes() != face_points.shape[0]:
        raise ValueError("Number of nodes is not equal to number of faces")

    face_idx_sequence = list(nx.bfs_tree(G, 0))

    return torch.tensor(face_idx_sequence, dtype=torch.long, device=edge_face_connectivity.device)


def normalize_coord(v_points):
    points = v_points[..., :3]
    normals = v_points[..., 3:]
    shape = points.shape
    num_items = shape[0]
    points = points.reshape(num_items, -1, 3)
    target_points = points + normals.reshape(num_items, -1, 3)

    center = points.mean(dim=1, keepdim=True)
    scale = points.max(dim=1, keepdim=True)[0] - points.min(dim=1, keepdim=True)[0]
    points = (points - center) / (scale + 1e-6)
    target_points = (target_points - center) / (scale + 1e-6)
    normals = target_points - points
    normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))

    points = points.reshape(shape)
    normals = normals.reshape(shape)

    return points, normals, center[:, 0], scale[:, 0]


def denormalize_coord(points, center, scale):
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    return points


def denormalize_coord2(points, bbox):
    normal = points[..., 3:]
    points = points[..., :3]
    target_points = points + normal
    center = bbox[..., :3]
    scale = bbox[..., 3:]
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    target_points = target_points * scale + center
    normal = target_points - points
    normal = normal / (1e-6 + torch.linalg.norm(normal, dim=-1, keepdim=True))
    points = torch.cat((points, normal), dim=-1)
    return points


def discrete_coord(points, center, scale, v_dim):
    points = torch.round((points + 0.5) * v_dim)
    points = torch.clamp(points, 0, v_dim - 1).to(torch.long)

    center = torch.round((center + 1.) * v_dim / 2)
    center = torch.clamp(center, 0, v_dim - 1).to(torch.long)

    scale = torch.round(scale * v_dim / 2)
    scale = torch.clamp(scale, 0, v_dim - 1).to(torch.long)
    return points, center, scale


def continuous_coord(points, center, scale, v_dim):
    points = points / v_dim - 0.5
    scale = scale / v_dim * 2
    center = center / v_dim * 2 - 1
    return points, center, scale


class AutoEncoder_geo_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_geo_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        self.max_intersection = 500
        if v_training_mode == "testing":
            listfile = v_conf['test_dataset']
        elif v_training_mode == "training":
            listfile = v_conf['train_dataset']
        elif v_training_mode == "validation":
            listfile = v_conf['val_dataset']
        else:
            raise

        self.data_folders = [item.strip() for item in open(listfile).readlines()]
        self.root = Path(v_conf["data_root"])
        self.is_aug = v_conf["is_aug"]

        if v_conf["is_overfit"]:
            self.data_folders = self.data_folders[:100]
            if v_training_mode == "training":
                self.data_folders = self.data_folders * 100
        print(len(self.data_folders))

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        # idx = 0
        prefix = self.data_folders[idx]
        data_npz = np.load(str(self.root / prefix / "data.npz"))

        # Face sample points (num_faces*32*32*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord(face_points)
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord(line_points)

        face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
        edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)
        face_bbox = torch.cat((face_center, face_scale), dim=-1)
        edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

        return (
            prefix,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
        )

    @staticmethod
    def collate_fn(batch):
        (
            prefix,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
        ) = zip(*batch)
        bs = len(prefix)

        dtype = torch.float32

        return {
            "v_prefix" : prefix,
            "face_norm": torch.cat(face_norm, dim=0).to(dtype),
            "face_bbox": torch.cat(face_bbox, dim=0).to(dtype),
            "edge_norm": torch.cat(edge_norm, dim=0).to(dtype),
            "edge_bbox": torch.cat(edge_bbox, dim=0).to(dtype),
        }


class AutoEncoder_dataset2(AutoEncoder_geo_dataset):
    def __init__(self, v_training_mode, v_conf):
        super(AutoEncoder_dataset2, self).__init__(v_training_mode, v_conf)
        self.disable_half = v_conf["disable_half"]

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        # idx = 0
        prefix = self.data_folders[idx]
        data_npz = np.load(str(self.root / prefix / "data.npz"))

        # Face sample points (num_faces*32*32*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        if self.is_aug == 0:
            matrix = np.identity(3)
        if self.is_aug == 1:
            matrix = Rotation.from_euler('xyz', np.random.randint(0, 3, 3) * np.pi / 2).as_matrix()
        elif self.is_aug == 2:
            matrix = Rotation.from_euler('xyz', np.random.rand(3) * np.pi * 2).as_matrix()
        if self.is_aug != 0:
            matrix = torch.from_numpy(matrix).float()
            fp = face_points[..., :3].reshape(-1, 3)
            lp = line_points[..., :3].reshape(-1, 3)
            fp1 = (matrix @ fp.T).T
            lp1 = (matrix @ lp.T).T

            if face_points.shape[1] > 3:
                fn = face_points[..., 3:].reshape(-1, 3)
                ln = line_points[..., 3:].reshape(-1, 3)
                ft = fp + fn
                lt = lp + ln
                ft1 = (matrix @ ft.T).T
                lt1 = (matrix @ lt.T).T

                fn1 = ft1 - fp1
                ln1 = lt1 - lp1

                fn1 = fn1 / (1e-6 + torch.linalg.norm(fn, dim=-1, keepdim=True))
                ln1 = ln1 / (1e-6 + torch.linalg.norm(ln, dim=-1, keepdim=True))
                face_points[..., 3:] = fn1.reshape(face_points[..., 3:].shape)
                line_points[..., 3:] = ln1.reshape(line_points[..., 3:].shape)

            face_points[..., :3] = fp1.reshape(face_points[..., :3].shape)
            line_points[..., :3] = lp1.reshape(line_points[..., :3].shape)

        num_faces = face_points.shape[0]
        num_edges = line_points.shape[0]

        face_adj = torch.from_numpy(data_npz['face_adj'])
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]

        zero_positions = torch.from_numpy(data_npz['zero_positions'])

        if self.disable_half:
            cache = set()
            edge_face_connectivity2 = []
            for item in edge_face_connectivity:
                if (item[2].item(), item[1].item()) not in cache:
                    edge_face_connectivity2.append(item)
                    cache.add((item[1].item(), item[2].item()))

            edge_face_connectivity = torch.stack(edge_face_connectivity2, dim=0)
            line_points = line_points[edge_face_connectivity[:, 0]]

            edge_face_connectivity[:, 0] = torch.arange(edge_face_connectivity.shape[0])
            edge_face_connectivity = torch.cat((edge_face_connectivity, edge_face_connectivity[:, [0, 2, 1]]), dim=0)
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            face_adj[edge_face_connectivity[:, 1], edge_face_connectivity[:, 2]] = True
            face_adj[edge_face_connectivity[:, 2], edge_face_connectivity[:, 1]] = True
            zero_positions = torch.stack(torch.where(face_adj == False), dim=1)

        if zero_positions.shape[0] > edge_face_connectivity.shape[0]:
            index = np.random.choice(zero_positions.shape[0], edge_face_connectivity.shape[0], replace=False)
            zero_positions = zero_positions[index]

        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord(face_points)
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord(line_points)

        face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
        edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)

        face_bbox = torch.cat((face_center, face_scale), dim=-1)
        edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

        # face_points = denormalize_coord2(face_norm, face_bbox)
        # edge_points = denormalize_coord2(edge_norm, edge_bbox)

        return (
            prefix,
            face_points, line_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj
        )

    @staticmethod
    def collate_fn(batch):
        (
            prefix,
            face_points, edge_points,
            face_norm, edge_norm,
            face_bbox, edge_bbox,
            edge_face_connectivity, zero_positions, face_adj
        ) = zip(*batch)
        bs = len(prefix)

        flat_zero_positions = []
        num_face_record = []

        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces += face_norm[i].shape[0]
            num_edges += edge_norm[i].shape[0]
            num_face_record.append(face_norm[i].shape[0])
        num_face_record = torch.tensor(num_face_record, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
            id_cur += edge_conn_num[i]

        num_max_faces = num_face_record.max()
        valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
        for i in range(bs):
            valid_mask[i, :num_face_record[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur + face_norm[i].shape[0], id_cur: id_cur + face_norm[i].shape[0]] = False
            id_cur += face_norm[i].shape[0]

        dtype = torch.float32
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        dtype = torch.float32

        return {
            "v_prefix"              : prefix,
            "face_points"             : torch.cat(face_points, dim=0).to(dtype),
            "face_norm"             : torch.cat(face_norm, dim=0).to(dtype),
            "edge_points"             : torch.cat(edge_points, dim=0).to(dtype),
            "edge_norm"             : torch.cat(edge_norm, dim=0).to(dtype),
            "face_bbox"             : torch.cat(face_bbox, dim=0).to(dtype),
            "edge_bbox"             : torch.cat(edge_bbox, dim=0).to(dtype),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions"        : flat_zero_positions,
            "attn_mask"             : attn_mask,
            "edge_attn_mask"        : edge_attn_mask,

            "num_face_record"       : num_face_record,
            "valid_mask"            : valid_mask,
        }


def rotate_matrix(angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    return rotation_matrix


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

        self.data_folders = [folder for folder in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, folder))]
        if True:
            deduplicate_file = "src/brepnet/data/list/deduplicated_deepcad_{}.txt".format(v_training_mode)
            print("Use deduplicate list ", deduplicate_file)
            deduplicate_list = [item.strip() for item in open(deduplicate_file).readlines()]
            self.data_folders = set(self.data_folders) & set(deduplicate_list)
            self.data_folders = list(self.data_folders)

        self.data_folders.sort()
        self.data_folders = [os.path.join(self.dataset_path, item) for item in self.data_folders]

        self.src_data_sum = len(self.data_folders)

        self.check_data(self.dataset_path, v_training_mode)

        self.data_sum = len(self.data_folders)

        self.is_aug = v_conf["is_aug"]

        print("Dataset INFO")
        print("Src data_folders:", self.src_data_sum)
        print("After removing:", self.data_sum)
        print("Remove invalid data.npz folders:", self.src_data_sum - self.data_sum)

    def __len__(self):
        return len(self.data_folders)

    def check_data(self, v_path, v_training_mode):
        filepath = os.path.join(v_path, r"id_larger_than_64_faces.txt")
        ignore_ids = []
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
            try:
                self.data_folders.remove(folder_path)
            except:
                pass

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*32*32*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        if self.is_aug == 0:
            matrix = np.identity(3)
        if self.is_aug == 1:
            matrix = Rotation.from_euler('xyz', np.random.randint(0, 3, 3) * np.pi / 2).as_matrix()
        elif self.is_aug == 2:
            matrix = Rotation.from_euler('xyz', np.random.rand(3) * np.pi * 2).as_matrix()
        if self.is_aug != 0:
            matrix = torch.from_numpy(matrix).float()
            fp = face_points[..., :3].reshape(-1, 3)
            lp = line_points[..., :3].reshape(-1, 3)
            fp1 = (matrix @ fp.T).T
            lp1 = (matrix @ lp.T).T

            if face_points.shape[1] > 3:
                fn = face_points[..., 3:].reshape(-1, 3)
                ln = line_points[..., 3:].reshape(-1, 3)
                ft = fp + fn
                lt = lp + ln
                ft1 = (matrix @ ft.T).T
                lt1 = (matrix @ lt.T).T

                fn1 = ft1 - fp1
                ln1 = lt1 - lp1

                fn1 = fn1 / (1e-6 + torch.linalg.norm(fn, dim=-1, keepdim=True))
                ln1 = ln1 / (1e-6 + torch.linalg.norm(ln, dim=-1, keepdim=True))
                face_points[..., 3:] = fn1.reshape(face_points[..., 3:].shape)
                line_points[..., 3:] = ln1.reshape(line_points[..., 3:].shape)

            face_points[..., :3] = fp1.reshape(face_points[..., :3].shape)
            line_points[..., :3] = lp1.reshape(line_points[..., :3].shape)

        face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord(face_points)
        edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord(line_points)
        # face_points_discrete, face_center_discrete, face_scale_discrete = discrete_coord(face_points_norm, face_center, face_scale, 256)
        # edge_points_discrete, edge_center_discrete, edge_scale_discrete = discrete_coord(edge_points_norm, edge_center, edge_scale, 256)
        # face_points = continuous_coord(face_points_discrete, face_center, face_scale, 256)

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
            face_points_norm, face_center, face_scale,
            edge_points_norm, edge_center, edge_scale,
            edge_face_connectivity
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, face_points, edge_points,
            face_adj, zero_positions,
            face_points_norm, face_center, face_scale,
            edge_points_norm, edge_center, edge_scale,
            edge_face_connectivity
        ) = zip(*batch)
        bs = len(v_prefix)
        flat_zero_positions = []
        num_face_record = []

        num_faces = 0
        num_edges = 0
        edge_conn_num = []
        for i in range(bs):
            edge_face_connectivity[i][:, 0] += num_edges
            edge_face_connectivity[i][:, 1:] += num_faces
            edge_conn_num.append(edge_face_connectivity[i].shape[0])
            flat_zero_positions.append(zero_positions[i] + num_faces)
            num_faces += face_points[i].shape[0]
            num_edges += edge_points[i].shape[0]
            num_face_record.append(face_points[i].shape[0])
        num_face_record = torch.tensor(num_face_record, dtype=torch.long)
        num_sum_edges = sum(edge_conn_num)
        edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
        id_cur = 0
        for i in range(bs):
            edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
            id_cur += edge_conn_num[i]

        num_max_faces = num_face_record.max()
        valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
        for i in range(bs):
            valid_mask[i, :num_face_record[i]] = True
        attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
        id_cur = 0
        for i in range(bs):
            attn_mask[id_cur:id_cur + face_points[i].shape[0], id_cur: id_cur + face_points[i].shape[0]] = False
            id_cur += face_points[i].shape[0]

        dtype = torch.float32
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        return {
            "v_prefix"              : v_prefix,
            "edge_points"           : torch.cat(edge_points, dim=0),
            "face_points"           : torch.cat(face_points, dim=0),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions"        : flat_zero_positions,
            "attn_mask"             : attn_mask,
            "edge_attn_mask"        : edge_attn_mask,

            "num_face_record"       : num_face_record,
            "valid_mask"            : valid_mask,

            "face_norm"      : torch.cat(face_points_norm, dim=0),
            "face_center"           : torch.cat(face_center, dim=0),
            "face_scale"            : torch.cat(face_scale, dim=0),
            "edge_norm"      : torch.cat(edge_points_norm, dim=0),
            "edge_center"           : torch.cat(edge_center, dim=0),
            "edge_scale"            : torch.cat(edge_scale, dim=0),
        }


class Diffusion_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Diffusion_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        scale_factor = int(v_conf["scale_factor"])
        self.max_intersection = 500
        self.latent_root = v_conf['face_z']
        if v_training_mode == "testing":
            self.data_split = Path(v_conf['test_dataset'])
            scale_factor = 1
        elif v_training_mode == "training":
            self.data_split = Path(v_conf['train_dataset'])
        elif v_training_mode == "validation":
            self.data_split = Path(v_conf['val_dataset'])
            scale_factor = 1
        else:
            raise

        self.max_faces = v_conf["num_max_faces"]
        print("Use deduplicate list ", self.data_split)
        filelist = [item.strip() for item in open(self.data_split).readlines()]
        filelist.sort()

        self.condition = v_conf["condition"]
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        if v_conf["overfit"]:  # Overfitting mode
            self.data_folders = filelist[:100] * scale_factor
        else:
            self.data_folders = filelist * scale_factor
        print("Total data num:", len(self.data_folders))

        self.is_aug = v_conf["is_aug"]
        self.cached_condition = v_conf["cached_condition"]
        self.pad_method = v_conf["pad_method"]
        self.addition_tag = v_conf["addition_tag"]

        return

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        # idx = 0
        folder_path = self.data_folders[idx]
        data_npz = np.load(os.path.join(self.latent_root, folder_path + "/features.npy"))
        face_features = torch.from_numpy(data_npz)

        if self.pad_method == "zero":
            padded_face_features = torch.zeros((self.max_faces, 32), dtype=torch.float32)
            padded_face_features[:face_features.shape[0]] = face_features
        elif self.pad_method == "random":
            add_flag = torch.ones(self.max_faces, dtype=face_features.dtype)
            add_flag[face_features.shape[0]:] = -1
            add_flag = add_flag[:,None]

            index = torch.randperm(face_features.shape[0])
            num_repeats = math.ceil(self.max_faces / index.shape[0])
            index = index.repeat(num_repeats)[:self.max_faces]
            index2 = torch.randperm(self.max_faces)
            index = index[index2]
            padded_face_features = face_features[index]
            if self.addition_tag:
                padded_face_features = torch.cat((padded_face_features, add_flag[index2]), dim=-1)
        else:
            raise ValueError("Invalid pad method")
        condition = {

        }
        if self.condition == "single_img" or self.condition == "multi_img" or self.condition == "sketch":
            cache_data = self.cached_condition
            if self.condition == "single_img":
                idx = np.random.choice(np.arange(24), 1, replace=False)
            # elif self.condition == "sketch":
            # idx = np.random.choice(np.arange(24,48), 1, replace=False)
            else:
                idx = np.random.choice(np.arange(24), 4, replace=False)
            if cache_data:
                ori_data = np.load(self.data_split / folder_path / "img_feature_dinov2.npy")
                img_features = torch.from_numpy(ori_data[idx]).float()
                condition["img_features"] = img_features
            else:
                ori_data = np.load(self.data_split / folder_path / "data.npz")["imgs"]
                imgs = ori_data[idx]
                transformed_imgs = []
                for id in range(imgs.shape[0]):
                    transformed_imgs.append(self.transform(imgs[id]))
                transformed_imgs = torch.stack(transformed_imgs, dim=0)
                condition["ori_imgs"] = torch.from_numpy(ori_data[idx])
                condition["imgs"] = transformed_imgs
            condition["img_id"] = torch.from_numpy(idx)
        elif self.condition == "pc":
            pc = o3d.io.read_point_cloud(str(self.data_split / folder_path / "pc.ply"))
            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            condition["points"] = torch.from_numpy(np.concatenate((points, normals), axis=-1)).float()[None,]
        return (
            folder_path,
            padded_face_features,
            condition
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, v_face_features, conditions
        ) = zip(*batch)

        face_features = torch.stack(v_face_features, dim=0)

        keys = conditions[0].keys()
        condition_out = {key: [] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0)

        return {
            "v_prefix"     : v_prefix,
            "face_features": face_features,
            "conditions"   : condition_out
        }


class Dummy_dataset(torch.utils.data.Dataset):
    def __init__(self, v_mode, v_conf):
        self.length = v_conf["length"]

    def __len__(self,):
        return self.length

    def __getitem__(self, idx):
        return "{:08d}".format(idx)
    
    @staticmethod
    def collate_fn(batch):
        return {
            "v_prefix"     : batch,
        }
