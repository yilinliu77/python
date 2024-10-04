import math
import os.path
from pathlib import Path
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

from shared.common_utils import export_point_cloud, check_dir

from typing import Final
from einops import rearrange
import torchvision.transforms as T
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

def normalize_coord(v_points):
    shape = v_points.shape
    num_items = shape[0]
    v_points = v_points.reshape(num_items, -1, 3)
    center = v_points.mean(dim=1, keepdim=True)
    scale = v_points.max(dim=1, keepdim=True)[0] - v_points.min(dim=1, keepdim=True)[0]
    points = (v_points - center) / (scale+1e-6)
    points = points.reshape(shape)
    return points, center[:,0], scale[:,0]

def denormalize_coord(points, center, scale):
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    return points

def discrete_coord(points, center, scale, v_dim):
    points = torch.round((points + 0.5) * v_dim)
    points = torch.clamp(points, 0, v_dim-1).to(torch.long)

    center = torch.round((center + 1.) * v_dim / 2)
    center = torch.clamp(center, 0, v_dim-1).to(torch.long)

    scale = torch.round(scale * v_dim / 2)
    scale = torch.clamp(scale, 0, v_dim-1).to(torch.long)
    return points, center, scale

def continuous_coord(points, center, scale, v_dim):
    points = points / v_dim - 0.5
    scale = scale / v_dim * 2
    center = center / v_dim * 2 - 1
    return points, center, scale

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
        face_points = torch.from_numpy(data_npz['sample_points_faces'])[...,:-3]
        line_points = torch.from_numpy(data_npz['sample_points_lines'])[...,:-3]

        face_points_norm, face_center, face_scale = normalize_coord(face_points)
        edge_points_norm, edge_center, edge_scale = normalize_coord(line_points)
        face_points_discrete, face_center_discrete, face_scale_discrete = discrete_coord(face_points_norm, face_center, face_scale, 256)
        edge_points_discrete, edge_center_discrete, edge_scale_discrete = discrete_coord(edge_points_norm, edge_center, edge_scale, 256)
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
            face_points_discrete, face_center_discrete, face_scale_discrete,
            edge_points_discrete, edge_center_discrete, edge_scale_discrete,
            edge_face_connectivity
        )

    @staticmethod
    def collate_fn(batch):
        (
            v_prefix, face_points, edge_points,
            face_adj, zero_positions,
            face_points_norm, face_center, face_scale,
            edge_points_norm, edge_center, edge_scale,
            face_points_discrete, face_center_discrete, face_scale_discrete,
            edge_points_discrete, edge_center_discrete, edge_scale_discrete,
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
            
        dtype=torch.float32
        flat_zero_positions = torch.cat(flat_zero_positions, dim=0)

        return {
            "v_prefix": v_prefix,
            "edge_points": torch.cat(edge_points, dim=0),
            "face_points": torch.cat(face_points, dim=0),

            "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0),
            "zero_positions": flat_zero_positions,
            "attn_mask": attn_mask,
            "edge_attn_mask": edge_attn_mask,

            "num_face_record": num_face_record,
            "valid_mask": valid_mask,

            "face_points_norm": torch.cat(face_points_norm, dim=0),
            "face_center": torch.cat(face_center, dim=0),
            "face_scale": torch.cat(face_scale, dim=0),
            "edge_points_norm": torch.cat(edge_points_norm, dim=0),
            "edge_center": torch.cat(edge_center, dim=0),
            "edge_scale": torch.cat(edge_scale, dim=0),

            # "face_points_discrete": flat_face_points_discrete,
            # "edge_points_discrete": flat_edge_points_discrete,
            # "face_center_discrete": flat_face_center_discrete,
            # "face_scale_discrete": flat_face_scale_discrete,
            # "edge_center_discrete": flat_edge_center_discrete,
            # "edge_scale_discrete": flat_edge_scale_discrete,
        }

class Diffusion_dataset(torch.utils.data.Dataset):
    def __init__(self, v_training_mode, v_conf):
        super(Diffusion_dataset, self).__init__()
        self.mode = v_training_mode
        self.conf = v_conf
        scale_factor = int(v_conf["scale_factor"])
        self.max_intersection = 500
        self.face_z_dataset = v_conf['face_z']
        if v_training_mode == "testing":
            self.dataset_path = Path(v_conf['test_dataset'])
            scale_factor = 1
        elif v_training_mode == "training":
            self.dataset_path = Path(v_conf['train_dataset'])
        elif v_training_mode == "validation":
            self.dataset_path = Path(v_conf['val_dataset'])
            scale_factor = 1
        else:
            raise
        filelist1 = os.listdir(self.dataset_path)
        filelist2 = [item[:8] for item in os.listdir(self.face_z_dataset) if os.path.exists(os.path.join(self.face_z_dataset, item+"/feature.npy"))]
        filelist = list(set(filelist1) & set(filelist2))
        filelist.sort()

        self.condition = v_conf["condition"]
        self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        self.max_faces = 64
        if False: # Overfitting mode
            self.data_folders = filelist[:100] * scale_factor
        else:
            self.data_folders = filelist * scale_factor
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
        try:
            data_npz = np.load(os.path.join(self.face_z_dataset, folder_path + "_feature.npz"))['face_features']
        except:
            data_npz = np.load(os.path.join(self.face_z_dataset, folder_path + "/feature.npy"))
        face_features = torch.from_numpy(data_npz)

        padded_face_features = torch.zeros((self.max_faces, *face_features.shape[1:]), dtype=torch.float32)
        padded_face_features[:face_features.shape[0]] = face_features

        condition = {

        }
        if self.condition == "single_img" or self.condition == "multi_img":
            cache_data = True
            if self.condition == "single_img":
                idx = np.random.randint(0, 23, 1)
            else:
                idx = np.random.randint(0, 23, 4)
            if cache_data:
                ori_data = np.load(self.dataset_path / folder_path / "img_feature_dinov2.npy")
                img_features = torch.from_numpy(ori_data[idx]).float()
                condition["img_features"] = img_features
            else:
                ori_data = np.load(self.dataset_path / folder_path / "data.npz")["imgs"]
                imgs = ori_data[idx]
                transformed_imgs = []
                for id in range(imgs.shape[0]):
                    transformed_imgs.append(self.transform(imgs[id]))
                transformed_imgs = torch.stack(transformed_imgs, dim=0)
                condition["imgs"] = transformed_imgs
            condition["img_id"] = torch.from_numpy(idx)
        elif self.condition == "pc":
            pc = o3d.io.read_point_cloud(str(self.dataset_path / folder_path / "pc.ply"))
            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            condition["points"] = torch.from_numpy(np.concatenate((points,normals),axis=-1)).float()[None,]
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
        condition_out = {key:[] for key in keys}
        for idx in range(len(conditions)):
            for key in keys:
                condition_out[key].append(conditions[idx][key])

        for key in keys:
            condition_out[key] = torch.stack(condition_out[key], dim=0)

        return {
            "v_prefix": v_prefix,
            "face_features": face_features,
            "conditions": condition_out
        }
