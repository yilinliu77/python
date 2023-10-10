import math
import os.path
from pathlib import Path
import sys
import time

import faiss
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d

from shared.common_utils import export_point_cloud

try:
    sys.path.append("thirdparty")
    import cuda_distance
    import open3d as o3d
except:
    print("Cannot import cuda_distance, ignore this if you don't use 'ABC_dataset_test_mesh'")


def normalize_points(v_points):
    min_xyz = v_points.min(axis=0)
    max_xyz = v_points.max(axis=0)
    diag = np.linalg.norm(max_xyz - min_xyz)
    center_xyz = (min_xyz + max_xyz) / 2
    v_points = (v_points - center_xyz[None, :]) / diag * 2
    return v_points


def generate_coords(v_resolution):
    coords = np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution), indexing="ij")
    coords = np.stack(coords, axis=3) / (v_resolution - 1)
    coords = (coords * 2 - 1).astype(np.float32)
    return coords


def angle2vector(v_angles):
    angles = (v_angles / 65535 * np.pi * 2)
    dx = np.cos(angles[..., 0]) * np.sin(angles[..., 1])
    dy = np.sin(angles[..., 0]) * np.sin(angles[..., 1])
    dz = np.cos(angles[..., 1])
    gradients = np.stack([dx, dy, dz], axis=-1)
    return gradients


# Dataset for patch training
class ABC_patch(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch, self).__init__()
        self.data_root = v_data_root
        self.mode = v_training_mode
        self.conf = v_conf
        self.patch_size = 32
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        assert self.resolution % self.patch_size == 0
        self.num_patch = self.resolution // self.patch_size
        self.num_patches = self.num_patch ** 2

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
        else:
            raise ""
        self.index = self.index.reshape((-1, 2))

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()

        ps = self.patch_size
        x_start = (v_id_patch // self.num_patch) * ps
        y_start = (v_id_patch % self.num_patch) * ps

        with h5py.File(self.data_root, "r") as f:
            features = f["features"][
                       v_id_item,
                       x_start:x_start + ps,
                       y_start:y_start + ps,
                       ].astype(np.float32)
            flags = (f["flags"][
                     v_id_item,
                     x_start:x_start + ps,
                     y_start:y_start + ps,
                     ]).astype(bool).astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()

        features = features.reshape(ps, ps, self.num_patch, ps, -1).transpose(2, 0, 1, 3, 4)
        flags = flags.reshape(ps, ps, self.num_patch, ps).transpose(2, 0, 1, 3)

        times[1] += time.time() - cur_time
        return features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        feat_data, flag_data = self.get_patch(id_object, id_patch)
        return feat_data, flag_data, self.names[id_object], np.arange(flag_data.shape[0], dtype=np.int64) + id_patch * \
                                                            flag_data.shape[0]

    @staticmethod
    def collate_fn(v_batches):
        feat_data, flag_data, names, id_patch = [], [], [], []
        for item in v_batches:
            feat_data.append(item[0])
            flag_data.append(item[1])
            names.append(item[2])
            id_patch.append(item[3])
        feat_data = np.stack(feat_data, axis=0)
        flag_data = np.stack(flag_data, axis=0)
        id_patch = np.stack(id_patch, axis=0)
        names = np.asarray(names)

        return (
            (torch.from_numpy(feat_data), torch.zeros(feat_data.shape[0], dtype=torch.float32)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(id_patch),
        )


# Do overlap during the training
class ABC_patch_overlap(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_patch_overlap, self).__init__()
        self.data_root = v_data_root
        self.mode = v_training_mode
        self.conf = v_conf
        self.patch_size = 32
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        assert self.resolution % self.patch_size == 0
        self.num_patch = self.resolution // (self.patch_size // 2) - 1
        self.num_patches = self.num_patch ** 2

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_patches), indexing="ij"), axis=2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
        else:
            raise ""
        self.index = self.index.reshape((-1, 2))

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        times = [0] * 10
        cur_time = time.time()
        patch_size_2 = self.patch_size // 2
        patch_size_4 = patch_size_2 // 2

        x_start = v_id_patch // self.num_patch * patch_size_2
        y_start = v_id_patch % self.num_patch * patch_size_2

        with h5py.File(self.data_root, "r") as f:
            features = f["features"][
                       v_id_item,
                       x_start:x_start + self.patch_size,
                       y_start:y_start + self.patch_size,
                       ].astype(np.float32)
            # Only predict the central part of this cell
            flags = (f["flags"][
                     v_id_item,
                     x_start:x_start + self.patch_size,
                     y_start:y_start + self.patch_size,
                     ]).astype(bool).astype(np.float32)

        times[0] += time.time() - cur_time
        cur_time = time.time()
        features = np.lib.stride_tricks.sliding_window_view(
            features, window_shape=self.patch_size, axis=2
        )[:, :, ::patch_size_2].transpose(2, 0, 1, 4, 3)
        flags = np.lib.stride_tricks.sliding_window_view(
            flags, window_shape=self.patch_size, axis=2
        )[:, :, ::patch_size_2].transpose(2, 0, 1, 3)

        flags = flags[:,
                patch_size_4:-patch_size_4,
                patch_size_4:-patch_size_4,
                patch_size_4:-patch_size_4]
        times[1] += time.time() - cur_time
        return features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        feat_data, flag_data = self.get_patch(id_object, id_patch)
        return feat_data, flag_data, self.names[id_object], np.arange(flag_data.shape[0], dtype=np.int64) + id_patch * \
                                                            flag_data.shape[0]

    @staticmethod
    def collate_fn(v_batches):
        feat_data, flag_data, names, id_patch = [], [], [], []
        for item in v_batches:
            feat_data.append(item[0])
            flag_data.append(item[1])
            names.append(item[2])
            id_patch.append(item[3])
        feat_data = np.stack(feat_data, axis=0)
        flag_data = np.stack(flag_data, axis=0)
        id_patch = np.stack(id_patch, axis=0)
        names = np.asarray(names)

        return (
            torch.from_numpy(feat_data),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(id_patch),
        )


# Read pointcloud and features
class ABC_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_pc, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        self.batch_size = 64
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        assert self.resolution % self.batch_size == 0
        self.num_batches_per_item = ((self.resolution // self.batch_size) ** 3)

        if self.mode == "training":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[:self.validation_start],
                np.arange(1), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "validation":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items)[self.validation_start:],
                np.arange(self.num_batches_per_item), indexing="ij"), axis=2).reshape(-1, 2)
        elif self.mode == "testing":
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_batches_per_item), indexing="ij"), axis=2).reshape(-1, 2)
        else:
            raise ""

        self.coords = np.mgrid[:self.resolution, :self.resolution, :self.resolution] / (self.resolution - 1) * 2 - 1
        self.coords = np.transpose(self.coords, (1, 2, 3, 0)).astype(np.float32)

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, id_patch):
        times = [0] * 10
        cur_time = time.time()
        if self.mode == "training":
            id_item = v_id_item
            id_patch = 0
            index = np.random.randint(0, self.resolution - 1, (self.batch_size, self.batch_size, self.batch_size, 3))
            with h5py.File(self.data_root, "r") as f:
                points = f["poisson_points"][v_id_item].astype(np.float32)
                flags = f["flags"][v_id_item][index[..., 0], index[..., 1], index[..., 2]].astype(bool).astype(
                    np.float32)
                coords = self.coords[index[..., 0], index[..., 1], index[..., 2]]
        else:
            nums_per_dim = self.resolution // self.batch_size
            bs = self.batch_size
            xs = id_patch // nums_per_dim // nums_per_dim * bs
            ys = id_patch // nums_per_dim % nums_per_dim * bs
            zs = id_patch % nums_per_dim * bs

            with h5py.File(self.data_root, "r") as f:
                flags = f["flags"][v_id_item][xs:xs + bs, ys:ys + bs, zs:zs + bs].astype(bool).astype(np.float32)
                points = f["poisson_points"][v_id_item].astype(np.float32)
                coords = self.coords[xs:xs + bs, ys:ys + bs, zs:zs + bs]

        times[0] += time.time() - cur_time
        cur_time = time.time()
        times[1] += time.time() - cur_time
        return points[None, :], coords[None, :], flags[None, :], id_patch

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object, id_patch)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )


# Dynamically generate point features
class ABC_pc_dynamic(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_pc_dynamic, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.resolution = f["features"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            distances = f["features"][v_id_item][:, :, :, 0].astype(np.float32) / 65535 * 2
            distance_flag = distances < 0.1
            flags = f["flags"][v_id_item].astype(bool)
            points = f["poisson_points"][v_id_item].astype(np.float32)
            coords = self.coords[distance_flag]
            flags = flags[distance_flag].astype(np.float32)

        if self.mode == "training":
            if coords.shape[0] > self.max_training_sample:
                index = np.arange(coords.shape[0])
                np.random.shuffle(index)
                coords = coords[index[:self.max_training_sample]]
                flags = flags[index[:self.max_training_sample]]

        times[0] += time.time() - cur_time
        cur_time = time.time()
        return points[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )


# Whole field
class ABC_whole_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_whole_pc, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["flags"].shape[0]
            self.resolution = f["flags"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and not self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            flags = f["flags"][v_id_item].astype(bool).astype(np.float32)
            point_features = f["point_features"][v_id_item].astype(np.float32)
            coords = self.coords

        times[0] += time.time() - cur_time
        cur_time = time.time()
        return point_features[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )


# Calculate the point features online using kdtree
# Not used because it is so slow
class ABC_whole_pc_dynamic(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_conf):
        super(ABC_whole_pc_dynamic, self).__init__()
        self.data_root = v_data_root
        self.pc_dir = str(Path(v_data_root).parent)
        self.mode = v_training_mode
        self.conf = v_conf
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["flags"].shape[0]
            self.resolution = f["flags"].shape[1]
            self.names = np.asarray(
                ["{:08d}_{}".format(f["names"][i], f["ids"][i]) for i in range(f["names"].shape[0])])
        self.validation_start = max(self.num_items // 10 * 9, self.num_items - 1000)

        if self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)
        elif self.mode == "training" and self.conf["overfit"]:
            self.index = np.arange(self.num_items)[:self.validation_start]
        elif self.mode == "validation":
            self.index = np.arange(self.num_items)[self.validation_start:]
        elif self.mode == "testing":
            self.index = np.arange(self.num_items)
        else:
            raise ""

        self.coords = generate_coords(self.resolution)
        self.max_training_sample = self.conf["max_training_sample"]

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item):
        times = [0] * 10
        cur_time = time.time()
        with h5py.File(self.data_root, "r") as f:
            flags = f["flags"][v_id_item].astype(bool).astype(np.float32)
            points = f["poisson_points"][v_id_item].astype(np.float32) / 65535 * 2 - 1
            coords = self.coords

        times[0] += time.time() - cur_time
        cur_time = time.time()

        kdtree = faiss.IndexFlatL2(3)
        res = faiss.StandardGpuResources()
        gpu_kdtree = faiss.index_cpu_to_gpu(res, 0, kdtree)

        gpu_kdtree.add(points[:, :3])
        dists, indices = gpu_kdtree.search(coords.reshape(-1, 3), 1)

        times[1] += time.time() - cur_time
        cur_time = time.time()

        udf = np.sqrt(dists).reshape(self.resolution, self.resolution, self.resolution, 1)
        gradients = points[indices.reshape(-1), :3] - coords.reshape(-1, 3)
        gradients = (gradients / np.linalg.norm(gradients, axis=1, keepdims=True)).reshape(
            self.resolution, self.resolution, self.resolution, 3)
        normals = points[indices.reshape(-1), 3:6].reshape(
            self.resolution, self.resolution, self.resolution, 3)

        point_features = np.concatenate([udf, gradients, normals], axis=-1)

        times[2] += time.time() - cur_time
        cur_time = time.time()

        return point_features[None, :], coords[None, :], flags[None, :], 0

    def __getitem__(self, idx):
        id_object = self.index[idx]

        points, feat_data, flag_data, id_patch = self.get_patch(id_object)
        return points, feat_data, flag_data, self.names[id_object], id_patch

    @staticmethod
    def collate_fn(v_batches):
        points, feat_data, flag_data, names, ids = [], [], [], [], []
        for item in v_batches:
            points.append(item[0])
            feat_data.append(item[1])
            flag_data.append(item[2])
            names.append(item[3])
            ids.append(item[4])
        points = np.concatenate(points, axis=0)
        feat_data = np.concatenate(feat_data, axis=0)
        flag_data = np.concatenate(flag_data, axis=0)
        names = np.stack(names, axis=0)
        ids = np.stack(ids, axis=0)

        return (
            (torch.from_numpy(points), torch.from_numpy(feat_data)),
            torch.from_numpy(flag_data),
            names,
            torch.from_numpy(ids)
        )


class ABC_test_mesh(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_output_features=4, v_resolution=256, v_output_root=None):
        super(ABC_test_mesh, self).__init__()
        self.batch_size = v_batch_size
        self.output_features = v_output_features
        self.data_root = v_data_root
        self.resolution = v_resolution

        assert v_resolution % 32 == 0
        num_patches_per_dim = v_resolution // 32
        self.coords = generate_coords(v_resolution).reshape(-1, 3)
        self.num_patches = (v_resolution // 32) ** 3
        assert self.num_patches % v_resolution == 0

        print("Prepare mesh data")
        if not os.path.exists(v_data_root):
            print("Cannot find ", v_data_root)
        mesh = o3d.io.read_triangle_mesh(v_data_root)
        mesh.compute_triangle_normals()
        # Normalize
        points = np.asarray(mesh.vertices)
        min_xyz = points.min(axis=0)
        center_xyz = (min_xyz + points.max(axis=0)) / 2
        bbox = points.max(axis=0) - min_xyz
        diagonal = np.linalg.norm(bbox)
        points = (points - center_xyz) / diagonal * 2
        mesh.vertices = o3d.utility.Vector3dVector(points)
        if v_output_root is not None:
            o3d.io.write_triangle_mesh(os.path.join(v_output_root, "temp.ply"), mesh)
        use_dense_feature = True
        if use_dense_feature:
            points = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.triangle_normals)
            triangles = points[faces]
            num_triangles = triangles.shape[0]
            num_queries = self.coords.shape[0]
            query_result = cuda_distance.query(triangles.reshape(-1), self.coords.reshape(-1), 512, 512 ** 3)

            udf = np.asarray(query_result[0]).astype(np.float32)
            closest_points = np.asarray(query_result[1]).reshape((num_queries, 3)).astype(np.float32)
            dir = closest_points - self.coords
            dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)

            normals = normals[query_result[2]].astype(np.float32)
        else:
            pc = mesh.sample_points_poisson_disk(10000)
            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            index = faiss.IndexFlatL2(3)
            index.add(points)
            dists, indices = index.search(self.coords, 1)
            closest_points = points[indices.reshape(-1)]
            dir = closest_points - self.coords
            dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)
            udf = np.sqrt(dists).reshape(-1)
            normals = normals[indices.reshape(-1)]
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Revised at 1004
        feat_data = np.concatenate([udf[:, None], dir, normals], axis=1)
        self.feat_data = feat_data.reshape(
            (v_resolution, v_resolution, v_resolution, 7)).astype(np.float32)[:, :, :, :self.output_features]

        self.patch_size = 32
        self.patch_list = []
        for x in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
            for y in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                for z in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                    self.patch_list.append((x, y, z))

        pass

    def __len__(self):
        return math.ceil(len(self.patch_list) / self.batch_size)

    def __getitem__(self, idx):
        features = []
        id_list = []
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            if id >= len(self.patch_list):
                break
            feat_data = self.feat_data[
                        self.patch_list[id][0]:self.patch_list[id][0] + self.patch_size,
                        self.patch_list[id][1]:self.patch_list[id][1] + self.patch_size,
                        self.patch_list[id][2]:self.patch_list[id][2] + self.patch_size,
                        ]
            features.append(np.transpose(feat_data, [3, 0, 1, 2]))
            id_list.append(self.patch_list[id])
        features = np.stack(features, axis=0)
        return features, id_list


class ABC_test_pc(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_output_features=4, v_resolution=256):
        super(ABC_test_pc, self).__init__()
        self.batch_size = v_batch_size
        self.output_features = v_output_features
        self.data_root = v_data_root
        self.resolution = v_resolution

        assert v_resolution % 32 == 0
        num_patches_per_dim = v_resolution // 32
        self.coords = generate_coords(v_resolution)
        self.num_patches = (v_resolution // 32) ** 3
        assert self.num_patches % v_resolution == 0

        print("Prepare mesh data")
        if not os.path.exists(v_data_root):
            print("Cannot find ", v_data_root)
        pcd = o3d.io.read_point_cloud(v_data_root)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        index = faiss.IndexFlatL2(3)
        index.add(points)
        dists, indices = index.search(self.coords, 1)
        closest_points = points[indices.reshape(-1)]
        dir = closest_points - self.coords
        dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)
        udf = np.sqrt(dists).reshape(-1)
        normals = normals[indices.reshape(-1)]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Revised at 1004
        feat_data = np.concatenate([udf[:, None], dir, normals], axis=1)
        self.feat_data = feat_data.reshape(
            (v_resolution, v_resolution, v_resolution, self.output_features)).astype(np.float32)

        self.patch_size = 32
        self.patch_list = []
        for x in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
            for y in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                for z in range(0, v_resolution - self.patch_size + 1, self.patch_size // 2):
                    self.patch_list.append((x, y, z))

        pass

    def __len__(self):
        return math.ceil(len(self.patch_list) / self.batch_size)

    def __getitem__(self, idx):
        features = []
        id_list = []
        for i in range(self.batch_size):
            id = idx * self.batch_size + i
            if id >= len(self.patch_list):
                break
            feat_data = self.feat_data[
                        self.patch_list[id][0]:self.patch_list[id][0] + self.patch_size,
                        self.patch_list[id][1]:self.patch_list[id][1] + self.patch_size,
                        self.patch_list[id][2]:self.patch_list[id][2] + self.patch_size,
                        ]
            features.append(np.transpose(feat_data, [3, 0, 1, 2]))
            id_list.append(self.patch_list[id])
        features = np.stack(features, axis=0)
        return features, id_list


class ABC_test_voxel(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_resolution=256, v_output_root=None):
        super(ABC_test_voxel, self).__init__()
        self.batch_size = v_batch_size
        self.data_root = v_data_root
        self.resolution = v_resolution

        print("Prepare mesh data")
        if not os.path.exists(v_data_root):
            print("Cannot find ", v_data_root)
        mesh = o3d.io.read_triangle_mesh(v_data_root)
        vertices = np.asarray(mesh.vertices)
        vertices = normalize_points(vertices)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        if v_output_root is not None:
            o3d.io.write_triangle_mesh(os.path.join(v_output_root, "temp.ply"), mesh)
        # poisson sampling
        pcd = mesh.sample_points_poisson_disk(10000)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # Revised at 1004
        feat_data = np.concatenate([points, normals], axis=1)
        self.feat_data = feat_data.astype(np.float32)

        self.query_coords = generate_coords(v_resolution).reshape(-1, 3)
        self.bs_1d = self.batch_size ** 3  # actual batch size 1D
        self.num_patches = math.ceil(self.query_coords.shape[0] / self.bs_1d)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        return (
            self.feat_data,
            self.query_coords[idx * self.bs_1d:min(self.query_coords.shape[0], (idx + 1) * self.bs_1d)].reshape(
                self.batch_size, self.batch_size, self.batch_size, 3)
        )
