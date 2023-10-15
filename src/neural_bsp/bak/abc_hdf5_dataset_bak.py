import math
import os.path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    sys.path.append("thirdparty")
    import cuda_distance
    import open3d as o3d
except:
    print("Cannot import cuda_distance, ignore this if you don't use 'ABC_dataset_test_mesh'")


class ABC_dataset_patch_hdf5(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode, v_batch_size, v_output_features=4):
        super(ABC_dataset_patch_hdf5, self).__init__()
        self.data_root = v_data_root
        self.batch_size = v_batch_size
        self.output_features = v_output_features
        if os.path.isdir(self.data_root):
            self.names = list(set([item[:8] for item in os.listdir(v_data_root)]))
            self.names = sorted(self.names, key=lambda x: int(x))
            self.names = np.asarray(self.names)
            self.objects = [os.path.join(v_data_root, item) for item in self.names]
            self.num_items = len(self.objects)
            self.num_patches = np.load(self.objects[0] + "_feat.npy").shape[0]
        else:
            with h5py.File(self.data_root, "r") as f:
                self.num_items = f["features"].shape[0]
                self.num_patches = f["features"].shape[1]
                self.names = np.asarray(["{:08d}".format(item) for item in np.asarray(f["names"])])
        self.mode = v_training_mode
        self.validation_start = self.num_items // 4 * 3

        assert self.num_patches % self.batch_size == 0

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
        self.index = self.index.reshape((self.index.shape[0], -1, self.batch_size, 2)).reshape((-1, self.batch_size, 2))

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][v_id_item[0], v_id_patch[0]:v_id_patch[-1] + 1]
            flags = f["flags"][v_id_item[0], v_id_patch[0]:v_id_patch[-1] + 1]
        return v_id_item, v_id_patch, features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, :, 0]
        id_patch = self.index[idx, :, 1]

        times = [0] * 10
        cur_time = time.time()
        id_obj, id_patch, feat_data, flag_data = self.get_patch(id_object, id_patch)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        if self.output_features == 4:
            feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3)) * np.pi * 2
            dx = np.cos(feat_data[:, 1]) * np.sin(feat_data[:, 2])
            dy = np.sin(feat_data[:, 1]) * np.sin(feat_data[:, 2])
            dz = np.cos(feat_data[:, 2])
            feat_data = np.concatenate([dx[:, None], dy[:, None], dz[:, None], feat_data[:, 0:1]], axis=1)
        elif self.output_features == 1:
            feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3)) * np.pi * 2
            feat_data = feat_data[:, 0:1]
        elif self.output_features == 3:
            feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3))
        flag_data = flag_data.astype(np.float32)[:, None, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_obj], id_patch

    @staticmethod
    def collate_fn(v_batches):
        return [torch.from_numpy(v_batches[0][0]),
                torch.from_numpy(v_batches[0][1]),
                v_batches[0][2],
                torch.from_numpy(v_batches[0][3])]


class ABC_dataset_patch_hdf5_sample(ABC_dataset_patch_hdf5):
    def __init__(self, v_data_root, v_training_mode, v_batch_size, v_output_features=4):
        super(ABC_dataset_patch_hdf5_sample, self).__init__(
            v_data_root, v_training_mode, v_batch_size, v_output_features)

    def get_patch(self, v_id_item, v_id_patch):
        if self.mode == "training":
            index = np.unique(np.random.choice(np.arange(self.num_patches), self.batch_size, replace=False))
            index = np.sort(index)
            id_object = v_id_item[0]
            id_patch = index
        elif self.mode == "validation":
            id_object = v_id_item[0]
            id_patch = v_id_patch
        else:
            raise ""
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][id_object, id_patch]
            flags = f["flags"][id_object, id_patch]
        return np.asarray([id_object] * self.batch_size), id_patch, features, flags


class ABC_dataset_patch_hdf5_test(ABC_dataset_patch_hdf5):
    def __init__(self, v_data_root, v_training_mode, v_batch_size, v_output_features=4, v_test_list=[]):
        super(ABC_dataset_patch_hdf5_test, self).__init__(v_data_root, v_training_mode, v_batch_size, v_output_features)

        if len(v_test_list) > 0:
            self.idxs = np.zeros(len(v_test_list), dtype=np.int32)
            for id, name in enumerate(v_test_list):
                if name in self.names:
                    self.idxs[id] = np.where(self.names == name)[0][0]
                else:
                    print("Cannot find ", name, " in the dataset")
                    raise ""
            self.names = np.asarray(v_test_list)
            self.total_num_items = len(v_test_list)
            self.num_items = self.total_num_items
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
            self.index = self.index.reshape((self.index.shape[0], -1, self.batch_size, 2)).reshape(
                (-1, self.batch_size, 2))


class ABC_dataset_patch_npy(ABC_dataset_patch_hdf5):
    def __init__(self, v_data_root, v_training_mode, v_batch_size, v_output_features=4, ):
        super(ABC_dataset_patch_npy, self).__init__(v_data_root, v_training_mode, v_batch_size, v_output_features)

    def get_patch(self, v_id_item, v_id_patch):
        if self.mode == "training":
            index = np.unique(np.random.choice(np.arange(self.num_patches), self.batch_size, replace=False))
            index = np.sort(index)
            id_object = v_id_item[0]
            id_patch = index
        elif self.mode == "validation" or self.mode == "testing":
            id_object = v_id_item[0]
            id_patch = v_id_patch
        else:
            raise ""
        features = np.load(self.objects[id_object] + "_feat.npy", mmap_mode="r")
        features = features[id_patch]
        flags = np.load(self.objects[id_object] + "_flag.npy", mmap_mode="r")
        flags = flags[id_patch]
        return np.asarray([id_object] * self.batch_size), id_patch, features, flags


class ABC_dataset_patch_test(ABC_dataset_patch_npy):
    def __init__(self, v_data_root, v_training_mode, v_batch_size, v_output_features=4, v_test_list=[]):
        super(ABC_dataset_patch_test, self).__init__(v_data_root, v_training_mode, v_batch_size, v_output_features)
        if len(v_test_list) > 0:
            self.idxs = np.zeros(len(v_test_list), dtype=np.int32)
            for id, name in enumerate(v_test_list):
                if name in self.names:
                    self.idxs[id] = np.where(self.names == name)[0][0]
                else:
                    print("Cannot find ", name, " in the dataset")
                    raise ""
            self.names = np.asarray(v_test_list)
            self.total_num_items = len(v_test_list)
            self.num_items = self.total_num_items
            self.index = np.stack(np.meshgrid(
                np.arange(self.num_items),
                np.arange(self.num_patches), indexing="ij"), axis=2)
            self.index = self.index.reshape((self.index.shape[0], -1, self.batch_size, 2)).reshape(
                (-1, self.batch_size, 2))


class ABC_dataset_test_raw(torch.utils.data.Dataset):
    def __init__(self, v_data_root=None, v_training_mode=None):
        super(ABC_dataset_test_raw, self).__init__()
        self.data_root = v_data_root
        if v_data_root is None:
            return
        self.names = list(set([item[:8] for item in os.listdir(v_data_root)]))
        self.names = sorted(self.names, key=lambda x: int(x))
        self.objects = [os.path.join(v_data_root, item) for item in self.names]

        self.num_items = len(self.objects)

        self.mode = v_training_mode

        pass

    def __len__(self):
        if self.mode == "training":
            return self.num_items // 4 * 3
        elif self.mode == "validation":
            return self.num_items // 4
        elif self.mode == "testing":
            return self.num_items
        raise

    def get_total(self, v_idx):
        features = np.load(self.objects[v_idx] + "_feat.npy")
        features = features.reshape((8, 8, 8, 32, 32, 32, 3)).transpose((0, 3, 1, 4, 2, 5, 6)).reshape(256, 256, 256, 3)
        flags = np.load(self.objects[v_idx] + "_flag.npy")
        flags = flags.reshape((8, 8, 8, 32, 32, 32)).transpose((0, 3, 1, 4, 2, 5)).reshape(256, 256, 256)
        return features, flags

    def __getitem__(self, idx):
        if self.mode == "training" or self.mode == "testing":
            id_dummy = 0
        else:
            id_dummy = self.num_items // 4 * 3
        times = [0] * 10
        cur_time = time.time()
        feat_data, flag_data = self.get_total(idx + id_dummy)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (3, 0, 1, 2))
        flag_data = flag_data.astype(np.float32)[None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[idx + id_dummy]

    @staticmethod
    def collate_fn(v_batches):
        input_features = []
        consistent_flags = []
        for item in v_batches:
            input_features.append(item[0])
            consistent_flags.append(item[1])

        input_features = np.stack(input_features, axis=0)
        # input_features = torch.from_numpy(np.stack(input_features,axis=0).astype(np.float32)).permute(0, 4, 1, 2, 3)
        consistent_flags = torch.from_numpy(np.stack(consistent_flags, axis=0))

        return input_features, consistent_flags


def generate_test_coords(v_resolution):
    coords = np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution), indexing="ij")
    coords = np.stack(coords, axis=3)
    coords = coords.reshape((-1, 3))
    return coords


class ABC_dataset_test_mesh(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_batch_size, v_output_features=4, v_resolution=256):
        super(ABC_dataset_test_mesh, self).__init__()
        self.batch_size = v_batch_size
        self.output_features = v_output_features
        self.data_root = v_data_root
        self.resolution = v_resolution

        assert v_resolution % 32 == 0
        num_patches_per_dim = v_resolution // 32
        coords = generate_test_coords(v_resolution)
        self.coords = coords / (v_resolution - 1) * 2 - 1
        self.num_patches = (v_resolution // 32) ** 3
        assert self.num_patches % v_resolution == 0

        print("Prepare mesh data")
        if not os.path.exists(v_data_root):
            print("Cannot find ", v_data_root)
        mesh = o3d.io.read_triangle_mesh(v_data_root)
        points = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        min_xyz = points.min(axis=0)
        center_xyz = (min_xyz + points.max(axis=0)) / 2
        bbox = points.max(axis=0) - min_xyz
        diagonal = np.linalg.norm(bbox)
        points = (points - center_xyz) / diagonal * 2
        triangles = points[faces]
        num_triangles = triangles.shape[0]
        num_queries = self.coords.shape[0]
        query_result = cuda_distance.query(triangles.reshape(-1), self.coords.reshape(-1), 512, 512**3)

        udf = np.asarray(query_result[0]).astype(np.float32)
        closest_points = np.asarray(query_result[1]).reshape((num_queries, 3)).astype(np.float32)
        dir = closest_points - self.coords
        dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)

        if self.output_features == 4:
            feat_data = np.concatenate([dir, udf[:, None] * np.pi], axis=1)
        elif self.output_features == 1:
            feat_data = udf[:, None] * np.pi
        else:
            raise
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

