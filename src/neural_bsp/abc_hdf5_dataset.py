import math
import os.path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


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
            self.num_patches = np.load(self.objects[0]+"_feat.npy").shape[0]
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
