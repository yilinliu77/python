import math
import os.path
import random
import time

import h5py
import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ray
import scipy
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pytorch_lightning as pl
import faiss
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from shared.fast_dataloader import FastDataLoader
from src.neural_bsp.model import AttU_Net_3D, U_Net_3D
from shared.common_utils import export_point_cloud, sigmoid
from src.neural_bsp.train_model import ABC_dataset, Base_model
import torch.distributed as dist


class ABC_dataset_patch_hdf5(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode, v_batch_size):
        super(ABC_dataset_patch_hdf5, self).__init__(None, None)
        self.data_root = v_data_root
        self.batch_size = v_batch_size
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.num_patches = f["features"].shape[1]
            self.names = np.asarray(["{:08d}".format(item) for item in np.asarray(f["names"])])
        self.mode = v_training_mode
        self.validation_start = self.num_items // 4 * 3

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
        self.index = self.index.reshape(-1, 2)

    def __len__(self):
        return self.index.shape[0]

    def get_patch(self, v_id_item, v_id_patch):
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][v_id_item, v_id_patch]
            flags = f["flags"][v_id_item, v_id_patch]
        return features, flags

    def __getitem__(self, idx):
        id_object = self.index[idx, 0]
        id_patch = self.index[idx, 1]

        times = [0] * 10
        cur_time = time.time()
        feat_data, flag_data = self.get_patch(id_object, id_patch)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (3, 0, 1, 2))
        flag_data = flag_data.astype(np.float32)[None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_object], id_patch


class ABC_dataset_patch_hdf5_sample(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode, v_batch_size):
        super(ABC_dataset_patch_hdf5_sample, self).__init__(None, None)
        self.data_root = v_data_root
        self.batch_size = v_batch_size
        self.mode = v_training_mode

        with h5py.File(self.data_root, "r") as f:
            self.num_patches_per_item = f["features"].shape[1]
            self.total_num_items = f["features"].shape[0]
            self.names = np.asarray(["{:08d}".format(item) for item in np.asarray(f["names"])])

        self.validation_start = self.total_num_items // 4 * 3

        if self.mode == "training":
            self.num_items = self.validation_start
        elif self.mode == "validation":
            assert self.num_patches_per_item % self.batch_size == 0
            self.id_object = np.tile(np.arange(self.validation_start, self.total_num_items)[:, None],
                                     (1, self.num_patches_per_item))
            self.id_patch = np.tile(np.arange(self.num_patches_per_item)[None, :],
                                    (self.id_object.shape[0], 1)).reshape(-1)
            self.id_object = self.id_object.reshape(-1)
            self.num_items = math.ceil(self.id_object.shape[0] / self.batch_size)
        else:
            raise ""

    def __len__(self):
        return self.num_items

    def get_patch(self, idx):
        if self.mode == "training":
            index = np.unique(np.random.choice(np.arange(self.num_patches_per_item), self.batch_size, replace=False))
            index = np.sort(index)
            id_object = idx
            id_patch = index
        elif self.mode == "validation":
            id_start = idx * self.batch_size
            id_end = min((idx + 1) * self.batch_size, self.id_patch.shape[0])
            id_object = self.id_object[id_start]
            id_patch = self.id_patch[id_start:id_end]
        else:
            raise ""
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][id_object, id_patch]
            flags = f["flags"][id_object, id_patch]
        return np.asarray([id_object] * self.batch_size), id_patch, features, flags

    def __getitem__(self, idx):
        times = [0] * 10
        cur_time = time.time()
        id_object, id_patch, feat_data, flag_data = self.get_patch(idx)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3))
        flag_data = flag_data.astype(np.float32)[:, None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_object], id_patch

    def collate_fn(v_batches):
        return [torch.from_numpy(v_batches[0][0]),
                torch.from_numpy(v_batches[0][1]),
                v_batches[0][2],
                torch.from_numpy(v_batches[0][3])]


class ABC_dataset_patch_hdf5_test(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode, v_batch_size):
        super(ABC_dataset_patch_hdf5_test, self).__init__(None, None)
        self.data_root = v_data_root
        self.batch_size = v_batch_size
        self.mode = v_training_mode

        with h5py.File(self.data_root, "r") as f:
            self.num_patches_per_item = f["features"].shape[1]
            self.total_num_items = f["features"].shape[0]
            self.names = np.asarray(["{:08d}".format(item) for item in np.asarray(f["names"])])

        assert v_batch_size == self.num_patches_per_item

        self.num_items = self.total_num_items

    def __len__(self):
        return self.num_items

    def get_patch(self, idx):
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][idx, :]
            flags = f["flags"][idx, :]
        return np.asarray([idx] * self.batch_size), np.arange(self.num_patches_per_item), features, flags

    def __getitem__(self, idx):
        times = [0] * 10
        cur_time = time.time()
        id_object, id_patch, feat_data, flag_data = self.get_patch(idx)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3))
        flag_data = flag_data.astype(np.float32)[:, None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_object], id_patch

    def collate_fn(v_batches):
        return [torch.from_numpy(v_batches[0][0]),
                torch.from_numpy(v_batches[0][1]),
                v_batches[0][2],
                torch.from_numpy(v_batches[0][3])]


class ABC_dataset_patch_train(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode, v_batch_size):
        super(ABC_dataset_patch_train, self).__init__(None, None)
        self.data_root = v_data_root
        self.batch_size = v_batch_size
        self.mode = v_training_mode

        self.data_root = v_data_root

        self.names = list(set([item[:8] for item in os.listdir(v_data_root)]))
        self.names = np.asarray(sorted(self.names, key=lambda x: int(x)))
        self.total_num_items = len(self.names)
        self.objects = [os.path.join(v_data_root, item) for item in self.names]
        self.num_patches_per_item = np.load(self.objects[0] + "_feat.npy").shape[0]

        self.validation_start = self.total_num_items // 4 * 3

        if self.mode == "training":
            self.num_items = self.validation_start
        elif self.mode == "validation":
            assert self.num_patches_per_item % self.batch_size == 0
            self.id_object = np.tile(np.arange(self.validation_start, self.total_num_items)[:, None],
                                     (1, self.num_patches_per_item))
            self.id_patch = np.tile(np.arange(self.num_patches_per_item)[None, :],
                                    (self.id_object.shape[0], 1)).reshape(-1)
            self.id_object = self.id_object.reshape(-1)
            self.num_items = math.ceil(self.id_object.shape[0] / self.batch_size)
        else:
            raise ""

    def __len__(self):
        return self.num_items

    def get_patch(self, idx):
        if self.mode == "training":
            index = np.unique(np.random.choice(np.arange(self.num_patches_per_item), self.batch_size, replace=False))
            index = np.sort(index)
            id_object = idx
            id_patch = index
        elif self.mode == "validation":
            id_start = idx * self.batch_size
            id_end = min((idx + 1) * self.batch_size, self.id_patch.shape[0])
            id_object = self.id_object[id_start]
            id_patch = self.id_patch[id_start:id_end]
        else:
            raise ""
        features = np.load(self.objects[id_object] + "_feat.npy", mmap_mode="r")
        choice = np.random.choice(features.shape[0], self.batch_size)
        features = features[choice]
        flags = np.load(self.objects[id_object] + "_flag.npy")
        flags = flags[choice]
        return np.asarray([id_object] * self.batch_size), id_patch, features, flags

    def __getitem__(self, idx):
        times = [0] * 10
        cur_time = time.time()
        id_object, id_patch, feat_data, flag_data = self.get_patch(idx)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (0, 4, 1, 2, 3))
        flag_data = flag_data.astype(np.float32)[:, None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_object], id_patch

    def collate_fn(v_batches):
        return [torch.from_numpy(v_batches[0][0]),
                torch.from_numpy(v_batches[0][1]),
                v_batches[0][2],
                torch.from_numpy(v_batches[0][3])]


class ABC_dataset_patch_test(ABC_dataset_patch_train):
    def __init__(self, v_data_root, v_training_mode, v_batch_size):
        super(ABC_dataset_patch_test, self).__init__(v_data_root, "training", v_batch_size)
        assert v_batch_size == self.num_patches_per_item
        self.num_objects = len(self.names)

    def get_patch(self, v_id_item):
        features = np.load(self.objects[v_id_item] + "_feat.npy")
        features = features
        flags = np.load(self.objects[v_id_item] + "_flag.npy")
        flags = flags
        return np.asarray([v_id_item] * self.batch_size), np.arange(self.num_patches_per_item), features, flags
