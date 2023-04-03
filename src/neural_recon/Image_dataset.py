import math
from typing import List

import cv2
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from functools import partial
from multiprocessing import Pool
from typing import List

import open3d as o3d
import numpy as np
import os

from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm


class Image_dataset(torch.utils.data.Dataset):
    def __init__(self, v_img, v_num_samples, v_batch_size, v_mode, v_sampling_strategy, v_query_strategy):
        super(Image_dataset, self).__init__()
        self.img: torch.Tensor = torch.from_numpy(v_img) / 255.
        self.batch_size: int = v_batch_size
        self.mode = v_mode
        self.num_sample = v_num_samples

        # Used for validation
        self.original_sample_points = torch.from_numpy(np.stack(
                np.meshgrid(
                    (np.arange(self.img.shape[1], dtype=np.float32) + 0.5) / self.img.shape[1],
                    (np.arange(self.img.shape[0], dtype=np.float32) + 0.5) / self.img.shape[0],
                    indexing="xy"), axis=2).reshape([-1, 2]))
        index = (self.original_sample_points *
                     torch.flip(torch.tensor(self.img.shape[:2]), dims=[0])).long()
        index[:, 0] = torch.clamp_max(index[:, 0], self.img.shape[1] - 1)
        index[:, 1] = torch.clamp_max(index[:, 1], self.img.shape[0] - 1)
        self.original_pixel_values = self.img[index[:, 1], index[:, 0]]

        self.sampling_strategy = v_sampling_strategy
        self.query_strategy = v_query_strategy
        if self.mode == "training":
            self.generate_data()
        pass

    def generate_data(self,):
        # self.sample_points: [0,1]
        if self.sampling_strategy == "pixel":
            sample_points = np.stack(
                np.meshgrid(
                    (np.arange(self.img.shape[1], dtype=np.float32) + 0.5) / self.img.shape[1],
                    (np.arange(self.img.shape[0], dtype=np.float32) + 0.5) / self.img.shape[0],
                    indexing="xy"), axis=2).reshape([-1, 2])
            np.random.shuffle(sample_points)
            self.sample_points = torch.from_numpy(sample_points)
        elif self.sampling_strategy == "random":
            sample_points = np.random.rand(
                int(self.num_sample), 2).astype(np.float32)
            self.sample_points = torch.from_numpy(sample_points)
        else:
            raise

        if self.query_strategy == "fix":
            index = (self.sample_points * torch.flip(torch.tensor(self.img.shape[:2]), dims=[0])).long()
            index[:, 0] = torch.clamp_max(index[:, 0], self.img.shape[1] - 1)
            index[:, 1] = torch.clamp_max(index[:, 1], self.img.shape[0] - 1)
            self.pixel_values = self.img[index[:, 1], index[:, 0]]
        elif self.query_strategy == "interpolate":
            self.pixel_values = F.grid_sample(
                self.img.permute(2, 0, 1).unsqueeze(0),
                (self.sample_points*2-1).unsqueeze(0).unsqueeze(0), mode="bilinear",
                padding_mode="border", align_corners=True)[0,:,0,:].transpose(0,1)
        else:
            raise

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        if self.mode == "training":
            end_index = min(idx * self.batch_size + self.batch_size, self.sample_points.shape[0])
            return self.sample_points[start_index:end_index], self.pixel_values[start_index:end_index]
        else:
            end_index = min(idx * self.batch_size + self.batch_size, self.original_pixel_values.shape[0])
            return self.original_sample_points[start_index:end_index], self.original_pixel_values[start_index:end_index]

    def __len__(self):
        return math.ceil((self.sample_points.shape[0] if self.mode == "training" else self.original_sample_points.shape[0]) / self.batch_size)

class Image_dataset_with_transform(torch.utils.data.Dataset):
    def __init__(self, v_img, v_num_samples, v_batch_size, v_mode, v_sampling_strategy, v_query_strategy):
        super(Image_dataset_with_transform, self).__init__()
        self.img: torch.Tensor = torch.from_numpy(v_img)
        self.batch_size: int = v_batch_size
        self.mode = v_mode
        self.num_sample = v_num_samples

        # Used for validation
        self.original_sample_points = torch.from_numpy(np.stack(
                np.meshgrid(
                    (np.arange(self.img.shape[1], dtype=np.float32) + 0.5) / self.img.shape[1],
                    (np.arange(self.img.shape[0], dtype=np.float32) + 0.5) / self.img.shape[0],
                    indexing="xy"), axis=2).reshape([-1, 2]))
        index = (self.original_sample_points *
                     torch.flip(torch.tensor(self.img.shape[:2]), dims=[0])).long()
        index[:, 0] = torch.clamp_max(index[:, 0], self.img.shape[1] - 1)
        index[:, 1] = torch.clamp_max(index[:, 1], self.img.shape[0] - 1)
        self.original_pixel_values = self.img[index[:, 1], index[:, 0]]

        self.transform = Compose([
            transforms.ToPILImage(),
            transforms.ToPILImage(),
        ])

        self.sampling_strategy = v_sampling_strategy
        self.query_strategy = v_query_strategy
        if self.mode == "training":
            self.generate_data()
        pass

    def generate_data(self,):
        # self.sample_points: [0,1]
        if self.sampling_strategy == "pixel":
            sample_points = np.stack(
                np.meshgrid(
                    (np.arange(
                        self.img.shape[1], dtype=np.float32) + 0.5) / self.img.shape[1],
                    (np.arange(
                        self.img.shape[0], dtype=np.float32) + 0.5) / self.img.shape[0],
                    indexing="xy"), axis=2).reshape([-1, 2])
            np.random.shuffle(sample_points)
            self.sample_points = torch.from_numpy(sample_points)
        elif self.sampling_strategy == "random":
            sample_points = np.random.rand(
                int(self.num_sample), 2).astype(np.float32)
            self.sample_points = torch.from_numpy(sample_points)
        else:
            raise

        self.img = (self.img / 255.)
        if self.query_strategy == "fix":
            index = (self.sample_points *
                     torch.flip(torch.tensor(self.img.shape[:2]), dims=[0])).long()
            index[:, 0] = torch.clamp_max(index[:, 0], self.img.shape[1] - 1)
            index[:, 1] = torch.clamp_max(index[:, 1], self.img.shape[0] - 1)
            self.pixel_values = self.img[index[:, 1], index[:, 0]]
        elif self.query_strategy == "interpolate":
            self.pixel_values = F.grid_sample(self.img.permute(2, 0, 1).unsqueeze(0), (self.sample_points*2-1).unsqueeze(0).unsqueeze(0), mode="bilinear",
                                              padding_mode="border", align_corners=True)[0,:,0,:].transpose(0,1)
        else:
            raise

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        if self.mode == "training":
            end_index = min(idx * self.batch_size + self.batch_size, self.sample_points.shape[0])
            return self.sample_points[start_index:end_index], self.pixel_values[start_index:end_index]
        else:
            end_index = min(idx * self.batch_size + self.batch_size, self.original_pixel_values.shape[0])
            return self.original_sample_points[start_index:end_index], self.original_pixel_values[start_index:end_index]

    def __len__(self):
        return math.ceil((self.sample_points.shape[0] if self.mode == "training" else self.original_sample_points.shape[0]) / self.batch_size)