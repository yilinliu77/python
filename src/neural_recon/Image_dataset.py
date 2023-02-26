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

from tqdm import tqdm


class Image_dataset(torch.utils.data.Dataset):
    def __init__(self, v_img, v_num_samples, v_batch_size, v_mode):
        super(Image_dataset, self).__init__()
        self.img: torch.Tensor = torch.from_numpy(v_img)
        self.batch_size: int = v_batch_size

        # if v_mode == "training":
        #     self.sample_points = np.random.random((int(v_num_samples), 2)).astype(np.float32)
        # else:
        self.sample_points = np.stack(
            np.meshgrid(
                np.arange(self.img.shape[1],dtype=np.float32) / (self.img.shape[1] - 1),
                np.arange(self.img.shape[0],dtype=np.float32) / (self.img.shape[0] - 1),
                indexing="xy"), axis=2).reshape([-1,2])

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = min(idx * self.batch_size + self.batch_size, self.sample_points.shape[0])
        sample_points = torch.from_numpy(self.sample_points[start_index:end_index])
        index = (sample_points*torch.flip(torch.tensor(self.img.shape[:2]) - 1,dims=[0])).long()
        return sample_points.to(torch.float16), (self.img[index[:,1],index[:,0]] / 255.).to(torch.float16)

    def __len__(self):
        return math.ceil(self.sample_points.shape[0] / self.batch_size)
