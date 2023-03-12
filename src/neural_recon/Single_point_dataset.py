from typing import List
from scipy.spatial.distance import cdist
import cv2
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from functools import partial
from multiprocessing import Pool
from typing import List

import open3d as o3d
import os

from tqdm import tqdm

import pysdf

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
from src.neural_recon.sample import sample_points_cpu
from torchvision import transforms

from src.neural_recon.generate_gt import extract_roi_region, calculate_ncc_batch, compute_direction_similarity, extract_roi_rectangle, compute_loss

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 600)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]
)


class Single_point_dataset(torch.utils.data.Dataset):
    def __init__(self, v_raw_data, v_training_data, v_mode):
        super(Single_point_dataset, self).__init__()
        self.trainer_mode = v_mode
        self.raw_data = v_raw_data
        self.training_data = v_training_data
        self.points = self.raw_data["sample_points"]
        self.points_visibility = self.raw_data["final_visibility"]
        pass

    def __getitem__(self, index):
        point = self.points[index].astype(np.float32)

        gt_loss = self.training_data["gt_loss"][index]
        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["sample_point"] = torch.from_numpy(point.astype(np.float16))
        data["gt_loss"] = torch.from_numpy(gt_loss.astype(np.float16))
        return data


    def __len__(self):
        return self.points.shape[0]

    @staticmethod
    def collate_fn(batch):
        id = torch.stack([item["id"] for item in batch], dim=0)
        sample_point = torch.stack([item["sample_point"] for item in batch], dim=0)
        gt_loss = torch.stack([item["gt_loss"] for item in batch], dim=0)

        return {
            'id': id,
            'sample_point': sample_point,
            'gt_loss': gt_loss,
        }
