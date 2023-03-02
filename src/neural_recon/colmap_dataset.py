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

from generate_gt import extract_roi_region, calculate_ncc_batch, compute_direction_similarity, extract_roi_rectangle, compute_loss

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 600)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]
)


class Colmap_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data, v_mode):
        super(Colmap_dataset, self).__init__()
        self.trainer_mode = v_mode
        self.imgs = v_data["img_database"]
        self.sdf = v_data["sample_distances"]
        self.sample_points = v_data["sample_points"]
        self.final_visibility = v_data["final_visibility"]
        pass

    def __getitem__(self, index):
        sample_point = self.sample_points[index]
        sdf = self.sdf[index]
        visibility_mask = self.final_visibility[:, index]
        visible_imgs = [item for idx, item in enumerate(self.imgs) if visibility_mask[idx]]

        # Read original img
        if False:
            original_imgs = [
                cv2.resize(cv2.cvtColor(cv2.imread(item.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3], cv2.COLOR_BGR2RGB),
                           (800, 600)) for item in visible_imgs]
            tensor_imgs = [img_transform(item) for item in original_imgs]
            tensor_imgs = torch.stack(tensor_imgs, dim=0)
            original_imgs = torch.from_numpy(np.stack(original_imgs, axis=0))

        # Read image model
        img_names = [item.img_name for item in visible_imgs]

        projection_matrix_ = [item.projection for item in visible_imgs]
        projection_matrix = torch.from_numpy(np.stack(projection_matrix_, axis=0).astype(np.float32)) if len(
            projection_matrix_) > 1 else 0

        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["sample_point"] = torch.from_numpy(sample_point.astype(np.float32))
        data["sdf"] = torch.from_numpy(sdf.astype(np.float32))
        data["img_names"] = img_names
        # data["original_imgs"] = original_imgs
        # data["tensor_imgs"] = tensor_imgs
        data["projection_matrix"] = projection_matrix
        return data

    def __len__(self):
        return self.sample_points.shape[0]

    @staticmethod
    def collate_fn(batch):
        id_points = [item["id"] for item in batch]
        id_imgs_ = [item["id_imgs"] for item in batch]
        keypoints_ = [item["keypoints"] for item in batch]
        projection_matrix_ = [item["projection_matrix"] for item in batch]

        # keypoints = pad_sequence(keypoints_,batch_first=True,padding_value=-1)
        id_imgs = pad_sequence(id_imgs_, batch_first=True, padding_value=-1)
        projection_matrix = pad_sequence(projection_matrix_, batch_first=True, padding_value=-1)
        valid_views = torch.logical_not(torch.all(torch.flatten(projection_matrix, start_dim=2) == -1, dim=2))

        return {
            'id_points': torch.stack(id_points, dim=0),
            'id_imgs': id_imgs,
            'keypoints': keypoints_,
            'projection_matrix': projection_matrix,
            'valid_views': valid_views,
        }


class Blender_Segment_dataset(torch.utils.data.Dataset):
    def __init__(self, v_raw_data, v_training_data, v_mode):
        super(Blender_Segment_dataset, self).__init__()
        self.trainer_mode = v_mode
        self.raw_data = v_raw_data
        self.training_data = v_training_data
        self.segments = self.raw_data["segments"]
        self.segments_visibility = self.raw_data["segments_visibility"]

        self.cache = {}
        pass

    def __getitem__(self, index):
        segment = self.segments[index].astype(np.float32)
        projected_segment = self.training_data["projected_segments"][index]

        final_ncc,final_edge_similarity,final_edge_magnitude,lbd_similarity = self.training_data["gt_loss"][index]
        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["sample_segment"] = torch.from_numpy(segment.astype(np.float32))
        # data["img_names"] = img_names
        data["projected_coordinates"] = projected_segment
        data["projected_coordinates_tensor"] = torch.from_numpy(projected_segment)
        data["ncc"] = final_ncc
        data["edge_similarity"] = final_edge_similarity
        data["edge_magnitude"] = final_edge_magnitude
        data["lbd_similarity"] = lbd_similarity
        # self.cache[index] = data
        return data


    def __len__(self):
        return self.segments.shape[0]

    @staticmethod
    def collate_fn(batch):
        id = torch.stack([item["id"] for item in batch], dim=0)
        sample_segment = torch.stack([item["sample_segment"] for item in batch], dim=0)
        # img_names = np.asarray([item["img_names"] for item in batch], dtype=object)
        projected_coordinates = np.asarray([item["projected_coordinates"] for item in batch], dtype=object)
        # projected_coordinates_tensor = [item["projected_coordinates_tensor"] for item in batch]
        ncc = torch.tensor([item["ncc"] for item in batch], dtype=torch.float32).unsqueeze(1)
        edge_similarity = torch.tensor([item["edge_similarity"] for item in batch], dtype=torch.float32).unsqueeze(1)
        edge_magnitude = torch.tensor([item["edge_magnitude"] for item in batch], dtype=torch.float32).unsqueeze(1)
        lbd_similarity = torch.tensor([item["lbd_similarity"] for item in batch], dtype=torch.float32).unsqueeze(1)

        return {
            'id': id,
            'sample_segment': sample_segment,
            # 'img_names': img_names,
            'projected_coordinates': projected_coordinates,
            # 'projected_coordinates_tensor': projected_coordinates_tensor,
            'ncc': ncc,
            'edge_similarity': edge_similarity,
            'edge_magnitude': edge_magnitude,
            'lbd_similarity': lbd_similarity,
        }
