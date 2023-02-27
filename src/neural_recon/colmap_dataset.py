from typing import List
import sys
sys.path.append(r"D:\repo\python\thirdparty\pylbd\build\Release")
import pytlbd
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

from generate_gt import extract_roi_region, calculate_ncc_batch, compute_direction_similarity, extract_roi_rectangle

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
    def __init__(self, v_data, v_imgs, v_mode):
        super(Blender_Segment_dataset, self).__init__()
        self.trainer_mode = v_mode
        self.img_database = v_data["img_database"]
        self.imgs = v_imgs
        self.segments = v_data["segments"]
        self.segments_visibility = v_data["segments_visibility"]

        self.cache = {}
        pass

    def __getitem__(self, index):
        if index not in self.cache:
            # Segment and visibility
            segment = self.segments[index]
            visibility_mask = self.segments_visibility[:, index]
            visible_imgs = [item for idx, item in enumerate(self.img_database) if visibility_mask[idx]]

            # Read original img
            if False:
                original_imgs = [
                    cv2.resize(cv2.cvtColor(cv2.imread(item.img_path, cv2.IMREAD_UNCHANGED)[:, :, :3], cv2.COLOR_BGR2RGB),
                               (800, 600)) for item in visible_imgs]
                tensor_imgs = [img_transform(item) for item in original_imgs]
                tensor_imgs = torch.stack(tensor_imgs, dim=0)
                original_imgs = torch.from_numpy(np.stack(original_imgs, axis=0))

            img_names = tuple(item.img_name for item in visible_imgs)

            # Project the segments on image
            projection_matrix_ = [item.projection for item in visible_imgs]
            homogeneous_sample_segment = np.insert(segment, 3, 1, axis=1)
            num_view = len(img_names)
            # We should at least have two views to reconstruct a target
            # or we will predict it as a negative sample
            if num_view <= 1:
                projected_segment = np.zeros(0)
                final_ncc = 0.
                final_edge_similarity = 0.
                final_edge_magnitude = 1.
                lbd_similarity = -1.
            else:
                # Calculate the projected segment
                projected_segment = np.matmul(np.asarray(projection_matrix_),
                                              np.transpose(homogeneous_sample_segment)).swapaxes(1, 2)
                projected_segment = projected_segment[:, :, :2] / projected_segment[:, :, 2:3]

                # Make sure the start point of the segment is on the left
                is_y_larger_than_x = projected_segment[:, 0, 0] > projected_segment[:, 1, 0]
                projected_segment[is_y_larger_than_x] = projected_segment[is_y_larger_than_x][:, ::-1]

                # Compute the related region (resize to (W*H)) on the image according to the projected segment
                # W is the length of the projected segment
                # H is a pre-defined parameter (10 by default)
                roi_regions = []
                edge_similarities = []
                edge_magnitudes = []
                lbd_descriptors = []
                # It might not equal to `num_view`. Some projections might be a single pixel on image. Just skip them
                for id_view in range(num_view):
                    # Take the data
                    img = self.imgs[img_names[id_view]][0]
                    projected_segment_per_view = projected_segment[id_view]
                    img_size = img.shape[:2][::-1]
                    ori_coords = (projected_segment_per_view * img_size).astype(np.int64)

                    if True:
                        viz_img = cv2.line(img.copy(), ori_coords[0],ori_coords[1],(255,0,0),2)
                        cv2.imwrite("output/loss_test/0_{}_original.png".format(id_view), viz_img)
                    # Compute the related region using a rotated rectangle
                    roi_size, roi_angle, roi_center = extract_roi_rectangle(projected_segment_per_view, img_size)
                    # Check the size
                    if roi_size[0] <= 1 or roi_size[1] <= 1:
                        continue

                    # Edge similarity
                    img_direction = self.imgs[img_names[id_view]][1]
                    edge_similarity, edge_magnitude = compute_direction_similarity(projected_segment_per_view, img_direction)
                    edge_similarities.append(edge_similarity)
                    edge_magnitudes.append(edge_magnitude)

                    # ROI region
                    roi_resized = extract_roi_region(img, roi_center, roi_angle, roi_size, id_view)
                    roi_regions.append(roi_resized)

                    descriptor = pytlbd.lbd_single_scale(img, ori_coords.reshape((1,4)), 9, 7)
                    lbd_descriptors.append(descriptor)
                if len(roi_regions) <= 1:
                    final_ncc = 0.
                    final_edge_similarity = 0.
                    final_edge_magnitude = 0.
                    lbd_similarity = -1.
                else:
                    nccs = calculate_ncc_batch(roi_regions) # [-1,1]
                    final_ncc = np.mean(nccs) / 2 + 0.5 # [0,1]
                    final_edge_similarity = np.mean(edge_similarities) # [0,1]
                    final_edge_magnitude = np.mean(edge_magnitudes) # [0,1]
                    lbd_similarities = []
                    for id_view1 in range(len(roi_regions)):
                        for id_view2 in range(id_view1 + 1, len(roi_regions)):
                            lbd_similarity = cdist(lbd_descriptors[id_view1], lbd_descriptors[id_view2])
                            lbd_similarities.append(lbd_similarity)
                    lbd_similarity = 1 - np.mean(lbd_similarities)
                    pass

            data = {}
            data["id"] = torch.tensor(index, dtype=torch.long)
            data["sample_segment"] = torch.from_numpy(segment.astype(np.float32))
            data["img_names"] = img_names
            data["projected_coordinates"] = projected_segment
            data["projected_coordinates_tensor"] = torch.from_numpy(projected_segment)
            data["ncc"] = final_ncc
            data["edge_similarity"] = final_edge_similarity
            data["edge_magnitude"] = final_edge_magnitude
            data["lbd_similarity"] = lbd_similarity
            self.cache[index] = data
            return data
        else:
            return self.cache[index]


    def __len__(self):
        return self.segments.shape[0]

    @staticmethod
    def collate_fn(batch):
        id = torch.stack([item["id"] for item in batch], dim=0)
        sample_segment = torch.stack([item["sample_segment"] for item in batch], dim=0)
        img_names = np.asarray([item["img_names"] for item in batch], dtype=object)
        projected_coordinates = np.asarray([item["projected_coordinates"] for item in batch], dtype=object)
        # projected_coordinates_tensor = [item["projected_coordinates_tensor"] for item in batch]
        ncc = torch.tensor([item["ncc"] for item in batch], dtype=torch.float32).unsqueeze(1)
        edge_similarity = torch.tensor([item["edge_similarity"] for item in batch], dtype=torch.float32).unsqueeze(1)
        edge_magnitude = torch.tensor([item["edge_magnitude"] for item in batch], dtype=torch.float32).unsqueeze(1)
        lbd_similarity = torch.tensor([item["lbd_similarity"] for item in batch], dtype=torch.float32).unsqueeze(1)

        return {
            'id': id,
            'sample_segment': sample_segment,
            'img_names': img_names,
            'projected_coordinates': projected_coordinates,
            # 'projected_coordinates_tensor': projected_coordinates_tensor,
            'ncc': ncc,
            'edge_similarity': edge_similarity,
            'edge_magnitude': edge_magnitude,
            'lbd_similarity': lbd_similarity,
        }
