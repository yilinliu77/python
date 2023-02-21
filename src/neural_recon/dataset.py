import os, sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset

import open3d as o3d
import mesh2sdf

class Single_img_dataset(torch.utils.data.Dataset):
    def __init__(self, v_imgs, v_world_points, v_mode):
        super(Single_img_dataset, self).__init__()
        self.trainer_mode = v_mode

        self.imgs: List[Image] = v_imgs
        self.world_points: List[Point_3d] = v_world_points
        pass

    def __getitem__(self, index):
        point_3d = self.world_points[index]
        projection_matrix_ = np.zeros((len(point_3d.tracks), 4, 4))
        keypoints_: List[torch.Tensor] = []
        for id_track, track in enumerate(point_3d.tracks):
            id_img = track[0]
            projection_matrix_[id_track] = self.imgs[id_img].projection
            keypoints_.append(torch.from_numpy(self.imgs[id_img].detected_points))

        projection_matrix = torch.from_numpy(projection_matrix_)
        keypoints = pad_sequence(keypoints_, batch_first=True, padding_value=-1)
        keypoints = torch.cat([keypoints, torch.logical_not(torch.all(keypoints == -1, dim=2, keepdim=True))], dim=2)

        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["keypoints"] = keypoints
        data["projection_matrix"] = projection_matrix
        return data

    def __len__(self):
        return len(self.world_points)

    @staticmethod
    def collate_fn(batch):
        id_points = [item["id"] for item in batch]
        keypoints_ = [item["keypoints"] for item in batch]
        projection_matrix_ = [item["projection_matrix"] for item in batch]

        keypoints = pad_sequence(keypoints_, batch_first=True, padding_value=-1)
        projection_matrix = pad_sequence(projection_matrix_, batch_first=True, padding_value=-1)
        valid_views = torch.logical_not(torch.all(torch.flatten(projection_matrix, start_dim=2) == -1, dim=2))

        return {
            'id_points': torch.stack(id_points, dim=0),
            'keypoints': keypoints,
            'projection_matrix': projection_matrix,
            'valid_views': valid_views,
        }


class Single_img_dataset_with_kdtree_index(torch.utils.data.Dataset):
    def __init__(self, v_imgs, v_world_points, v_mode):
        super(Single_img_dataset_with_kdtree_index, self).__init__()
        self.trainer_mode = v_mode

        self.imgs: List[Image] = v_imgs
        self.world_points: List[Point_3d] = v_world_points
        pass

    def __getitem__(self, index):
        index = 1
        point_3d = self.world_points[index]
        projection_matrix_ = np.zeros((len(point_3d.tracks), 4, 4), dtype=np.float32)
        id_imgs: torch.Tensor = torch.zeros(len(point_3d.tracks), dtype=torch.long)
        keypoints_: List[torch.Tensor] = []
        for id_track, track in enumerate(point_3d.tracks):
            id_img = track[0]
            id_imgs[id_track] = id_img
            projection_matrix_[id_track] = self.imgs[id_img].projection
            keypoints_.append(torch.from_numpy(self.imgs[id_img].detected_points))

        projection_matrix = torch.from_numpy(projection_matrix_)
        keypoints = pad_sequence(keypoints_, batch_first=True, padding_value=-1)
        keypoints = torch.cat([keypoints, torch.logical_not(torch.all(keypoints == -1, dim=2, keepdim=True))], dim=2)

        data = {}
        data["id"] = torch.tensor(index, dtype=torch.long)
        data["id_imgs"] = id_imgs
        data["keypoints"] = keypoints
        data["projection_matrix"] = projection_matrix
        return data

    def __len__(self):
        return len(self.world_points)

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


class Geometric_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 v_sdf_computer,
                 v_num_sample,
                 v_batch_size,
                 v_mode
                 ):
        super().__init__()
        self.trainer_mode = v_mode
        self.batch_size = v_batch_size

        if True:
            # print("Start to query")
            query_points = v_sdf_computer.compute_sdf(int(v_num_sample[0]), int(v_num_sample[1]),
                                                              int(v_num_sample[2]), False)
            # print("Done query")
            indexes = np.random.permutation(query_points.shape[0])
            query_points=query_points[indexes]
            # print("Done Shuffle")
            if False:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(self.query_points[:,:3])
                o3d.io.write_point_cloud("/mnt/d/Projects/NeuralRecon/tmp/t.ply" ,pc)
            self.samples_points = torch.tensor(query_points[:, :3], dtype=torch.half)
            self.sdf = torch.tensor(query_points[:, 3:4], dtype=torch.half)
        else:
            self.bounds_center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
            self.bounds_size = np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
            vertices = (vertices - self.bounds_center) / self.bounds_size * 2

            with torch.no_grad():
                vertices = torch.tensor(vertices, dtype=torch.float32, device="cuda")
                faces = torch.tensor(faces, dtype=torch.long, device="cuda")

                samples = []
                num_samples = v_num_sample
                distrib = area_weighted_distribution(vertices, faces)
                samples.append(sample_surface(vertices, faces, num_samples, distrib=distrib)[0])
                samples.append(sample_surface(vertices, faces, num_samples, distrib=distrib)[0])
                samples.append(sample_near_surface(vertices, faces, num_samples, distrib=distrib))
                samples.append(sample_near_surface(vertices, faces, num_samples, distrib=distrib))
                samples.append(sample_uniform(num_samples).to(vertices.device))
                self.samples_points = torch.cat(samples, dim=0).contiguous()

                self.sdf = mesh2sdf.mesh2sdf_gpu(self.samples_points, vertices[faces])[0].cpu()[..., None]
                self.samples_points = self.samples_points.cpu()
        self.num_iter = self.sdf.shape[0] // self.batch_size + 1
        # print("Done dataset construction")
        return

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        return self.samples_points[idx * self.batch_size:min(self.sdf.shape[0], (idx + 1) * self.batch_size)], \
            self.sdf[idx * self.batch_size:min(self.sdf.shape[0], (idx + 1) * self.batch_size)]

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


class Geometric_dataset_inference(torch.utils.data.Dataset):
    def __init__(self,
                 n_resolution=128,
                 v_batch_size=1,
                 ):
        super().__init__()
        X, Y, Z = np.meshgrid(np.arange(n_resolution), np.arange(n_resolution), np.arange(n_resolution), indexing="xy")
        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        pts = pts.reshape([-1, 3])
        # self.pts = torch.tensor((pts / (n_resolution - 1) - 0.5) * 2)
        self.pts = torch.tensor(pts / (n_resolution - 1))
        self.batch_size = v_batch_size
        self.num_iter = self.pts.shape[0] // self.batch_size + 1

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = min(self.pts.shape[0], (idx + 1) * self.batch_size)
        query_coordinates = self.pts[start_index:end_index].half() # Half is important!!!!!!!!!!!!!!!!
        return [query_coordinates, torch.zeros_like(query_coordinates[:,:1])]
