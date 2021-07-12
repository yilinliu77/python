import os
import pickle
from copy import deepcopy

import cv2
import hydra
import torch
import torchvision
import numpy as np
from PIL import Image
from visualDet3D.data.kitti.kittidata import KittiData

from src.utils import project_box3d_to_img


class KittiMonoDataset(torch.utils.data.Dataset):

    def __init__(self, v_params, v_mode, v_transform):
        super(KittiMonoDataset, self).__init__()

        imdb_file_path = os.path.join(
            hydra.utils.get_original_cwd(),
            v_params["det_3d"]["preprocessed_path"],
            '{}/imdb.pkl'.format(v_mode))
        self.imdb = pickle.load(open(imdb_file_path, 'rb'))  # list of kittiData
        self.mode = v_mode
        self.transform = v_transform
        self.params = v_params

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        data_frame = KittiData(self.params["trainer"]["train_dataset"], kitti_data.index_name, {
            "calib": True,
            "image": True,
            "label": True,
            "velodyne": False,
        })
        calib, image, label, velo = data_frame.read_data()
        image, P2, label_tr = self.transform(image, labels=deepcopy(label.data), p2=deepcopy(calib.P2))
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Do the filtering
        label_tr = [
            item for item in label_tr if item.type == "Car" and item.z < 200]

        if len(label_tr) == 0:
            bbox3d = torch.zeros((0, 4)).float()
            bbox2d = torch.zeros((0, 4)).float()
            bbox3d_img_center = torch.zeros((0, 2)).float()

        else:
            corners_in_camera_coordinates = project_box3d_to_img(label_tr,P2,
                                                                 self.params["det_3d"][
                                                                     "rotate_pitch_when_project"])
            bbox3d_img_center = torch.stack(
                [item[8, :] for item in corners_in_camera_coordinates],
                dim=0
            )
            bbox2d = torch.stack([torch.tensor((
                item.bbox_l,
                item.bbox_t,
                item.bbox_r,
                item.bbox_b,)
            ) for item in label_tr
            ], dim=0
            )
            # for item in label_tr:
            #     item.y = item.y - item.h * 0.5
            bbox3d = torch.stack([torch.tensor((
                item.z, item.w, item.h, item.l, item.alpha
            )) for item in label_tr
            ], dim=0)

        output_dict = {
            'original_calib': calib.P2,
            'calib': P2,
            'label': [0 for _ in label_tr],
            'original_label': [0 for _ in label.data],
            'image': image,
            # 'bbox2d': kitti_data.bbox2d,  # [N, 4] [x1, y1, x2, y2]
            # 'bbox3d': kitti_data.bbox3d,  # [N, 7] [z, sin2a, cos2a, w, h, l]
            # 'bbox3d_img_center': kitti_data.bbox3d_img_center,
            'training_data': torch.cat([
                bbox2d, bbox3d_img_center, bbox3d
            ], dim=1),
        }
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        original_calib = [item["original_calib"] for item in batch]
        calib = [item["calib"] for item in batch]
        image = [item["image"] for item in batch]
        label = [item['label'] for item in batch]
        original_label = [item['original_label'] for item in batch]
        # bbox2ds = [item['bbox2d'] for item in batch]
        # bbox3ds = [item['bbox3d'] for item in batch]
        # bbox3d_img_center = [item['bbox3d_img_center'] for item in batch]
        training_data = [item['training_data'] for item in batch]

        return {
            'original_calib': torch.tensor(original_calib).float(),
            'calib': torch.tensor(calib).float(),
            'image': torch.stack(image, dim=0),
            'label': label,
            'original_label': original_label,
            # 'bbox2d': bbox2ds,  # [N, 4] [x1, y1, x2, y2]
            # 'bbox3d': bbox3ds,  # [N, 7] [z, sin2a, cos2a, w, h, l, alpha]
            # 'bbox3d_img_center': bbox3d_img_center,
            'training_data': training_data,
        }
