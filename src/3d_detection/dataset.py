import os
import pickle

import cv2
import hydra
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import List, Tuple
from visualDet3D.utils.utils import alpha2theta_3d, theta2alpha_3d
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj, KittiCalib
from visualDet3D.networks.utils import BBox3dProjector
from visualDet3D.data.kitti.kittidata import KittiData
from copy import deepcopy


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
        self.projector = BBox3dProjector()
        self.is_reproject = True

    def _reproject(self, P2: np.ndarray, transformed_label: List[KittiObj]) -> Tuple[List[KittiObj], np.ndarray]:
        bbox3d_state = np.zeros([len(transformed_label), 7])  # [camera_x, camera_y, z, w, h, l, alpha]
        for obj in transformed_label:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
        bbox3d_origin = torch.tensor(
            [[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label],
            dtype=torch.float32)
        abs_corner, homo_corner, _ = self.projector(bbox3d_origin, bbox3d_origin.new(P2))
        for i, obj in enumerate(transformed_label):
            extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1])[:, np.newaxis]  # [4, 1]
            extended_bottom = np.array([obj.x, obj.y, obj.z, 1])[:, np.newaxis]  # [4, 1]
            image_center = (P2 @ extended_center)[:, 0]  # [3]
            image_center[0:2] /= image_center[2]

            image_bottom = (P2 @ extended_bottom)[:, 0]  # [3]
            image_bottom[0:2] /= image_bottom[2]

            bbox3d_state[i] = np.concatenate([image_center,
                                              [obj.w, obj.h, obj.l, obj.alpha]])  # [7]

        max_xy, _ = homo_corner[:, :, 0:2].max(dim=1)  # [N,2]
        min_xy, _ = homo_corner[:, :, 0:2].min(dim=1)  # [N,2]

        result = torch.cat([min_xy, max_xy], dim=-1)  # [:, 4]

        bbox2d = result.cpu().numpy()

        if self.is_reproject:
            for i in range(len(transformed_label)):
                transformed_label[i].bbox_l = bbox2d[i, 0]
                transformed_label[i].bbox_t = bbox2d[i, 1]
                transformed_label[i].bbox_r = bbox2d[i, 2]
                transformed_label[i].bbox_b = bbox2d[i, 3]

        return transformed_label, bbox3d_state

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        # image = torch.from_numpy(kitti_data.image).permute(2, 0, 1)
        #image = np.asarray(Image.open(kitti_data.image2_path)).copy()
        #image = self.transform(image)[0]

        data_frame = KittiData(self.params["trainer"]["train_dataset"], kitti_data.index_name, {
            "calib": True,
            "image": True,
            "label": True,
            "velodyne": False,
        })

        calib, image, label, velo = data_frame.read_data()
        image, P2, label_tr = self.transform(image, labels=deepcopy(label.data), p2=deepcopy(calib.P2))
        image = torch.from_numpy(image).permute(2, 0, 1)

        label_tr = [
            item for item in label_tr if item.type == "Car"]

        bbox3d_state = np.zeros([len(label_tr), 7])  # [camera_x, camera_y, z, w, h, l, alpha]
        if len(label_tr) > 0:
            label_tr, bbox3d_state = self._reproject(P2, label_tr)

        bbox2d = torch.tensor(
            [[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label_tr])

        bbox3d_state = torch.tensor(bbox3d_state)

        output_dict = {
            'original_calib': calib.P2,
            'calib': P2,
            'image': image,
            'label': [0 for _ in label_tr],
            'original_label': [0 for _ in label.data],
            'bbox2d': bbox2d,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': bbox3d_state,  # [N, 7] [z, sin2a, cos2a, w, h, l. alpha]
            'training_data': torch.cat([
                bbox2d, bbox3d_state
            ],dim=1),
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
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        #bbox3d_img_center = [item['bbox3d_img_center'] for item in batch]
        training_data = [item['training_data'] for item in batch]

        return {
            'original_calib': torch.tensor(original_calib).float(),
            'calib': torch.tensor(calib).float(),
            'image': torch.stack(image, dim=0),
            'label': label,
            'bbox2d': bbox2ds,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': bbox3ds,  # [N, 6] [z, sin2a, cos2a, w, h, l]
            #'bbox3d_img_center': bbox3d_img_center,
            'training_data': training_data,
        }

class KittiMonoTestDataset(KittiMonoDataset):

    def __init__(self, v_params, v_mode, v_transform):
        super(KittiMonoTestDataset, self).__init__(v_params, 'testing', v_transform)
        imdb_file_path = os.path.join(
            hydra.utils.get_original_cwd(),
            v_params["det_3d"]["preprocessed_path"],
            '{}/imdb.pkl'.format(v_mode))
        self.imdb = pickle.load(open(imdb_file_path, 'rb'))  # list of kittiData
        self.mode = v_mode

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]

        data_frame = KittiData(self.params["trainer"]["test_dataset"], kitti_data.index_name, {
            "calib": True,
            "image": True,
            "label": False,
            "velodyne": False,
        })

        calib, image, _, _ = data_frame.read_data()
        image, P2 = self.transform(image, p2=deepcopy(calib.P2))
        image = torch.from_numpy(image).permute(2, 0, 1)

        output_dict = {
            'original_calib': calib.P2,
            'calib': P2,
            'image': image,
        }
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        original_calib = [item["original_calib"] for item in batch]
        calib = [item["calib"] for item in batch]
        image = [item["image"] for item in batch]

        return {
            'original_calib': torch.tensor(original_calib).float(),
            'calib': torch.tensor(calib).float(),
            'image': torch.stack(image, dim=0),
        }
