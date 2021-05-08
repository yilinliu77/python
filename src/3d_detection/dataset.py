import os
import pickle

import hydra
import torch
import torchvision
import numpy as np
from PIL import Image


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

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        # image = torch.from_numpy(kitti_data.image).permute(2, 0, 1)
        image = np.asarray(Image.open(kitti_data.image2_path)).copy()
        image = self.transform(image)[0]
        image = torch.from_numpy(image).permute(2, 0, 1)

        output_dict = {
            'original_calib': kitti_data.original_calib.P2,
            'calib': kitti_data.calib.P2,
            'image': image,
            'label': [0 for _ in kitti_data.label.data],
            'bbox2d': kitti_data.bbox2d,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': kitti_data.bbox3d,  # [N, 7] [x, y, z, w, h, l, ry]
            'bbox3d_img_center': kitti_data.bbox3d_img_center,
            'training_data': kitti_data.data,
            'gt_index_per_anchor': kitti_data.gt_index_per_anchor,
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
        bbox3d_img_center = [item['bbox3d_img_center'] for item in batch]
        training_data = [item['training_data'] for item in batch]
        gt_index_per_anchor = [item['gt_index_per_anchor'] for item in batch]

        return {
            'original_calib': torch.tensor(original_calib).float(),
            'calib': torch.tensor(calib).float(),
            'image': torch.stack(image, dim=0),
            'label': label,
            'bbox2d': bbox2ds,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': bbox3ds,  # [N, 7] [x, y, z, w, h, l, ry]
            'bbox3d_img_center': bbox3d_img_center,
            'training_data': training_data,
            'gt_index_per_anchor': gt_index_per_anchor,
        }
