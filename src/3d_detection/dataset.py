import os
import pickle

import hydra
import torch
import torchvision
import numpy as np

kitti_mono_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(800, 800),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
]
)


class KittiMonoDataset(torch.utils.data.Dataset):

    def __init__(self, v_params, v_mode='training'):
        super(KittiMonoDataset, self).__init__()
        imdb_file_path = os.path.join(hydra.utils.get_original_cwd(), 'temp/det3d/anchors/{}/imdb.pkl'.format(v_mode))
        self.imdb = pickle.load(open(imdb_file_path, 'rb'))  # list of kittiData
        self.transform = kitti_mono_transform
        self.mode = v_mode

    def __getitem__(self, index):
        kitti_data = self.imdb[index % len(self.imdb)]
        kitti_data.output_dict = {
            "calib": False,
            "image": True,
            "image_3": False,
            "label": False,
            "velodyne": False
        }
        _, image, _, _ = kitti_data.read_data()
        image=self.transform(image)
        output_dict = {
            'calib': kitti_data.calib.P2,
            'image': image,
            'label': [0 for _ in kitti_data.label.data],
            'bbox2d': kitti_data.bbox2d,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': kitti_data.bbox3d,  # [N, 7] [x, y, z, w, h, l, ry]
            'bbox3d_img_center': kitti_data.bbox3d_img_center,
        }
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        calib = [item["calib"] for item in batch]
        image = [item["image"] for item in batch]
        label = [item['label'] for item in batch]
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        bbox3d_img_center = [item['bbox3d_img_center'] for item in batch]

        return {
            'calib': torch.tensor(calib).float(),
            'image': torch.stack(image,dim=0),
            'label': label,
            'bbox2d': bbox2ds,  # [N, 4] [x1, y1, x2, y2]
            'bbox3d': bbox3ds,  # [N, 7] [x, y, z, w, h, l, ry]
            'bbox3d_img_center': bbox3d_img_center,
        }
