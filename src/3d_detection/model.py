import math
import os

import cv2
import hydra
import torch
from easydict import EasyDict
from torchvision.models import resnet101
from torch import nn
import numpy as np

from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.utils.utils import cfg_from_file

from scipy.spatial.transform import Rotation as R

from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file


class Yolo3D(nn.Module):
    def __init__(self, v_cfg):
        super(Yolo3D, self).__init__()
        det_3d_cfg = cfg_from_file(v_cfg["det_3d"]["visualDet3D_config_path"])

        anchors_cfg = EasyDict({
            "pyramid_levels": [4],
            "strides": [16],
            "sizes": [24],
            "ratios": [.5, 1.],
            "scales": np.array([2 ** (i / 4.0) for i in range(16)]),
            "obj_types": ['Car'],
            "filter_anchors": True,
            "filter_y_threshold_min_max": None,
            "filter_x_threshold": None,
            "anchor_prior_channel": 6,
        })

        head_loss = EasyDict(
            filter_anchor=False,
            fg_iou_threshold=0.5,
            bg_iou_threshold=0.4,
            L1_regression_alpha=5 ** 2,
            focal_loss_gamma=2.0,
            match_low_quality=False,
            balance_weight=[20.0],
            regression_weight=[1, 1, 1, 1, 1, 1, 3, 1, 1, 0.5, 0.5, 0.5, 1],
            # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
        )

        # self.bbox_head = AnchorBasedDetection3DHead(num_features_in=1024,
        #                                             num_classes=3,
        #                                             num_regression_loss_terms=12,
        #                                             preprocessed_path="temp/anchors",
        #                                             anchors_cfg=anchors_cfg,
        #                                             layer_cfg=EasyDict(),
        #                                             loss_cfg=EasyDict(),
        #                                             test_cfg=EasyDict(),
        #                                             read_precompute_anchor=True)

        if v_cfg["model"]["compute_anchor"] == True:
            # Get the training split
            train_index_names = [item.strip() for item in open(
                os.path.join(hydra.utils.get_original_cwd(), "temp/anchors/train_split.txt")).readlines()]

            anchor_manager = Anchors(os.path.join(hydra.utils.get_original_cwd(), "temp/anchors/"),
                                     readConfigFile=False,
                                     **anchors_cfg)
            # Statistics about the anchor
            total_objects = 0
            total_usable_objects = 0
            uniform_sum_each_type = []
            uniform_square_each_type = []
            len_scale = len(anchor_manager.scales)
            len_ratios = len(anchor_manager.ratios)
            len_level = len(anchor_manager.pyramid_levels)
            examine = np.zeros([len_level * len_scale, len_ratios])
            sums = np.zeros([len_level * len_scale, len_ratios, 3])
            squared = np.zeros([len_level * len_scale, len_ratios, 3], dtype=np.float64)
            for i, index_name in enumerate(train_index_names):
                data_frame = KittiData(v_cfg["trainer"]["train_dataset"], index_name, {
                    "calib": True,
                    "image": True,
                    "label": True,
                    "velodyne": False,
                })
                calib, image, label, velo = data_frame.read_data()
                data_frame.calib = calib
                data_frame.label = label
                # Replace the 2d bounding box with re-projection
                # NOTE: Currently fix the shape
                data_frame.bbox2d = self.project_box3d_to_img(data_frame)

                ### Debug
                from matplotlib import pyplot as plt
                min_x = torch.min(data_frame.bbox2d[0, :, 0]).item()
                min_y = torch.min(data_frame.bbox2d[0, :, 1]).item()
                max_x = torch.max(data_frame.bbox2d[0, :, 0]).item()
                max_y = torch.max(data_frame.bbox2d[0, :, 1]).item()
                x_min, y_min, x_max, y_max = map(int, [min_x,min_y,max_x,max_y])
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [0, 255, 0],5)
                plt.imshow(image)
                plt.show()
                ###

                total_objects += len(data_frame.label.data)
                data = np.array(
                    [
                        [obj.x, obj.y, obj.z, obj.w, obj.h, obj.l, obj.ry] for obj in data_frame.label.data
                    ]
                )
                if data.any():
                    uniform_sum_each_type.append(np.sum(data, axis=0))
                    uniform_square_each_type.append(np.sum(data ** 2, axis=0))

                # NOTE: Currently fix the shape
                anchors, _ = anchor_manager(torch.ones((1, 3, 800, 800)), torch.tensor(calib.P2).reshape([-1, 3, 4]))
                bbox3d = torch.tensor(data).cuda()
                usable_anchors = anchors[0]

                IoUs = calc_iou(usable_anchors, data_frame.bbox2d)  # [N, K]
                IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                num_usable_object = torch.sum(IoU_max > head_loss.fg_iou_threshold).item()
                total_usable_objects += num_usable_object

                positive_anchors_mask = IoU_max_anchor > head_loss.fg_iou_threshold
                positive_ground_truth_3d = bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy()  # [x1, y1, x2, y2]

                sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                for k in range(len(sizes_int)):
                    examine[sizes_int[k], ratio_int[k]] += 1
                    sums[sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                    squared[sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

        self.core = YoloMono3DCore()
        self.network_cfg = det_3d_cfg
        self.obj_types = det_3d_cfg.obj_types

    # NOTE: Add fixed 45 degree in pitch (X axis)
    def project_box3d_to_img(self, v_dataframe, v_width=None, v_height=None):
        camera_corners = []
        for id_batch in range(len(v_dataframe.label.data)):
            corners = torch.tensor(
                [
                    [-v_dataframe.label.data[id_batch].w / 2,
                     0,
                     -v_dataframe.label.data[id_batch].l / 2],
                    [-v_dataframe.label.data[id_batch].w / 2,
                     0,
                     v_dataframe.label.data[id_batch].l / 2],
                    [v_dataframe.label.data[id_batch].w / 2,
                     0,
                     -v_dataframe.label.data[id_batch].l / 2],
                    [v_dataframe.label.data[id_batch].w / 2,
                     0,
                     v_dataframe.label.data[id_batch].l / 2],
                    [-v_dataframe.label.data[id_batch].w / 2,
                     -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].l / 2],
                    [-v_dataframe.label.data[id_batch].w / 2,
                     -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].l / 2],
                    [v_dataframe.label.data[id_batch].w / 2,
                     -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].l / 2],
                    [v_dataframe.label.data[id_batch].w / 2,
                     -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].l / 2],
                ]
            ).float()
            rotation_matrix = torch.tensor(
                R.from_euler("XYZ", [-math.pi / 4, -v_dataframe.label.data[id_batch].ry, 0]).as_matrix()).float()
            rotated_corners = torch.matmul(rotation_matrix, corners.T).T
            abs_corners = rotated_corners + torch.tensor([
                v_dataframe.label.data[id_batch].x,
                v_dataframe.label.data[id_batch].y,
                v_dataframe.label.data[id_batch].z
            ]).float()  # [N, 8, 3]
            abs_corners=abs_corners
            camera_corners.append(torch.cat([abs_corners,
                                             torch.ones_like(abs_corners[:, 0:1])],
                                            dim=-1))  # [N, 8, 4]
            with open("{}.xyz".format(id_batch), "w") as f:
                for point in camera_corners[-1]:
                    f.write("{} {} {}\n".format(point[0].item(), point[1].item(), point[2].item()))
            pass
        camera_corners = torch.stack(camera_corners, dim=0)
        # camera_corners = torch.matmul(torch.tensor(v_dataframe.calib.P2).float(),
        #                             camera_corners.transpose(1, 2)).transpose(1, 2)  # [N, 8, 3]

        homo_coord = (camera_corners / (camera_corners[:, :, 2:] + 1e-6))[:, :, :2]  # [N, 8, 3]
        if v_width is not None:
            homo_coord = torch.stack([
                torch.clamp(homo_coord[:, :, 0], 0, v_width),
                torch.clamp(homo_coord[:, :, 1], 0, v_height),
            ], dim=-1)
        return homo_coord

    def training_forward(self, img_batch, annotations, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """

        features = self.core(dict(image=img_batch, P2=P2))
        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2, image=img_batch))

        anchors = self.bbox_head.get_anchor(img_batch, P2)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        return cls_loss, reg_loss, loss_dict

    def test_forward(self, img_batch, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1  # we recommmend image batch size = 1 for testing

        features = self.core(dict(image=img_batch, P2=P2))
        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2))

        anchors = self.bbox_head.get_anchor(img_batch, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, img_batch)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations, calib)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)
