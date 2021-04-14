import torch
from torchvision.models import resnet101
from torch import nn
import numpy as np

from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.utils.utils import cfg_from_file

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
        if v_cfg["det_3d"]["with_anchor"]==True:
            anchor_manager = Anchors(
                "temp/anchors",
                readConfigFile=False,
                pyramid_levels=[4],
                strides=[16],
                sizes=[24],
                ratios=[.5,1.],
                scales=np.array([2 ** (i / 4.0) for i in range(16)]),
                obj_types=det_3d_cfg.obj_types,
                filter_anchors=True,
                filter_y_threshold_min_max=None,
                filter_x_threshold=None,
                anchor_prior_channel=6,)
            preprocess = build_augmentator(det_3d_cfg.data.test_augmentation)

            train_index_names=[item.strip() for item in open("temp/train_split.txt").readlines()]
            for i, index_name in enumerate(train_index_names):
                data_frame = KittiData(det_3d_cfg.path.data_path, index_name, {
                                "calib": True,
                                "image": True,
                                "label": True,
                                "velodyne": False,
                            })
                calib, image, label, velo = data_frame.read_data()
                for j in range(len(det_3d_cfg.obj_types)):
                    total_objects[j] += len([obj for obj in data_frame.label if obj.type==cfg.obj_types[j]])
                    data = np.array(
                        [
                            [obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha), obj.w, obj.h, obj.l]
                                for obj in data_frame.label if obj.type==cfg.obj_types[j]
                        ]
                    ) #[N, 6]
                    if data.any():
                        uniform_sum_each_type[j, :] += np.sum(data, axis=0)
                        uniform_square_each_type[j, :] += np.sum(data ** 2, axis=0)

        self.bbox_head = AnchorBasedDetection3DHead(num_features_in=1024,
                                                    num_classes=3,
                                                    num_regression_loss_terms=12,
                                                    preprocessed_path='',
                                                    anchors_cfg=EasyDict(),
                                                    layer_cfg=EasyDict(),
                                                    loss_cfg=EasyDict(),
                                                    test_cfg=EasyDict(),
                                                    read_precompute_anchor=True)
        self.core = YoloMono3DCore()
        self.network_cfg = det_3d_cfg
        self.obj_types = det_3d_cfg.obj_types

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