import torch
from easydict import EasyDict
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

        self.bbox_head = AnchorBasedDetection3DHead(num_features_in=1024,
                                                    num_classes=3,
                                                    num_regression_loss_terms=12,
                                                    preprocessed_path='',
                                                    anchors_cfg=EasyDict(),
                                                    layer_cfg=EasyDict(),
                                                    loss_cfg=EasyDict(),
                                                    test_cfg=EasyDict(),
                                                    read_precompute_anchor=True)

        if v_cfg["det_3d"]["with_anchor"]==True:
            anchor_manager = Anchors(
                "temp/anchors",
                readConfigFile=False,
                pyramid_levels=[4],
                strides=[16],
                sizes=[24],
                ratios=[.5,1.],
                scales=np.array([2 ** (i / 4.0) for i in range(16)]),
                obj_types=['Car'],
                filter_anchors=True,
                filter_y_threshold_min_max=None,
                filter_x_threshold=None,
                anchor_prior_channel=6,)
            # Get the training split
            train_index_names=[item.strip() for item in open("temp/train_split.txt").readlines()]

            # Statistics about the anchor
            total_objects = 0
            total_usable_objects = 0
            uniform_sum_each_type = []
            uniform_square_each_type = []
            len_scale = len(anchor_manager.scales)
            len_ratios = len(anchor_manager.ratios)
            len_level = len(anchor_manager.pyramid_levels)
            examine = [list() for i in range(len_level * len_scale)]
            sums = [list() for i in range(len_level * len_scale)]
            squared = [list() for i in range(len_level * len_scale)]
            for i, index_name in enumerate(train_index_names):
                data_frame = KittiData(det_3d_cfg.path.data_path, index_name, {
                                "calib": True,
                                "image": True,
                                "label": True,
                                "velodyne": False,
                            })
                calib, image, label, velo = data_frame.read_data()
                data_frame.calib = calib
                # Replace the 2d bounding box with re-projection
                data_frame.bbox2d=None

                total_objects += len(data_frame.label)
                data = np.array(
                    [
                        [obj.x, obj.y, obj.z, obj.w, obj.h, obj.l,obj.theta]for obj in data_frame.label
                    ]
                )
                if data.any():
                    uniform_sum_each_type.append(np.sum(data, axis=0))
                    uniform_square_each_type.append(np.sum(data ** 2, axis=0))

                anchors, _ = anchor_manager(image[np.newaxis].transpose([0,3,1,2]), torch.tensor(calib.P2).reshape([-1, 3, 4]))
                bbox3d = torch.tensor([[obj.x, obj.y, obj.z, obj.theta] for obj in label]).cuda()
                usable_anchors = anchors[0]

                IoUs = calc_iou(usable_anchors, data_frame.bbox2d)  # [N, K]
                IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                num_usable_object = torch.sum(IoU_max > cfg.detector.head.loss_cfg.fg_iou_threshold).item()
                total_usable_objects[j] += num_usable_object

                positive_anchors_mask = IoU_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
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