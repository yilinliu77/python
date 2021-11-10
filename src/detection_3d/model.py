import math
import os
import pickle
import shutil
from copy import deepcopy
import sys
# from lib.DCNv2_latest.dcn_v2 import DCNv2, DCN

# import win32file

# sys.path.append('C:/Users/zihan/Desktop/visualDet3D/visualDet3D')

import cv2
import hydra
import torch
from easydict import EasyDict
from torch import nn
import numpy as np
from torchvision.ops import nms
from tqdm import tqdm

from visualDet3D.networks.heads.losses import SigmoidFocalLoss
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from thirdparty.DCNv2_latest.dcn_v2 import DCN
from visualDet3D.networks.utils.utils import ClipBoxes

from scipy.spatial.transform import Rotation as R

from src.detection_3d.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj
from visualDet3D.utils.utils import alpha2theta_3d, theta2alpha_3d
from typing import List, Tuple
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file
from torch.nn import functional as F


class Yolo3D(nn.Module):
    def __init__(self, v_cfg):
        super(Yolo3D, self).__init__()
        self.hparams = v_cfg

        anchors_cfg = EasyDict({
            "pyramid_levels": list(v_cfg["anchor"]["pyramid_levels"]),
            "v_strides": list(v_cfg["anchor"]["v_strides"]),
            "sizes": list(v_cfg["anchor"]["sizes"]),  # Base size of the anchors (in original image shape)
            "ratios": list(v_cfg["anchor"]["ratios"]),  # Different ratio of the anchors
            "scales": list(v_cfg["anchor"]["scales"]),
        })

        self.head_loss_cfg = EasyDict(
            filter_anchor=False,
            fg_iou_threshold=0.5,
            bg_iou_threshold=0.4,
            match_low_quality=False,
            regression_weight=[1, 1, 1, 1, 1, 1, 3, 1, 1, 0.5, 0.5, 0.5, 1],
        )

        self.network_cfg = EasyDict(
            reg_feature_size=1024,
            cls_feature_size=1024,

            num_cls_output=1 + 1,  # A flag for alpha angle
            num_reg_output=12,  # (x1, y1, x2, y2, x3d_center, y3d_center, z, sin2a, cos2a, w, h, l)
        )

        self.stds = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], requires_grad=False).float()

        self.clipper = ClipBoxes()
        self.projector = BBox3dProjector()

        self.is_reproject = True

        train_transform = [
            EasyDict(type_name='ConvertToFloat'),
            EasyDict(type_name='PhotometricDistort',
                     keywords=EasyDict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5,
                                       saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
            EasyDict(type_name='CropTop', keywords=EasyDict(crop_top_index=100)),
            EasyDict(type_name='Resize',
                     keywords=EasyDict(size=(v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))),
            EasyDict(type_name='RandomMirror', keywords=EasyDict(mirror_prob=0.5)),
            EasyDict(type_name='Normalize',
                     keywords=EasyDict(mean=np.array([0.485, 0.456, 0.406]), stds=np.array([0.229, 0.224, 0.225])))
        ]
        test_transform = [
            EasyDict(type_name='ConvertToFloat'),
            EasyDict(type_name='CropTop', keywords=EasyDict(crop_top_index=100)),
            EasyDict(type_name='Resize',
                     keywords=EasyDict(size=(v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))),
            EasyDict(type_name='Normalize',
                     keywords=EasyDict(mean=np.array([0.485, 0.456, 0.406]), stds=np.array([0.229, 0.224, 0.225])))
        ]
        self.train_preprocess = build_augmentator(train_transform)
        self.test_preprocess = build_augmentator(test_transform)

        preprocessed_path = os.path.join(hydra.utils.get_original_cwd(), v_cfg["det_3d"]["preprocessed_path"])
        if v_cfg["det_3d"]["compute_anchor"] == True:
            with torch.no_grad():
                if not v_cfg["trainer"]["evaluate"]:
                    for data_split in ["training", "validation", ]:
                        print("Start the precomputing {}".format(data_split))
                        # Get the training split
                        train_index_names = [item.strip() for item in open(
                            os.path.join(hydra.utils.get_original_cwd(),
                                         v_cfg["trainer"]["train_split"] if data_split == "training" else v_cfg[
                                             "trainer"]["valid_split"])).readlines()]

                        anchor_manager = Anchors(preprocessed_path="",
                                                 **anchors_cfg)

                        len_scale = len(anchor_manager.scales)
                        len_ratios = len(anchor_manager.ratios)
                        len_level = len(anchor_manager.pyramid_levels)

                        ### modified
                        examine = np.zeros([1, len_level * len_scale, len_ratios])
                        sums = np.zeros([1, len_level * len_scale, len_ratios, 3])
                        squared = np.zeros([1, len_level * len_scale, len_ratios, 3], dtype=np.float64)

                        uniform_sum_each_type = np.zeros((1, 6),
                                                         dtype=np.float64)  # [z, sina, cosa, w, h, l]
                        uniform_square_each_type = np.zeros((1, 6), dtype=np.float64)
                        ###

                        # Statistics about the anchor
                        total_objects = 0
                        total_usable_objects = 0
                        data_statics = []
                        data_frames = []
                        for i, index_name in tqdm(enumerate(train_index_names)):
                            data_frame = KittiData(v_cfg["trainer"]["train_dataset"], index_name, {
                                "calib": True,
                                "image": True,
                                "label": True,
                                "velodyne": False,
                            })
                            calib, image, label, velo = data_frame.read_data()

                            data_frame.index_name = index_name
                            # data_frame.image_file = data_frame.image2_path
                            # data_frame.original_calib = deepcopy(calib)
                            # data_frame.original_shape = image.shape
                            # data_frame.label = label

                            # Do the filtering
                            max_occlusion = 2
                            min_z = 3
                            if data_split == "training":
                                label.data = [
                                    item for item in label.data if
                                    item.type == "Car" and item.occluded < max_occlusion and item.z > min_z]
                                # label.data = [
                                #    item for item in label.data if item.type == "Car" and item.z < 200]

                                total_objects += len(label.data)
                                data = np.array(
                                    [
                                        [obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha), obj.w, obj.h, obj.l]
                                        for obj in label.data
                                    ]
                                )  # [N, 6]
                                if data.any():
                                    uniform_sum_each_type[0, :] += np.sum(data, axis=0)
                                    uniform_square_each_type[0, :] += np.sum(data ** 2, axis=0)
                            else:
                                label.data = [
                                    item for item in label.data if item.type == "Car"]

                            preprocess = self.train_preprocess if data_split == "training" else self.test_preprocess
                            image, P2, label_tr = preprocess(image, labels=deepcopy(label.data), p2=deepcopy(calib.P2))

                            # data_frame.transformed_image = image
                            calib.P2 = P2
                            # data_frame.calib = calib
                            label.data = label_tr
                            # data_frame.label = label

                            bbox2d = torch.tensor(
                                [[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label.data])

                            bbox3d = torch.tensor(np.zeros([len(label.data), 7]))

                            if len(label.data) > 0:
                                if data_split == "training":
                                    bbox3d = torch.tensor(
                                        [[obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)] for obj in
                                         label.data]).cuda()

                                    anchors = anchor_manager(
                                        (v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))

                                    ###
                                    usable_anchors = anchors.to(bbox2d.device)

                                    IoUs = calc_iou(usable_anchors, bbox2d)  # [num_anchors, num_gt]
                                    IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                                    IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                                    num_usable_object = torch.sum(IoU_max > self.head_loss_cfg.fg_iou_threshold).item()
                                    total_usable_objects += num_usable_object

                                    positive_anchors_mask = IoU_max_anchor > self.head_loss_cfg.fg_iou_threshold
                                    positive_ground_truth_3d = bbox3d[
                                        IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                                    ### modified
                                    used_anchors = usable_anchors[
                                        positive_anchors_mask].cpu().numpy()  # [x1, y1, x2, y2]
                                    sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                                    for k in range(len(sizes_int)):
                                        examine[0, sizes_int[k], ratio_int[k]] += 1
                                        sums[0, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                                        squared[0, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

                                label.data, bbox3d = self._reproject(P2, label.data)
                                bbox3d = torch.tensor(bbox3d)

                                data = torch.cat([
                                    bbox2d,
                                    bbox3d,
                                ], dim=1)  # [x1, y1, x2, y2, cx, cy, z, w, h, l, alpha]
                            data_frames.append(data_frame)

                            # for box in data_frame.bbox2d:
                            #     pts=list(map(int,[box[0].item(),box[1].item(),box[2].item(),box[3].item()]))
                            #     viz_img = cv2.rectangle(image,
                            #                             (pts[0], pts[1]),
                            #                             (pts[2], pts[3]),
                            #                             (255, 0, 0),
                            #                             5
                            #                             )
                            # cv2.imshow("",viz_img)
                            # cv2.waitKey()

                        print("Total objects:{}, total usable objects:{}".format(total_objects, total_usable_objects))
                        if not os.path.exists(os.path.join(preprocessed_path, data_split)):
                            os.makedirs(os.path.join(preprocessed_path, data_split))
                        if data_split == "training":
                            global_mean = uniform_sum_each_type[0] / total_objects
                            global_var = np.sqrt(uniform_square_each_type[0] / total_objects - global_mean ** 2)

                            avg = sums[0] / (examine[0][:, :, np.newaxis] + 1e-8)
                            EX_2 = squared[0] / (examine[0][:, :, np.newaxis] + 1e-8)
                            std = np.sqrt(EX_2 - avg ** 2)

                            avg[examine[0] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
                            std[examine[0] < 10, :] = 1e10
                            avg[np.isnan(std)] = -100
                            std[np.isnan(std)] = 1e10
                            avg[std < 1e-3] = -100
                            std[std < 1e-3] = 1e10

                            whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
                            whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var[3:6]

                            avg = np.concatenate([avg, whl_avg], axis=2)
                            std = np.concatenate([std, whl_std], axis=2)

                            print("Pre calculation done, save it now")

                            npy_file = os.path.join(preprocessed_path, data_split, 'anchor_mean_std_Car.npy')
                            mean_std = np.stack([avg, std, ])
                            np.save(npy_file, mean_std)
                            # [[mean, std], scale, ratio, [z, sin2a, cos2a, w,h,l]] -- shape: [2, 16, 2, 6]

                        pkl_file = os.path.join(preprocessed_path, data_split, 'imdb.pkl')
                        pickle.dump(data_frames, open(pkl_file, 'wb'))

                else:
                    ### add for testing set
                    data_split = 'testing'
                    print("Start the precomputing {}".format(data_split))
                    # Get the testing split
                    train_index_names = [item.strip() for item in open(
                        os.path.join(hydra.utils.get_original_cwd(),
                                     v_cfg["trainer"]["test_split"])).readlines()]

                    data_frames = []

                    for i, index_name in tqdm(enumerate(train_index_names)):
                        data_frame = KittiData(v_cfg["trainer"]["test_dataset"], index_name, {
                            "calib": True,
                            "image": True,
                            "label": False,
                            "velodyne": False,
                        })

                        data_frame.index_name = index_name
                        calib, image, _, _ = data_frame.read_data()
                        data_frame.original_calib = deepcopy(calib)
                        data_frame.original_shape = image.shape

                        preprocess = self.test_preprocess
                        image, P2 = preprocess(image, p2=deepcopy(calib.P2))
                        # data_frame.image = image
                        data_frame.image_file = data_frame.image2_path
                        calib.P2 = P2
                        data_frame.calib = calib

                        data_frames.append(data_frame)

                    if not os.path.exists(os.path.join(preprocessed_path, data_split)):
                        os.makedirs(os.path.join(preprocessed_path, data_split))

                    pkl_file = os.path.join(preprocessed_path, data_split, 'imdb.pkl')
                    pickle.dump(data_frames, open(pkl_file, 'wb'))
                ###

        anchor_manager_with_distribution = Anchors(
            preprocessed_path=os.path.join(preprocessed_path, "training", 'anchor_mean_std_Car.npy'),
            **anchors_cfg)
        self.anchors = anchor_manager_with_distribution(
            (v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"])).float()
        self.anchors_distribution = anchor_manager_with_distribution.anchors_mean_std.float()

        self.init_layers(self.network_cfg["cls_feature_size"],
                         self.network_cfg["reg_feature_size"], anchor_manager_with_distribution.num_anchor_per_scale,
                         self.network_cfg["num_cls_output"], self.network_cfg["num_reg_output"])

        self.focal_loss = SigmoidFocalLoss(gamma=2.0, balance_weights=torch.tensor([20]))
        self.alpha_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.classification_loss = nn.CrossEntropyLoss(reduction="mean")
        self.regression_loss = nn.L1Loss(reduction="none")

    def init_layers(self,
                    v_cls_feature_size, v_reg_feature_size, v_num_anchors,
                    v_num_cls_output, v_num_reg_output):
        v_num_features_in = -1
        if self.hparams["model"]["backbone"] == "resnet101":
            from visualDet3D.networks.backbones import resnet101, resnet
            self.core = resnet(depth=101,
                               pretrained=True,
                               frozen_stages=-1,
                               num_stages=3,
                               out_indices=(2,),
                               norm_eval=False,
                               dilations=(1, 1, 1), )
            v_num_features_in = 1024
            self.cls_feature_extraction = nn.Sequential(
                nn.Conv2d(v_num_features_in, v_cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(v_cls_feature_size, v_cls_feature_size, kernel_size=3, padding=1),
                nn.Dropout2d(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(v_cls_feature_size, v_num_anchors * v_num_cls_output, kernel_size=3, padding=1),
                AnchorFlatten(v_num_cls_output)
            )

            self.cls_feature_extraction[-2].weight.data.fill_(0)
            self.cls_feature_extraction[-2].bias.data.fill_(0)

            self.reg_feature_extraction = nn.Sequential(
                DCN(v_num_features_in, v_reg_feature_size, 3, stride=1, padding=1),
                nn.BatchNorm2d(v_reg_feature_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(v_reg_feature_size, v_reg_feature_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(v_reg_feature_size),
                nn.ReLU(inplace=True),

                nn.Conv2d(v_reg_feature_size, v_num_anchors * v_num_reg_output, kernel_size=3, padding=1),
                AnchorFlatten(v_num_reg_output)
            )

            self.reg_feature_extraction[-2].weight.data.fill_(0)
            self.reg_feature_extraction[-2].bias.data.fill_(0)
        elif self.hparams["model"]["backbone"] == "resnet101_fpn":
            # from fpn_resnet import fpn_resnet
            # self.core = fpn_resnet(depth=101,
            #                        pretrained=True,
            #                        frozen_stages=-1,
            #                        num_stages=4,
            #                        out_indices=(2,),
            #                        norm_eval=False,
            #                        dilations=(1, 1, 1, 1), )
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,BackboneWithFPN
            # 1
            from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
            from torchvision.models import resnet
            from torchvision.ops import misc as misc_nn_ops
            backbone = resnet.__dict__["resnet101"](
                pretrained=True,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d)

            # select layers that wont be frozen
            assert 0 <= 3 <= 5
            layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:3]
            for name, parameter in backbone.named_parameters():
                if all([not name.startswith(layer) for layer in layers_to_train]):
                    parameter.requires_grad_(False)
            extra_blocks = LastLevelMaxPool()
            returned_layers = [1, 2, 3, 4]
            assert min(returned_layers) > 0 and max(returned_layers) < 5
            return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
            in_channels_stage2 = backbone.inplanes // 8
            in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
            out_channels = 1024
            self.core = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

            # 2
            # self.core = resnet_fpn_backbone('resnet101', pretrained=True, trainable_layers=3,)
            v_num_features_in = 1024
            self.cls_feature_extraction=nn.ModuleList()
            self.reg_feature_extraction=nn.ModuleList()
            for i in range(3):
                self.cls_feature_extraction.append(nn.Sequential(
                    nn.Conv2d(v_num_features_in, v_cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(v_cls_feature_size, v_cls_feature_size, kernel_size=3, padding=1),
                    nn.Dropout2d(0.3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(v_cls_feature_size, v_num_anchors * v_num_cls_output, kernel_size=3, padding=1),
                    AnchorFlatten(v_num_cls_output)
                ))

                self.cls_feature_extraction[-1][-2].weight.data.fill_(0)
                self.cls_feature_extraction[-1][-2].bias.data.fill_(0)

                self.reg_feature_extraction.append(nn.Sequential(
                    DCN(v_num_features_in, v_reg_feature_size, 3, stride=1, padding=1),
                    nn.BatchNorm2d(v_reg_feature_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(v_reg_feature_size, v_reg_feature_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v_reg_feature_size),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(v_reg_feature_size, v_num_anchors * v_num_reg_output, kernel_size=3, padding=1),
                    AnchorFlatten(v_num_reg_output)
                ))

                self.reg_feature_extraction[-1][-2].weight.data.fill_(0)
                self.reg_feature_extraction[-1][-2].bias.data.fill_(0)

        elif self.hparams["model"]["backbone"] == "tiny_swin_transformer":
            from TransformerDet import Swin_Transformer, Swin_Transformer_FPN
            self.core = Swin_Transformer(v_is_tiny=True)
            v_num_features_in = 512
        elif self.hparams["model"]["backbone"] == "tiny_swin_transformer_fpn":
            from TransformerDet import Swin_Transformer, Swin_Transformer_FPN
            v_num_features_in = 512
            self.core = Swin_Transformer_FPN(v_is_tiny=True, v_output_features=512)

        # self.core.requires_grad_(False)
        # self.core = resnet_fpn_backbone(
        #     "resnet50",
        #     pretrained=True,
        #     trainable_layers=5,
        #     output_channel=1024
        # )



    """ 
    Input:  Dataframe with the standard label
    Output: (9,4) torch tensor. 8 corners and 1 center. (x1,y1,x2,y2)
    Note:
        - Add fixed 45 degree in pitch (X axis)
        - May have negative coordinate if v_width and v_height is None
    """
    def project_box3d_to_img(self, v_dataframe, v_pitch_45, v_width=None, v_height=None):
        camera_corners = []
        for id_batch in range(len(v_dataframe.label.data)):
            corners = torch.tensor(
                [
                    [-v_dataframe.label.data[id_batch].l / 2, 0, -v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2, 0, v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2, 0, -v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2, 0, v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2, -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2, -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2, -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2, -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].w / 2],
                    [0, -v_dataframe.label.data[id_batch].h / 2, 0],

                ]
            ).float()

            # Rotate through Y axis
            # Both upper of lower case is accept. The box is currently at the origin
            yaw_rotation_matrix = torch.tensor(
                R.from_euler("xyz", [0, -v_dataframe.label.data[id_batch].ry, 0]).as_matrix()).float()
            corners = torch.matmul(yaw_rotation_matrix, corners.T).T

            corners = corners + torch.tensor([
                v_dataframe.label.data[id_batch].x,
                v_dataframe.label.data[id_batch].y,
                v_dataframe.label.data[id_batch].z
            ]).float()  # [N, 8, 3]

            # Rotate through Y axis
            # Should be global coordinates, upper case in scipy's Rotation
            if v_pitch_45:
                rotation_matrix = torch.tensor(
                    R.from_euler("XYZ", [math.pi / 4, 0, 0]).as_matrix()).float()
                corners = torch.matmul(rotation_matrix, corners.T).T

            camera_corners.append(torch.cat([corners,
                                             torch.ones_like(corners[:, 0:1])],
                                            dim=-1))  # [N, 8, 4]
            # with open("{}.xyz".format(id_batch), "w") as f:
            #     for point in camera_corners[-1]:
            #         f.write("{} {} {}\n".format(point[0].item(), point[1].item(), point[2].item()))
            pass
        camera_corners = torch.stack(camera_corners, dim=0)
        camera_corners = torch.matmul(torch.tensor(v_dataframe.calib.P2).float(),
                                      camera_corners.transpose(1, 2)).transpose(1, 2)  # [N, 8, 3]

        homo_coord = (camera_corners / (camera_corners[:, :, 2:] + 1e-6))[:, :, :2]  # [N, 8, 3]
        if v_width is not None:
            homo_coord = torch.stack([
                torch.clamp(homo_coord[:, :, 0], 0, v_width),
                torch.clamp(homo_coord[:, :, 1], 0, v_height),
            ], dim=-1)
        return homo_coord

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

    def _assign(self, anchor, annotation,
                bg_iou_threshold=0.0,
                fg_iou_threshold=0.5,
                min_iou_threshold=0.0,
                match_low_quality=True,
                gt_max_assign_all=True):
        """
            :param
                anchor: [N, 4]
                annotation: [num_gt, 4]
            :return
                num_gt: box num of gt
                assigned_gt_inds:
                max_overlaps=max_overlaps
                labels=assigned_label
        """
        N = anchor.shape[0]
        num_gt = annotation.shape[0]
        assigned_gt_inds = anchor.new_full(
            (N,),
            -1, dtype=torch.long
        )  # [N, ] torch.long
        max_overlaps = anchor.new_zeros((N,))
        assigned_labels = anchor.new_full((N,),
                                          -1,
                                          dtype=torch.long)

        if num_gt == 0:
            assigned_gt_inds = anchor.new_full(
                (N,),
                0, dtype=torch.long
            )
            return assigned_gt_inds

        IoU = calc_iou(anchor, annotation[:, :4])  # num_anchors x num_annotations

        # max for anchor
        max_overlaps, argmax_overlaps = IoU.max(dim=1)  # num_anchors

        # max for gt
        gt_max_overlaps, gt_argmax_overlaps = IoU.max(dim=0)  # num_gt

        # assign negative
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < bg_iou_threshold)] = 0

        # assign positive
        pos_inds = max_overlaps >= fg_iou_threshold
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if match_low_quality:
            for i in range(num_gt):
                if gt_max_overlaps[i] >= min_iou_threshold:
                    if gt_max_assign_all:
                        max_iou_inds = IoU[:, i] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        assigned_labels = assigned_gt_inds.new_full((N,), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False
        ).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = torch.zeros_like(
                annotation[assigned_gt_inds[pos_inds] - 1, 0]).long()  # Only 1 class here

        # return_dict = dict(
        #     num_gt=num_gt,
        #     assigned_gt_inds=assigned_gt_inds,
        #     max_overlaps=max_overlaps,
        #     labels=assigned_labels
        # )
        return assigned_gt_inds

    """
    sampled_gt_bboxes: (x1, y1, x2, y2, cx, cy, z, w, h, l, alpha)
    anchors_3d_mean_std: (z, sin2a, cos2a, w, h, l)(mean, std)
    outputs: (x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l)
    """

    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
        assert sampled_anchors.shape[0] == sampled_gt_bboxes.shape[0]

        sampled_anchors = sampled_anchors.float()
        sampled_gt_bboxes = sampled_gt_bboxes.float()
        px = (sampled_anchors[..., 0] + sampled_anchors[..., 2]) * 0.5  # center x of the predicted 2d bbox
        py = (sampled_anchors[..., 1] + sampled_anchors[..., 3]) * 0.5  # center y of the predicted 2d bbox
        pw = sampled_anchors[..., 2] - sampled_anchors[..., 0]  # width of the predicted 2d bbox
        ph = sampled_anchors[..., 3] - sampled_anchors[..., 1]  # height of the predicted 2d bbox

        gx = (sampled_gt_bboxes[..., 0] + sampled_gt_bboxes[..., 2]) * 0.5  # center x of the gt 2d bbox
        gy = (sampled_gt_bboxes[..., 1] + sampled_gt_bboxes[..., 3]) * 0.5  # center y of the gt 2d bbox
        gw = sampled_gt_bboxes[..., 2] - sampled_gt_bboxes[..., 0]  # width of the gt 2d bbox
        gh = sampled_gt_bboxes[..., 3] - sampled_gt_bboxes[..., 1]  # height of the gt 2d bbox

        targets_dx = (gx - px) / pw  # shift of the gt 2d bbox center x
        targets_dy = (gy - py) / ph  # shift of the gt 2d bbox center y
        targets_dw = torch.log(gw / pw)  # shift of the gt 2d bbox width
        targets_dh = torch.log(gh / ph)  # shift of the gt 2d bbox height

        ### modified
        targets_cdx_projected = (sampled_gt_bboxes[:, 4] - px) / pw
        targets_cdy_projected = (sampled_gt_bboxes[:, 5] - py) / ph

        targets_cdz = (sampled_gt_bboxes[:, 6] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        targets_cd_sin = (torch.sin(sampled_gt_bboxes[:, 10] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[
                                                                                                    :, 1, 1]
        targets_cd_cos = (torch.cos(sampled_gt_bboxes[:, 10] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[
                                                                                                    :, 2, 1]
        targets_w3d = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 8] - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 9] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_cdx_projected, targets_cdy_projected, targets_cdz,
                               targets_cd_sin, targets_cd_cos,
                               targets_w3d, targets_h3d, targets_l3d), dim=1)

        targets = targets.div_(self.stds.to(targets.device))

        targets_alpha_cls = (torch.cos(sampled_gt_bboxes[:, 10]) > 0).float()
        return targets, targets_alpha_cls

    def _decode(self, v_boxes_2d, v_prediction, anchors_3d_mean_std, v_alpha):
        std = self.stds.to(v_boxes_2d.device)

        # 2D Bounding box
        widths = v_boxes_2d[..., 2] - v_boxes_2d[..., 0]
        heights = v_boxes_2d[..., 3] - v_boxes_2d[..., 1]
        ctr_x = v_boxes_2d[..., 0] + 0.5 * widths
        ctr_y = v_boxes_2d[..., 1] + 0.5 * heights

        dx = v_prediction[..., 0] * std[0]
        dy = v_prediction[..., 1] * std[1]
        dw = v_prediction[..., 2] * std[2]
        dh = v_prediction[..., 3] * std[3]

        pred_centre_x = ctr_x + dx * widths
        pred_centre_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_centre_x - 0.5 * pred_w
        pred_boxes_y1 = pred_centre_y - 0.5 * pred_h
        pred_boxes_x2 = pred_centre_x + 0.5 * pred_w
        pred_boxes_y2 = pred_centre_y + 0.5 * pred_h

        # 2D Projected centre
        cdx = v_prediction[..., 4] * std[4]
        cdy = v_prediction[..., 5] * std[5]
        pred_center_x1_projected = ctr_x + cdx * widths
        pred_center_y1_projected = ctr_y + cdy * heights

        ### modified
        pred_z = v_prediction[..., 6] * anchors_3d_mean_std[:, 0, 1] + anchors_3d_mean_std[:, 0, 0]  # [N, 6]
        pred_sin = v_prediction[..., 7] * anchors_3d_mean_std[:, 1, 1] + anchors_3d_mean_std[:, 1, 0]
        pred_cos = v_prediction[..., 8] * anchors_3d_mean_std[:, 2, 1] + anchors_3d_mean_std[:, 2, 0]
        pred_alpha = torch.atan2(pred_sin, pred_cos) / 2.0

        pred_w = v_prediction[..., 9] * anchors_3d_mean_std[:, 3, 1] + anchors_3d_mean_std[:, 3, 0]
        pred_h = v_prediction[..., 10] * anchors_3d_mean_std[:, 4, 1] + anchors_3d_mean_std[:, 4, 0]
        pred_l = v_prediction[..., 11] * anchors_3d_mean_std[:, 5, 1] + anchors_3d_mean_std[:, 5, 0]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                  pred_center_x1_projected, pred_center_y1_projected, pred_z,
                                  pred_w, pred_h, pred_l, pred_alpha], dim=1)
        pred_boxes[v_alpha[:, 0] < 0.5, -1] += np.pi

        return pred_boxes

    def _sample(self, v_gt_index_per_anchor, anchors, v_gt_data):
        """
            Pseudo sampling
        """
        pos_inds = torch.nonzero(
            v_gt_index_per_anchor > 0, as_tuple=False
        ).unsqueeze(-1).unique()
        neg_inds = torch.nonzero(
            v_gt_index_per_anchor == 0, as_tuple=False
        ).unsqueeze(-1).unique()

        pos_assigned_gt_inds = v_gt_index_per_anchor - 1

        if v_gt_data.numel() == 0:
            ### modified
            pos_gt_bboxes = v_gt_data.new_zeros([0, 12])
            ###
            # pos_gt_bboxes = v_gt_data.new_zeros([0, 13])
        else:
            pos_gt_bboxes = v_gt_data[pos_assigned_gt_inds[pos_inds]]

        return_dict = dict(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            pos_bboxes=anchors[pos_inds],
            neg_bboxes=anchors[neg_inds],
            pos_gt_bboxes=pos_gt_bboxes,
            pos_assigned_gt_inds=pos_assigned_gt_inds[pos_inds],
        )
        return return_dict

    def get_training_data(self, v_anchors, v_data, v_anchor_mean_std, v_cls_prediction, v_reg_prediction):
        # modify: new GT box after sample: (x1, y1, x2, y2, x3d_projected, y3d_projected, z3d, sin2a, cos2a, w3d, h3d, l3d)
        gt_index_per_anchor = self._assign(v_anchors, v_data[:, :4],
                                           bg_iou_threshold=self.head_loss_cfg["bg_iou_threshold"],
                                           fg_iou_threshold=self.head_loss_cfg["fg_iou_threshold"],
                                           min_iou_threshold=0.0,
                                           match_low_quality=self.head_loss_cfg[
                                               "match_low_quality"],
                                           gt_max_assign_all=True,
                                           )
        sampling_result_dict = self._sample(gt_index_per_anchor, v_anchors, v_data)

        labels = v_anchors.new_full((self.anchors.shape[0], 1),
                                    -1,  # -1 not computed, binary for each class
                                    dtype=torch.long)

        pos_inds = sampling_result_dict['pos_inds']
        neg_inds = sampling_result_dict['neg_inds']

        regression_weight = torch.tensor(self.head_loss_cfg["regression_weight"], dtype=torch.float).to(
            v_anchors.device)

        if len(pos_inds) > 0:
            pos_bboxes = sampling_result_dict['pos_bboxes']
            pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes']
            v_anchor_mean_std = v_anchor_mean_std[pos_inds, :]

            # Double check if anchor has negative z mean
            valid_flag = v_anchor_mean_std[:, 0, 0] > 0
            pos_bboxes = pos_bboxes[valid_flag]
            pos_gt_bboxes = pos_gt_bboxes[valid_flag]
            v_anchor_mean_std = v_anchor_mean_std[valid_flag]
            pos_inds = pos_inds[valid_flag]

            ###
            pos_bbox_targets, alpha_score = self._encode(
                pos_bboxes, pos_gt_bboxes, v_anchor_mean_std
            )  # [N, 12], [N, 1]
            labels[pos_inds, 0] = 1
            alpha_loss = self.alpha_loss(v_cls_prediction[pos_inds][:, 1], alpha_score).reshape(1)
            reg_loss = self.regression_loss(v_reg_prediction[pos_inds][:, :],
                                            pos_bbox_targets[:, :],
                                            ).sum(dim=0)
            reg_loss = torch.cat([reg_loss, alpha_loss])  # [N, 13]
            reg_loss = reg_loss * regression_weight
            num_positives = pos_inds.shape[0]

        else:
            reg_loss = v_anchors.new_zeros((13), requires_grad=True)
            num_positives = 0
        if len(neg_inds) > 0:
            labels[neg_inds, 0] = 0

        cls_loss = self.focal_loss(v_cls_prediction[:, 0:1], labels).mean()

        return cls_loss, reg_loss, num_positives

    def training_forward(self, v_data):
        v_data["features"] = self.core(v_data["image"])

        if self.hparams["model"]["backbone"] == "resnet101":
            cls_preds = torch.cat([self.cls_feature_extraction(feature) for feature in (v_data["features"])], dim=1)
            reg_preds = torch.cat([self.reg_feature_extraction(feature) for feature in v_data["features"]], dim=1)
        elif self.hparams["model"]["backbone"] == "resnet101_fpn":
            cls_preds = [
                self.cls_feature_extraction[0](v_data["features"]['1']),
                self.cls_feature_extraction[1](v_data["features"]['2']),
                self.cls_feature_extraction[2](v_data["features"]['3']),
            ]
            reg_preds = [
                self.reg_feature_extraction[0](v_data["features"]['1']),
                self.reg_feature_extraction[1](v_data["features"]['2']),
                self.reg_feature_extraction[2](v_data["features"]['3']),
            ]
            cls_preds = torch.cat(cls_preds, dim=1)
            reg_preds = torch.cat(reg_preds, dim=1)

        reg_loss = cls_preds.new_full((13,), 0, dtype=torch.float32)
        cls_loss = cls_preds.new_tensor(0., requires_grad=True)

        gt_prediction = None
        num_positives = 1e-6
        for id_batch in range(len(v_data["label"])):
            if len(v_data["label"][id_batch]) > 0:
                cls_loss_item, reg_loss_item, num_positive_item = self.get_training_data(
                    self.anchors.to(cls_preds.device),
                    v_data["training_data"][id_batch],
                    self.anchors_distribution.to(cls_preds.device),
                    cls_preds[id_batch],
                    reg_preds[id_batch],
                )

                reg_loss = reg_loss + reg_loss_item
                cls_loss = cls_loss + cls_loss_item
                num_positives += num_positive_item
                # img = v_data["image"][id_batch].cpu().permute(1, 2, 0).numpy()
                # img=(img-img.min())/(img.max()-img.min())*255
                # img=img.astype(np.uint8)
                # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # for anchor in self.anchors[gt_index_per_anchor > 0]:
                #     pts=list(map(int,anchor))
                #     cv2.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),(255,0,0),3)
                # cv2.imshow("", img)
                # cv2.waitKey()

        reg_loss = reg_loss / num_positives
        cls_loss = cls_loss / (len(v_data["label"]) + 1e-6)

        # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
        ### modified
        loss_2d = torch.mean(reg_loss[:4])
        loss_3d_xyz = torch.mean(reg_loss[4:7])  # cx, cy, z
        loss_3d_sin_cos = torch.mean(reg_loss[7:9])  # sin & cos
        loss_3d_whl = torch.mean(reg_loss[9:12])  # whl
        loss_alpha = reg_loss[12]
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_sin_cos, loss_3d_whl, loss_alpha]

        return cls_loss, reg_loss.mean(), loss_sep

    def get_boxes(self, v_cls_preds, v_reg_preds, v_anchors, v_anchor_mean_std, v_data, v_nms_threshold=0.5):
        assert v_cls_preds.shape[0] == 1
        cls_score = v_cls_preds.sigmoid()[0]
        alpha_score = cls_score[..., 1:1 + 1]
        cls_score = cls_score[:, 0:1]

        anchor = v_anchors  # [N, 4]
        anchor_mean_std_3d = v_anchor_mean_std  # [N, K, 2] -- [2, 16, 2, 6]

        score_thr = self.hparams["det_3d"]["score_threshold"]
        max_score, label = cls_score.max(dim=-1)
        high_score_mask = (max_score > score_thr)
        anchor = anchor[high_score_mask, :]
        cls_score = cls_score[high_score_mask, :]
        alpha_score = alpha_score[high_score_mask, :]
        reg_pred = v_reg_preds[0][high_score_mask, :]
        max_score = max_score[high_score_mask]
        label = label[high_score_mask]
        ### modified
        anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        ###

        bboxes = self._decode(anchor, reg_pred, anchor_mean_std_3d, alpha_score)

        bboxes = self.clipper(bboxes, v_data['image'])

        # BUG HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 0.75->0.5
        keep_inds = nms(bboxes[:, :4], max_score, v_nms_threshold)

        bboxes = bboxes[keep_inds]
        max_score = max_score[keep_inds]
        label = label[keep_inds]

        return max_score, bboxes, label

    def rectify_2d_box(self, v_box2d, v_original_calib, v_calib):
        original_P = v_original_calib
        P2 = v_calib
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]

        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d = v_box2d
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        return bbox_2d

    def test_forward(self, v_data):
        v_data["features"] = self.core(v_data["image"])

        if self.hparams["model"]["backbone"] == "resnet101":
            cls_preds = torch.cat([self.cls_feature_extraction(feature) for feature in (v_data["features"])], dim=1)
            reg_preds = torch.cat([self.reg_feature_extraction(feature) for feature in v_data["features"]], dim=1)
        elif self.hparams["model"]["backbone"] == "resnet101_fpn":
            cls_preds=[
                self.cls_feature_extraction[0](v_data["features"]['1']),
                self.cls_feature_extraction[1](v_data["features"]['2']),
                self.cls_feature_extraction[2](v_data["features"]['3']),
            ]
            reg_preds=[
                self.reg_feature_extraction[0](v_data["features"]['1']),
                self.reg_feature_extraction[1](v_data["features"]['2']),
                self.reg_feature_extraction[2](v_data["features"]['3']),
            ]
            cls_preds = torch.cat(cls_preds, dim=1)
            reg_preds = torch.cat(reg_preds, dim=1)

        reg_loss = cls_preds.new_full((13,), 0, dtype=torch.float32)
        cls_loss = cls_preds.new_tensor(0., requires_grad=True)

        gt_prediction = None
        num_positives = 1e-6
        for id_batch in range(len(v_data["label"])):
            if len(v_data["label"][id_batch]) > 0:
                cls_loss_item, reg_loss_item, num_positive_item = self.get_training_data(
                    self.anchors.to(cls_preds.device),
                    v_data["training_data"][id_batch],
                    self.anchors_distribution.to(cls_preds.device),
                    cls_preds[id_batch],
                    reg_preds[id_batch],
                )

                reg_loss = reg_loss + reg_loss_item
                cls_loss = cls_loss + cls_loss_item
                num_positives += num_positive_item
                # img = v_data["image"][id_batch].cpu().permute(1, 2, 0).numpy()
                # img=(img-img.min())/(img.max()-img.min())*255
                # img=img.astype(np.uint8)
                # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # for anchor in self.anchors[gt_index_per_anchor > 0]:
                #     pts=list(map(int,anchor))
                #     cv2.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),(255,0,0),3)
                # cv2.imshow("", img)
                # cv2.waitKey()

        reg_loss = reg_loss / num_positives
        cls_loss = cls_loss / (len(v_data["label"]) + 1e-6)

        # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
        ### modified
        loss_2d = torch.mean(reg_loss[:4])
        loss_3d_xyz = torch.mean(reg_loss[4:7])  # cx, cy, z
        loss_3d_sin_cos = torch.mean(reg_loss[7:9])  # sin & cos
        loss_3d_whl = torch.mean(reg_loss[9:12])  # whl
        loss_alpha = reg_loss[12]
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_sin_cos, loss_3d_whl, loss_alpha]

        ###

        """
        Debug
        """

        # cls_preds = cls_preds.new_zeros((1, cls_preds.shape[1], 1))
        # reg_preds = cls_preds.new_zeros((1, cls_preds.shape[1], 13))
        # if gt_prediction is not None:
        #     reg_preds[0][v_data["gt_index_per_anchor"][0] > 0] = gt_prediction
        # cls_preds[0][v_data["gt_index_per_anchor"][0] > 0] = 5

        scores, bboxes, cls_indexes = [],[],[]
        for id_batch in range(len(v_data["label"])):
            score, bbox, cls_index = self.get_boxes(cls_preds[id_batch:id_batch+1], reg_preds[id_batch:id_batch+1],  # [1, 58880, 1], [1, 58880, 13]
                                                     self.anchors.to(cls_preds.device),  # [58880, 4]
                                                     self.anchors_distribution.to(cls_preds.device),
                                                     # [2, 16, 2, 6] --> [z, sin2a, cos2a, w,h,l]
                                                     v_data)
            valid_mask = score > self.hparams["det_3d"]["test_score_threshold"]

            scores.append(score[valid_mask])
            bboxes.append(bbox[valid_mask])
            cls_indexes.append(cls_index[valid_mask])

        return {
            "scores": scores,
            "bboxes": bboxes,
            "cls_indexes": cls_indexes,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss.mean(),
            "loss_sep": loss_sep
        }

    def test_set_forward(self, v_data):
        """
        v_data["features"] = self.core(v_data["image"])[0]

        cls_preds = self.cls_feature_extraction(v_data["features"])  # [1, 58880, 1]
        reg_preds = self.reg_feature_extraction(v_data["features"])  # [1, 58880, 12]
        """

        v_data["features"] = self.core(v_data["image"])

        cls_preds = []
        reg_preds = []

        for i in self.hparams["anchor"]["pyramid_levels"]:
            i = i - 3
            cls_pred = self.cls_feature_extraction(v_data["features"][i])  # [1, 58880, 1]
            reg_pred = self.reg_feature_extraction(v_data["features"][i])  # [1, 58880, 12]

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        scores, bboxes, cls_indexes = self.get_boxes(cls_preds, reg_preds,  # [1, 58880, 1], [1, 58880, 13]
                                                     self.anchors.to(cls_preds.device),  # [58880, 4]
                                                     self.anchors_distribution.to(cls_preds.device),
                                                     # [2, 16, 2, 6] --> [z, sin2a, cos2a, w,h,l]
                                                     v_data)

        valid_mask = scores > self.hparams["det_3d"]["test_score_threshold"]

        return {
            "scores": scores[valid_mask],
            "bboxes": bboxes[valid_mask],
            "cls_indexes": cls_indexes[valid_mask],
        }

    def forward(self, v_inputs):
        if self.hparams["trainer"]["evaluate"]:
            return self.test_set_forward(v_inputs)
        if self.training:
            return self.training_forward(v_inputs)
        else:
            return self.test_forward(v_inputs)
