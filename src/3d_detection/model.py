import math
import os
import pickle
import shutil
from copy import deepcopy
import sys

sys.path.append('C:/Users/zihan/Desktop/visualDet3D/visualDet3D')

import cv2
import hydra
import torch
from easydict import EasyDict
from torchvision.models import resnet101, resnet18, resnet50
from torch import nn
import numpy as np
from torchvision.ops import nms
from tqdm import tqdm

from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.networks.heads.losses import SigmoidFocalLoss
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from visualDet3D.utils.utils import cfg_from_file

from scipy.spatial.transform import Rotation as R

# from visualDet3D.networks.heads.anchors import Anchors
from anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file

from torch.nn import functional as F

#from include.resnet_fpn import resnet_fpn_backbone


class Yolo3D(nn.Module):
    def __init__(self, v_cfg):
        super(Yolo3D, self).__init__()
        self.hparams = v_cfg
        # anchors_cfg = EasyDict({
        #     "pyramid_levels": [5], # Final shape of the images 800(original_width) / 2^5(levels) = 25(feature_map_shape)
        #     "strides": [32], # Shift the anchor from the single pixel. 800(original_width) / 25(feature_map_shape) = 32
        #     "sizes": [24], # Base size of the anchors (in original image shape)
        #     "ratios": [.5, 1., 2.], # Different ratio of the anchors
        #     "scales": np.array([2 ** (i / 4.0) for i in range(16)]), # Different area of the anchors, will multiply the base size
        #     "obj_types": ['Car'],
        #     "filter_anchors": False,
        #     "filter_y_threshold_min_max": None,
        #     "filter_x_threshold": None,
        #     "anchor_prior_channel": 14, #(x, y, z, w, h, l, ry) *2
        # })

        anchors_cfg = EasyDict({
            "pyramid_levels": list(v_cfg["anchor"]["pyramid_levels"]),
            "v_strides": list(v_cfg["anchor"]["v_strides"]),
            # Shift the anchor from the single pixel. 800(original_width) / 25(feature_map_shape) = 32
            "sizes": list(v_cfg["anchor"]["sizes"]),  # Base size of the anchors (in original image shape)
            "ratios": list(v_cfg["anchor"]["ratios"]),  # Different ratio of the anchors
            "scales": list(v_cfg["anchor"]["scales"]),
            # Different area of the anchors, will multiply the base size
        })

        self.head_loss_cfg = EasyDict(
            filter_anchor=False,
            fg_iou_threshold=0.5,
            bg_iou_threshold=0.4,
            # L1_regression_alpha=5 ** 2,
            # focal_loss_gamma=2.0,
            match_low_quality=False,
            # balance_weight=[20.0],
            # regression_weight=[1, 1, 1, 1, 1, 1, 1, 1, 3, 0.5, 0.5, 0.5, 1],
        )

        self.network_cfg = EasyDict(
            num_features_in=1024,
            #reg_feature_size=512,
            #cls_feature_size=256,
            reg_feature_size=1024,
            cls_feature_size=512,

            num_cls_output=1,
            #num_reg_output=13,  # (x1, y1, x2, y2, x3d_center, y3d_center, x, y, z, w, h, l, ry)
            num_reg_output=12,  # (x1, y1, x2, y2, x3d_center, y3d_center, z, sin2a, cos2a, w, h, l)
        )

        # self.stds = [1, 1, 1, 1, 1, 1, .1, .1, .1, .1, .1, .1, 1]
        self.stds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        train_transform = [
            EasyDict(type_name='ConvertToFloat'),
            ###
            #EasyDict(type_name='PhotometricDistort',
            #         keywords=EasyDict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5,
            #                           saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
            #EasyDict(type_name='CropTop', keywords=EasyDict(crop_top_index=100)),
            ###
            EasyDict(type_name='Resize',
                     keywords=EasyDict(size=(v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))),
            ###
            #EasyDict(type_name='RandomMirror', keywords=EasyDict(mirror_prob=0.5)),
            ###
            EasyDict(type_name='Normalize',
                     keywords=EasyDict(mean=np.array([0.485, 0.456, 0.406]), stds=np.array([0.229, 0.224, 0.225])))
        ]
        test_transform = [
            EasyDict(type_name='ConvertToFloat'),
            ###
            #EasyDict(type_name='CropTop', keywords=EasyDict(crop_top_index=100)),
            ###
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
                                                     dtype=np.float64)  # [z, sin2a, cos2a, w, h, l]
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
                        data_frame.original_calib = deepcopy(calib)
                        data_frame.original_shape = image.shape

                        preprocess = self.train_preprocess if data_split == "training" else self.test_preprocess
                        image, P2, label_tr = preprocess(image, labels=deepcopy(label.data), p2=deepcopy(calib.P2))
                        #data_frame.image = image
                        data_frame.image_file = data_frame.image2_path
                        calib.P2 = P2
                        data_frame.calib = calib
                        label.data = label_tr
                        data_frame.label = label

                        # Do the filtering
                        data_frame.label.data = [
                            item for item in data_frame.label.data if item.type == "Car" and item.z < 200]

                        #data_frame.label.data = [
                        #    item for item in data_frame.label.data if item.type == "Car" and item.occluded < 2 and item.z > 3]

                        if len(data_frame.label.data) == 0:
                            data_frame.corners_in_camera_coordinates = torch.zeros((0, 9, 4)).float()
                            data_frame.bbox2d = torch.zeros((0, 4)).float()
                            data_frame.bbox3d_img_center = torch.zeros((0, 2)).float()
                            ### modified
                            #data_frame.bbox3d = torch.zeros((0, 7)).float()
                            data_frame.bbox3d = torch.zeros((0, 6)).float()
                            ###
                            data_frame.data = torch.zeros((0, 13)).float()
                            data_frames.append(data_frame)
                            continue

                        # Replace the 2d bounding box with re-projection
                        # NOTE: Currently fix the shape
                        data_frame.corners_in_camera_coordinates = self.project_box3d_to_img(data_frame,
                                                                                             v_cfg["det_3d"][
                                                                                                 "rotate_pitch_when_project"])
                        if False:
                            data_frame.bbox2d = torch.stack([torch.tensor((
                                torch.min(item[:8, 0]),
                                torch.min(item[:8, 1]),
                                torch.max(item[:8, 0]),
                                torch.max(item[:8, 1]))
                            ) for item in data_frame.corners_in_camera_coordinates
                            ], dim=0
                            )
                        else:
                            data_frame.bbox2d = torch.stack([torch.tensor((
                                item.bbox_l,
                                item.bbox_t,
                                item.bbox_r,
                                item.bbox_b,)
                            ) for item in data_frame.label.data
                            ], dim=0
                            )

                        data_frame.bbox3d_img_center = torch.stack(
                            [item[8, :] for item in data_frame.corners_in_camera_coordinates],
                            dim=0
                        )

                        # ### Debug
                        # from matplotlib import pyplot as plt
                        # for box in data_frame.bbox2d:
                        #     x_min, y_min, x_max, y_max = map(int, box.numpy().tolist())
                        #     image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [0, 255, 0],5)
                        # plt.imshow(image)
                        # plt.show()
                        # ###

                        # Fix the height of KITTI label. (Y is the bottom of the car)
                        for item in data_frame.label.data:
                            item.y = item.y - item.h * 0.5
                        # Find the 3d bbox
                        ### modified
                        data_frame.bbox3d = torch.stack([torch.tensor((
                            item.z, np.sin(2 * item.alpha), np.cos(2 * item.alpha),
                            item.w, item.h, item.l
                        )) for item in data_frame.label.data
                        ], dim=0)

                        ###
                        """
                        data_frame.bbox3d = torch.stack([torch.tensor((
                            item.x, item.y, item.z,
                            item.w, item.h, item.l,
                            item.ry
                        )) for item in data_frame.label.data
                        ], dim=0)
                        """
                        total_objects += len(data_frame.label.data)

                        ### modified
                        data = np.array(
                            [
                                [item.z, np.sin(2 * item.alpha), np.cos(2 * item.alpha), item.w, item.h, item.l] # [z, sin(2a), cos(2a), w, h, l]
                                    for item in data_frame.label.data
                            ]
                        ) #[N, 6]


                        if data.any():
                            uniform_sum_each_type[0, :] += np.sum(data, axis=0)
                            uniform_square_each_type[0, :] += np.sum(data ** 2, axis=0)
                        ###

                        """
                        data = np.array(
                            [
                                [item.x, item.y, item.z, item.w, item.h, item.l, item.ry] for item in
                                data_frame.label.data
                            ]
                        )
                        if data.any():
                            # uniform_sum_each_type.append(np.sum(data, axis=0))
                            # uniform_square_each_type.append(np.sum(data ** 2, axis=0))
                            data_statics.append(data)
                        """


                        # NOTE: Currently fix the shape
                        # anchors, _ = anchor_manager(torch.ones((1, 3, 800, 800)), torch.tensor(calib.P2).reshape([-1, 3, 4]))
                        anchors = anchor_manager((v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))
                        ### modified
                        #bbox3d = torch.tensor(data)
                        bbox3d = torch.tensor(
                            [[item.x, item.y, item.z, np.sin(2 * item.alpha), np.cos(2 * item.alpha)] for item in data_frame.label.data]).cuda()

                        ###
                        usable_anchors = anchors

                        IoUs = calc_iou(usable_anchors, data_frame.bbox2d)  # [num_anchors, num_gt]
                        IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                        IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                        num_usable_object = torch.sum(IoU_max > self.head_loss_cfg.fg_iou_threshold).item()
                        total_usable_objects += num_usable_object

                        positive_anchors_mask = IoU_max_anchor > self.head_loss_cfg.fg_iou_threshold
                        positive_ground_truth_3d = bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                        ### modified
                        used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy()  # [x1, y1, x2, y2]
                        sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                        for k in range(len(sizes_int)):
                            examine[0, sizes_int[k], ratio_int[k]] += 1
                            sums[0, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                            squared[0, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

                        ###

                        data_frame.data = torch.cat([
                            data_frame.bbox2d,
                            data_frame.bbox3d_img_center,
                            data_frame.bbox3d,
                        ], dim=1) # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
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

                        ### modified
                        """
                        data_statics = np.concatenate(data_statics, axis=0).astype(np.float32)
                        global_mean = (data_statics.max(0) + data_statics.min(0)) / 2
                        global_var = (data_statics.max(0) - data_statics.min(0)) / 2
                        """

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


                        ###
                        print("Pre calculation done, save it now")

                        npy_file = os.path.join(preprocessed_path, data_split, 'anchor_mean_std_Car.npy')
                        ### modified
                        #mean_std = np.stack([global_mean, global_var, ])
                        mean_std = np.stack([avg, std, ])
                        ###
                        np.save(npy_file, mean_std)
                        # [[mean, std], scale, ratio, [z, sin2a, cos2a, w,h,l]] -- shape: [2, 16, 2, 6]
                    anchor_manager_with_distribution = Anchors(
                        preprocessed_path=os.path.join(preprocessed_path, "training", 'anchor_mean_std_Car.npy'),
                        **anchors_cfg)
                    anchors = anchor_manager_with_distribution(
                        (v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))
                    anchors = anchors
                    for _, data_frame in tqdm(enumerate(data_frames)):
                        gt_index_per_anchor = self._assign(anchors, data_frame.bbox2d,
                                                           bg_iou_threshold=self.head_loss_cfg["bg_iou_threshold"],
                                                           fg_iou_threshold=self.head_loss_cfg["fg_iou_threshold"],
                                                           min_iou_threshold=0.0,
                                                           match_low_quality=self.head_loss_cfg[
                                                               "match_low_quality"],
                                                           gt_max_assign_all=True,
                                                           )

                        # if (gt_index_per_anchor>0).sum()>0:
                        #     img = cv2.imread(data_frame.image2_path)
                        #     for box in data_frame.bbox2d:
                        #         pts = list(map(int, box))
                        #         cv2.rectangle(img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 3)
                        #
                        #     for box in anchors[gt_index_per_anchor > 0]:
                        #         pts = list(map(int, box))
                        #         cv2.rectangle(img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 0, 255), 1)

                        # cv2.imshow("",img)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        pass
                        data_frame.gt_index_per_anchor = gt_index_per_anchor

                    pkl_file = os.path.join(preprocessed_path, data_split, 'imdb.pkl')
                    pickle.dump(data_frames, open(pkl_file, 'wb'))

        anchor_manager_with_distribution = Anchors(
            preprocessed_path=os.path.join(preprocessed_path, "training", 'anchor_mean_std_Car.npy'),
            **anchors_cfg)
        self.anchors = anchor_manager_with_distribution((v_cfg["model"]["img_shape_y"], v_cfg["model"]["img_shape_x"]))
        self.anchors_distribution = anchor_manager_with_distribution.anchors_mean_std

        self.init_layers(self.network_cfg["num_features_in"], self.network_cfg["cls_feature_size"],
                         self.network_cfg["reg_feature_size"], anchor_manager_with_distribution.num_anchor_per_scale,
                         self.network_cfg["num_cls_output"], self.network_cfg["num_reg_output"])

        self.focal_loss = SigmoidFocalLoss(gamma=2.0, balance_weights=torch.tensor([.5]))
        # self.focal_loss = SigmoidFocalLoss()

    def init_layers(self,
                    v_num_features_in, v_cls_feature_size, v_reg_feature_size, v_num_anchors,
                    v_num_cls_output, v_num_reg_output):
        resnet = resnet101(pretrained=True)
        self.core = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4,
        )
        # self.core.requires_grad_(False)
        # self.core = resnet_fpn_backbone(
        #     "resnet50",
        #     pretrained=True,
        #     trainable_layers=5,
        #     output_channel=1024
        # )

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
        self.reg_feature_extraction = nn.Sequential(
            ModulatedDeformConvPack(v_num_features_in, v_reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(v_reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(v_reg_feature_size, v_reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(v_reg_feature_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(v_reg_feature_size, v_num_anchors * v_num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(v_num_reg_output)
        )

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
                    [-v_dataframe.label.data[id_batch].l / 2,
                     0,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2,
                     0,
                     v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2,
                     0,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2,
                     0,
                     v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2,
                     -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [-v_dataframe.label.data[id_batch].l / 2,
                     -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2,
                     -v_dataframe.label.data[id_batch].h,
                     -v_dataframe.label.data[id_batch].w / 2],
                    [v_dataframe.label.data[id_batch].l / 2,
                     -v_dataframe.label.data[id_batch].h,
                     v_dataframe.label.data[id_batch].w / 2],
                    [0,
                     -v_dataframe.label.data[id_batch].h / 2,
                     0],

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
    # modified
    sampled_gt_bboxes: (x1, y1, x2, y2, cx, cy, z, w, h, l, alpha)
    anchors_3d_mean_std: (z, sin2a, cos2a, w, h, l)
    outputs: (x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l)
    """

    def _encode(self, sampled_anchors, sampled_gt_bboxes, selected_anchors_3d):
        # Sampled_gt_bboxes: GT Box after sample: x1, y1, x2, y2, x3d_projected, y3d_projected, x3d, y3d, z3d, w3d, h3d, l3d, ry
        # modify: new GT box after sample: (x1, y1, x2, y2, x3d_projected, y3d_projected, z3d, w3d, h3d, l3d, ry)
        # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
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
        targets_cdx_projected = (sampled_gt_bboxes[:, 5] - px) / pw
        targets_cdy_projected = (sampled_gt_bboxes[:, 6] - py) / ph

        targets_cdz = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[:, 0, 0]) / selected_anchors_3d[:, 0, 1]
        targets_cd_sin = (torch.sin(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 1, 0]) / selected_anchors_3d[
                                                                                                    :, 1, 1]
        targets_cd_cos = (torch.cos(sampled_gt_bboxes[:, 11] * 2) - selected_anchors_3d[:, 2, 0]) / selected_anchors_3d[
                                                                                                    :, 2, 1]
        targets_w3d = (sampled_gt_bboxes[:, 8] - selected_anchors_3d[:, 3, 0]) / selected_anchors_3d[:, 3, 1]
        targets_h3d = (sampled_gt_bboxes[:, 9] - selected_anchors_3d[:, 4, 0]) / selected_anchors_3d[:, 4, 1]
        targets_l3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[:, 5, 0]) / selected_anchors_3d[:, 5, 1]

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_cdx_projected, targets_cdy_projected, targets_cdz,
                               targets_cd_sin, targets_cd_cos,
                               targets_w3d, targets_h3d, targets_l3d), dim=1)

        stds = targets.new([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1])

        targets = targets.div_(stds)
        ###

        """
        targets_cdx_projected = (sampled_gt_bboxes[:, 4] - px) / pw  # shift of the projected gt 3d bbox center x
        targets_cdy_projected = (sampled_gt_bboxes[:, 5] - py) / ph  # shift of the projected gt 3d bbox center y

        targets_cdz = (sampled_gt_bboxes[:, 8] - selected_anchors_3d[0, 2]) / selected_anchors_3d[1, 2]
        targets_cdy = (sampled_gt_bboxes[:, 7] - selected_anchors_3d[0, 1]) / selected_anchors_3d[1, 1]
        targets_cdx = (sampled_gt_bboxes[:, 6] - selected_anchors_3d[0, 0]) / selected_anchors_3d[1, 0]
        targets_w3d = (sampled_gt_bboxes[:, 9] - selected_anchors_3d[0, 3]) / selected_anchors_3d[1, 3]
        targets_h3d = (sampled_gt_bboxes[:, 10] - selected_anchors_3d[0, 4]) / selected_anchors_3d[1, 4]
        targets_l3d = (sampled_gt_bboxes[:, 11] - selected_anchors_3d[0, 5]) / selected_anchors_3d[1, 5]
        targets_ry = (sampled_gt_bboxes[:, 12] - selected_anchors_3d[0, 6]) / selected_anchors_3d[1, 6]
        # targets_cdz = sampled_gt_bboxes[:, 8] / 100
        # targets_cdy = sampled_gt_bboxes[:, 7] / 100
        # targets_cdx = sampled_gt_bboxes[:, 6] / 100
        # targets_w3d = sampled_gt_bboxes[:, 9] / 10
        # targets_h3d = sampled_gt_bboxes[:, 10] / 10
        # targets_l3d = sampled_gt_bboxes[:, 11] / 10
        # targets_ry = sampled_gt_bboxes[:, 12]

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_cdx_projected, targets_cdy_projected,
                               targets_cdx, targets_cdy, targets_cdz,
                               targets_w3d, targets_h3d, targets_l3d,
                               targets_ry
                               ), dim=1)

        stds = targets.new(self.stds)

        targets = targets.div_(stds)   
        """

        return targets

    """
    :param
    v_prediction: (x1,y1,x2,y2,x3d_projected,y3d_projected, x3d, y3d, z3d, w3d, h3d, l3d, ry)
    anchors_3d_mean_std: (x3d, y3d, z3d, w3d, h3d, l3d, ry)
    """

    """
    # modified
    v_prediction: (x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l)
    anchors_3d_mean_std: (z, sin2a, cos2a, w, h, l)
    outputs: (x1, y1, x2, y2, cx, cy, z, w, h, l, alpha)
    """

    def _decode(self, v_boxes_2d, v_prediction, anchors_3d_mean_std, label_index):
        #std = torch.tensor(self.stds, dtype=torch.float32, device=v_boxes_2d.device)
        std = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 1, 1, 1, 1, 1, 1], dtype=torch.float32, device=v_boxes_2d.device)

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

        # 3D Bounding box
        # pred_x = v_prediction[..., 6] * 100 * std[6]
        # pred_y = v_prediction[..., 7] * 100 * std[7]
        # pred_z = v_prediction[..., 8] * 100 * std[8]
        #
        # pred_w = v_prediction[..., 9] * 10 * std[9]
        # pred_h = v_prediction[..., 10] * 10 * std[10]
        # pred_l = v_prediction[..., 11] * 10 * std[11]
        #
        # pred_ry = v_prediction[..., 12] * std[12]

        # 3D Bounding box
        """ # newer version
        pred_x = (v_prediction[..., 6] * anchors_3d_mean_std[1, 0] + anchors_3d_mean_std[0, 0]) * std[6]  # [N, 6]
        pred_y = (v_prediction[..., 7] * anchors_3d_mean_std[1, 1] + anchors_3d_mean_std[0, 1]) * std[7]  # [N, 6]
        pred_z = (v_prediction[..., 8] * anchors_3d_mean_std[1, 2] + anchors_3d_mean_std[0, 2]) * std[8]  # [N, 6]

        pred_w = (v_prediction[..., 9] * anchors_3d_mean_std[1, 3] + anchors_3d_mean_std[0, 3]) * std[9]
        pred_h = (v_prediction[..., 10] * anchors_3d_mean_std[1, 4] + anchors_3d_mean_std[0, 4]) * std[10]
        pred_l = (v_prediction[..., 11] * anchors_3d_mean_std[1, 5] + anchors_3d_mean_std[0, 5]) * std[11]

        pred_ry = (v_prediction[..., 12] * anchors_3d_mean_std[1, 6] + anchors_3d_mean_std[0, 6]) * std[12]

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2,
                                  pred_center_x1_projected, pred_center_y1_projected,
                                  pred_x, pred_y, pred_z,
                                  pred_w, pred_h, pred_l, pred_ry], dim=1)
        """
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

        ###
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
            #pos_gt_bboxes = v_gt_data.new_zeros([0, 13])
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

    def _get_anchor_3d(self, anchors, anchor_mean_std_3d, assigned_labels):
        """
            anchors: [N_pos, 4] only positive anchors
            anchor_mean_std_3d: [N_pos, C, K=6, 2]
            assigned_labels: torch.Longtensor [N_pos, ]

            return:
                selected_mask = torch.bool [N_pos, ]
                selected_anchors_3d:  [N_selected, K, 2]
        """
        num_classes = 1
        one_hot_mask = torch.nn.functional.one_hot(assigned_labels, num_classes).bool()
        selected_anchor_3d = anchor_mean_std_3d[one_hot_mask]

        return torch.ones_like(selected_anchor_3d[:, 0, 0]).long(), selected_anchor_3d

    def loss(self, cls_scores, reg_preds, anchors, v_anchor_mean_std, v_data):
        batch_size = cls_scores.shape[0]

        cls_loss = []
        reg_loss = []
        gt_pos_bbox_targets = []
        number_of_positives = []
        for id_batch in range(batch_size):
            reg_pred = reg_preds[id_batch]
            cls_score = cls_scores[id_batch]

            if len(v_data["label"][id_batch]) == 0:
                cls_loss.append(reg_preds.new_tensor(0, requires_grad=True))
                reg_loss.append(reg_preds.new_zeros(self.network_cfg["num_reg_output"]))
                # reg_loss.append(reg_preds.new_zeros(4,requires_grad=True))
                number_of_positives.append(0)
                continue

            gt_index_per_anchor = self._assign(anchors[0], v_data["bbox2d"][id_batch],
                                               bg_iou_threshold=self.head_loss_cfg["bg_iou_threshold"],
                                               fg_iou_threshold=self.head_loss_cfg["fg_iou_threshold"],
                                               min_iou_threshold=0.0,
                                               match_low_quality=self.head_loss_cfg["match_low_quality"],
                                               gt_max_assign_all=True,
                                               )
            # GT Box after sample: x1,y1,x2,y2,x3d_projected,y3d_projected, x3d, y3d, z3d, w3d, h3d, l3d, ry
            sampling_result_dict = self._sample(assignement_result_dict, anchors[0], v_data, id_batch)

            num_valid_anchors = anchors[0].shape[0]
            labels = anchors[0].new_full((num_valid_anchors, 1),
                                         -1,  # -1 not computed, binary for each class
                                         dtype=torch.float)

            pos_inds = sampling_result_dict['pos_inds']
            neg_inds = sampling_result_dict['neg_inds']

            if len(pos_inds) > 0:
                pos_assigned_gt_label = torch.tensor(
                    v_data["label"][id_batch], device=sampling_result_dict['pos_assigned_gt_inds'].device)[
                    sampling_result_dict['pos_assigned_gt_inds']].long()

                """
                Only one class here, so do not need to select
                """
                # selected_mask, selected_anchor_3d = self._get_anchor_3d(
                #     sampling_result_dict['pos_bboxes'],
                #     v_anchor_mean_std[pos_inds],
                #     pos_assigned_gt_label,
                # )
                selected_anchor_3d = v_anchor_mean_std[pos_inds][:, 0]
                if len(selected_anchor_3d) > 0:
                    pos_inds = pos_inds
                    pos_bboxes = sampling_result_dict['pos_bboxes']
                    pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes']
                    pos_assigned_gt = sampling_result_dict['pos_assigned_gt_inds']

                    pos_bbox_targets = self._encode(
                        pos_bboxes, pos_gt_bboxes, selected_anchor_3d
                    )  # [N, 12], [N, 1]
                    label_index = pos_assigned_gt_label
                    labels[pos_inds, :] = 0
                    labels[pos_inds, label_index] = 1

                    if False:
                        pos_prediction_decoded = self._decode(pos_anchor, reg_pred[pos_inds], v_anchor_mean_std,
                                                              label_index, pos_alpha_score)
                        pos_target_decoded = self._decode(pos_anchor, pos_bbox_targets, anchors_3d_mean_std,
                                                          label_index, pos_alpha_score)

                        reg_loss.append(
                            (self.loss_bbox(pos_prediction_decoded, pos_target_decoded) * self.regression_weight).mean(
                                dim=0))
                    else:
                        # reg_loss_j = F.l1_loss(pos_bbox_targets[:, :4], reg_pred[pos_inds][:, :4], reduction='none')
                        reg_loss_j = F.l1_loss(pos_bbox_targets, reg_pred[pos_inds], reduction='none')
                        # alpha_loss_j = self.alpha_loss(pos_alpha_score, targets_alpha_cls)
                        # loss_j = torch.cat([reg_loss_j, alpha_loss_j], dim=1) * self.regression_weight  # [N, 13]
                        reg_loss.append(reg_loss_j.mean(dim=0))  # [13]
                        number_of_positives.append(len(v_data["label"][id_batch]))
                        # gt_pos_bbox_targets.append(pos_bbox_targets)
            else:
                reg_loss.append(reg_preds.new_zeros(self.network_cfg["num_reg_output"]))
                # reg_loss.append(reg_preds.new_zeros(4))
                number_of_positives.append(len(v_data["label"][id_batch]))

            if len(neg_inds) > 0:
                labels[neg_inds, :] = 0

            cls_loss.append(self.focal_loss(cls_score, labels).sum() / (len(pos_inds) + len(neg_inds)))

        weights = reg_pred.new(number_of_positives).unsqueeze(1)  # [B, 1]
        cls_loss = torch.stack(cls_loss).mean()
        reg_loss = torch.stack(reg_loss, dim=0)  # [B, 13]

        weighted_regression_losses = torch.sum(weights * reg_loss / (torch.sum(weights) + 1e-6), dim=0)
        reg_loss = weighted_regression_losses.mean(dim=0)
        return cls_loss, reg_loss

    def get_training_data(self, v_gt_index_per_anchor, v_anchors, v_data, v_anchor_mean_std):

        # modify: new GT box after sample: (x1, y1, x2, y2, x3d_projected, y3d_projected, z3d, w3d, h3d, l3d, ry)

        # GT Box after sample: x1,y1,x2,y2,x3d_projected,y3d_projected, x3d, y3d, z3d, w3d, h3d, l3d, ry
        sampling_result_dict = self._sample(v_gt_index_per_anchor, v_anchors, v_data)

        labels = v_anchors.new_full((self.anchors.shape[0], 1),
                                    -1,  # -1 not computed, binary for each class
                                    dtype=torch.float)

        pos_inds = sampling_result_dict['pos_inds']
        neg_inds = sampling_result_dict['neg_inds']

        if len(pos_inds) > 0:
            pos_bboxes = sampling_result_dict['pos_bboxes']
            pos_gt_bboxes = sampling_result_dict['pos_gt_bboxes']
            ### modified
            v_anchor_mean_std = v_anchor_mean_std[pos_inds, :]
            ###
            pos_bbox_targets = self._encode(
                pos_bboxes, pos_gt_bboxes, v_anchor_mean_std
            )  # [N, 12], [N, 1]
            labels[pos_inds, 0] = 1

        else:
            ### modified
            pos_bbox_targets = v_anchors.new_zeros((0, 12))
            ###
            #pos_bbox_targets = v_anchors.new_zeros((0, 13))

        if len(neg_inds) > 0:
            labels[neg_inds, 0] = 0

        return pos_bbox_targets, labels

    def training_forward(self, v_data):
        """
        Args: v_data
            calib: (B,3,4) P2 matrix
            image: (B,3,800,800) images
            label: [[],[]] list of indexes of the label
            bbox2d: [(N1, 4), (N2, 4)] list of objects' 2d bounding boxes. N1,N2... is the number of objects contained
                in one image. (x1,y1,x2,y2)
            bbox3d: [(N1, 7), (N2, 7)] list of objects' 3d bounding boxes. N1,N2... is the number of objects contained
                in one image. (x, y, z, w, h, l, ry)
            bbox3d_img_center: [(N1, 2), (N2, 7)] list of 3d bounding boxes projected center. N1,N2... is the number of
            objects contained in one image. (x, y)

        Returns:
            cls_loss, reg_loss: tensor of losses
        """
        v_data["features"] = self.core(v_data["image"])

        cls_preds = self.cls_feature_extraction(v_data["features"])
        reg_preds = self.reg_feature_extraction(v_data["features"])

        # cls_preds = []
        # reg_preds = []
        # for feature_name, feature_map in v_data["features"].items():
        #     if feature_name in ["2"]:
        #         cls_pred = self.cls_feature_extraction(feature_map)
        #         reg_pred = self.reg_feature_extraction(feature_map)
        #         cls_preds.append(cls_pred)
        #         reg_preds.append(reg_pred)
        # cls_preds = torch.cat(cls_preds, dim=1)
        # reg_preds = torch.cat(reg_preds, dim=1)

        cls_loss = cls_preds.new_tensor(0., requires_grad=True)
        reg_loss = cls_preds.new_tensor(0., requires_grad=True)

        #regression_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 3, 0.5, 0.5, 0.5, 1], device='cuda:0')
        #regression_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')

        #reg_loss_mat = cls_preds.new_tensor(0., requires_grad=True)

        ### modified
        reg_loss_mat = torch.zeros(12, device='cuda:0', requires_grad=True)
        ###
        #reg_loss_mat = torch.zeros(13, device='cuda:0', requires_grad=True)

        num_positives = 1
        for id_batch in range(len(v_data["label"])):
            if len(v_data["label"][id_batch]) > 0:
                gt_index_per_anchor = v_data["gt_index_per_anchor"][id_batch]
                if (gt_index_per_anchor > 0).sum() == 0:
                    continue
                pos_bbox_targets, labels = self.get_training_data(
                    gt_index_per_anchor,
                    self.anchors.to(cls_preds.device),
                    v_data["training_data"][id_batch],
                    self.anchors_distribution.to(cls_preds.device),
                )
                cls_loss = cls_loss + self.focal_loss(cls_preds[id_batch], labels).mean()
                """
                reg_loss_batch = F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                                pos_bbox_targets[:, :],
                                                reduction='none'
                                                )[0]

                reg_loss = reg_loss + reg_loss_batch.mean(dim=0) * (gt_index_per_anchor > 0).sum()

                reg_loss_mat = reg_loss_mat + reg_loss_batch * (gt_index_per_anchor > 0).sum()

                """
                reg_loss = reg_loss + F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                                pos_bbox_targets[:, :]
                                                ) * (gt_index_per_anchor > 0).sum()

                reg_loss_mat = reg_loss_mat + F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                                pos_bbox_targets[:, :],
                                                reduction='none'
                                                )[0] * (gt_index_per_anchor > 0).sum()

                num_positives += (gt_index_per_anchor > 0).sum()

        reg_loss = reg_loss / num_positives
        cls_loss = cls_loss / len(v_data["label"]) * 100 # divide with batch size

        # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
        ### modified
        loss_2d = torch.mean(reg_loss_mat[:4]) / num_positives
        loss_3d_xyz = torch.mean(reg_loss_mat[6]) / num_positives  # z
        loss_3d_whl = torch.mean(reg_loss_mat[7:9]) / num_positives  # sin_cos
        loss_3d_ry = torch.mean(reg_loss_mat[9:]) / num_positives  # whl
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_whl, loss_3d_ry]
        ###

        """
        loss_2d = torch.mean(reg_loss_mat[:4]) / num_positives
        loss_3d_xyz = torch.mean(reg_loss_mat[6:9]) / num_positives
        loss_3d_whl = torch.mean(reg_loss_mat[9:12]) / num_positives
        loss_3d_ry = torch.mean(reg_loss_mat[12]) / num_positives
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_whl, loss_3d_ry]
        """

        return cls_loss, reg_loss, loss_sep

    def get_boxes(self, v_cls_preds, v_reg_preds, v_anchors, v_anchor_mean_std, v_data):
        assert v_cls_preds.shape[0] == 1
        cls_score = v_cls_preds.sigmoid()[0]

        anchor = v_anchors  # [N, 4]
        anchor_mean_std_3d = v_anchor_mean_std  # [N, K, 2] -- [2, 16, 2, 6]

        score_thr = self.hparams["det_3d"]["score_threshold"]
        max_score, label = cls_score.max(dim=-1)
        high_score_mask = (max_score > score_thr)
        anchor = anchor[high_score_mask, :]
        cls_score = cls_score[high_score_mask, :]
        reg_pred = v_reg_preds[0][high_score_mask, :]
        max_score = max_score[high_score_mask]
        label = label[high_score_mask]
        ### modified
        anchor_mean_std_3d = anchor_mean_std_3d[high_score_mask, :]
        ###

        bboxes = self._decode(anchor, reg_pred, anchor_mean_std_3d, label)

        keep_inds = nms(bboxes[:, :4], max_score, 0.5)

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
        """
        Args: v_data
            calib: (B,3,4) P2 matrix
            image: (B,3,800,800) images
            label: [[],[]] list of indexes of the label
            bbox2d: [(N1, 4), (N2, 4)] list of objects' 2d bounding boxes. N1,N2... is the number of objects contained
                in one image. (x1,y1,x2,y2)
            bbox3d: [(N1, 7), (N2, 7)] list of objects' 3d bounding boxes. N1,N2... is the number of objects contained
                in one image. (x, y, z, w, h, l, ry)
            bbox3d_img_center: [(N1, 2), (N2, 7)] list of 3d bounding boxes projected center. N1,N2... is the number of
            objects contained in one image. (x, y)

        Returns:
            cls_loss, reg_loss: tensor of losses
        """
        v_data["features"] = self.core(v_data["image"])

        cls_preds = self.cls_feature_extraction(v_data["features"]) # [1, 58880, 1]
        reg_preds = self.reg_feature_extraction(v_data["features"]) # [1, 58880, 12]

        # cls_preds = []
        # reg_preds = []
        # for feature_name, feature_map in v_data["features"].items():
        #     if feature_name in ["2"]:
        #         cls_pred = self.cls_feature_extraction(feature_map)
        #         reg_pred = self.reg_feature_extraction(feature_map)
        #         cls_preds.append(cls_pred)
        #         reg_preds.append(reg_pred)
        # cls_preds = torch.cat(cls_preds, dim=1)
        # reg_preds = torch.cat(reg_preds, dim=1)

        #regression_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 3, 0.5, 0.5, 0.5, 1], device='cuda:0')
        #regression_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')

        reg_loss = cls_preds.new_tensor(0.)
        cls_loss = cls_preds.new_tensor(0.)
        reg_loss_mat = torch.zeros(12, device='cuda:0')

        gt_prediction = None
        num_positives=1
        for id_batch in range(len(v_data["label"])):
            if len(v_data["label"][id_batch]) > 0:
                gt_index_per_anchor = v_data["gt_index_per_anchor"][id_batch]
                if (gt_index_per_anchor > 0).sum() == 0:
                    continue
                pos_bbox_targets, labels = self.get_training_data(
                    gt_index_per_anchor,
                    self.anchors.to(cls_preds.device),
                    v_data["training_data"][id_batch],
                    self.anchors_distribution.to(cls_preds.device),
                )
                gt_prediction = pos_bbox_targets

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
                """
                cls_loss = cls_loss + self.focal_loss(cls_preds[id_batch], labels).mean()

                reg_loss_batch = F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                           pos_bbox_targets[:, :],
                                           reduction='none'
                                           )[0]

                reg_loss = reg_loss + reg_loss_batch.mean(dim=0) * (gt_index_per_anchor > 0).sum()

                reg_loss_mat = reg_loss_mat + reg_loss_batch * (gt_index_per_anchor > 0).sum()

                """
                reg_loss = reg_loss + F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                                pos_bbox_targets[:, :]
                                                ) * (gt_index_per_anchor > 0).sum()

                reg_loss_mat = reg_loss_mat + F.l1_loss(reg_preds[id_batch][gt_index_per_anchor > 0][:, :],
                                            pos_bbox_targets[:, :],
                                            reduction='none'
                                            )[0] * (gt_index_per_anchor > 0).sum()

                num_positives += (gt_index_per_anchor > 0).sum()

        reg_loss = reg_loss / num_positives
        cls_loss = cls_loss / len(v_data["label"]) * 100 # divide with batch size

        """
        loss_2d = torch.mean(reg_loss_mat[:4]) / num_positives
        loss_3d_xyz = torch.mean(reg_loss_mat[6:9]) / num_positives
        loss_3d_whl = torch.mean(reg_loss_mat[9:12]) / num_positives
        loss_3d_ry = torch.mean(reg_loss_mat[12]) / num_positives
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_whl, loss_3d_ry]
        """

        # [x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
        ### modified
        loss_2d = torch.mean(reg_loss_mat[:4]) / num_positives
        loss_3d_xyz = torch.mean(reg_loss_mat[6]) / num_positives  # z
        loss_3d_whl = torch.mean(reg_loss_mat[7:9]) / num_positives # sin_cos
        loss_3d_ry = torch.mean(reg_loss_mat[9:]) / num_positives # whl
        loss_sep = [loss_2d, loss_3d_xyz, loss_3d_whl, loss_3d_ry]

        ###

        """
        Debug
        """
        # cls_preds = cls_preds.new_zeros((1, cls_preds.shape[1], 1))
        # reg_preds = cls_preds.new_zeros((1, cls_preds.shape[1], 13))
        # if gt_prediction is not None:
        #     reg_preds[0][v_data["gt_index_per_anchor"][0] > 0] = gt_prediction
        # cls_preds[0][v_data["gt_index_per_anchor"][0] > 0] = 5

        scores, bboxes, cls_indexes = self.get_boxes(cls_preds, reg_preds,                           # [1, 58880, 1], [1, 58880, 13]
                                                     self.anchors.to(cls_preds.device),              # [58880, 4]
                                                     self.anchors_distribution.to(cls_preds.device), # [2, 16, 2, 6] --> [z, sin2a, cos2a, w,h,l]
                                                     v_data)

        """
        Compute alpha (needed in evaluation)
        
        (x1,y1,x2,y2,x3d_projected,y3d_projected, x3d, y3d, z3d, w3d, h3d, l3d, ry)
        (x1,y1,x2,y2,x3d_projected,y3d_projected, z3d, w3d, h3d, l3d, ry)
        """
        """
        from visualDet3D.utils.utils import theta2alpha_3d
        alpha = theta2alpha_3d(bboxes[:, 12], bboxes[:, 6], bboxes[:, 8],
                               v_data["calib"][0]).unsqueeze(1)
        # alpha[:, :] = -10
        bboxes = torch.cat([bboxes, alpha], dim=1)
        bboxes[:, 12] += 0.01
        bboxes[:, :4] += 1
        bboxes[:, 7:12] += 0.1
        """
        valid_mask = scores > self.hparams["det_3d"]["test_score_threshold"]

        return {
            "scores": scores[valid_mask],
            "bboxes": bboxes[valid_mask],
            "cls_indexes": cls_indexes[valid_mask],
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "loss_sep": loss_sep
        }

    def forward(self, v_inputs):
        if self.training:
            return self.training_forward(v_inputs)
        else:
            return self.test_forward(v_inputs)
