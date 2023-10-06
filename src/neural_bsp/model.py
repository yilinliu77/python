import os
from functools import partial

import sys

import numpy as np

from src.neural_bsp.abc_hdf5_dataset import generate_coords

sys.path.append("thirdparty/pvcnn")
from thirdparty.pvcnn.modules import functional as pvcnn_F

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchvision.ops import sigmoid_focal_loss

from shared.common_utils import sigmoid, export_point_cloud


def focal_loss(v_predictions, labels, v_alpha=0.75):
    loss = sigmoid_focal_loss(v_predictions, labels,
                              alpha=v_alpha,
                              reduction="mean"
                              )
    return loss


def BCE_loss(v_predictions, labels, v_alpha=0.75):
    loss = nn.functional.binary_cross_entropy_with_logits(v_predictions, labels,
                                                          reduction="mean"
                                                          )
    return loss


class Residual_fc(nn.Module):
    def __init__(self, v_input, v_output):
        super().__init__()
        self.fc = nn.Linear(v_input, v_output)
        self.relu = nn.LeakyReLU()

    def forward(self, v_data):
        feature = self.relu(self.fc(v_data))
        return feature + v_data


def de_normalize_angles(v_angles):
    angles = (v_angles / 65535 * torch.pi * 2)
    dx = torch.cos(angles[..., 0]) * torch.sin(angles[..., 1])
    dy = torch.sin(angles[..., 0]) * torch.sin(angles[..., 1])
    dz = torch.cos(angles[..., 1])
    gradients = torch.stack([dx, dy, dz], dim=-1)
    return gradients


def de_normalize_udf(v_udf):
    return v_udf / 65535 * 2

def de_normalize_points(v_points):
    return v_points / 65535 * 2 - 1

#################################################################################################################
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, with_bn=True):
        super(conv_block, self).__init__()
        if with_bn:
            self.conv1 = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True),

            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x).contiguous()
        return x


class U_Net_3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, v_depth=5, v_pool_first=True, base_channel=16, with_bn=True):
        super(U_Net_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.depths = v_depth

        self.conv = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        if v_pool_first:
            self.conv1 = nn.Sequential(
                conv_block(ch_in=img_ch, ch_out=base_channel, with_bn=with_bn),
                nn.MaxPool3d(kernel_size=4, stride=4),
                conv_block(ch_in=base_channel, ch_out=base_channel, with_bn=with_bn),
            )
        elif with_bn:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(base_channel),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        cur_channel = base_channel
        for i in range(v_depth):
            self.conv.append(conv_block(ch_in=cur_channel, ch_out=cur_channel * 2, with_bn=with_bn))

            if i == 1:
                self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
                self.up_conv.append(conv_block(ch_in=cur_channel, ch_out=cur_channel, with_bn=with_bn))
            elif i > 1:
                self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
                self.up_conv.append(conv_block(ch_in=cur_channel * 2, ch_out=cur_channel, with_bn=with_bn))

            cur_channel = cur_channel * 2

        self.Conv_1x1 = nn.Conv3d(base_channel * 2, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, v_input):
        # encoding path
        x1 = self.conv1(v_input)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 2, -1, -1):
            item = self.up[i](up_x[-1])
            if i > 0:
                item = torch.cat((item, x[i + 1]), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class Base_model(nn.Module):
    def __init__(self, v_conf):
        super(Base_model, self).__init__()
        self.need_normalize = v_conf["need_normalize"]
        self.encoder = U_Net_3D(
            img_ch=v_conf["channels"],
            output_ch=1,
            v_pool_first=False,
            v_depth=v_conf["depths"],
            base_channel=8
        )
        self.loss_func = focal_loss
        self.loss_alpha = 0.75
        self.num_features = v_conf["channels"]

    def forward(self, v_data, v_training=False):
        feat_data, _ = v_data
        bs = feat_data.shape[0]
        num_mini_batch = feat_data.shape[1]
        feat_data = feat_data.reshape((bs * num_mini_batch,) + feat_data.shape[2:])

        if self.need_normalize:
            udf = de_normalize_udf(feat_data[..., 0:1])
            gradients = de_normalize_angles(feat_data[..., 1:3])
            if feat_data.shape[-1] == 5 and self.num_features == 7:
                normal = de_normalize_angles(feat_data[..., 3:5])
                x = torch.cat([udf, gradients, normal], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            else:
                x = torch.cat([udf, gradients], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
        else:
            x = feat_data

        prediction = self.encoder(x)

        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:])

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = self.loss_func(v_predictions, labels[:, :, None, :, :, :], self.loss_alpha)
        return {"total_loss": loss}

    def compute_pr(self, v_pred, v_gt):
        bs = v_pred.shape[0]
        prob = torch.sigmoid(v_pred).reshape(bs, -1)
        gt = v_gt.reshape(bs, -1).to(torch.long)
        return prob, gt

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt):
        assert gathered_prediction.shape[0] == 225
        v_resolution = 256
        query_points = generate_coords(v_resolution).reshape(-1, 3)

        predicted_labels = gathered_prediction.reshape(
            (-1, 15, 15, 15, 16, 16, 16)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(240, 240, 240)
        gt_labels = gathered_gt.reshape(
            (-1, 15, 15, 15, 16, 16, 16)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(240, 240, 240)

        predicted_labels = np.pad(predicted_labels, 8, mode="constant", constant_values=0)
        gt_labels = np.pad(gt_labels, 8, mode="constant", constant_values=0)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = sigmoid(gt_labels) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return
    

class PC_model(nn.Module):
    def __init__(self, v_conf):
        super(PC_model, self).__init__()
        self.need_normalize = v_conf["need_normalize"]
        self.encoder = U_Net_3D(
            img_ch=v_conf["channels"],
            output_ch=1,
            v_pool_first=False,
            v_depth=v_conf["depths"],
            base_channel=8
        )
        self.encoder.Conv_1x1 = nn.Conv3d(240, 1, 1)
        self.resolution = 64
        self.loss_func = focal_loss
        self.loss_alpha = 0.75
        self.num_features = v_conf["channels"]

    def forward(self, v_data, v_training=False):
        (points,feat_data), labels = v_data
        query_coords = feat_data[..., :3]
        bs = feat_data.shape[0] # Batch size
        
        if self.need_normalize:
            points = de_normalize_points(points)
        else:
            points = points

        points = points.permute(0,2,1)
        vox_coords = torch.round((points[:, :3] + torch.ones_like(points[:, :3])) / 2 * self.resolution).to(torch.int32)
        voxel_features = pvcnn_F.avg_voxelize(points, vox_coords, self.resolution)
        x1 = self.encoder.conv1(voxel_features)

        x = [x1, ]
        for i in range(self.encoder.depths):
            x.append(self.encoder.Maxpool(self.encoder.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.encoder.depths - 2, -1, -1):
            item = self.encoder.up[i](up_x[-1])
            if i > 0:
                item = torch.cat((item, x[i + 1]), dim=1)
            up_x.append(self.encoder.up_conv[i](item))

        sampled_features = []
        for feature in up_x:
            sampled_feature = F.grid_sample(feature, query_coords, mode="bilinear", align_corners=True)
            sampled_features.append(sampled_feature)
        sampled_features = torch.cat(sampled_features, dim=1)
        prediction = self.encoder.Conv_1x1(sampled_features)

        return prediction

    def loss(self, v_predictions, v_input):
        (points,feat_data), labels = v_input

        loss = self.loss_func(v_predictions, labels[:, None, :, :, :], self.loss_alpha)
        return {"total_loss": loss}

    def compute_pr(self, v_pred, v_gt):
        bs = v_pred.shape[0]
        prob = torch.sigmoid(v_pred).reshape(bs, -1)
        gt = v_gt.reshape(bs, -1).to(torch.long)
        return prob, gt

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt):
        assert gathered_prediction.shape[0] == 8
        v_resolution = 256
        query_points = generate_coords(v_resolution).reshape(-1, 3)

        predicted_labels = gathered_prediction.reshape(
            (-1, 2, 2, 2, 128, 128, 128)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(v_resolution, v_resolution, v_resolution)
        gt_labels = gathered_gt.reshape(
            (-1, 2, 2, 2, 128, 128, 128)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(v_resolution, v_resolution, v_resolution)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = sigmoid(gt_labels) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return