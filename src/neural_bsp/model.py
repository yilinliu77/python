import os

import sys

import numpy as np
from torchvision.ops import sigmoid_focal_loss

from src.neural_bsp.abc_hdf5_dataset import generate_coords
from thirdparty.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

sys.path.append("thirdparty/pvcnn")
from thirdparty.pvcnn.modules import functional as pvcnn_F

from torch import nn
from torch.nn import functional as F
import torch

from shared.common_utils import sigmoid, export_point_cloud

# Adopt the implementation in pytorch, but prevent NaN values
def focal_loss(inputs, targets, v_alpha=0.75, gamma: float = 2,):
    # loss = sigmoid_focal_loss(v_predictions, labels,
    #                           alpha=v_alpha,
    #                           reduction="mean"
    #                           )

    p = torch.sigmoid(inputs.to(torch.float32))
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if v_alpha >= 0:
        alpha_t = v_alpha * targets + (1 - v_alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    loss = loss.mean()
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
    def __init__(self, ch_in, ch_out, with_bn=True, kernel_size=3, padding=1, stride=1, dilate=1):
        super(conv_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, dilation=dilate),
            nn.BatchNorm3d(ch_out) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True, dilation=1),
            nn.BatchNorm3d(ch_out) if with_bn else nn.Identity(),
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
        (feat_data, _), _ = v_data
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

        x = x[:,:self.num_features]
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


class Base_model_wo_pooling(nn.Module):
    def __init__(self, v_conf):
        super(Base_model_wo_pooling, self).__init__()
        self.need_normalize = v_conf["need_normalize"]

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        ic = v_conf["channels"] # input_channels
        bs = 8 # base_channel
        with_bn=False

        self.conv1 = conv_block(ch_in=ic, ch_out=bs * 2, with_bn=with_bn, kernel_size=7, padding=3)
        self.conv2 = conv_block(ch_in=bs * 2, ch_out=bs * 4, with_bn=with_bn, kernel_size=5, padding=2)
        self.conv3 = conv_block(ch_in=bs * 4, ch_out=bs * 8, with_bn=with_bn, kernel_size=3, padding=1)
        self.conv4 = conv_block(ch_in=bs * 8, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=1)
        self.conv5 = conv_block(ch_in=bs * 4, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=1)
        self.fc = nn.Conv3d(bs * 2, 1, kernel_size=3, padding=1)

        self.offset = 8

        self.loss_func = focal_loss
        self.loss_alpha = 0.75
        self.num_features = v_conf["channels"]

    def forward(self, v_data, v_training=False):
        (feat_data, _), _ = v_data
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
            x = feat_data[:,:self.num_features]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        prediction = self.fc(x)

        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:])

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        with torch.autocast(device_type="cuda"):
            loss = self.loss_func(v_predictions, labels[:, :, None, :, :, :], self.loss_alpha)
        return {"total_loss": loss}

    def compute_pr(self, v_pred, v_gt):
        bs = v_pred.shape[0]
        prob = torch.sigmoid(v_pred).reshape(bs, -1)
        gt = v_gt.reshape(bs, -1).to(torch.long)
        return prob, gt

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
        assert gathered_prediction.shape[0] == 64
        v_resolution = 256
        query_points = generate_coords(v_resolution).reshape(-1, 3)

        predicted_labels = gathered_prediction.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(256, 256, 256)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(256, 256, 256)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = gt_labels > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return


class Base_model_dilated(Base_model_wo_pooling):
    def __init__(self, v_conf):
        super(Base_model_dilated, self).__init__(v_conf)
        self.need_normalize = v_conf["need_normalize"]

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        ic = v_conf["channels"] # input_channels
        bs = 8 # base_channel
        with_bn=False

        self.conv1 = conv_block(ch_in=ic, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=5, dilate=5)
        self.conv2 = conv_block(ch_in=bs * 2, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=2, dilate=2)
        self.conv3 = conv_block(ch_in=bs * 4, ch_out=bs * 8, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv4 = conv_block(ch_in=bs * 8, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv5 = conv_block(ch_in=bs * 4, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.fc = nn.Conv3d(bs * 2, 1, kernel_size=3, padding=1)

        self.offset = 8

        self.loss_func = focal_loss
        self.loss_alpha = 0.75
        self.num_features = v_conf["channels"]


class PC_local_global(nn.Module):
    def __init__(self, v_conf):
        super(PC_local_global, self).__init__()
        self.need_normalize = v_conf["need_normalize"]

        self.num_features = v_conf["channels"]

        # Point cloud
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[6, 32, 32, 64],
                use_xyz=True,
                bn=False
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
                bn=False
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                bn=False
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
                bn=False
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 6, 128, 128, 128], bn=False))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128], bn=False))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256], bn=False))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], bn=False))

        # Convolutional network
        with_bn= False
        self.conv1 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn, dilate=2, padding=2)
        self.conv2 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn, dilate=2, padding=2)
        self.conv3 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn, dilate=1)
        self.conv4 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn, dilate=1)

        self.up4 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up_conv4 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn)
        self.up3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up_conv3 = conv_block(ch_in=128, ch_out=128, with_bn=with_bn)
        self.up2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv3d(in_channels=128, out_channels=16, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=4)
        self.up_conv1 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1)

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.resolution = v_conf["voxelize_res"]
        self.loss_func = focal_loss
        self.loss_alpha = 0.75
        self.offset = -1/self.resolution

    def point_features(self, points):
        xyz = points[:, :3].permute(0,2,1).contiguous()
        features = points

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

    def forward(self, v_data, v_training=False):
        (points,feat_data), labels = v_data
        query_coords = feat_data[..., :3]
        bs = feat_data.shape[0] # Batch size
        
        if self.need_normalize:
            points = de_normalize_points(points)
        else:
            points = points

        points = points.permute(0,2,1).contiguous()

        point_features = self.point_features(points)

        vox_coords = torch.round(
            (self.offset + points[:, :3] + torch.ones_like(points[:, :3])) / 2 * self.resolution).to(torch.int32)
        voxel_features = pvcnn_F.avg_voxelize(point_features, vox_coords, self.resolution)
        if False:
            p = generate_coords(self.resolution)
            v = vox_coords.cpu().numpy()[0]
            p = p[v[0], v[1], v[2]]
            export_point_cloud("test.ply", p)
        if False:
            p = generate_coords(self.resolution)
            p = p[(voxel_features[0]!=0).max(dim=0)[0].cpu().numpy()]
            export_point_cloud("test.ply", p)

        x1 = self.conv1(voxel_features)
        x2 = self.Maxpool(self.conv2(x1))
        x3 = self.Maxpool(self.conv3(x2))
        x4 = self.Maxpool(self.conv4(x3))

        up_x4 = self.up4(x4)
        up_x4 = self.up_conv4(up_x4 + x3)
        up_x3 = self.up3(up_x4)
        up_x3 = self.up_conv3(up_x3 + x2)
        up_x2 = self.up2(up_x3)
        up_x1 = self.up_conv2(up_x2 + x1)

        prediction = self.up1(up_x1)
        prediction = self.up_conv1(prediction)

        return prediction

    def loss(self, v_predictions, v_input):
        (_,_), labels = v_input

        loss = self.loss_func(v_predictions, labels[:, None], self.loss_alpha)
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


class PC_model2(PC_local_global):
    def __init__(self, v_conf):
        super(PC_model2, self).__init__(v_conf)
        self.encoder=None
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        input_feature = self.num_features
        base_channel = 8
        with_bn = False
        self.conv = nn.ModuleList()
        self.conv.append(
            conv_block(ch_in=input_feature, ch_out=base_channel, with_bn=with_bn, kernel_size=7, padding=3))
        self.conv.append(
            conv_block(ch_in=base_channel * 1, ch_out=base_channel * 2, with_bn=with_bn, kernel_size=7, padding=3))
        self.conv.append(
            conv_block(ch_in=base_channel * 2, ch_out=base_channel * 4, with_bn=with_bn, kernel_size=5, padding=2))
        self.conv.append(
            conv_block(ch_in=base_channel * 4, ch_out=base_channel * 8, with_bn=with_bn, kernel_size=3, padding=1))
        # self.conv.append(
        #     conv_block(ch_in=base_channel * 8, ch_out=base_channel * 16, with_bn=with_bn, kernel_size=3, padding=1))
        # self.conv.append(
        #     conv_block(ch_in=base_channel * 16, ch_out=base_channel * 32, with_bn=with_bn, kernel_size=3, padding=1))

        self.fc = nn.Sequential(
            # nn.Linear(504, 256),
            # nn.ReLU(inplace=True),
            Residual_fc(120, 120),
            nn.Linear(120, 1),
        )

    def pos_encoding(self, v_points):
        freq = 2 ** torch.arange(16, dtype=torch.float32, device=v_points.device) * np.pi  # [L]
        spectrum = v_points[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*v_points.shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, v_data, v_training=False):
        (points, feat_data), labels = v_data
        query_coords = feat_data[..., :3]
        bs = feat_data.shape[0]  # Batch size

        if self.need_normalize:
            points = de_normalize_points(points)
        else:
            points = points

        points = points.permute(0, 2, 1)
        vox_coords = torch.round(
            (self.offset + points[:, :3] + torch.ones_like(points[:, :3])) / 2 * self.resolution).to(torch.int32)
        voxel_features = pvcnn_F.avg_voxelize(points, vox_coords, self.resolution)
        if False:
            p = generate_coords(self.resolution)
            v = vox_coords.cpu().numpy()[0]
            p = p[v[0], v[1], v[2]]
            export_point_cloud("test.ply", p)
        if False:
            p = generate_coords(self.resolution)
            p = p[(voxel_features[0]!=0).max(dim=0)[0].cpu().numpy()]
            export_point_cloud("test.ply", p)

        features = [voxel_features, ]
        for i in range(len(self.conv)):
            features.append(self.Maxpool(self.conv[i](features[-1])))

        sampled_features = []
        for feature in features[1:]:
            sampled_feature = F.grid_sample(
                feature, query_coords[:,None,None,], mode="bilinear", align_corners=True)[:,:,0,0]
            sampled_features.append(sampled_feature)

        sampled_features = torch.cat(sampled_features, dim=1).permute((0,2,1))
        prediction = self.fc(sampled_features)

        return prediction

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
        query_points = gathered_queries.reshape(-1,3)

        predicted_labels = gathered_prediction.reshape(-1)
        gt_labels = gathered_gt.reshape(-1)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = sigmoid(gt_labels) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return


##################################################################
class U_Net_3D2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, base_channel=16, with_bn=True):
        super(U_Net_3D2, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.conv1 = conv_block(ch_in=img_ch, ch_out=base_channel, with_bn=with_bn, dilate=2, padding=2)
        self.conv2 = conv_block(ch_in=base_channel, ch_out=base_channel * 2, with_bn=with_bn, dilate=2, padding=2)
        self.conv3 = conv_block(ch_in=base_channel * 2, ch_out=base_channel * 4, with_bn=with_bn, dilate=1)
        self.conv4 = conv_block(ch_in=base_channel * 4, ch_out=base_channel * 8, with_bn=with_bn, dilate=1)

        self.up4 = nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=2, stride=2)
        self.up_conv4 = conv_block(ch_in=base_channel * 4, ch_out=base_channel * 4, with_bn=with_bn)
        self.up3 = nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=2, stride=2)
        self.up_conv3 = conv_block(ch_in=base_channel * 2, ch_out=base_channel * 2, with_bn=with_bn)
        self.up2 = nn.ConvTranspose3d(base_channel * 2, base_channel * 1, kernel_size=2, stride=2)
        self.up_conv2 = nn.Conv3d(in_channels=base_channel * 1, out_channels=output_ch, kernel_size=1)


    def forward(self, v_input):
        x1 = self.conv1(v_input)
        x2 = self.Maxpool(self.conv2(x1))
        x3 = self.Maxpool(self.conv3(x2))
        x4 = self.Maxpool(self.conv4(x3))

        up_x4 = self.up4(x4)
        up_x4 = self.up_conv4(up_x4 + x3)
        up_x3 = self.up3(up_x4)
        up_x3 = self.up_conv3(up_x3 + x2)
        up_x2 = self.up2(up_x3)
        prediction = self.up_conv2(up_x2 + x1)

        return prediction


class PC_model_whole_voxel(PC_local_global):
    def __init__(self, v_conf):
        super(PC_model_whole_voxel, self).__init__(v_conf)
        self.encoder=None
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        input_feature = self.num_features
        base_channel = 8
        with_bn = False
        self.conv = U_Net_3D2(
            img_ch=input_feature,
            output_ch=1,
            base_channel=base_channel,
            with_bn=with_bn
        )

    def forward(self, v_data, v_training=False):
        (feature, query_coords), labels = v_data
        bs = feature.shape[0]  # Batch size

        if self.need_normalize:
            udf = de_normalize_udf(feature[..., 0:1])
            gradients = de_normalize_angles(feature[..., 1:3])
            normals = de_normalize_angles(feature[..., 3:5])
            feature = torch.cat((udf, gradients, normals), dim=-1)
        else:
            feature = feature

        feature = feature.permute(0, 4,1,2,3)

        prediction = self.conv(feature)

        return prediction

    def loss(self, v_predictions, v_input):
        (_,_), labels = v_input

        loss = self.loss_func(v_predictions, labels[:, None], self.loss_alpha)
        return {"total_loss": loss}

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
        query_points = gathered_queries.reshape(-1,3)

        predicted_labels = gathered_prediction.reshape(-1)
        gt_labels = gathered_gt.reshape(-1)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = sigmoid(gt_labels) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return