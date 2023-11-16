import os

import sys
import time

import numpy as np
from torchvision.ops import sigmoid_focal_loss

from src.neural_bsp.abc_hdf5_dataset import generate_coords
# from thirdparty.Pointnet2_PyTorch.pointnet2.models.pointnet2_msg_sem import PointNet2SemSegMSG
from thirdparty.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, \
    PointnetSAModule, PointnetSAModuleMSG

sys.path.append("thirdparty/pvcnn")
from thirdparty.pvcnn.modules import functional as pvcnn_F

from torch import nn
from torch.nn import functional as F
import torch

from shared.common_utils import sigmoid, export_point_cloud


# Adopt the implementation in pytorch, but prevent NaN values
def focal_loss(inputs, targets, v_alpha=0.75, gamma: float = 2, ):
    loss = sigmoid_focal_loss(inputs, targets,
                              alpha=v_alpha,
                              reduction="mean"
                              )

    # p = torch.sigmoid(inputs.to(torch.float32))
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    #
    # if v_alpha >= 0:
    #     alpha_t = v_alpha * targets + (1 - v_alpha) * (1 - targets)
    #     loss = alpha_t * loss
    #
    # Check reduction option and return loss accordingly
    # loss = loss.mean()
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
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,
                      dilation=dilate),
            nn.InstanceNorm3d(ch_out) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True, dilation=1),
            nn.InstanceNorm3d(ch_out) if with_bn else nn.Identity(),
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
            # nn.BatchNorm3d(ch_out),
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

            self.up.append(up_conv(ch_in=cur_channel * 2, ch_out=cur_channel))
            self.up_conv.append(conv_block(ch_in=cur_channel * 2, ch_out=cur_channel, with_bn=with_bn))

            cur_channel = cur_channel * 2

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, v_input):
        # encoding path
        x1 = self.conv1(v_input)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 1, -1, -1):
            item = self.up[i](up_x[-1])
            if i >= 0:
                item = torch.cat((item, x[i]), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class Base_model(nn.Module):
    def __init__(self, v_conf):
        super(Base_model, self).__init__()
        self.need_normalize = v_conf["need_normalize"]
        self.loss_func = focal_loss
        self.loss_alpha = v_conf["focal_alpha"]
        self.num_features = v_conf["channels"]
        self.ic = v_conf["channels"]  # input_channels
        self.output_c = v_conf["output_channels"]

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = self.loss_func(v_predictions, labels, self.loss_alpha)
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
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        predicted_labels = predicted_labels.max(axis=-1)
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = gt_labels > 0.5
        gt_labels = gt_labels.max(axis=-1)
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return


class Base_model_UNet(Base_model):
    def __init__(self, v_conf):
        super(Base_model_UNet, self).__init__(v_conf)
        self.encoder = U_Net_3D(
            img_ch=v_conf["channels"],
            output_ch=self.output_c,
            v_pool_first=False,
            v_depth=4,
            base_channel=8,
            with_bn=v_conf["with_bn"]
        )

    def forward(self, v_data, v_training=False):
        (feat_data, _), _ = v_data
        bs = feat_data.shape[0]
        num_mini_batch = feat_data.shape[1]
        feat_data = feat_data.reshape((bs * num_mini_batch,) + feat_data.shape[2:])

        if self.need_normalize:
            udf = de_normalize_udf(feat_data[..., 0:1])
            if self.ic == 7:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                normal = de_normalize_angles(feat_data[..., 3:5])
                x = torch.cat([udf, gradients, normal], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            elif self.ic == 4:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                x = torch.cat([udf, gradients], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            else:
                x = udf.permute((0, 4, 1, 2, 3)).contiguous()
        else:
            x = feat_data[:, :self.num_features]

        prediction = self.encoder(x)

        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:])


class Base_model_k7(Base_model):
    def __init__(self, v_conf):
        super(Base_model_k7, self).__init__(v_conf)
        bs = 8  # base_channel
        with_bn = v_conf["with_bn"]

        self.conv1 = conv_block(ch_in=self.ic, ch_out=bs * 2, with_bn=with_bn, kernel_size=7, padding=3)
        self.conv2 = conv_block(ch_in=bs * 2, ch_out=bs * 4, with_bn=with_bn, kernel_size=5, padding=2)
        self.conv3 = conv_block(ch_in=bs * 4, ch_out=bs * 8, with_bn=with_bn, kernel_size=3, padding=1)
        self.conv4 = conv_block(ch_in=bs * 8, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=1)
        self.conv5 = conv_block(ch_in=bs * 4, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=1)
        self.fc = nn.Conv3d(bs * 2, self.output_c, kernel_size=1, padding=0)

    def forward(self, v_data, v_training=False):
        (feat_data, _), _ = v_data
        bs = feat_data.shape[0]
        num_mini_batch = feat_data.shape[1]
        feat_data = feat_data.reshape((bs * num_mini_batch,) + feat_data.shape[2:])

        if self.need_normalize:
            udf = de_normalize_udf(feat_data[..., 0:1])
            if self.ic == 7:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                normal = de_normalize_angles(feat_data[..., 3:5])
                x = torch.cat([udf, gradients, normal], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            elif self.ic == 4:
                gradients = de_normalize_angles(feat_data[..., 1:3])
                x = torch.cat([udf, gradients], dim=-1).permute((0, 4, 1, 2, 3)).contiguous()
            else:
                x = udf.permute((0, 4, 1, 2, 3)).contiguous()
        else:
            x = feat_data[:, :self.num_features]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        prediction = self.fc(x)

        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:])


class Base_model_dilated(Base_model_k7):
    def __init__(self, v_conf):
        super(Base_model_dilated, self).__init__(v_conf)
        bs = 8
        with_bn = v_conf["with_bn"]

        self.conv1 = conv_block(ch_in=self.ic, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=5, dilate=5)
        self.conv2 = conv_block(ch_in=bs * 2, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=2, dilate=2)
        self.conv3 = conv_block(ch_in=bs * 4, ch_out=bs * 8, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv4 = conv_block(ch_in=bs * 8, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv5 = conv_block(ch_in=bs * 4, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.fc = nn.Conv3d(bs * 2, self.output_c, kernel_size=1, padding=0)


##################################################################

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = (coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return pvcnn_F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')

class U_Net_3D_2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, v_pool_first=True, base_channel=[8,16,32,64,128], with_bn=True):
        super(U_Net_3D_2, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.depths = len(base_channel)-1

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
                nn.Conv3d(img_ch, base_channel[0], kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(base_channel[0]),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(img_ch, base_channel[0], kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
        for i in range(self.depths):
            self.conv.append(conv_block(ch_in=base_channel[i], ch_out=base_channel[i+1], with_bn=with_bn))

            self.up.append(up_conv(ch_in=base_channel[i+1], ch_out=base_channel[i]))
            self.up_conv.append(conv_block(ch_in=base_channel[i], ch_out=base_channel[i], with_bn=with_bn))

        self.Conv_1x1 = nn.Conv3d(base_channel[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, v_input):
        # encoding path
        x1 = self.conv1(v_input)

        x = [x1, ]
        for i in range(self.depths):
            x.append(self.Maxpool(self.conv[i](x[-1])))

        up_x = [x[-1]]
        for i in range(self.depths - 1, -1, -1):
            item = self.up[i](up_x[-1])
            if i >= 0:
                item = item + x[i]
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class PC_model(Base_model):
    def __init__(self, v_conf):
        super(PC_model, self).__init__(v_conf)
        hidden_dim = 32
        channels = v_conf["channels"]
        with_bn = v_conf["with_bn"]

        self.SA_modules = nn.ModuleList()
        # PointNet2
        if True:
            c_in = channels
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=1024,
                    radii=[0.05, 0.1],
                    nsamples=[16, 32],
                    mlps=[[c_in, 16], [c_in, 32]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_0 = 16 + 32

            c_in = c_out_0
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=256,
                    radii=[0.1, 0.2],
                    nsamples=[16, 32],
                    mlps=[[c_in, 32], [c_in, 64]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_1 = 32 + 64

            c_in = c_out_1
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=64,
                    radii=[0.2, 0.4],
                    nsamples=[16, 32],
                    mlps=[[c_in, 64], [c_in, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_2 = 64 + 128

            c_in = c_out_2
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=16,
                    radii=[0.4, 0.8],
                    nsamples=[16, 32],
                    mlps=[[c_in, 128], [c_in, 128]],
                    use_xyz=True,
                    bn=with_bn
                )
            )
            c_out_3 = 128 + 128

            self.FP_modules = nn.ModuleList()
            self.FP_modules.append(PointnetFPModule(mlp=[32 + channels, 32], bn=with_bn))
            self.FP_modules.append(PointnetFPModule(mlp=[64 + c_out_0, 32], bn=with_bn))
            self.FP_modules.append(PointnetFPModule(mlp=[128 + c_out_1, 64], bn=with_bn))
            self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 128], bn=with_bn))

            self.fc_lyaer = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=1, bias=False),
                nn.BatchNorm1d(32) if with_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Conv1d(32, hidden_dim, kernel_size=1),
            )

        self.encoder = U_Net_3D_2(
            img_ch=hidden_dim,
            output_ch=self.output_c,
            v_pool_first=False,
            base_channel=[32,64,128,128],
            with_bn=with_bn
        )

        self.voxelizer = Voxelization(32, False)

    def forward(self, v_data, v_training=False):
        (feat_data, _), _ = v_data
        bs = feat_data.shape[0]
        num_mini_batch = feat_data.shape[1]
        feat_data = feat_data.reshape((bs * num_mini_batch,) + feat_data.shape[2:])

        time_statics = [0] * 10
        cur_time = time.time()
        points = feat_data[..., 0:3]
        points_flags = feat_data[..., 3:4].to(torch.bool)

        if self.need_normalize:
            points = points / 32767

        time_statics[0]+=time.time() - cur_time
        cur_time = time.time()

        valid_numbers = (points_flags[...,0] == 1).sum(dim=1)
        target_length = points.shape[1]
        # Step 1: Create a tensor that represents a range from 0 to the maximum valid index for each example
        max_indices = valid_numbers.squeeze() - 1
        range_tensors = torch.arange(max_indices.max() + 1, device=points.device).unsqueeze(0).repeat(
            valid_numbers.size(0), 1)
        # Step 2: Truncate the range tensors to the maximum valid index for each example
        truncated_range_tensors = range_tensors % (max_indices.unsqueeze(1) + 1)
        # Step 3: Repeat and truncate to the target length
        repeated_tensors = truncated_range_tensors.repeat(1, target_length // truncated_range_tensors.size(1) + 1)
        index_tensors = repeated_tensors[:, :target_length]
        xyz = torch.gather(points, 1, index_tensors[:, :, None].tile((1, 1, 3)))

        time_statics[1]+=time.time() - cur_time
        cur_time = time.time()
        features = points.permute(0,2,1).contiguous()
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        features =  self.fc_lyaer(l_features[0])
        time_statics[2]+=time.time() - cur_time
        cur_time = time.time()

        voxel_features, voxel_coords = self.voxelizer(features, points)

        prediction = self.encoder(voxel_features)
        time_statics[3]+=time.time() - cur_time
        return prediction.reshape((bs, num_mini_batch,) + prediction.shape[1:])

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
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


class PC_model_whole_voxel(nn.Module):
    def __init__(self, v_conf):
        super(PC_model_whole_voxel, self).__init__()
        self.encoder = None
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.need_normalize = v_conf["need_normalize"]
        input_feature = v_conf["channels"]
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

        feature = feature.permute(0, 4, 1, 2, 3)

        prediction = self.conv(feature)

        return prediction

    def loss(self, v_predictions, v_input):
        (_, _), labels = v_input

        loss = focal_loss(v_predictions, labels[:, None])
        return {"total_loss": loss}

    def compute_pr(self, v_pred, v_gt):
        bs = v_pred.shape[0]
        prob = torch.sigmoid(v_pred).reshape(bs, -1)
        gt = v_gt.reshape(bs, -1).to(torch.long)
        return prob, gt

    def valid_output(self, idx, log_root, target_viz_name,
                     gathered_prediction, gathered_gt, gathered_queries):
        query_points = generate_coords(256).reshape(-1, 3)

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
class Base_model_dilated_backup(nn.Module):
    def __init__(self, v_conf):
        super(Base_model_dilated_backup, self).__init__()
        self.need_normalize = v_conf["need_normalize"]

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        ic = v_conf["channels"]  # input_channels
        bs = 8  # base_channel
        with_bn = v_conf["with_bn"]
        self.output_c = v_conf["output_channels"]

        self.conv1 = conv_block(ch_in=ic, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=5, dilate=5)
        self.conv2 = conv_block(ch_in=bs * 2, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=2, dilate=2)
        self.conv3 = conv_block(ch_in=bs * 4, ch_out=bs * 8, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv4 = conv_block(ch_in=bs * 8, ch_out=bs * 4, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.conv5 = conv_block(ch_in=bs * 4, ch_out=bs * 2, with_bn=with_bn, kernel_size=3, padding=1, dilate=1)
        self.fc = nn.Conv3d(bs * 2, self.output_c, kernel_size=3, padding=1)

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
            x = feat_data[:, :self.num_features]

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
            loss = self.loss_func(v_predictions, labels, self.loss_alpha)
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
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, self.output_c, 32, 32, 32)).transpose((0, 1, 5, 2, 6, 3, 7, 4)).reshape(256, 256, 256, -1)

        predicted_labels = sigmoid(predicted_labels) > 0.5
        predicted_labels = predicted_labels.max(axis=-1)
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_pred.ply".format(idx, target_viz_name)),
                           query_points[mask])

        gt_labels = gt_labels > 0.5
        gt_labels = gt_labels.max(axis=-1)
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(log_root, "{}_{}_gt.ply".format(idx, target_viz_name)),
                           query_points[mask])
        return
