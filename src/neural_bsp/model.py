from functools import partial

import sys

import numpy as np

# sys.path.append("thirdparty/pvcnn")
# from thirdparty.pvcnn.modules.pvconv import PVConv
# from thirdparty.pvcnn.modules.voxelization import Voxelization

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchvision.ops import sigmoid_focal_loss


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

#################################################################################################################

"""
points: N*3
query_and_data: M*8 (udf, gradient, flag) 
"""
class PVCNN(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)

        cur_channels = 16
        self.encoders = nn.ModuleList()
        hidden_dim_out = 0
        fc_hidden_dim =  v_conf["fc_hidden_dim"]
        conv_hidden_dim =  v_conf["max_voxel_dim"]
        for i in range(5):
            if i == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv3d(7, cur_channels, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv3d(cur_channels, cur_channels, 3, padding=1),
                    nn.LeakyReLU(),
                    # nn.BatchNorm3d(cur_channels)
                ))
            else:
                target_channels = min(conv_hidden_dim, cur_channels * 2)
                self.encoders.append(nn.Sequential(
                    nn.Conv3d(cur_channels, target_channels, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv3d(target_channels, target_channels, 3, padding=1),
                    nn.LeakyReLU(),
                    # nn.BatchNorm3d(target_channels)
                ))
                cur_channels = target_channels
            hidden_dim_out += cur_channels

        self.udf_out = nn.Sequential(
            nn.Linear(hidden_dim_out, fc_hidden_dim),
            nn.LeakyReLU(),
            Residual_fc(fc_hidden_dim,fc_hidden_dim),
            Residual_fc(fc_hidden_dim,fc_hidden_dim),
            nn.Linear(fc_hidden_dim,1),
        )

        self.gradient_out = nn.Sequential(
            nn.Linear(hidden_dim_out, fc_hidden_dim),
            nn.LeakyReLU(),
            Residual_fc(fc_hidden_dim, fc_hidden_dim),
            Residual_fc(fc_hidden_dim, fc_hidden_dim),
            nn.Linear(fc_hidden_dim, 3),
        )

        self.flag_out = nn.Sequential(
            nn.Linear(hidden_dim_out, fc_hidden_dim),
            nn.LeakyReLU(),
            Residual_fc(fc_hidden_dim, fc_hidden_dim),
            Residual_fc(fc_hidden_dim, fc_hidden_dim),
            nn.Linear(fc_hidden_dim, 1),
        )

    def forward(self, v_data, v_training=False):
        sparse_features, query_features = v_data

        voxel_feature = sparse_features.permute(0,4,1,2,3)
        features= []
        for layer in self.encoders:
            voxel_feature = layer(voxel_feature)
            features.append(voxel_feature)
            voxel_feature = self.maxpool(voxel_feature)

        sampled_feature = self.embedding(query_features[:,:,:3], features)
        pred_udf = self.udf_out(sampled_feature.permute(0,2,1))
        pred_gradient = self.gradient_out(sampled_feature.permute(0,2,1))
        pred_flag = self.flag_out(sampled_feature.permute(0,2,1))
        return torch.cat((pred_udf, pred_gradient, pred_flag), dim=2)

    def embedding(self, p, v_features):
        sampled_features = []
        for feature in v_features:
            sampled_feature = F.grid_sample(feature, p.unsqueeze(1).unsqueeze(1), align_corners=True)[:,:,0,0]
            sampled_features.append(sampled_feature)
        sampled_features = torch.cat(sampled_features, dim=1)
        return sampled_features

    def loss(self, v_prediction, v_gt):
        gt_labels = v_gt[1]
        gt_gradient = gt_labels[:,:,3:6]
        gt_udf = gt_labels[:,:,6:7]
        gt_flag = gt_labels[:,:,7:8].to(gt_gradient.dtype)

        udf_loss = F.l1_loss(v_prediction[:,:,0:1], gt_udf, reduction="mean")
        gradient_loss = F.l1_loss(v_prediction[:,:,1:4], gt_gradient, reduction="mean")
        flag_loss = focal_loss(v_prediction[:,:,4:5], gt_flag, 0.75)
        total_loss = udf_loss + gradient_loss + flag_loss * 10
        return {
            "udf_loss": udf_loss,
            "gradient_loss": gradient_loss,
            "flag_loss": flag_loss,
            "total_loss": total_loss
        }

class PVCNN_local(PVCNN):
    def __init__(self,
                 v_conf,
                 ):
        super(PVCNN_local, self).__init__(v_conf)
        self.maxpool = nn.Identity()

        cur_channels = 16
        self.encoders = nn.ModuleList()
        conv_hidden_dim =  v_conf["max_voxel_dim"]
        for i in range(5):
            if i == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv3d(7, cur_channels, 7, padding=3),
                    nn.LeakyReLU(),
                    nn.Conv3d(cur_channels, cur_channels, 7, padding=3),
                    nn.LeakyReLU(),
                ))
            elif i <= 2:
                target_channels = min(conv_hidden_dim, cur_channels * 2)
                self.encoders.append(nn.Sequential(
                    nn.Conv3d(cur_channels, target_channels, 7, padding=3),
                    nn.LeakyReLU(),
                    nn.Conv3d(target_channels, target_channels, 7, padding=3),
                    nn.LeakyReLU(),
                ))
                cur_channels = target_channels
            else:
                target_channels = min(conv_hidden_dim, cur_channels * 2)
                self.encoders.append(nn.Sequential(
                    nn.Conv3d(cur_channels, target_channels, 1, padding=0),
                    nn.LeakyReLU(),
                    nn.Conv3d(target_channels, target_channels, 1, padding=0),
                    nn.LeakyReLU(),
                ))
                cur_channels = target_channels

#################################################################################################################

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        Args:
            num_channels (int): No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        """
        Args:
            weights (torch.Tensor): weights for few shot learning
            x: X, shape = (batch_size, num_channels, D, H, W)

        Returns:
            (torch.Tensor): output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, is3d):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, is3d=True):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, is3d):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1,
                 is3d=True):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding, is3d=is3d))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding, is3d=is3d))


class ResNetBlock(nn.Module):
    """
    Residual block that can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, is3d=True, **kwargs):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                is3d=is3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, is3d=is3d)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, se_module='scse', **kwargs):
        super(ResNetBlockSE, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse', 'cse', 'sse']
        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1, is3d=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         is3d=is3d)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (bool): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True, is3d=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         is3d=is3d)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size, is3d):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              is3d=is3d)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              is3d=is3d)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, is3d):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          is3d=is3d)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class TestNetDoubleConv(nn.Module):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5,
                 v_input_channel=3, *kwargs
                 ):
        super(TestNetDoubleConv, self).__init__()
        self.f_maps = v_phase
        self.encoders = create_encoders(
            v_input_channel,
            self.f_maps,
            DoubleConv,
            3,
            1,
            "cbr",
            8,
            2,
            True)
        self.decoders = create_decoders(
            self.f_maps,
            DoubleConv,
            3,
            1,
            "cbr",
            8,
            True)
        self.final_conv = nn.Conv3d(self.f_maps[0], 1, 1)
        self.loss_func = globals()[v_loss_type]
        self.loss_alpha = v_alpha

    def forward(self, v_data, v_training=False):
        x, labels = v_data
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        prediction = self.final_conv(x)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = self.loss_func(v_predictions, labels, self.loss_alpha)
        return loss


class TestNetResidual(TestNetDoubleConv):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5,
                 v_input_channel=3, *kwargs
                 ):
        super().__init__(v_phase, v_loss_type, v_alpha)
        self.encoders = create_encoders(
            v_input_channel,
            self.f_maps,
            ResNetBlock,
            3,
            1,
            "cbr",
            8,
            2,
            True)
        self.decoders = create_decoders(
            self.f_maps,
            ResNetBlock,
            3,
            1,
            "cbr",
            8,
            True)


class TestNetResidualSE(TestNetDoubleConv):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5,
                 v_input_channel=3, *kwargs
                 ):
        super().__init__(v_phase, v_loss_type, v_alpha)
        self.encoders = create_encoders(
            v_input_channel,
            self.f_maps,
            ResNetBlockSE,
            3,
            1,
            "cbr",
            8,
            2,
            True)
        self.decoders = create_decoders(
            self.f_maps,
            ResNetBlockSE,
            3,
            1,
            "cbr",
            8,
            True)


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
        x = self.up(x)
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
            item = torch.cat((item, x[i]), dim=1)
            up_x.append(self.up_conv[i](item))

        d1 = self.Conv_1x1(up_x[-1])

        return d1


class Base_model(nn.Module):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5, v_input_channel=3, v_depth=4, v_base_channel=16):
        super(Base_model, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(
            img_ch=v_input_channel, output_ch=1, v_pool_first=False, v_depth=v_depth, base_channel=v_base_channel)
        self.loss_func = globals()[v_loss_type]
        self.loss_alpha = v_alpha

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = self.loss_func(v_predictions, labels, self.loss_alpha)
        return loss


class Base_patch_model_deeper_wo_bn(Base_model):
    def __init__(self, v_phase=0, v_loss_type="focal", v_alpha=0.5, v_input_channel=3, v_depth=4, v_base_channel=16):
        super(Base_patch_model_deeper_wo_bn, self).__init__(v_phase, v_loss_type, v_alpha)
        self.phase = v_phase
        self.encoder = U_Net_3D(
            img_ch=v_input_channel, output_ch=1, v_pool_first=False,
            v_depth=v_depth, base_channel=v_base_channel, with_bn=False)
