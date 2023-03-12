import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numba as nb

import tinycudann as tcnn


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest",
                                           recompute_scale_factor=True) + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest",
                                           recompute_scale_factor=True) + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest",
                                           recompute_scale_factor=True) + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs


class Explorer(nn.Module):
    def __init__(self, v_img_models):
        super(Explorer, self).__init__()
        self.feature_extractor = v_img_models
        self.fuser = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.mlp = nn.Linear(32, 1)
        pass

    def forward(self, v_data):
        sample_point = v_data["sample_point"][0]
        if len(v_data["img_names"]) <= 1:
            return torch.tensor([0], dtype=torch.float16), torch.tensor([0], dtype=torch.float16)
        features = []
        pixels = []
        for id_img in range(len(v_data["img_names"])):
            img_name = v_data["img_names"][id_img]
            img_model = self.feature_extractor[img_name[0]]
            projected_pixel_pos = torch.matmul(v_data["projection_matrix"][0][id_img],
                                               torch.cat([sample_point, torch.ones_like(sample_point[0:1])], dim=-1))
            projected_pixel_pos = projected_pixel_pos[:2] / projected_pixel_pos[2]
            sampled_feature = img_model.model1(projected_pixel_pos.unsqueeze(0))
            sampled_pixel = img_model.model2(sampled_feature)
            features.append(sampled_feature)
            pixels.append(sampled_pixel)
        pixels = torch.concatenate(pixels, dim=0).unsqueeze(0)
        features = torch.stack(features, dim=1)
        fused_features = self.fuser(key=features, value=features, query=features)[0]
        mean_features = fused_features.mean(dim=1)
        predicted_probability = self.mlp(mean_features)
        predicted_probability = torch.sigmoid(predicted_probability)

        gt_probability = self.prepare_gt(pixels)

        return predicted_probability, gt_probability

    def prepare_gt(self, v_pixels):
        return torch.var(v_pixels, dim=[1, 2]).unsqueeze(1)

    def loss(self, v_predicted_probability, v_gt_probability):
        inv_predicted_probability = 1 - v_predicted_probability
        loss = torch.mean(inv_predicted_probability * v_gt_probability)
        return loss


@nb.njit
def bresenham(x0, y0, x1, y1, thickness=0):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    result = []
    is_thickness_larger = thickness != 0
    for x in range(dx + 1):
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy
        result.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy))
        if is_thickness_larger:
            for i in range(1, thickness + 1):
                result.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy - i))
                result.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy + i))
    return result


class Segment_explorer(nn.Module):
    def __init__(self, v_imgs):
        super(Segment_explorer, self).__init__()
        # self.fuser = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        # self.mlp = nn.Linear(32, 1)
        self.imgs = v_imgs

        self.model1 = tcnn.Encoding(
            n_input_dims=6,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 12,
            })
        self.model2 = tcnn.Network(
            n_input_dims=self.model1.n_output_dims,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            })
        pass

    def forward(self, v_data):
        if False:
            cv2.namedWindow("1", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1", 1600, 900)
            cv2.moveWindow("1", 0, 0)
            for idx, item in enumerate(v_data["img_names"]):
                img = self.imgs[item[0]]
                img = cv2.line(img,
                               (v_data["projected_coordinates"][0, idx, 0].cpu().numpy() * np.array((600, 400))).astype(
                                   int),
                               (v_data["projected_coordinates"][0, idx, 1].cpu().numpy() * np.array((600, 400))).astype(
                                   int),
                               (0, 0, 255),
                               5,
                               )
                cv2.imshow("1", img)
                cv2.waitKey()
            pass
        # for id_batch in range(v_data["id"].shape[0]):
        #     sample_segment = v_data["sample_segment"][id_batch]
        #     if len(v_data["img_names"]) <= 1:
        #         return torch.tensor([0], dtype=torch.float16), torch.tensor([0], dtype=torch.float16)
        #     projected_segment = v_data["projected_coordinates"][id_batch]
        #     roi_regions = []
        nif_feature = self.model1(v_data["sample_segment"].reshape([-1, 6]))
        occupancy = torch.sigmoid(self.model2(nif_feature))
        return occupancy

    def prepare_gt(self, v_pixels):
        return torch.var(v_pixels, dim=[1, 2]).unsqueeze(1)

    # v_data["gt_loss"] = (batch_size, (ncc, edge_similarity, edge_magnitude, lbd_similarity))
    def loss(self, v_predicted_probability, v_data):
        gt_loss = torch.clip(v_data["gt_loss"], 0, 1)
        gt_probability = (gt_loss[:, 0] + gt_loss[:, 1] + gt_loss[:, 2] + gt_loss[:, 3]) / 4
        loss = torch.nn.functional.mse_loss(v_predicted_probability[:, 0], gt_probability)
        return loss


class Point_explorer(nn.Module):
    def __init__(self, v_imgs):
        super(Point_explorer, self).__init__()
        self.imgs = v_imgs

        self.model1 = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 12,
            })
        self.model2 = tcnn.Network(
            n_input_dims=self.model1.n_output_dims,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            })
        pass

    def forward(self, v_data):
        if False:
            cv2.namedWindow("1", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1", 1600, 900)
            cv2.moveWindow("1", 0, 0)
            for idx, item in enumerate(v_data["img_names"]):
                img = self.imgs[item[0]]
                img = cv2.line(img,
                               (v_data["projected_coordinates"][0, idx, 0].cpu().numpy() * np.array((600, 400))).astype(
                                   int),
                               (v_data["projected_coordinates"][0, idx, 1].cpu().numpy() * np.array((600, 400))).astype(
                                   int),
                               (0, 0, 255),
                               5,
                               )
                cv2.imshow("1", img)
                cv2.waitKey()
            pass
        # for id_batch in range(v_data["id"].shape[0]):
        #     sample_segment = v_data["sample_segment"][id_batch]
        #     if len(v_data["img_names"]) <= 1:
        #         return torch.tensor([0], dtype=torch.float16), torch.tensor([0], dtype=torch.float16)
        #     projected_segment = v_data["projected_coordinates"][id_batch]
        #     roi_regions = []
        nif_feature = self.model1(v_data["sample_point"].reshape([-1, 3]))
        occupancy = torch.sigmoid(self.model2(nif_feature))
        return occupancy

    # v_data["gt_loss"] = (batch_size, ncc,)
    def loss(self, v_predicted_probability, v_data):
        loss = torch.nn.functional.mse_loss(v_predicted_probability[:, 0], v_data["gt_loss"][:, 0])
        return loss


if __name__ == '__main__':
    pixels = np.array(bresenham(10, 10, 0, 0, 2))
    pixels = np.clip(pixels, 0, 19)
    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.moveWindow("1", 0, 0)
    cv2.resizeWindow("1", 900, 900)
    img = np.zeros((20, 20), dtype=np.uint8)
    img[pixels[:, 1], pixels[:, 0]] = 255
    cv2.imshow("1", img)
    cv2.waitKey()
