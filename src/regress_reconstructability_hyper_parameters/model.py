import math
import time
from typing import Dict, Optional, Tuple, List

import torch
import torchvision
from scipy.stats import stats
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, init, MultiheadAttention, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import _get_activation_fn

from torchvision.models import resnet18
import torchsort
import sys

from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, \
    PointNetFeaturePropagation
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder


def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


def normal_distribution(v_x, v_sigma, v_exponential):
    return torch.exp(-torch.pow(v_x, v_exponential) / (2 * v_sigma * v_sigma))


class Regress_hyper_parameters_Model(nn.Module):
    def __init__(self, hparams):
        super(Regress_hyper_parameters_Model, self).__init__()
        self.hydra_conf = hparams

        self.alpha_weights_sigma1 = nn.Parameter(torch.tensor(5. / 90, dtype=torch.float))
        self.alpha_weights_sigma2 = nn.Parameter(torch.tensor(30. / 90, dtype=torch.float))
        self.alpha_weights_exponential1 = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)
        self.alpha_weights_exponential2 = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.gsd_weights_sigma = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
        self.gsd_weights_exponential = nn.Parameter(torch.tensor(4., dtype=torch.float), requires_grad=False)

        self.scale_weights_sigma = nn.Parameter(torch.tensor(.25, dtype=torch.float))
        self.scale_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.angle_to_normal_weights_sigma = nn.Parameter(torch.tensor(30. / 90, dtype=torch.float))
        self.angle_to_normal_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float),
                                                                requires_grad=False)

        self.distortion_weights_sigma = nn.Parameter(torch.tensor(1., dtype=torch.float))
        self.distortion_weights_exponential = nn.Parameter(torch.tensor(2., dtype=torch.float), requires_grad=False)

        self.model_weights_sigma = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.model_weights_exponential = nn.Parameter(torch.tensor(2, dtype=torch.float), requires_grad=False)

    def forward(self, v_data):
        # alpha_in_degree, d1/d_max, d2/d_max, 1-min(d1,d2)/max(d1,d2), normal_angle1, normal_angle2, central_angle1, central_angle2
        views = v_data["views"]
        view_pairs = v_data["view_pairs"]
        point_attribute = v_data["point_attribute"]

        mask = v_data["views"][:, :, 0] == 1

        step_alpha_sigma = torch.zeros_like(view_pairs[:, :, 1])
        step_alpha_sigma[view_pairs[:, :, 1] < 20. / 90] = self.alpha_weights_sigma1
        step_alpha_sigma[view_pairs[:, :, 1] > 20. / 90] = self.alpha_weights_sigma2
        alpha_weights_exponential = torch.zeros_like(view_pairs[:, :, 1])
        alpha_weights_exponential[view_pairs[:, :, 1] < 20. / 90] = self.alpha_weights_exponential1
        alpha_weights_exponential[view_pairs[:, :, 1] > 20. / 90] = self.alpha_weights_exponential2

        base_score = normal_distribution(view_pairs[:, :, 1],
                                         step_alpha_sigma, alpha_weights_exponential)

        gsd_score = normal_distribution(views[:, :, 4], self.gsd_weights_sigma, self.gsd_weights_exponential)
        scale_score = normal_distribution(view_pairs[:, :, 2], self.scale_weights_sigma, self.scale_weights_exponential)
        normal_score = normal_distribution(views[:, :, 5],
                                           self.angle_to_normal_weights_sigma,
                                           self.angle_to_normal_weights_exponential)
        distortion_score = normal_distribution(views[:, :, 6],
                                               self.distortion_weights_sigma, self.distortion_weights_exponential)

        # Filter out the invalid view
        view_pairs_mask = view_pairs[:, :, 0] != 0
        base_score = base_score * view_pairs_mask
        gsd_score = gsd_score * mask
        scale_score = scale_score * view_pairs_mask
        normal_score = normal_score * mask
        distortion_score = distortion_score * mask

        # Sum up
        flatten_mask = torch.triu_indices(gsd_score.shape[1], gsd_score.shape[1], offset=1)

        gsd_score = torch.einsum('bi,bj->bij', (gsd_score, gsd_score))
        gsd_score = torch.triu(gsd_score, diagonal=1)
        gsd_score = gsd_score[:, flatten_mask[0], flatten_mask[1]]

        normal_score = torch.einsum('bi,bj->bij', (normal_score, normal_score))
        normal_score = torch.triu(normal_score, diagonal=1)
        normal_score = normal_score[:, flatten_mask[0], flatten_mask[1]]

        distortion_score = torch.einsum('bi,bj->bij', (distortion_score, distortion_score))
        distortion_score = torch.triu(distortion_score, diagonal=1)
        distortion_score = distortion_score[:, flatten_mask[0], flatten_mask[1]]

        # final_score = base_score
        final_score = base_score * gsd_score * scale_score * normal_score * distortion_score
        final_score = torch.sum(final_score, dim=1)
        # final_score_exp = normal_distribution(final_score,self.model_weights_sigma,self.model_weights_exponential)
        return final_score

    def loss(self, v_point_attribute, v_prediction):
        # v_error = torch.clamp_max(v_error,1.)
        gt_reconstructability = v_point_attribute[:, 0]
        gt_error = v_point_attribute[:, 1]
        valid_point = gt_error < 9999999
        num_valid_point = valid_point.sum()

        valid_reconstructability_exp = v_prediction[valid_point]
        valid_error_exp = gt_error[valid_point]

        soft_spearman = spearmanr(valid_reconstructability_exp.unsqueeze(0), valid_error_exp.unsqueeze(0),
                                  regularization_strength=1e-2,
                                  regularization="kl")
        gt_spearman = stats.spearmanr(valid_reconstructability_exp.detach().cpu().numpy(),
                                      valid_error_exp.detach().cpu().numpy())[0]

        return soft_spearman, gt_spearman, num_valid_point


class Brute_force_nn(nn.Module):
    def __init__(self, hparams):
        super(Brute_force_nn, self).__init__()
        self.hydra_conf = hparams

        self.reconstructability_predictor = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.reconstructability_to_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        batch_size = v_data["views"].shape[0]
        if len(v_data["views"].shape) == 4:
            view_attribute = v_data["views"].view(-1, v_data["views"].shape[2], v_data["views"].shape[3])
        else:
            view_attribute = v_data["views"]
        view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
                torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.reconstructability_predictor(
            view_attribute[:, :, 1:7])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].bool()] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask
        predict_reconstructability = torch.sum(predict_reconstructability_per_view,
                                               dim=1)  # Sum up all the view contribution of one point
        predict_reconstructability = predict_reconstructability / torch.sum(view_attribute[:, :, 0].bool(),
                                                                            dim=1).unsqueeze(
            -1)  # Normalize the features
        predict_error = self.reconstructability_to_error(predict_reconstructability)  # Map it to point error
        if predict_error.shape[0] != batch_size:
            predict_error = predict_error.reshape(batch_size, -1, predict_error.shape[1])
        return predict_error

    def loss(self, v_point_attribute, v_prediction):
        gt_reconstructability = v_point_attribute[:, 0]
        gt_error = v_point_attribute[:, 1:2]

        loss = torch.nn.functional.mse_loss(v_prediction, gt_error)
        gt_spearman = stats.spearmanr(v_prediction.detach().cpu().numpy(),
                                      gt_error.detach().cpu().numpy())[0]

        return loss, gt_spearman, 0


class Correlation_nn(nn.Module):
    def __init__(self, hparams, v_drop_out=0.5):
        super(Correlation_nn, self).__init__()
        self.hydra_conf = hparams
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Dropout(v_drop_out),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(v_drop_out),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.view_feature_fusioner1 = nn.MultiheadAttention(embed_dim=256, num_heads=2, dropout=v_drop_out)
        self.view_feature_fusioner_linear1 = nn.Linear(256, 256)
        self.view_feature_fusioner_relu1 = nn.LeakyReLU()
        self.view_feature_fusioner2 = nn.MultiheadAttention(embed_dim=256, num_heads=2, dropout=v_drop_out)
        self.view_feature_fusioner_linear2 = nn.Linear(256, 256)
        self.view_feature_fusioner_relu2 = nn.LeakyReLU()

        self.reconstructability_to_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        for m in self.parameters():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        batch_size = v_data["views"].shape[0]
        if len(v_data["views"].shape) == 4:
            view_attribute = v_data["views"].view(-1, v_data["views"].shape[2], v_data["views"].shape[3])
        else:
            view_attribute = v_data["views"]
        view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
                torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:7])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].bool()] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        predict_reconstructability = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break
            feature_item = predict_reconstructability_per_view[
                           i * 1024:min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])]
            feature_item = torch.transpose(feature_item, 0, 1)
            #
            attention_result = self.view_feature_fusioner1(feature_item, feature_item, feature_item)
            attention_result = self.view_feature_fusioner_linear1(torch.transpose(attention_result[0], 0, 1))
            attention_result = self.view_feature_fusioner_relu1(attention_result)
            feature_item = torch.transpose(attention_result, 0, 1)
            attention_result = self.view_feature_fusioner2(feature_item, feature_item, feature_item)
            attention_result = self.view_feature_fusioner_linear2(torch.transpose(attention_result[0], 0, 1))
            attention_result = self.view_feature_fusioner_relu2(attention_result)

            predict_reconstructability.append(attention_result)
        predict_reconstructability = torch.cat(predict_reconstructability, dim=0)
        predict_reconstructability = torch.sum(predict_reconstructability,
                                               dim=1)  # Sum up all the view contribution of one point
        predict_reconstructability = predict_reconstructability / torch.sum(view_attribute[:, :, 0].bool(),
                                                                            dim=1).unsqueeze(
            -1)  # Normalize the features

        predict_error = self.reconstructability_to_error(predict_reconstructability)  # Map it to point error
        if predict_error.shape[0] != batch_size:
            predict_error = predict_error.reshape(batch_size, -1, predict_error.shape[1])
        return predict_error

    def loss(self, v_point_attribute, v_prediction):
        gt_reconstructability = v_point_attribute[:, 0]
        gt_error = v_point_attribute[:, 1:2]

        loss = torch.nn.functional.mse_loss(v_prediction, gt_error)
        gt_spearman = stats.spearmanr(v_prediction.detach().cpu().numpy(),
                                      gt_error.detach().cpu().numpy())[0]

        return loss, gt_spearman, 0


class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(128, 0.3, 32, 35 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(64, 0.6, 32, 64 + 3, [64, 64, 128], False)
        self.fp2 = PointNetFeaturePropagation(128 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l4_xyz, l4_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l4_xyz, l1_points, l4_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # x = F.leaky_relu_(self.conv1(l0_points))
        x = self.drop1(F.leaky_relu_(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return x, l4_points


class PointNet1(nn.Module):
    def __init__(self, v_num_channel_input, v_num_channel_output):
        super(PointNet1, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=v_num_channel_input)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, v_num_channel_output, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, -1)
        return x, trans_feat


class ViewFeatureFuser(nn.Module):
    def __init__(self):
        super(ViewFeatureFuser, self).__init__()
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True)
        self.view_feature_fusioner2 = nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True)

    # valid_flag, delta_theta, delta_phi, distance_ratio, normal_angle, central_angle
    def forward(self, v_data):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        # Normalize the view direction, no longer needed since we use phi and theta now
        # view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
        #         torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].type(torch.bool)] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            attn_mask_item = torch.ones((feature_item.shape[0], feature_item.shape[1], feature_item.shape[1],),
                                        dtype=torch.bool, device=feature_item.device)
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            for i_batch in range(feature_item.shape[0]):
                attn_mask_item[i_batch] = torch.logical_not(valid_mask_item)[i_batch, :, 0].float().unsqueeze(1) @ (
                    torch.logical_not(valid_mask_item)[i_batch, :, 0].float().T).unsqueeze(0)
            attn_mask_item = torch.logical_not(attn_mask_item)

            # feature_item_filter = torch.transpose(feature_item, 0, 1)
            attention_result = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
                # src_mask=attn_mask_item.repeat_interleave(2,0), # The output will be NAN, which bring problem when backpropogate the gradient
            )
            attention_result[valid_mask_item] = 0
            attention_result = self.view_feature_fusioner2(
                attention_result,
                src_key_padding_mask=valid_mask_item[:, :, 0],
                # src_mask=attn_mask_item.repeat_interleave(2,0),
            )
            attention_result[
                valid_mask_item] = 0  # Set invalid feature to 0, because we will do sum operator on this feature

            view_features.append(attention_result)

        view_features = torch.cat(view_features, dim=0)
        view_features[torch.logical_not(valid_mask.type(torch.bool))] = 0
        view_features = torch.sum(view_features,
                                  dim=1)  # Sum up all the view contribution of one point
        view_features = view_features / 50  # Normalize the features, encode the view num here
        if len(v_data.shape) == 4:
            view_features = view_features.reshape(v_data.shape[0], -1,
                                                  view_features.shape[1])  # B * num_point * 256
        return view_features


class ViewFeatureFuser2(nn.Module):
    def __init__(self):
        super(ViewFeatureFuser2, self).__init__()
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True)
        self.view_feature_fusioner2 = nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True)
        self.max_error = 0.2
        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )

    # valid_flag, delta_theta, delta_phi, distance_ratio, normal_angle, central_angle
    def forward(self, v_data):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        # Normalize the view direction, no longer needed since we use phi and theta now
        # view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
        #         torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].type(torch.bool)] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            attn_mask_item = torch.ones((feature_item.shape[0], feature_item.shape[1], feature_item.shape[1],),
                                        dtype=torch.bool, device=feature_item.device)
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            for i_batch in range(feature_item.shape[0]):
                attn_mask_item[i_batch] = torch.logical_not(valid_mask_item)[i_batch, :, 0].float().unsqueeze(1) @ (
                    torch.logical_not(valid_mask_item)[i_batch, :, 0].float().T).unsqueeze(0)
            attn_mask_item = torch.logical_not(attn_mask_item)

            # feature_item_filter = torch.transpose(feature_item, 0, 1)
            attention_result = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
                # src_mask=attn_mask_item.repeat_interleave(2,0), # The output will be NAN, which bring problem when backpropogate the gradient
            )
            attention_result[valid_mask_item] = 0
            attention_result = self.view_feature_fusioner2(
                attention_result,
                src_key_padding_mask=valid_mask_item[:, :, 0],
                # src_mask=attn_mask_item.repeat_interleave(2,0),
            )
            attention_result[
                valid_mask_item] = 0  # Set invalid feature to 0, because we will do sum operator on this feature

            view_features.append(attention_result)

        view_features = torch.cat(view_features, dim=0)
        error_per_view = self.features_to_error(view_features)
        error_per_view[torch.logical_not(valid_mask[:, :, 0].type(torch.bool))] = 0
        error_per_view = self.max_error - torch.sum(error_per_view,
                                                    dim=1)  # Sum up all the view contribution of one point
        error_per_view = nn.functional.leaky_relu(error_per_view)
        if len(v_data.shape) == 4:
            error_per_view = error_per_view.reshape(v_data.shape[0], -1, 1)  # B * num_point * 1
        return error_per_view


class ViewFeatureFuser4(nn.Module):
    def __init__(self):
        super(ViewFeatureFuser4, self).__init__()
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = nn.TransformerEncoderLayer(256, 2, 256, 0.1, batch_first=True)
        self.view_feature_fusioner2 = nn.TransformerEncoderLayer(256, 2, 256, 0.1, batch_first=True)
        self.max_error = 0.2
        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)

    # valid_flag, delta_theta, delta_phi, distance_ratio, normal_angle, central_angle
    def forward(self, v_data):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].type(torch.bool)] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            attention_result = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[valid_mask_item] = 0
            attention_result = self.view_feature_fusioner2(
                attention_result,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[
                valid_mask_item] = 0  # Set invalid feature to 0, because we will do sum operator on this feature

            view_features.append(attention_result)

        view_features = torch.cat(view_features, dim=0)
        error_per_view = self.features_to_error(view_features)
        error_per_view = self.max_error - torch.sum(error_per_view,
                                                    dim=1)  # Sum up all the view contribution of one point
        if len(v_data.shape) == 4:
            error_per_view = error_per_view.reshape(v_data.shape[0], -1, 1)  # B * num_point * 1
        return error_per_view


class ViewFeatureFuser3(nn.Module):
    def __init__(self):
        super(ViewFeatureFuser3, self).__init__()
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = nn.TransformerEncoderLayer(256, 2, 256, 0.1, batch_first=True)
        self.view_feature_fusioner2 = nn.TransformerEncoderLayer(256, 2, 256, 0.1, batch_first=True)
        self.max_error = 0.2
        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)

        # for m in self.modules():
        #     if isinstance(m, (nn.Linear,)):
        #         nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight)

    # valid_flag, delta_theta, delta_phi, distance_ratio, normal_angle, central_angle
    def forward(self, v_data):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        # Normalize the view direction, no longer needed since we use phi and theta now
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predict_reconstructability_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predict_reconstructability_per_view.shape[0], 1, 1]),
            predict_reconstructability_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            attention_result = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[valid_mask_item] = 0
            attention_result = self.view_feature_fusioner2(
                attention_result,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[
                valid_mask_item] = 0  # Set invalid feature to 0, because we will do sum operator on this feature

            view_features.append(attention_result)

        view_features = torch.cat(view_features, dim=0)
        error_per_view = self.features_to_error(view_features[:, 0])
        if len(v_data.shape) == 4:
            error_per_view = error_per_view.reshape(v_data.shape[0], -1, 1)  # B * num_point * 1
        return error_per_view


class ViewFeatureFuserWithPoints(nn.Module):
    def __init__(self):
        super(ViewFeatureFuserWithPoints, self).__init__()
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.view_feature_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True),
            2
        )
        self.point_feature_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True),
            2
        )

        self.point_feature_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, 2, 256, 0., F.leaky_relu_, batch_first=True),
            2
        )

    # v_data: valid_flag, delta_theta, delta_phi, distance_ratio, normal_angle, central_angle
    # v_points: n * 256
    def forward(self, v_data, v_points):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        if len(v_points.shape) == 3:
            point_attribute = v_points.view(-1, v_points.shape[2])
        else:
            point_attribute = v_points

        # Normalize the view direction, no longer needed since we use phi and theta now
        # view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
        #         torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].type(torch.bool)] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            point_feature_item = point_attribute[start_index:end_index]
            attn_mask_item = torch.ones((feature_item.shape[0], feature_item.shape[1], feature_item.shape[1],),
                                        dtype=torch.bool, device=feature_item.device)
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            for i_batch in range(feature_item.shape[0]):
                attn_mask_item[i_batch] = torch.logical_not(valid_mask_item)[i_batch, :, 0].float().unsqueeze(1) @ (
                    torch.logical_not(valid_mask_item)[i_batch, :, 0].float().T).unsqueeze(0)
            attn_mask_item = torch.logical_not(attn_mask_item)

            # feature_item_filter = torch.transpose(feature_item, 0, 1)
            fused_view_features = self.view_feature_encoder(feature_item, src_key_padding_mask=valid_mask_item[:, :, 0])
            fused_point_features = self.point_feature_encoder(point_feature_item.unsqueeze(0))
            attention_result = self.point_feature_decoder(
                fused_point_features[0].unsqueeze(1), fused_view_features,
                memory_key_padding_mask=valid_mask_item[:, :, 0],
            )

            view_features.append(attention_result[:, 0, :])

        view_features = torch.cat(view_features, dim=0)
        if len(v_data.shape) == 4:
            view_features = view_features.reshape(v_data.shape[0], -1,
                                                  view_features.shape[1])  # B * num_point * 256
        return view_features


class ImgFeatureFuser(nn.Module):
    def __init__(self):
        super(ImgFeatureFuser, self).__init__()
        self.view_feature_fusioner1 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear1 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu1 = nn.LeakyReLU()
        self.view_feature_fusioner2 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear2 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu2 = nn.LeakyReLU()

    """
    point_features: B * N_point * N_view * 32
    """

    def forward(self, point_features, point_features_mask):
        img_features = []
        for id_batch, item_batch in enumerate(point_features):
            pixel_position_features = item_batch

            # Fuse the image features
            pixel_position_features = pixel_position_features.transpose(0, 1)
            pixel_position_features = self.view_feature_fusioner1(pixel_position_features,
                                                                  pixel_position_features,
                                                                  pixel_position_features,
                                                                  # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                  key_padding_mask=
                                                                  point_features_mask[
                                                                      id_batch])[0]
            pixel_position_features = self.view_feature_fusioner_linear1(
                pixel_position_features.transpose(0, 1))
            pixel_position_features = self.view_feature_fusioner_relu1(pixel_position_features)
            pixel_position_features = pixel_position_features.transpose(0, 1)
            pixel_position_features = self.view_feature_fusioner2(pixel_position_features,
                                                                  pixel_position_features,
                                                                  pixel_position_features,
                                                                  # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                  key_padding_mask=
                                                                  point_features_mask[
                                                                      id_batch])[0]
            pixel_position_features = self.view_feature_fusioner_linear2(
                pixel_position_features.transpose(0, 1))
            pixel_position_features = self.view_feature_fusioner_relu2(pixel_position_features)

            img_features.append(torch.mean(pixel_position_features, dim=1))

        img_features = torch.stack(img_features, dim=0)
        return img_features


def loss_l2_error(v_point_attribute, v_prediction, v_is_img_involved=False):
    predicted_recon_error = v_prediction[:, :, 0:1]
    predicted_gt_error = v_prediction[:, :, 1:2]

    smith_reconstructability = v_point_attribute[:, :, 0]

    gt_recon_error = v_point_attribute[:, :, 1:2]
    recon_mask = (gt_recon_error != -1).bool()
    gt_gt_error = v_point_attribute[:, :, 2:3]
    gt_mask = (gt_gt_error != -1).bool()

    scaled_gt_recon_error = torch.clamp(gt_recon_error, -1., 1.)
    scaled_gt_gt_error = torch.clamp(gt_gt_error, -1., 1.)

    recon_loss = torch.nn.functional.l1_loss(predicted_recon_error[recon_mask], scaled_gt_recon_error[recon_mask])
    gt_loss = torch.zeros_like(recon_loss)
    if v_is_img_involved:
        gt_loss = torch.nn.functional.l1_loss(predicted_gt_error[gt_mask], scaled_gt_gt_error[gt_mask])

    return recon_loss, gt_loss, gt_loss if v_is_img_involved else recon_loss


def loss_spearman_error(v_point_attribute, v_prediction, v_is_img_involved=False, method="l2", normalized_factor=1.):
    predicted_recon_error = v_prediction[:, :, 0:1]
    predicted_gt_error = v_prediction[:, :, 1:2]

    smith_reconstructability = v_point_attribute[:, 0]

    gt_recon_error = v_point_attribute[:, :, 1:2]
    recon_mask = (gt_recon_error != -1).bool()
    gt_gt_error = v_point_attribute[:, :, 2:3]
    gt_mask = (gt_gt_error != -1).bool()

    scaled_gt_recon_error = torch.clamp(gt_recon_error, -1., 1.)
    scaled_gt_gt_error = torch.clamp(gt_gt_error, -1., 1.)

    recon_loss = spearmanr(
                predicted_recon_error[recon_mask].unsqueeze(0),
                gt_recon_error[recon_mask].unsqueeze(0),
                regularization=method, regularization_strength=normalized_factor
            )
    recon_loss = 1 - recon_loss  # We want to minimize the loss, which is maximizing the correlation factor
    gt_loss = torch.zeros_like(recon_loss)
    if v_is_img_involved:
        gt_loss = spearmanr(
            predicted_gt_error[gt_mask].unsqueeze(0),
            gt_gt_error[gt_mask].unsqueeze(0),
            regularization=method, regularization_strength=normalized_factor
        )
        gt_loss = 1 - gt_loss  # We want to minimize the loss, which is maximizing the correlation factor

    return recon_loss, gt_loss, gt_loss if v_is_img_involved else recon_loss


# class Uncertainty_Modeling_v2(torch.jit.ScriptModule):
class Uncertainty_Modeling_v2(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_v2, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]
        self.phase_1_extractor = PointNet1(3 + 256, 1)  # xyz + view_features
        self.phase_2_extractor = PointNet1(3 + 256 + 256 + 32 + 1,
                                           256)  # xyz + view_features + img_view_features + img_features + predicted recon
        self.phase_2_recon = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_inconsistency = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        # Img features
        self.img_feature_fusioner = ImgFeatureFuser()
        self.view_feature_fusioner = ViewFeatureFuser()
        # self.img_view_feature_fusioner = ViewFeatureFuser()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 0, 0].type(torch.bool)
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        view_features = self.view_feature_fusioner(v_data["views"])  # Note that some features are not reasonable
        view_features[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        points = v_data["points"]  # B * num_point * 5 (x,y,z, index, centre_point_index)
        points[torch.logical_not(valid_view_mask)] = 0
        point_features = torch.cat([points[:, :, :3], view_features], dim=2)
        # uncertainty = self.point_feature_extractor(point_features.transpose(1,2))[0].transpose(1,2) # PointNet++
        predict_reconstructability = self.phase_1_extractor(point_features.transpose(1, 2))[0]
        inconsistency_identifier = torch.zeros_like(predict_reconstructability)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        # Phase 2, only use viewpoint features to predict recon
        if self.is_involve_img:
            is_point_can_be_seen_with_at_least_one_view = (v_data["img_pose"][:, :, 0, 0]).type(torch.bool)
            img_view_features = torch.zeros_like(view_features)
            img_features = torch.zeros((batch_size, img_view_features.shape[1], 32),
                                       device=img_view_features.device)
            for id_batch in range(batch_size):
                valid_oblique_view_features_per_point = v_data["img_pose"][id_batch][
                    is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_per_point = v_data["point_features"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_mask_per_point = v_data["point_features_mask"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                # Extract view features of the pre-collected pattern
                # t = time.time()
                img_view_features_item = self.view_feature_fusioner(valid_oblique_view_features_per_point)
                # img_view_feature_time = time.time() - t

                # Calculate img features
                # t = time.time()
                img_features_item = self.img_feature_fusioner(
                    valid_oblique_img_features_per_point.unsqueeze(0),
                    valid_oblique_img_features_mask_per_point.unsqueeze(0))
                # img_feature_time = time.time() - t
                # t = time.time()

                img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_view_features_item \
                      + img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_features_item[0] \
                      + img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]

            # Phase 2, use img features to refine recon and predict proxy inconsistency
            point_features_plus = torch.cat(
                [
                    points[:, :, :3],
                    view_features,
                    predict_reconstructability,
                    img_view_features,
                    img_features], dim=2)
            predict_features = self.phase_2_extractor(point_features_plus.transpose(1, 2))[0]
            delta_recon = self.phase_2_recon(predict_features)
            inconsistency = self.phase_2_inconsistency(predict_features)

            # Extract the result
            predict_reconstructability = predict_reconstructability + delta_recon[:, :, 0:1]
            inconsistency_identifier = inconsistency_identifier + inconsistency[:, :, 0:1]
            inconsistency_identifier = torch.cat(
                [inconsistency_identifier, is_point_can_be_seen_with_at_least_one_view.unsqueeze(-1)], dim=-1)

        # Done
        predict_result = torch.cat([predict_reconstructability, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        elif self.hydra_conf["trainer"]["loss"] == "loss_truncated_entropy":
            return loss_truncated_entropy(v_point_attribute, v_prediction)
        elif self.hydra_conf["trainer"]["loss"] == "loss_l2_recon":
            return loss_l2_recon_error(v_point_attribute, v_prediction)
        else:
            raise


class Uncertainty_Modeling_wo_pointnet(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]
        self.phase_1_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        self.phase_2_recon = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_inconsistency = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        # Img features
        self.img_feature_fusioner = ImgFeatureFuser()
        self.view_feature_fusioner = ViewFeatureFuser()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        view_features = self.view_feature_fusioner(v_data["views"])  # Note that some features are not reasonable

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        view_features[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        predict_reconstructability = self.phase_1_extractor(view_features)
        inconsistency_identifier = torch.zeros_like(predict_reconstructability)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        # Phase 2, only use viewpoint features to predict recon
        if self.is_involve_img:
            is_point_can_be_seen_with_at_least_one_view = (v_data["img_pose"][:, :, 0, 0]).type(torch.bool)
            img_view_features = torch.zeros_like(view_features)
            img_features = torch.zeros((batch_size, img_view_features.shape[1], 32),
                                       device=img_view_features.device)
            for id_batch in range(batch_size):
                valid_oblique_view_features_per_point = v_data["img_pose"][id_batch][
                    is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_per_point = v_data["point_features"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_mask_per_point = v_data["point_features_mask"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                # Extract view features of the pre-collected pattern
                # t = time.time()
                img_view_features_item = self.view_feature_fusioner(valid_oblique_view_features_per_point)
                # img_view_feature_time = time.time() - t

                # Calculate img features
                # t = time.time()
                img_features_item = self.img_feature_fusioner(
                    valid_oblique_img_features_per_point.unsqueeze(0),
                    valid_oblique_img_features_mask_per_point.unsqueeze(0))
                # img_feature_time = time.time() - t
                # t = time.time()

                img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_view_features_item \
                      + img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_features_item[0] \
                      + img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]

            # Phase 2, use img features to refine recon and predict proxy inconsistency
            point_features_plus = torch.cat(
                [
                    # points[:, :, :3],
                    view_features,
                    predict_reconstructability,
                    img_view_features,
                    img_features], dim=2)
            predict_features = self.phase_2_extractor(point_features_plus)
            delta_recon = self.phase_2_recon(predict_features)
            inconsistency = self.phase_2_inconsistency(predict_features)

            # Extract the result
            predict_reconstructability = predict_reconstructability + delta_recon[:, :, 0:1]
            inconsistency_identifier = inconsistency_identifier + inconsistency[:, :, 0:1]
            inconsistency_identifier = torch.cat(
                [inconsistency_identifier, is_point_can_be_seen_with_at_least_one_view.unsqueeze(-1)], dim=-1)

        # Done
        predict_result = torch.cat([predict_reconstructability, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class Uncertainty_Modeling_wo_pointnet2(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet2, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]
        self.phase_1_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        self.phase_2_recon = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_inconsistency = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        # Img features
        self.img_feature_fusioner = ImgFeatureFuser()
        self.view_feature_fusioner = ViewFeatureFuser2()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        point_error = self.view_feature_fusioner(v_data["views"])  # Note that some features are not reasonable

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        inconsistency_identifier = torch.zeros_like(point_error)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)

        # Done
        predict_result = torch.cat([point_error, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class Uncertainty_Modeling_wo_pointnet3(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet3, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.view_feature_fusioner = ViewFeatureFuser3()

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        point_error = self.view_feature_fusioner(v_data["views"])  # Note that some features are not reasonable

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        inconsistency_identifier = torch.zeros_like(point_error)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)

        # Done
        predict_result = torch.cat([point_error, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class Uncertainty_Modeling_wo_pointnet4(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet4, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.view_feature_fusioner = ViewFeatureFuser4()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        point_error = self.view_feature_fusioner(v_data["views"])  # Note that some features are not reasonable

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        inconsistency_identifier = torch.zeros_like(point_error)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)

        # Done
        predict_result = torch.cat([point_error, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class TFEncorder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 batch_first=False, add_bias_kv=False,
                 add_norm=False) -> None:
        super(TFEncorder, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            add_bias_kv=add_bias_kv)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.add_norm = add_norm
        if self.add_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[
        Tensor, Optional[Tensor]]:
        x = src

        dx, weights = self._sa_block(x, src_mask, src_key_padding_mask)
        dx = torch.nan_to_num(dx)
        x = x + dx
        if self.add_norm:
            x = self.norm1(x)
        x = x + self._ff_block(x)
        if self.add_norm:
            x = self.norm2(x)
        return x, weights

    # self-attention block
    def _sa_block(self, x,
                  attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None):
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        return self.dropout1(x), weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Uncertainty_Modeling_wo_pointnet5(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet5, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 512, 0.1, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(256, 2, dropout=0.1, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        for m in self.view_feature_extractor.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.normal_(m.bias, -bound, bound)

        nn.init.kaiming_normal_(self.view_feature_fusioner1.self_attn.in_proj_weight)

        init.normal_(self.view_feature_fusioner1.self_attn.in_proj_bias)
        init.normal_(self.view_feature_fusioner1.self_attn.out_proj.bias)
        init.xavier_normal_(self.view_feature_fusioner1.self_attn.bias_k)
        init.xavier_normal_(self.view_feature_fusioner1.self_attn.bias_v)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.view(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predict_reconstructability_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predict_reconstructability_per_view.shape[0], 1, 1]),
            predict_reconstructability_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        weights: List[Tensor] = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            attention_result, weight_item = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[valid_mask_item] = 0
            view_features.append(attention_result)
            assert weight_item is not None
            weights.append(weight_item)

        view_features = torch.cat(view_features, dim=0)
        point_error = self.features_to_error(view_features[:, 0])
        if len(views.shape) == 4:
            point_error = point_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0

        inconsistency_identifier = torch.zeros_like(point_error)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        predict_result = torch.cat([point_error, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        return predict_result, weights

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class Uncertainty_Modeling_wo_pointnet6(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet6, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 512, 0.1, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(256, 2, dropout=0.1, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.max_error = 0.2

        for m in self.view_feature_extractor.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.normal_(m.bias, -bound, bound)

        nn.init.kaiming_normal_(self.view_feature_fusioner1.self_attn.in_proj_weight)

        init.normal_(self.view_feature_fusioner1.self_attn.in_proj_bias)
        init.normal_(self.view_feature_fusioner1.self_attn.out_proj.bias)
        init.xavier_normal_(self.view_feature_fusioner1.self_attn.bias_k)
        init.xavier_normal_(self.view_feature_fusioner1.self_attn.bias_v)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.view(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 0].type(torch.bool)] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        view_features = []
        weights = []
        for i in range(predict_reconstructability_per_view.shape[0] // 1024 + 1):
            if i * 1024 == predict_reconstructability_per_view.shape[0]:
                break

            start_index = i * 1024
            end_index = min(i * 1024 + 1024, predict_reconstructability_per_view.shape[0])

            feature_item = predict_reconstructability_per_view[start_index:end_index]
            valid_mask_item = torch.logical_not(valid_mask[start_index:end_index])
            attention_result, weight_item = self.view_feature_fusioner1(
                feature_item,
                src_key_padding_mask=valid_mask_item[:, :, 0],
            )
            attention_result[valid_mask_item] = 0
            view_features.append(attention_result)
            weights.append(weight_item)

        view_features = torch.cat(view_features, dim=0)
        point_error = self.features_to_error(view_features)
        point_error = self.max_error - point_error.sum(dim=1)
        if len(views.shape) == 4:
            point_error = point_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0

        inconsistency_identifier = torch.zeros_like(point_error)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        predict_result = torch.cat([point_error, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


class TFDecorder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 batch_first=False,
                 add_bias_kv=False, add_norm=False) -> None:
        super(TFDecorder, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            add_bias_kv=add_bias_kv)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 add_bias_kv=add_bias_kv)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.add_norm = add_norm
        if self.add_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, v_point_features_from_img: Tensor, v_fused_view_features: Tensor,
                v_point_features_mask: Optional[Tensor] = None):
        x = v_point_features_from_img

        dx, sa_weights = self._sa_block(x, key_padding_mask=v_point_features_mask)
        x = x + dx
        if self.add_norm:
            x = self.norm1(x)
        dx2, mha_weights = self.multihead_attn(v_fused_view_features, x, x,
                                               key_padding_mask=v_point_features_mask,
                                               need_weights=True)
        x = v_fused_view_features + dx2
        if self.add_norm:
            x = self.norm2(x)
        x = x + self._ff_block(x)
        if self.add_norm:
            x = self.norm3(x)
        return x, sa_weights, mha_weights

    def _sa_block(self, x,
                  attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None):
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        return self.dropout1(x), weights

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]):
        x, weights = self.multihead_attn(x, mem, mem,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=True)
        return self.dropout2(x), weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class Uncertainty_Modeling_wo_pointnet7(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet7, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 512, 0.1, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(256, 2, dropout=0.1, batch_first=True,
                                                                   add_bias_kv=True)

        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
        )
        self.img_feature_fusioner1 = TFDecorder(256, 2, 256, 0.1, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(256, 2, dropout=0.1, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_error = nn.Sequential(
            nn.Linear(256, 1),
        )
        self.features_to_uncertainty = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.view(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predict_reconstructability_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predict_reconstructability_per_view.shape[0], 1, 1]),
            predict_reconstructability_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predict_reconstructability_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predict_reconstructability_per_view = predict_reconstructability_per_view * valid_mask

        fused_view_features, weights = self.view_feature_fusioner1(
            predict_reconstructability_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
        )
        fused_view_features[torch.logical_not(valid_mask)] = 0

        fused_view_features = fused_view_features[:, 0]
        point_error = self.features_to_error(fused_view_features)
        if len(views.shape) == 4:
            point_error = point_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        point_error[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0

        uncertainty = torch.zeros_like(point_error)
        if self.is_involve_img:
            fused_view_features
            point_features_from_imgs = v_data["point_features"]
            point_features_mask = v_data["point_features_mask"]

            point_features_from_imgs = self.img_feature_expander(point_features_from_imgs)
            point_features_from_imgs = point_features_from_imgs * (1 - point_features_mask.float()).unsqueeze(-1).tile(
                1, 1, point_features_from_imgs.shape[2])

            fused_point_feature, point_feature_weight, cross_weight = self.img_feature_fusioner1(
                point_features_from_imgs, fused_view_features.unsqueeze(1),
                v_point_features_mask=point_features_mask)
            uncertainty = self.features_to_uncertainty(fused_point_feature)
        predict_result = torch.cat([point_error, uncertainty], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0

        return predict_result, weights

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction, 5)
        else:
            return loss_l2_recon_error(v_point_attribute, v_prediction)


# version 51
class Uncertainty_Modeling_wo_pointnet8(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet8, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        # ========================================Phase 0========================================
        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.reshape(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predicted_recon_error_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predicted_recon_error_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predicted_recon_error_per_view.shape[0], 1, 1]),
            predicted_recon_error_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predicted_recon_error_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predicted_recon_error_per_view = predicted_recon_error_per_view * valid_mask

        value_mask = torch.logical_not(valid_mask)[:, :, 0].unsqueeze(-1).tile([1, 1, valid_mask.shape[1]])
        value_mask = torch.logical_or(value_mask, torch.transpose(value_mask, 1, 2))

        fused_view_features, weights = self.view_feature_fusioner1(
            predicted_recon_error_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
            # src_mask = value_mask
        )
        fused_view_features = fused_view_features * valid_mask

        fused_view_features = fused_view_features[:, 0]
        predicted_recon_error = self.features_to_recon_error(fused_view_features)
        if len(views.shape) == 4:
            predicted_recon_error = predicted_recon_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        predicted_recon_error = valid_view_mask.unsqueeze(-1) * predicted_recon_error  # Mark these features to 0

        # ========================================Phase 1========================================
        predicted_gt_error = torch.zeros_like(predicted_recon_error)
        cross_weight: Optional[Tensor] = None
        if self.is_involve_img:
            fused_view_features
            point_features_from_imgs = v_data["point_features"]
            point_features_mask = v_data["point_features_mask"]

            point_features_from_imgs = self.img_feature_expander(point_features_from_imgs)
            point_features_from_imgs = point_features_from_imgs * (1 - point_features_mask.float()).unsqueeze(-1).tile(
                1, 1, 1, point_features_from_imgs.shape[3])

            point_features_from_imgs = point_features_from_imgs.reshape([
                -1,
                point_features_from_imgs.shape[2],
                point_features_from_imgs.shape[3]
            ])
            point_features_mask = point_features_mask.reshape([
                -1, point_features_mask.shape[2]
            ])

            fused_point_feature, point_feature_weight, cross_weight = self.img_feature_fusioner1(
                point_features_from_imgs, fused_view_features.unsqueeze(1),
                v_point_features_mask=point_features_mask)
            predicted_gt_error = self.features_to_gt_error(fused_point_feature)
            if len(views.shape) == 4:
                predicted_gt_error = predicted_gt_error.reshape(views.shape[0], -1, 1)
        predict_result = torch.cat([predicted_recon_error, predicted_gt_error], dim=2)
        predict_result = torch.tile(valid_view_mask.unsqueeze(-1), [1, 1, 2]) * predict_result

        return predict_result, (weights, cross_weight),

    def loss(self, v_point_attribute, v_prediction):
        if self.hydra_conf["trainer"]["loss"] == "loss_l2_error":
            return loss_l2_error(v_point_attribute, v_prediction, self.is_involve_img)
        else:
            return loss_spearman_error(v_point_attribute, v_prediction, self.is_involve_img)

    def init_linear(self, item):
        for m in item.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.normal_(m.bias, -bound, bound)

    def init_attention(self, item):
        nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

        init.normal_(item.self_attn.in_proj_bias)
        init.normal_(item.self_attn.out_proj.bias)
        if self.hydra_conf["model"]["add_bias_kv"]:
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)


# Delete dropout in the first few layer; useful; version 52
class Uncertainty_Modeling_wo_pointnet9(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet9, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 1, 256, 0.1, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(256, 1, dropout=0.1, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(256, 1),
        )

        # ========================================Phase 1========================================
        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.img_feature_fusioner1 = TFDecorder(256, 1, 256, 0.1, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(256, 1, dropout=0.1, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_gt_error = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_recon_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)


# Delete dropout in transformer; not useful; version 53
class Uncertainty_Modeling_wo_pointnet10(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet10, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 1, 256, 0.0, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(256, 1, dropout=0.0, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(256, 1),
        )

        # ========================================Phase 1========================================
        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.img_feature_fusioner1 = TFDecorder(256, 1, 256, 0.0, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(256, 1, dropout=0.0, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_gt_error = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_recon_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)


# Delete dropout in transformer and reduce feature dimension; useful; version 54
class Uncertainty_Modeling_wo_pointnet11(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet11, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.0, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(128, 1, dropout=0.0, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        # ========================================Phase 1========================================
        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.0, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(128, 1, dropout=0.0, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_gt_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_recon_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)


# version 55; version 56 (4 gpus)
# more dimension
# it is complicated to say the effect of multi gpu training
# Generally, multi-gpu version has lower loss
# more dimension is not useful
class Uncertainty_Modeling_wo_pointnet12(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet12, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )
        self.view_feature_fusioner1 = TFEncorder(512, 1, 512, 0.0, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(512, 1, dropout=0.0, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(512, 1),
        )

        # ========================================Phase 1========================================
        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )
        self.img_feature_fusioner1 = TFDecorder(512, 1, 512, 0.0, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(512, 1, dropout=0.0, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_gt_error = nn.Sequential(
            nn.Linear(512, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 512))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_recon_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)


# version 55; version 56
# 128 dimension with dropout in transformer
# version 83 with normalized_l2_loss: not good
# version 85 with l2_loss: better than net14
class Uncertainty_Modeling_wo_pointnet13(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet13, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.1, batch_first=True)
        self.view_feature_fusioner1.self_attn = MultiheadAttention(128, 1, dropout=0.1, batch_first=True,
                                                                   add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        # ========================================Phase 1========================================
        self.img_feature_expander = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.1, batch_first=True)
        self.img_feature_fusioner1.self_attn = MultiheadAttention(128, 1, dropout=0.1, batch_first=True,
                                                                  add_bias_kv=True)

        self.features_to_gt_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        for module in [self.view_feature_extractor, self.img_feature_expander]:
            for m in module.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        for transformer_module in [self.view_feature_fusioner1, self.img_feature_fusioner1]:
            nn.init.kaiming_normal_(transformer_module.self_attn.in_proj_weight)

            init.normal_(transformer_module.self_attn.in_proj_bias)
            init.normal_(transformer_module.self_attn.out_proj.bias)
            init.xavier_normal_(transformer_module.self_attn.bias_k)
            init.xavier_normal_(transformer_module.self_attn.bias_v)

        if self.hydra_conf["model"]["open_weights"] is False:
            self.view_feature_extractor.requires_grad_(False)
            self.view_feature_fusioner1.requires_grad_(False)
            self.features_to_recon_error.requires_grad_(False)
            self.magic_class_token.requires_grad_(False)


# version 83 with normalized_l2_loss: not good
# version 85 with l2_loss: better than net14
class Uncertainty_Modeling_wo_pointnet14(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet14, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 512, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        self.init_linear(self.view_feature_extractor)
        self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
            self.img_feature_fusioner1 = TFDecorder(256, 2, 512, 0.1, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.init_linear(self.img_feature_expander)
            self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)


# Lightweight version 14
class Uncertainty_Modeling_wo_pointnet15(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet15, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.2, batch_first=True, add_bias_kv=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        def init_linear(item):
            for m in item.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        def init_attention(item):
            nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

            init.normal_(item.self_attn.in_proj_bias)
            init.normal_(item.self_attn.out_proj.bias)
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        init_linear(self.view_feature_extractor)
        init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.2, batch_first=True, add_bias_kv=True)

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            init_linear(self.img_feature_expander)
            init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)


# Lightweight version 14 with norm
class Uncertainty_Modeling_wo_pointnet16(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet16, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.2, batch_first=True, add_bias_kv=True, add_norm=True)

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        def init_linear(item):
            for m in item.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        def init_attention(item):
            nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

            init.normal_(item.self_attn.in_proj_bias)
            init.normal_(item.self_attn.out_proj.bias)
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        init_linear(self.view_feature_extractor)
        init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.2, batch_first=True, add_bias_kv=True)

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            init_linear(self.img_feature_expander)
            init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)


# Spearman version
class Uncertainty_Modeling_wo_pointnet17(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet17, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 256, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        def init_linear(item):
            for m in item.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        def init_attention(item):
            nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

            init.normal_(item.self_attn.in_proj_bias)
            init.normal_(item.self_attn.out_proj.bias)
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
            self.img_feature_fusioner1 = TFDecorder(256, 2, 256, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

            init_linear(self.img_feature_expander)
            init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)

        init_linear(self.view_feature_extractor)
        init_attention(self.view_feature_fusioner1)

    def forward(self, v_data: Dict[str, torch.Tensor]):
        predict_result, (weights, cross_weight) = super(Uncertainty_Modeling_wo_pointnet17, self).forward(v_data)
        predict_result = 1 - torch.sigmoid(predict_result) * 2

        return predict_result, (weights, cross_weight)


# Lightweight Spearman version
class Uncertainty_Modeling_wo_pointnet18(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet18, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        self.init_linear(self.view_feature_extractor)
        self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            self.init_linear(self.img_feature_expander)
            self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)

    def forward(self, v_data: Dict[str, torch.Tensor]):
        predict_result, (weights, cross_weight) = super(Uncertainty_Modeling_wo_pointnet18, self).forward(v_data)
        predict_result = torch.sigmoid(predict_result)

        return predict_result, (weights, cross_weight)

    def loss(self, v_point_attribute, v_prediction):
        if self.hydra_conf["trainer"]["loss"] == "loss_l2_error":
            return loss_l2_error(v_point_attribute, v_prediction, self.is_involve_img)
        else:
            return loss_spearman_error(v_point_attribute, v_prediction, self.is_involve_img,
                                       method=self.hydra_conf["model"]["spearman_method"],
                                       normalized_factor=self.hydra_conf["model"]["spearman_factor"])


# Lightweight L2 version
class Uncertainty_Modeling_wo_pointnet19(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet19, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 256, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        def init_linear(item):
            for m in item.modules():
                if isinstance(m, (nn.Linear,)):
                    nn.init.kaiming_normal_(m.weight)
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.normal_(m.bias, -bound, bound)

        def init_attention(item):
            nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

            init.normal_(item.self_attn.in_proj_bias)
            init.normal_(item.self_attn.out_proj.bias)
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 256, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 1),
            )
            init_linear(self.img_feature_expander)
            init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)

        init_linear(self.view_feature_extractor)
        init_attention(self.view_feature_fusioner1)


# Lightweight L2 version with whole view features
class Uncertainty_Modeling_wo_pointnet20(Uncertainty_Modeling_wo_pointnet18):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet20, self).__init__(hparams)

    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        # ========================================Phase 0========================================
        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.reshape(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predicted_recon_error_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predicted_recon_error_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predicted_recon_error_per_view.shape[0], 1, 1]),
            predicted_recon_error_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predicted_recon_error_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predicted_recon_error_per_view = predicted_recon_error_per_view * valid_mask

        fused_view_features, weights = self.view_feature_fusioner1(
            predicted_recon_error_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
        )
        fused_view_features = fused_view_features * valid_mask

        fused_view_features_token = fused_view_features[:, 0]
        predicted_recon_error = self.features_to_recon_error(fused_view_features_token)
        if len(views.shape) == 4:
            predicted_recon_error = predicted_recon_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        predicted_recon_error = valid_view_mask.unsqueeze(-1) * predicted_recon_error  # Mark these features to 0

        # ========================================Phase 1========================================
        predicted_gt_error = torch.zeros_like(predicted_recon_error)
        cross_weight: Optional[Tensor] = None
        if self.is_involve_img:
            fused_view_features
            point_features_from_imgs = v_data["point_features"]
            point_features_mask = v_data["point_features_mask"]

            point_features_from_imgs = self.img_feature_expander(point_features_from_imgs)
            point_features_from_imgs = point_features_from_imgs * (1 - point_features_mask.float()).unsqueeze(-1).tile(
                1, 1, 1, point_features_from_imgs.shape[3])

            point_features_from_imgs = point_features_from_imgs.reshape([
                -1,
                point_features_from_imgs.shape[2],
                point_features_from_imgs.shape[3]
            ])
            point_features_mask = point_features_mask.reshape([
                -1, point_features_mask.shape[2]
            ])

            fused_point_feature, point_feature_weight, cross_weight = self.img_feature_fusioner1(
                point_features_from_imgs, fused_view_features,
                v_point_features_mask=point_features_mask)
            fused_point_feature_token = fused_point_feature[:, 0]
            predicted_gt_error = self.features_to_gt_error(fused_point_feature_token)
            if len(views.shape) == 4:
                predicted_gt_error = predicted_gt_error.reshape(views.shape[0], -1, 1)
        predict_result = torch.cat([predicted_recon_error, predicted_gt_error], dim=2)
        predict_result = torch.tile(valid_view_mask.unsqueeze(-1), [1, 1, 2]) * predict_result

        return predict_result, (weights, cross_weight),


# Lightweight Spearman version with monocity
class Uncertainty_Modeling_wo_pointnet21(Uncertainty_Modeling_wo_pointnet8):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_pointnet21, self).__init__(hparams)
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.0, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.init_linear(self.view_feature_extractor)
        self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            self.init_linear(self.img_feature_expander)
            self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)

    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        # ========================================Phase 0========================================
        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.reshape(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predicted_recon_error_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view

        valid_mask = torch.zeros_like(predicted_recon_error_per_view)  # Filter out the unusable view
        valid_mask[view_attribute[:, :, 1].type(torch.bool)] = 1
        predicted_recon_error_per_view = predicted_recon_error_per_view * valid_mask

        fused_view_features, weights = self.view_feature_fusioner1(
            predicted_recon_error_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
        )
        predicted_recon_error = self.features_to_recon_error(fused_view_features)
        predicted_recon_error = torch.sigmoid(predicted_recon_error)
        predicted_recon_error = predicted_recon_error * view_attribute[:, :, 1:2].type(torch.bool)
        predicted_recon_error = predicted_recon_error.sum(dim=-2)
        if len(views.shape) == 4:
            predicted_recon_error = predicted_recon_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        predicted_recon_error = valid_view_mask.unsqueeze(-1) * predicted_recon_error  # Mark these features to 0

        # ========================================Phase 1========================================
        predicted_gt_error = torch.zeros_like(predicted_recon_error)
        cross_weight: Optional[Tensor] = None
        if self.is_involve_img:
            fused_view_features
            point_features_from_imgs = v_data["point_features"]
            point_features_mask = v_data["point_features_mask"]

            point_features_from_imgs = self.img_feature_expander(point_features_from_imgs)
            point_features_from_imgs = point_features_from_imgs * (1 - point_features_mask.float()).unsqueeze(-1).tile(
                1, 1, 1, point_features_from_imgs.shape[3])

            point_features_from_imgs = point_features_from_imgs.reshape([
                -1,
                point_features_from_imgs.shape[2],
                point_features_from_imgs.shape[3]
            ])
            point_features_mask = point_features_mask.reshape([
                -1, point_features_mask.shape[2]
            ])

            fused_point_feature, point_feature_weight, cross_weight = self.img_feature_fusioner1(
                point_features_from_imgs, fused_view_features.unsqueeze(1),
                v_point_features_mask=point_features_mask)
            predicted_gt_error = self.features_to_gt_error(fused_point_feature)
            if len(views.shape) == 4:
                predicted_gt_error = predicted_gt_error.reshape(views.shape[0], -1, 1)
        predict_result = torch.cat([predicted_recon_error, predicted_gt_error], dim=2)
        predict_result = torch.tile(valid_view_mask.unsqueeze(-1), [1, 1, 2]) * predict_result

        return predict_result, (weights, cross_weight),

    def loss(self, v_point_attribute, v_prediction):
        if self.hydra_conf["trainer"]["loss"] == "loss_l2_error":
            return loss_l2_error(v_point_attribute, v_prediction, self.is_involve_img)
        else:
            return loss_spearman_error(v_point_attribute, v_prediction, self.is_involve_img,
                                       method=self.hydra_conf["model"]["spearman_method"],
                                       normalized_factor=self.hydra_conf["model"]["spearman_factor"])


class Correlation_net(nn.Module):
    def __init__(self, hparams):
        super(Correlation_net, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = v_data["views"][:, :, 1, 0] > 1e-3
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        # ========================================Phase 0========================================
        views = v_data["views"]

        if len(views.shape) == 4:
            view_attribute = views.reshape(-1, views.shape[2], views.shape[3])
        else:
            view_attribute = views
        # Normalize the view direction, no longer needed since we use phi and theta now
        predicted_recon_error_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:6])  # Compute the reconstructabilty of every single view
        predicted_recon_error_per_view = torch.cat([
            torch.tile(self.magic_class_token, [predicted_recon_error_per_view.shape[0], 1, 1]),
            predicted_recon_error_per_view
        ], dim=1)

        valid_mask = torch.zeros_like(predicted_recon_error_per_view)  # Filter out the unusable view
        valid_mask[
            torch.cat(
                [torch.tensor([True], device=view_attribute.device).reshape(-1, 1).tile([view_attribute.shape[0], 1]),
                 view_attribute[:, :, 0].type(torch.bool)], dim=1)
        ] = 1
        predicted_recon_error_per_view = predicted_recon_error_per_view * valid_mask

        value_mask = torch.logical_not(valid_mask)[:, :, 0].unsqueeze(-1).tile([1, 1, valid_mask.shape[1]])
        value_mask = torch.logical_or(value_mask, torch.transpose(value_mask, 1, 2))

        fused_view_features, weights = self.view_feature_fusioner1(
            predicted_recon_error_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
            # src_mask = value_mask
        )
        weights = weights[:,0]
        fused_view_features = fused_view_features * valid_mask

        fused_view_features = fused_view_features[:, 0]
        predicted_recon_error = self.features_to_recon_error(fused_view_features)
        if len(views.shape) == 4:
            predicted_recon_error = predicted_recon_error.reshape(views.shape[0], -1, 1)  # B * num_point * 1

        v_data["views"][torch.logical_not(valid_view_mask)] = 0
        predicted_recon_error = valid_view_mask.unsqueeze(-1) * predicted_recon_error  # Mark these features to 0

        # ========================================Phase 1========================================
        predicted_gt_error = torch.zeros_like(predicted_recon_error)
        cross_weight: Optional[Tensor] = None
        if self.is_involve_img:
            fused_view_features
            point_features_from_imgs = v_data["point_features"]
            point_features_mask = v_data["point_features_mask"]

            point_features_from_imgs = self.img_feature_expander(point_features_from_imgs)
            point_features_from_imgs = point_features_from_imgs * (1 - point_features_mask.float()).unsqueeze(-1).tile(
                1, 1, 1, point_features_from_imgs.shape[3])

            point_features_from_imgs = point_features_from_imgs.reshape([
                -1,
                point_features_from_imgs.shape[2],
                point_features_from_imgs.shape[3]
            ])
            point_features_mask = point_features_mask.reshape([
                -1, point_features_mask.shape[2]
            ])

            fused_point_feature, point_feature_weight, cross_weight = self.img_feature_fusioner1(
                point_features_from_imgs, fused_view_features.unsqueeze(1),
                v_point_features_mask=point_features_mask)
            predicted_gt_error = self.features_to_gt_error(fused_point_feature)
            if len(views.shape) == 4:
                predicted_gt_error = predicted_gt_error.reshape(views.shape[0], -1, 1)
        predict_result = torch.cat([predicted_recon_error, predicted_gt_error], dim=2)
        predict_result = torch.tile(valid_view_mask.unsqueeze(-1), [1, 1, 2]) * predict_result

        return predict_result, (weights, cross_weight),

    def init_linear(self, item):
        for m in item.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.normal_(m.bias, -bound, bound)

    def init_attention(self, item):
        nn.init.kaiming_normal_(item.self_attn.in_proj_weight)

        init.normal_(item.self_attn.in_proj_bias)
        init.normal_(item.self_attn.out_proj.bias)
        if self.hydra_conf["model"]["add_bias_kv"]:
            init.xavier_normal_(item.self_attn.bias_k)
            init.xavier_normal_(item.self_attn.bias_v)

    def loss(self, v_point_attribute, v_prediction):
        if self.hydra_conf["trainer"]["loss"] == "loss_l2_error":
            return loss_l2_error(v_point_attribute, v_prediction, self.is_involve_img)
        else:
            return loss_spearman_error(v_point_attribute, v_prediction, self.is_involve_img,
                                       method=self.hydra_conf["model"]["spearman_method"],
                                       normalized_factor=self.hydra_conf["model"]["spearman_factor"])


class Correlation_l2_error_net(Correlation_net):
    def __init__(self, hparams):
        super(Correlation_l2_error_net, self).__init__(hparams)

        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.view_feature_fusioner1 = TFEncorder(256, 2, 512, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 256))

        self.init_linear(self.view_feature_extractor)
        self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
            self.img_feature_fusioner1 = TFDecorder(256, 2, 512, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.init_linear(self.img_feature_expander)
            self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)


class Spearman_net(Correlation_net):
    def __init__(self, hparams):
        super(Spearman_net, self).__init__(hparams)
        # ========================================Phase 0========================================
        self.view_feature_extractor = nn.Sequential(
            nn.Linear(5, 128),
        )
        self.view_feature_fusioner1 = TFEncorder(128, 1, 128, 0.2, batch_first=True,
                                                 add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

        self.features_to_recon_error = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, 128))

        self.init_linear(self.view_feature_extractor)
        self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            self.img_feature_expander = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.img_feature_fusioner1 = TFDecorder(128, 1, 128, 0.2, batch_first=True,
                                                    add_bias_kv=self.hydra_conf["model"]["add_bias_kv"])

            self.features_to_gt_error = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            self.init_linear(self.img_feature_expander)
            self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)

    def forward(self, v_data: Dict[str, torch.Tensor]):
        predict_result, (weights, cross_weight) = super(Spearman_net, self).forward(v_data)
        predict_result = torch.sigmoid(predict_result)

        return predict_result, (weights, cross_weight)


class Uncertainty_Modeling_w_pointnet(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_w_pointnet, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]
        self.phase_1_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_extractor = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        self.phase_2_recon = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )
        self.phase_2_inconsistency = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 1),
        )

        # Img features
        self.img_feature_fusioner = ImgFeatureFuser()
        # self.view_feature_fusioner = ViewFeatureFuser()
        self.view_feature_fusioner = ViewFeatureFuserWithPoints()
        # self.img_view_feature_fusioner = ViewFeatureFuser()
        self.point_feature_extractor = PointNet1(6, 256)

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    # @torch.jit.script_method
    def forward(self, v_data: Dict[str, torch.Tensor]):
        batch_size = v_data["views"].shape[0]

        valid_view_mask = torch.abs(v_data["views"][:, :, 0, 0]) > .5
        # Fake generate 1 view for those point which can not been seen
        # in order to prevent NAN in attention module
        v_data["views"][torch.logical_not(valid_view_mask)] = 1

        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        # t = time.time()
        point_features = self.point_feature_extractor(torch.cat([
            v_data["points"][:, :, 0:3],  # xyz
            v_data["point_attribute"][:, :, 7:10]  # nx,ny,nz
        ], dim=2).transpose(1, 2))[0]

        view_features = self.view_feature_fusioner(v_data["views"],
                                                   point_features)  # Note that some features are not reasonable
        view_features[torch.logical_not(valid_view_mask)] = 0  # Mark these features to 0
        # view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        predict_reconstructability = self.phase_1_extractor(view_features)
        inconsistency_identifier = torch.zeros_like(predict_reconstructability)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        # Phase 2, only use viewpoint features to predict recon
        if self.is_involve_img:
            is_point_can_be_seen_with_at_least_one_view = (v_data["img_pose"][:, :, 0, 0]).type(torch.bool)
            img_view_features = torch.zeros_like(view_features)
            img_features = torch.zeros((batch_size, img_view_features.shape[1], 32),
                                       device=img_view_features.device)
            for id_batch in range(batch_size):
                valid_oblique_view_features_per_point = v_data["img_pose"][id_batch][
                    is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_per_point = v_data["point_features"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                valid_oblique_img_features_mask_per_point = v_data["point_features_mask"][
                    id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                # Extract view features of the pre-collected pattern
                # t = time.time()
                img_view_features_item = self.view_feature_fusioner(valid_oblique_view_features_per_point,
                                                                    point_features)
                # img_view_feature_time = time.time() - t

                # Calculate img features
                # t = time.time()
                img_features_item = self.img_feature_fusioner(
                    valid_oblique_img_features_per_point.unsqueeze(0),
                    valid_oblique_img_features_mask_per_point.unsqueeze(0))
                # img_feature_time = time.time() - t
                # t = time.time()

                img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_view_features_item \
                      + img_view_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]
                img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]] \
                    = img_features_item[0] \
                      + img_features[id_batch][is_point_can_be_seen_with_at_least_one_view[id_batch]]

            # Phase 2, use img features to refine recon and predict proxy inconsistency
            point_features_plus = torch.cat(
                [
                    # points[:, :, :3],
                    view_features,
                    predict_reconstructability,
                    img_view_features,
                    img_features], dim=2)
            predict_features = self.phase_2_extractor(point_features_plus)
            delta_recon = self.phase_2_recon(predict_features)
            inconsistency = self.phase_2_inconsistency(predict_features)

            # Extract the result
            predict_reconstructability = predict_reconstructability + delta_recon[:, :, 0:1]
            inconsistency_identifier = inconsistency_identifier + inconsistency[:, :, 0:1]
            inconsistency_identifier = torch.cat(
                [inconsistency_identifier, is_point_can_be_seen_with_at_least_one_view.unsqueeze(-1)], dim=-1)

        # Done
        predict_result = torch.cat([predict_reconstructability, inconsistency_identifier], dim=2)
        predict_result[torch.logical_not(valid_view_mask)] = 0.

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_gt_error(v_point_attribute, v_prediction)
        elif self.hydra_conf["trainer"]["loss"] == "loss_truncated_entropy":
            return loss_truncated_entropy(v_point_attribute, v_prediction)
        elif self.hydra_conf["trainer"]["loss"] == "loss_l2_recon":
            return loss_l2_recon_error(v_point_attribute, v_prediction)
        else:
            raise


class Uncertainty_Modeling_wo_dropout(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_wo_dropout, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.point_feature_extractor = PointNet1()

        # self.img_feature_extractor = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

        # self.geometrical_feature_extractor = Correlation_nn(hparams)
        # self.geometrical_feature_extractor = Brute_force_nn(hparams)
        self.geometrical_feature_extractor = Correlation_nn(hparams, 0)
        if not self.hydra_conf["model"]["open_weights"]:
            self.geometrical_feature_extractor.requires_grad_(False)

        self.view_feature_fusioner1 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear1 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu1 = nn.LeakyReLU()
        self.view_feature_fusioner2 = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        self.view_feature_fusioner_linear2 = nn.Linear(32, 32)
        self.view_feature_fusioner_relu2 = nn.LeakyReLU()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
        attention_time = 0
        pointnet_time = 0
        correlation_time = 0
        points = v_data["points"]
        if self.is_involve_img:
            img_features = []
            t = time.time()
            for id_batch, item_batch in enumerate(v_data["point_features"]):
                pixel_position_features = item_batch

                # Fuse the image features
                pixel_position_features = pixel_position_features.transpose(0, 1)
                pixel_position_features = self.view_feature_fusioner1(pixel_position_features,
                                                                      pixel_position_features,
                                                                      pixel_position_features,
                                                                      # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                      key_padding_mask=
                                                                      v_data["point_features_mask"][
                                                                          id_batch])[0]
                pixel_position_features = self.view_feature_fusioner_linear1(
                    pixel_position_features.transpose(0, 1))
                pixel_position_features = self.view_feature_fusioner_relu1(pixel_position_features)
                pixel_position_features = pixel_position_features.transpose(0, 1)
                pixel_position_features = self.view_feature_fusioner2(pixel_position_features,
                                                                      pixel_position_features,
                                                                      pixel_position_features,
                                                                      # attn_mask = v_data["point_features_mask"][id_batch])[0]
                                                                      key_padding_mask=
                                                                      v_data["point_features_mask"][
                                                                          id_batch])[0]
                pixel_position_features = self.view_feature_fusioner_linear2(
                    pixel_position_features.transpose(0, 1))
                pixel_position_features = self.view_feature_fusioner_relu2(pixel_position_features)

                img_features.append(torch.mean(pixel_position_features, dim=1))

            img_features = torch.stack(img_features, dim=0)
            point_features = torch.cat([points[:, :, :3], img_features], dim=2)
            attention_time += time.time() - t
            t = time.time()
            # uncertainty = self.point_feature_extractor(point_features.transpose(1,2))[0].transpose(1,2) # PointNet++
            uncertainty = self.point_feature_extractor(point_features.transpose(1, 2))[0][:, :, 0:1]
            inconsistency_identifier = self.point_feature_extractor(point_features.transpose(1, 2))[0][:, :, 1:2]
            pointnet_time += time.time() - t
            t = time.time()
        else:
            uncertainty = 1
            inconsistency_identifier = torch.zeros((points.shape[0], points.shape[1], 1), device=points.device)

        t = time.time()
        geometrical_feature = self.geometrical_feature_extractor(v_data)

        geometrical_feature_with_uncertainty = geometrical_feature * uncertainty

        correlation_time += time.time() - t
        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return torch.cat([geometrical_feature_with_uncertainty, inconsistency_identifier], dim=2)

    def loss(self, v_point_attribute, v_prediction):
        predicted_error = v_prediction[:, :, 0:1]
        predicted_inconsistency = v_prediction[:, :, 1:2]

        gt_reconstructability = v_point_attribute[:, 0]
        gt_max_error = v_point_attribute[:, :, 1:2]
        gt_avg_error = v_point_attribute[:, :, 2:3]
        gt_error_mask_error = (1 - v_point_attribute[:, :, 6:7]).bool()  # Good Point -> True(1)

        error_loss = torch.nn.functional.mse_loss(predicted_error[gt_error_mask_error],
                                                  gt_avg_error[gt_error_mask_error])
        if self.is_involve_img:
            inconsistency_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                predicted_inconsistency,
                v_point_attribute[:, :, 6:7])
        else:
            inconsistency_loss = torch.zeros_like(error_loss)
        # gt_spearmans=[]
        # for id_item in range(v_point_attribute.shape[0]):
        #     gt_spearman = stats.spearmanr(v_prediction[id_item,:,0].detach().cpu().numpy(),
        #                                   gt_avg_error[id_item,:,0].detach().cpu().numpy())[0]
        #     gt_spearmans.append(gt_spearman)

        # return loss, np.mean(gt_spearmans), 0
        return error_loss, inconsistency_loss, 100 * error_loss + inconsistency_loss
