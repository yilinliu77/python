import time
from typing import Dict

import torch
import torchvision
from scipy.stats import stats
from torch import nn
import numpy as np
import torch.nn.functional as F

from fast_soft_sort.pytorch_ops import soft_rank
from torchvision.models import resnet18

import sys

from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, \
    PointNetFeaturePropagation
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder


def spearmanr(pred, target, **kw):
    pred = soft_rank(pred.cpu(), **kw).cuda()
    target = soft_rank(target.cpu(), **kw).cuda()
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
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.view_feature_fusioner1 = nn.MultiheadAttention(embed_dim=256, num_heads=2, dropout=0.)
        self.view_feature_fusioner_linear1 = nn.Linear(256, 256)
        self.view_feature_fusioner_relu1 = nn.LeakyReLU()
        self.view_feature_fusioner2 = nn.MultiheadAttention(embed_dim=256, num_heads=2, dropout=0.)
        self.view_feature_fusioner_linear2 = nn.Linear(256, 256)
        self.view_feature_fusioner_relu2 = nn.LeakyReLU()

    # valid_flag, dx, dy, dz, distance_ratio, normal_angle, central_angle
    def forward(self, v_data):
        if len(v_data.shape) == 4:
            view_attribute = v_data.view(-1, v_data.shape[2], v_data.shape[3])
        else:
            view_attribute = v_data
        view_attribute[:, :, 1:4] = view_attribute[:, :, 1:4] / (
                torch.norm(view_attribute[:, :, 1:4], dim=2).unsqueeze(-1) + 1e-6)  # Normalize the view direction
        predict_reconstructability_per_view = self.view_feature_extractor(
            view_attribute[:, :, 1:7])  # Compute the reconstructabilty of every single view
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

            #
            feature_item_filter = torch.transpose(feature_item, 0, 1)
            attention_result = self.view_feature_fusioner1(
                feature_item_filter, feature_item_filter, feature_item_filter,
                key_padding_mask=valid_mask_item[:, :, 0])
            attention_result = torch.transpose(attention_result[0], 0, 1)
            attention_result = self.view_feature_fusioner_linear1(attention_result)
            attention_result = self.view_feature_fusioner_relu1(attention_result)
            feature_item_filter = torch.transpose(attention_result, 0, 1)
            attention_result = self.view_feature_fusioner2(
                feature_item_filter, feature_item_filter, feature_item_filter,
                key_padding_mask=valid_mask_item[:, :, 0])
            attention_result = torch.transpose(attention_result[0], 0, 1)
            attention_result = self.view_feature_fusioner_linear2(attention_result)
            attention_result = self.view_feature_fusioner_relu2(attention_result)

            view_features.append(attention_result)

        view_features = torch.cat(view_features, dim=0)
        view_features = torch.sum(view_features,
                                  dim=1)  # Sum up all the view contribution of one point
        view_features = view_features / (torch.sum(
            view_attribute[:, :, 0].type(torch.bool), dim=1).unsqueeze(-1) + 1e-6)  # Normalize the features
        if view_features.shape[0] != v_data.shape[0]:
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


def loss_l2_recon_entropy_identifier(v_point_attribute, v_prediction, v_l2_weights=100):
    predicted_error = v_prediction[:, :, 0:1]
    predicted_inconsistency = v_prediction[:, :, 1:2]

    point_has_at_least_one_view_see_it = v_prediction[:, :, 2:3].bool()

    gt_reconstructability = v_point_attribute[:, 0]
    gt_max_error = v_point_attribute[:, :, 1:2]
    gt_avg_error = v_point_attribute[:, :, 2:3]
    gt_error_mask_error = (1 - v_point_attribute[:, :, 6:7]).bool()  # Good Point -> True(1)

    error_loss = torch.nn.functional.mse_loss(predicted_error[gt_error_mask_error], gt_avg_error[gt_error_mask_error])
    inconsistency_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predicted_inconsistency[point_has_at_least_one_view_see_it],
        v_point_attribute[:, :, 6:7][point_has_at_least_one_view_see_it])

    # return error_loss, inconsistency_loss, inconsistency_loss
    return error_loss, inconsistency_loss, v_l2_weights * error_loss + inconsistency_loss


def loss_l2_recon(v_point_attribute, v_prediction):
    predicted_error = v_prediction[:, :, 0:1]
    predicted_inconsistency = v_prediction[:, :, 1:2]

    gt_reconstructability = v_point_attribute[:, 0]
    gt_max_error = v_point_attribute[:, :, 1:2]
    gt_avg_error = v_point_attribute[:, :, 2:3]
    gt_error_mask_error = (1 - v_point_attribute[:, :, 6:7]).bool()  # Good Point -> True(1)

    error_loss = torch.nn.functional.mse_loss(predicted_error[gt_error_mask_error], gt_avg_error[gt_error_mask_error])

    return error_loss, 0, error_loss + 0


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
        view_features = self.view_feature_fusioner(v_data["views"]) # Note that some features are not reasonable
        view_features[torch.logical_not(valid_view_mask)] = 0 # Mark these features to 0
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
        predict_result[torch.logical_not(valid_view_mask)][:,0] = 0

        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_recon_entropy_identifier(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon(v_point_attribute, v_prediction)


class Uncertainty_Modeling_release(nn.Module):
    def __init__(self, hparams):
        super(Uncertainty_Modeling_release, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]
        self.phase_1_extractor = PointNet1(3 + 256, 1)  # xyz + view_features
        self.phase_2_extractor = PointNet1(3 + 256 + 256 + 32 + 1,
                                           256)  # xyz + view_features + img_view_features + img_features + predicted recon
        self.phase_2_recon = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0),
            nn.Linear(256, 1),
        )
        self.phase_2_inconsistency = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(0),
            nn.Linear(256, 1),
        )

        # Img features
        self.img_feature_fusioner = ImgFeatureFuser()
        self.view_feature_fusioner = ViewFeatureFuser()
        # self.img_view_feature_fusioner = ViewFeatureFuser()

        for m in self.parameters():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, v_data):
        batch_size = v_data["views"].shape[0]
        img_feature_time = 0
        view_feature_time = 0
        correlation_time = 0

        # Phase 1
        # Extract view features
        t = time.time()
        view_features = self.view_feature_fusioner(v_data["views"])
        view_feature_time = time.time() - t

        # Phase 1, only use viewpoint features to predict recon
        points = v_data["points"]  # B * num_point * 5 (x,y,z, index, centre_point_index)
        point_features = torch.cat([points[:, :, :3], view_features], dim=2)
        # uncertainty = self.point_feature_extractor(point_features.transpose(1,2))[0].transpose(1,2) # PointNet++
        predict_reconstructability = self.phase_1_extractor(point_features.transpose(1, 2))[0]
        inconsistency_identifier = torch.zeros_like(predict_reconstructability)
        inconsistency_identifier = torch.cat(
            [inconsistency_identifier, torch.ones_like(inconsistency_identifier)], dim=-1)
        # Phase 2, only use viewpoint features to predict recon
        if self.is_involve_img:
            is_point_can_be_seen_with_at_least_one_view = (v_data["img_pose"][:, :, 0, 0]).bool()
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
                t = time.time()
                img_view_features_item = self.view_feature_fusioner(valid_oblique_view_features_per_point)
                img_view_feature_time = time.time() - t

                # Calculate img features
                t = time.time()
                img_features_item = self.img_feature_fusioner(
                    valid_oblique_img_features_per_point.unsqueeze(0),
                    valid_oblique_img_features_mask_per_point.unsqueeze(0))
                img_feature_time = time.time() - t
                t = time.time()

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
        # print("{}, {}, {}".format(attention_time,pointnet_time,correlation_time))
        return predict_result

    def loss(self, v_point_attribute, v_prediction):
        if self.is_involve_img:
            return loss_l2_recon_entropy_identifier(v_point_attribute, v_prediction)
        else:
            return loss_l2_recon(v_point_attribute, v_prediction)


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
