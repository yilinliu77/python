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

from src.regress_reconstructability_hyper_parameters.loss import loss_l2_error, loss_spearman_error
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, \
    PointNetFeaturePropagation
from thirdparty.Pointnet_Pointnet2_pytorch.models.pointnet_utils import PointNetEncoder


class TFEncorder(nn.Module):
    def __init__(self, d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 add_bias_kv=False,
                 add_norm=False) -> None:
        super(TFEncorder, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
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
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        self.activation = self.activation()

    """
    view_features: [B * 1, num_view, 8]
    """

    def forward(self, point_view_features,
                src_key_padding_mask) -> Tuple[Tensor, Optional[Tensor]]:

        fused_point_view_features, view_weights = self.self_attn(
            point_view_features, point_view_features, point_view_features,
            attn_mask=None,
            key_padding_mask=src_key_padding_mask,
            need_weights=True)
        fused_point_view_features = self.dropout1(fused_point_view_features)

        fused_point_view_features = fused_point_view_features + point_view_features
        fused_point_view_features = self.norm1(fused_point_view_features)

        output_view_point_features = self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(fused_point_view_features)))))
        output_view_point_features = output_view_point_features + fused_point_view_features
        output_view_point_features = self.norm2(output_view_point_features)
        return output_view_point_features, view_weights


class TFDecorder(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 add_bias_kv=False,
                 add_norm=False) -> None:
        super(TFDecorder, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            add_bias_kv=add_bias_kv)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                                 add_bias_kv=add_bias_kv)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if add_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.norm1 = self.norm2 = self.norm3 = nn.Identity()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        self.activation = self.activation()

    def forward(self,
                v_point_features_from_img: Tensor,
                v_fused_view_features: Tensor,
                v_point_features_mask):
        fused_point_feature_from_imgs, img_feature_weights = self.self_attn(
            v_point_features_from_img, v_point_features_from_img, v_point_features_from_img,
            attn_mask=None,
            key_padding_mask=v_point_features_mask,
            need_weights=True)
        fused_point_feature_from_imgs = self.dropout1(fused_point_feature_from_imgs)
        fused_point_feature_from_imgs = fused_point_feature_from_imgs + v_point_features_from_img
        fused_point_feature_from_imgs = self.norm1(fused_point_feature_from_imgs)

        fused_features, mha_weights = self.multihead_attn(
            v_fused_view_features, fused_point_feature_from_imgs, fused_point_feature_from_imgs,
            attn_mask=None,
            key_padding_mask=v_point_features_mask,
            need_weights=True)
        fused_features = self.dropout2(fused_features)
        fused_features = v_fused_view_features + fused_features
        fused_features = self.norm2(fused_features)

        fused_features = self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(fused_features)))))
        fused_features = fused_features + fused_features
        fused_features = self.norm3(fused_features)
        return fused_features, img_feature_weights, mha_weights


class Correlation_net(nn.Module):
    def __init__(self, hparams):
        super(Correlation_net, self).__init__()
        self.hydra_conf = hparams
        self.is_involve_img = self.hydra_conf["model"]["involve_img"]

        self.scaled_loss = self.hydra_conf["model"]["use_scaled_loss"]
        self.spearman_method = self.hydra_conf["model"]["spearman_method"]
        self.spearman_factor = self.hydra_conf["model"]["spearman_factor"]

        self.sigmoid = self.hydra_conf["model"]["sigmoid"]
        use_layer_norm = self.hydra_conf["model"]["use_layer_norm"]
        dropout_rate = self.hydra_conf["model"]["dropout_rate"]
        if self.hydra_conf["model"]["activation"]=="relu":
            activation = nn.ReLU
        else:
            activation = nn.LeakyReLU
        added_bias_kv = self.hydra_conf["model"]["add_bias_kv"]
        attn_nhead = self.hydra_conf["model"]["attn_nhead"]
        self_init = self.hydra_conf["model"]["self_init"]

        view_hidden_dim = self.hydra_conf["model"]["view_hidden_dim"]
        view_feature_layers = self.hydra_conf["model"]["view_feature_layers"]
        final_layers = self.hydra_conf["model"]["final_layers"]

        img_hidden_dim = self.hydra_conf["model"]["img_hidden_dim"]
        img_feature_layers = self.hydra_conf["model"]["img_feature_layers"]
        img_final_layers = self.hydra_conf["model"]["img_final_layers"]

        # ========================================Phase 0========================================
        view_feature_extractor=[]
        view_feature_extractor.append(nn.Linear(5, view_hidden_dim))
        view_feature_extractor.append(activation())
        if use_layer_norm:
            view_feature_extractor.append(nn.LayerNorm(view_hidden_dim))
        for i in range(view_feature_layers):
            view_feature_extractor.append(nn.Linear(view_hidden_dim, view_hidden_dim))
            view_feature_extractor.append(activation())
            if use_layer_norm:
                view_feature_extractor.append(nn.LayerNorm(view_hidden_dim))
        self.view_feature_extractor = nn.Sequential(*view_feature_extractor)
        self.view_feature_fusioner1 = TFEncorder(view_hidden_dim,
                                                 attn_nhead,
                                                 dim_feedforward = view_hidden_dim,
                                                 dropout = dropout_rate,
                                                 activation = activation,
                                                 add_bias_kv=added_bias_kv,
                                                 add_norm=use_layer_norm,
                                                 )
        features_to_recon_error=[]
        features_to_recon_error.append(nn.Linear(view_hidden_dim, view_hidden_dim))
        features_to_recon_error.append(activation())
        for i in range(final_layers):
            features_to_recon_error.append(nn.Linear(view_hidden_dim, view_hidden_dim))
            features_to_recon_error.append(activation())
        features_to_recon_error.append(nn.Linear(view_hidden_dim, 1))
        self.features_to_recon_error = nn.Sequential(*features_to_recon_error)

        self.magic_class_token = nn.Parameter(torch.randn(1, 1, view_hidden_dim))

        if self_init:
            self.init_linear(self.view_feature_extractor)
            self.init_attention(self.view_feature_fusioner1)

        # ========================================Phase 1========================================
        if self.is_involve_img:
            img_feature_expander = []
            img_feature_expander.append(nn.Linear(32, img_hidden_dim))
            img_feature_expander.append(activation())
            if use_layer_norm:
                img_feature_expander.append(nn.LayerNorm(img_hidden_dim))
            for i in range(img_feature_layers):
                img_feature_expander.append(nn.Linear(img_hidden_dim, img_hidden_dim))
                img_feature_expander.append(activation())
                if use_layer_norm:
                    img_feature_expander.append(nn.LayerNorm(img_hidden_dim))
            self.img_feature_expander = nn.Sequential(*img_feature_expander)

            self.img_feature_fusioner1 = TFDecorder(img_hidden_dim,
                                                    attn_nhead,
                                                    img_hidden_dim,
                                                    dropout=dropout_rate,
                                                    activation=activation,
                                                    add_bias_kv=added_bias_kv,
                                                    add_norm=use_layer_norm,
                                                    )

            features_to_gt_error = []
            features_to_gt_error.append(nn.Linear(img_hidden_dim, img_hidden_dim))
            features_to_gt_error.append(activation())
            for i in range(img_final_layers):
                features_to_gt_error.append(nn.Linear(img_hidden_dim, img_hidden_dim))
                features_to_gt_error.append(activation())
            features_to_gt_error.append(nn.Linear(img_hidden_dim, 1))
            self.features_to_gt_error = nn.Sequential(*features_to_gt_error)
            if self_init:
                self.init_linear(self.img_feature_expander)
                self.init_attention(self.img_feature_fusioner1)
            if self.hydra_conf["model"]["open_weights"] is False:
                self.view_feature_extractor.requires_grad_(False)
                self.view_feature_fusioner1.requires_grad_(False)
                self.features_to_recon_error.requires_grad_(False)
                self.magic_class_token.requires_grad_(False)


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
        # Compute the reconstructabilty of every single view
        predicted_recon_error_per_view = self.view_feature_extractor(view_attribute[:, :, 1:6])
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

        # Deprecated now, use for attention mask
        # value_mask = torch.logical_not(valid_mask)[:, :, 0].unsqueeze(-1).tile([1, 1, valid_mask.shape[1]])
        # value_mask = torch.logical_or(value_mask, torch.transpose(value_mask, 1, 2))

        fused_view_features, weights = self.view_feature_fusioner1(
            predicted_recon_error_per_view,
            src_key_padding_mask=torch.logical_not(valid_mask)[:, :, 0],
            # src_mask = value_mask
        )
        fused_view_features = fused_view_features * valid_mask

        # Only use the magic token
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
            point_features_from_imgs = v_data["point_features"]
            valid_point_mask = point_features_from_imgs[:, :, 0, 0] != 0  # At least has one view to see it
            point_features_mask = point_features_from_imgs[:, :, :, 0] == 0  # key mask
            point_features_mask[:, :, 0] = False  # Fake data to prevent the nan in the attention module
            if "point_features_mask" in v_data:
                assert torch.allclose(point_features_mask, v_data["point_features_mask"])

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
            predicted_gt_error = predicted_gt_error * valid_point_mask.unsqueeze(-1)
        predict_result = torch.cat([predicted_recon_error, predicted_gt_error], dim=2)

        if self.sigmoid:
            predict_result = torch.sigmoid(predict_result)

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
            return loss_l2_error(v_point_attribute, v_prediction, self.is_involve_img,
                                 self.scaled_loss)
        else:
            return loss_spearman_error(v_point_attribute, v_prediction, self.is_involve_img,
                                       method=self.spearman_method,
                                       normalized_factor=self.spearman_factor)

