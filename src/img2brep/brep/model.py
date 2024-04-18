import importlib
from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from torch_geometric.nn import SAGEConv, GATv2Conv

from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from shared.common_utils import record_time, profile_time
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.img2brep.brep.common import *
from src.img2brep.brep.cross_attention import MultiLayerCrossAttention


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class res_block_1D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1):
        super(res_block_1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x = x + self.conv(x)
        x = rearrange(x, '... c h -> ... h c')
        x = self.norm(x)
        x = rearrange(x, '... h c -> ... c h')
        return self.act(x)


class res_block_2D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1):
        super(res_block_2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x = x + self.conv(x)
        x = rearrange(x, '... c h w -> ... h w c')
        x = self.norm(x)
        x = rearrange(x, '... h w c -> ... c h w')
        return self.act(x)


################### Encoder
class Continuous_encoder(nn.Module):
    def __init__(self, dim_codebook_face=256, dim_codebook_edge=256, **kwargs):
        super().__init__()
        self.face_encoder = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=7, stride=1, padding=3),
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(256),
                Rearrange('b h w c -> b c h w'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                res_block_2D(256, 256, ks=3, st=1, pa=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                res_block_2D(256, 256, ks=3, st=1, pa=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, dim_codebook_face)
                )

        self.edge_encoder = nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=256, kernel_size=7, stride=1, padding=3),
                Rearrange('b c h -> b h c'),
                nn.LayerNorm(256),
                Rearrange('b h c -> b c h'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                res_block_1D(256, 256, ks=3, st=1, pa=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                res_block_1D(256, 256, ks=3, st=1, pa=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, dim_codebook_edge)
                )

    def encode_face(self, v_data):
        face_coords = v_data["face_points"]
        face_mask = (face_coords != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        return self.face_encoder(face_coords[face_mask].permute(0, 3, 1, 2)), face_mask

    def encode_edge(self, v_data):
        edge_coords = v_data["edge_points"]
        edge_mask = (edge_coords != -1).all(dim=-1).all(dim=-1)
        return self.edge_encoder(edge_coords[edge_mask].permute(0, 2, 1)), edge_mask

    def forward(self, v_data):
        face_coords, face_mask = self.encode_face(v_data)
        edge_coords, edge_mask = self.encode_edge(v_data)
        return face_coords, edge_coords, face_mask, edge_mask


class Discrete_encoder(Continuous_encoder):
    def __init__(self, dim_codebook_face=256, dim_codebook_edge=256,
                 bbox_discrete_dim=64,
                 coor_discrete_dim=64,
                 ):
        super().__init__()
        hidden_dim = 256
        self.bbox_embedding = nn.Embedding(bbox_discrete_dim - 1, 64)
        self.coords_embedding = nn.Embedding(coor_discrete_dim - 1, 64)

        self.bbox_encoder = nn.Sequential(
                nn.Linear(6 * 64, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, dim_codebook_face)
                )

        self.face_encoder = nn.Sequential(
                nn.Conv2d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(hidden_dim),
                Rearrange('b h w c -> b c h w'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_dim, dim_codebook_face)
                )

        self.face_fuser = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                )

        self.edge_encoder = nn.Sequential(
                nn.Conv1d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
                Rearrange('b c h -> b h c'),
                nn.LayerNorm(hidden_dim),
                Rearrange('b h c -> b c h'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, dim_codebook_edge)
                )

        self.edge_fuser = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                )

    def encode_face(self, v_data):
        face_coords = v_data["discrete_face_points"]
        face_bbox = v_data["discrete_face_bboxes"]
        face_mask = (face_coords != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        flatten_face_coords = face_coords[face_mask]
        flatten_face_bbox = face_bbox[face_mask]

        face_coords = rearrange(self.coords_embedding(flatten_face_coords), "... h w c d -> ... (c d) h w")
        face_bbox = rearrange(self.bbox_embedding(flatten_face_bbox), "... l c d -> ... l (c d)")

        face_coords_features = self.face_encoder(face_coords)
        face_bbox_features = self.bbox_encoder(face_bbox)

        face_features = self.face_fuser(face_coords_features + face_bbox_features)

        return face_features, face_mask

    def encode_edge(self, v_data):
        edge_coords = v_data["discrete_edge_points"]
        edge_bbox = v_data["discrete_edge_bboxes"]
        edge_mask = (edge_coords != -1).all(dim=-1).all(dim=-1)

        flatten_edge_coords = edge_coords[edge_mask]
        flatten_edge_bbox = edge_bbox[edge_mask]

        edge_coords = rearrange(self.coords_embedding(flatten_edge_coords), "... h c d -> ... (c d) h")
        edge_bbox = rearrange(self.bbox_embedding(flatten_edge_bbox), "... l c d -> ... l (c d)")

        edge_coords_features = self.edge_encoder(edge_coords)
        edge_bbox_features = self.bbox_encoder(edge_bbox)

        edge_features = self.edge_fuser(edge_coords_features + edge_bbox_features)

        return edge_features, edge_mask


################### Decoder
class Small_decoder(nn.Module):
    def __init__(self,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout,
                 **kwargs
                 ):
        super(Small_decoder, self).__init__()
        # For edges
        self.edge_decoder = nn.Sequential(
                nn.Linear(dim_codebook_edge, 384),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(384),
                nn.Linear(384, 384),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(384),
                nn.Linear(384, 384),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(384),
                nn.Linear(384, 20 * 3),
                Rearrange('... (v c) -> ... v c', v=20)
                )

        # For faces
        self.face_decoder = nn.Sequential(
                nn.Linear(dim_codebook_face, 384),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(384),
                nn.Linear(384, 768),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(768),
                nn.Linear(768, 768),
                nn.SiLU(),
                nn.Dropout(resnet_dropout),
                nn.LayerNorm(768),
                nn.Linear(768, 20 * 20 * 3),
                Rearrange('... (v w c) -> ... v w c', v=20, w=20)
                )

    def forward(self, v_face_embeddings, v_edge_embeddings):
        recon_edges = self.decode_edge(v_edge_embeddings)
        recon_faces = self.decode_face(v_face_embeddings)
        recon_edges.update(recon_faces)
        return recon_edges

    def decode_edge(self, v_edge_embeddings):
        return {"edge_coords": self.edge_decoder(v_edge_embeddings)}

    def decode_face(self, v_face_embeddings):
        return {"face_coords": self.face_decoder(v_face_embeddings)}

    def inference(self, v_data):
        return v_data["face_coords"], v_data["edge_coords"]

    def loss_edge(self, v_pred, v_data, v_edge_mask, v_used_edge_indexes):
        gt_edge = v_data["edge_points"][v_edge_mask][v_used_edge_indexes]

        loss_edge_coords = nn.functional.mse_loss(
                gt_edge,
                v_pred["edge_coords"],
                reduction='mean')

        return {
            "edge_coords": loss_edge_coords,
            }

    def loss(self, v_pred, v_data, v_face_mask, v_edge_mask, v_used_edge_indexes):
        loss_edge = self.loss_edge(v_pred, v_data, v_edge_mask, v_used_edge_indexes)

        gt_face = v_data["face_points"][v_face_mask]
        loss_face_coords = nn.functional.mse_loss(
                gt_face,
                v_pred["face_coords"],
                reduction='mean')

        loss_edge.update({
            "total_loss" : loss_face_coords + loss_edge["edge_coords"],
            "face_coords": loss_face_coords,
            })

        return loss_edge


class Small_decoder_plus(Small_decoder):
    def __init__(self,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout,
                 **kwargs
                 ):
        super(Small_decoder_plus, self).__init__(dim_codebook_edge, dim_codebook_face, resnet_dropout)
        # For edges
        self.edge_decoder = nn.Sequential(
                Rearrange('... c -> ... c 1'),
                nn.Upsample(scale_factor=4, mode="linear"),
                res_block_1D(dim_codebook_edge, 256),
                nn.Upsample(scale_factor=2, mode="linear"),
                res_block_1D(256, 256),
                nn.Upsample(scale_factor=2, mode="linear"),
                res_block_1D(256, 256),
                nn.Upsample(size=20, mode="linear"),
                res_block_1D(256, 256),
                nn.Conv1d(256, 3, kernel_size=3, stride=1, padding=1),
                Rearrange('... c v -> ... v c', c=3),
                )

        # For faces
        self.face_decoder = nn.Sequential(
                Rearrange('... c -> ... c 1 1'),
                nn.Upsample(scale_factor=4, mode="bilinear"),
                res_block_2D(dim_codebook_face, 256),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                res_block_2D(256, 256),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                res_block_2D(256, 256),
                nn.Upsample(size=(20, 20), mode="bilinear"),
                res_block_2D(256, 256),
                nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
                Rearrange('... c w h -> ... w h c', c=3),
                )


class Discrete_decoder(Small_decoder):
    def __init__(self,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout,
                 bbox_discrete_dim=64,
                 coor_discrete_dim=64,
                 ):
        super(Discrete_decoder, self).__init__(dim_codebook_edge, dim_codebook_face, resnet_dropout)
        hidden_dim = 256
        self.bd = bbox_discrete_dim - 1  # discrete_dim
        self.cd = coor_discrete_dim - 1  # discrete_dim

        self.bbox_decoder = nn.Sequential(
                Rearrange('... c -> ... c 1'),
                res_block_1D(dim_codebook_face, hidden_dim, ks=1, st=1, pa=0),
                res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
                res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
                nn.Conv1d(hidden_dim, 6 * self.bd, kernel_size=1, stride=1, padding=0),
                Rearrange('...(p c) 1-> ... p c', p=6, c=self.bd),
                )

        # For faces
        self.face_coords = nn.Sequential(
                Rearrange('... c -> ... c 1 1'),
                nn.Upsample(scale_factor=4, mode="bilinear"),
                res_block_2D(dim_codebook_face, hidden_dim),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                res_block_2D(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                res_block_2D(hidden_dim, hidden_dim),
                nn.Upsample(size=(20, 20), mode="bilinear"),
                res_block_2D(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
                Rearrange('... (p c) w h -> ... w h p c', p=3, c=self.cd),
                )

        # For edges
        self.edge_coords = nn.Sequential(
                Rearrange('... c -> ... c 1'),
                nn.Upsample(scale_factor=4, mode="linear"),
                res_block_1D(dim_codebook_edge, hidden_dim),
                nn.Upsample(scale_factor=2, mode="linear"),
                res_block_1D(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2, mode="linear"),
                res_block_1D(hidden_dim, hidden_dim),
                nn.Upsample(size=20, mode="linear"),
                res_block_1D(hidden_dim, hidden_dim),
                nn.Conv1d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
                Rearrange('... (p c) w -> ... w p c', p=3, c=self.cd),
                )

    def decode_face(self, v_face_embeddings):
        face_coords_logits = self.face_coords(v_face_embeddings)
        face_bbox_logits = self.bbox_decoder(v_face_embeddings)
        return {
            "face_coords_logits": face_coords_logits,
            "face_bbox_logits"  : face_bbox_logits,
            }

    def decode_edge(self, v_edge_embeddings):
        edge_coords_logits = self.edge_coords(v_edge_embeddings)
        edge_bbox_logits = self.bbox_decoder(v_edge_embeddings)
        return {
            "edge_coords_logits": edge_coords_logits,
            "edge_bbox_logits"  : edge_bbox_logits,
            }

    def cross_entropy_loss(self, pred, gt, is_blur=True):
        if not is_blur:
            return nn.functional.cross_entropy(pred, gt, reduction='mean')
        else:
            pred_log_prob = pred.log_softmax(dim=-1)
            gt = nn.functional.one_hot(gt, num_classes=pred.shape[-1])
            gt = gaussian_blur_1d(gt.float(), sigma=0.3, guassian_kernel_width=5)
            return -torch.sum(gt * pred_log_prob) / pred.shape[0]

    def cross_entropy_loss_with_blur(self, pred, gt, bin_smooth_blur_sigma=0.3, guassian_kernel_width=5):
        pred_log_prob = pred.log_softmax(dim=-1)
        gt = nn.functional.one_hot(gt, num_classes=pred.shape[-1])
        gt = gaussian_blur_1d(gt.float(), bin_smooth_blur_sigma, guassian_kernel_width)
        return -torch.sum(gt * pred_log_prob) / pred.shape[0]

    def loss_edge(self, v_pred, v_data, v_edge_mask, v_used_edge_indexes):
        gt_edge_bbox = v_data["discrete_edge_bboxes"][v_edge_mask][v_used_edge_indexes]
        gt_edge_coords = v_data["discrete_edge_points"][v_edge_mask][v_used_edge_indexes]

        loss_edge_coords = self.cross_entropy_loss(v_pred["edge_coords_logits"].flatten(0, -2),
                                                   gt_edge_coords.flatten())
        loss_edge_bbox = self.cross_entropy_loss(v_pred["edge_bbox_logits"].flatten(0, -2),
                                                 gt_edge_bbox.flatten())

        return {
            "edge_coords": loss_edge_coords,
            "edge_bbox"  : loss_edge_bbox,
            }

    def loss(self, v_pred, v_data, v_face_mask, v_edge_mask, v_used_edge_indexes):
        loss_edge = self.loss_edge(v_pred, v_data, v_edge_mask, v_used_edge_indexes)

        gt_face_bbox = v_data["discrete_face_bboxes"][v_face_mask]
        gt_face_coords = v_data["discrete_face_points"][v_face_mask]

        loss_face_coords = self.cross_entropy_loss(v_pred["face_coords_logits"].flatten(0, -2),
                                                   gt_face_coords.flatten())
        loss_face_bbox = self.cross_entropy_loss(v_pred["face_bbox_logits"].flatten(0, -2),
                                                 gt_face_bbox.flatten())

        loss_edge.update({
            "total_loss" : loss_face_coords + loss_face_bbox + loss_edge["edge_coords"] + loss_edge["edge_bbox"],
            "face_coords": loss_face_coords,
            "face_bbox"  : loss_face_bbox,
            })

        return loss_edge

    def inference(self, v_data):
        bbox_shifts = (self.bd + 1) // 2 - 1
        coord_shifts = (self.cd + 1) // 2 - 1

        face_bbox = (v_data["face_bbox_logits"].argmax(dim=-1) - bbox_shifts) / bbox_shifts
        face_center = (face_bbox[:, 3:] + face_bbox[:, :3]) / 2
        face_length = (face_bbox[:, 3:] - face_bbox[:, :3])
        face_coords = (v_data["face_coords_logits"].argmax(dim=-1) - coord_shifts) / coord_shifts / 2
        face_coords = face_coords * face_length[:, None, None] + face_center[:, None, None]

        edge_bbox = (v_data["edge_bbox_logits"].argmax(dim=-1) - bbox_shifts) / bbox_shifts
        edge_center = (edge_bbox[:, 3:] + edge_bbox[:, :3]) / 2
        edge_length = (edge_bbox[:, 3:] - edge_bbox[:, :3])
        edge_coords = (v_data["edge_coords_logits"].argmax(dim=-1) - coord_shifts) / coord_shifts / 2
        edge_coords = edge_coords * edge_length[:, None] + edge_center[:, None]

        return face_coords, edge_coords


################### Intersector
class Intersector(nn.Module):
    def __init__(self, num_max_items=None):
        super().__init__()
        self.num_max_items = num_max_items

    def prepare_data(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        # True intersection
        intersection_embedding = v_face_embeddings[v_edge_face_connectivity[:, 1:]]

        # Construct features for false intersection
        face_adj = v_face_adj.clone()
        face_adj[v_face_adj == 0] = 1
        face_adj[v_face_adj == 1] = 0
        torch.diagonal(face_adj, dim1=1, dim2=2).fill_(0)

        face_embeddings = v_face_embeddings.new_zeros((*v_face_mask.shape, v_face_embeddings.shape[-1]))
        face_embeddings = face_embeddings.masked_scatter(rearrange(v_face_mask, '... -> ... 1'), v_face_embeddings)

        zero_positions = (face_adj == 1).nonzero()
        face_embeddings1_idx = zero_positions[:, [0, 1]]
        face_embeddings2_idx = zero_positions[:, [0, 2]]

        # False itersection
        face_embeddings1 = face_embeddings[face_embeddings1_idx[:, 0], face_embeddings1_idx[:, 1], :]
        face_embeddings2 = face_embeddings[face_embeddings2_idx[:, 0], face_embeddings2_idx[:, 1], :]
        null_intersection_embedding = torch.stack([face_embeddings1, face_embeddings2], dim=1)

        if self.num_max_items is not None and null_intersection_embedding.shape[0] > self.num_max_items:
            indices = torch.randperm(null_intersection_embedding.shape[0])[:self.num_max_items]
            null_intersection_embedding = null_intersection_embedding[indices]

        return intersection_embedding, null_intersection_embedding

    def forward(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        return

    def loss(self, edge_features, null_features):
        return 0

    def inference(self, v_features):
        return

    def inference_label(self, v_features):
        return torch.cosine_similarity(v_features,
                                       self.null_intersection, dim=-1)


class Attn_intersector(Intersector):
    def __init__(self, num_max_items=None):
        super().__init__(num_max_items)
        hidden_dim = 256
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            ])
        self.intersection_token = nn.Parameter(torch.rand(hidden_dim))
        self.null_intersection = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        intersection_embedding, null_intersection_embedding = self.prepare_data(
                v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask)
        edge_features = self.inference(intersection_embedding)
        null_features = self.inference(null_intersection_embedding)

        return edge_features, null_features

    def inference(self, v_features):
        x = self.intersection_token[None, None].repeat(v_features.shape[0], 1, 1)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(key=v_features, value=v_features,
                                     query=x)
                x = x + out
        edge_features = x[:, 0]
        return edge_features

    def inference_label(self, v_features):
        return torch.cosine_similarity(v_features,
                                       self.null_intersection, dim=-1) < 0.5

    def loss(self, edge_features, null_features):
        intersection_feature = torch.cat([edge_features, null_features])
        gt_label = torch.cat([-torch.ones_like(edge_features[:, 0]),
                              torch.ones_like(null_features[:, 0])])
        loss_intersection = F.cosine_embedding_loss(
                intersection_feature, self.null_intersection[None, :], gt_label, margin=0.5)
        return loss_intersection


class Attn_intersector_classifier(Intersector):
    def __init__(self, num_max_items=None):
        super().__init__(num_max_items)
        hidden_dim = 256
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),

            ])
        self.intersection_token = nn.Parameter(torch.rand(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)

    def inference(self, v_features):
        x = self.intersection_token[None, None].repeat(v_features.shape[0], 1, 1)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(key=v_features, value=v_features,
                                     query=x)
                x = x + out
        edge_features = x[:, 0]
        return edge_features

    def forward(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        intersection_embedding, null_intersection_embedding = self.prepare_data(
                v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask)
        edge_features = self.inference(intersection_embedding)
        null_features = self.inference(null_intersection_embedding)

        return edge_features, null_features

    def loss(self, edge_features, null_features):
        intersection_feature = torch.cat([edge_features, null_features])
        gt_label = torch.cat([torch.ones_like(edge_features[:, 0]),
                              torch.zeros_like(null_features[:, 0])])
        loss_intersection = F.binary_cross_entropy_with_logits(
                self.classifier(intersection_feature), gt_label[:, None])
        return loss_intersection

    def inference_label(self, v_features):
        return torch.sigmoid(self.classifier(v_features))[:, 0] > 0.5


################### Fuser
### Add edge features to the corresponding faces
class Fuser(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, v_face_edge_loop, v_face_mask,
                v_edge_embedding, v_face_embedding):
        return


### Add edge features to the corresponding faces through cross attention
class Attn_fuser(Fuser):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.atten = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            ])
        pass

    def forward(self, v_face_edge_loop, v_face_mask,
                v_edge_embedding, v_face_embedding):
        L, _ = v_face_embedding.shape
        S, _ = v_edge_embedding.shape
        face_edge_attn_mask = torch.ones(
                L, S + 1, device=v_face_embedding.device, dtype=torch.bool
                )
        face_edge_relations = v_face_edge_loop[v_face_mask].clone()
        valid_relation_mask = torch.logical_and(face_edge_relations != -1, face_edge_relations != -2)
        face_edge_relations[~valid_relation_mask] = S
        face_edge_attn_mask = face_edge_attn_mask.scatter(1, face_edge_relations, False)
        face_edge_attn_mask = face_edge_attn_mask[:, :S]

        x = v_face_embedding
        for layer in self.atten:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(
                        query=x,
                        key=v_edge_embedding,
                        value=v_edge_embedding,
                        attn_mask=face_edge_attn_mask,
                        )
                x = x + out

        return x


### Self attention across faces
class Self_atten(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.atten = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 2, 0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(hidden_dim, 2, 0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            ])

    def forward(self, v_embedding, v_mask):
        B, _ = v_mask.shape
        L, _ = v_embedding.shape
        attn_mask = v_embedding.new_ones(L, L, device=v_embedding.device, dtype=torch.bool)
        num_valid = v_mask.long().sum(dim=1)
        num_valid = torch.cat((torch.zeros_like(num_valid[:1]), num_valid.cumsum(dim=0)))
        for i in range(num_valid.shape[0] - 1):
            attn_mask[num_valid[i]:num_valid[i + 1], num_valid[i]:num_valid[i + 1]] = 0

        x = v_embedding
        for layer in self.atten:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(x, x, x, attn_mask=attn_mask, need_weights=True)
                x = x + out
        return x


class SAGE_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels_list, normalize=False, project=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        in_channels_layer = in_channels

        for idx, out_channels in enumerate(out_channels_list):
            self.layers.append(SAGEConv(in_channels_layer, out_channels, normalize=normalize, project=project))
            if idx != len(out_channels_list) - 1:
                norm = nn.Sequential(
                        nn.ReLU(),
                        nn.LayerNorm(out_channels)
                        )
                self.layers.append(norm)
            in_channels_layer = out_channels

    def forward(self, x, edge_index, **kwargs):
        for layer in self.layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GAT_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels_list, edge_dim=None, num_heads=8, concat=False,
                 negative_slope=0.2, fill_value='mean', dropout=0.0, bias=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        in_channels_layer = in_channels

        for idx, out_channels in enumerate(out_channels_list):
            self.layers.append(
                    GATv2Conv(in_channels_layer, out_channels, edge_dim=edge_dim, heads=num_heads,
                              concat=concat, negative_slope=negative_slope, fill_value=fill_value,
                              dropout=dropout, bias=bias))

            self.layers.append(nn.LayerNorm(out_channels * num_heads if concat else out_channels))
            self.layers.append(nn.ReLU())
            in_channels_layer = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        for layer in self.layers:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self,
                 v_conf,
                 max_length=100,
                 dim_codebook_edge=256,
                 dim_codebook_face=256,
                 encoder_dims_through_depth_edges: Tuple[int, ...] = (
                         256, 256, 256, 256
                         ),
                 encoder_dims_through_depth_faces: Tuple[int, ...] = (
                         256, 256, 256, 256
                         ),
                 ):
        super(AutoEncoder, self).__init__()
        self.max_length = max_length
        self.dim_codebook_edge = dim_codebook_edge
        self.dim_codebook_face = dim_codebook_face
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # 1. Convolutional encoder
        # Out: `dim_codebook_edge` and `dim_codebook_face`
        mod = importlib.import_module('src.img2brep.brep.model')
        self.encoder = getattr(mod, v_conf["encoder"])(
                dim_codebook_face=dim_codebook_face,
                bbox_discrete_dim=v_conf["bbox_discrete_dim"],
                coor_discrete_dim=v_conf["coor_discrete_dim"],
                )

        if v_conf["graphconv"] == "GAT":
            GraphConv = GAT_GraphConv
        else:
            GraphConv = SAGE_GraphConv

        # 2. GCN to distribute edge features to the nearby edges
        # Out: curr_dim
        self.gcn_on_edges = GraphConv(dim_codebook_edge, encoder_dims_through_depth_edges)

        # 3. attention to aggregate the others edges features
        self.edge_fuser = Self_atten()

        # 4. Fuser to distribute edge features to the corresponding face
        # Inject edge features to the corresponding face
        # This is the true latent code we want to obtain during the generation
        # self.fuser_edges_to_faces = Simple_fuser()
        self.fuser_edges_to_faces = Attn_fuser()

        # 5. GCN to distribute edge features to the nearby faces
        # Out: curr_dim
        self.gcn_on_faces = GraphConv(dim_codebook_face, encoder_dims_through_depth_faces, edge_dim=dim_codebook_edge)

        # 6. attention to aggregate the others faces features
        self.face_fuser = Self_atten()

        # 6. Intersection
        # Use face features and connectivity to obtain the edge latent
        self.intersector = getattr(mod, v_conf["intersector"])(500)
        # self.intersector = Attn_intersector(500)
        # self.intersector = Attn_intersector_classifier(500)

        # 7. Decoder
        # Get BSpline surfaces and edges based on the true latent code
        self.decoder = getattr(mod, v_conf["decoder"])(
                dim_codebook_edge=dim_codebook_edge,
                dim_codebook_face=dim_codebook_face,
                resnet_dropout=0.0,
                bbox_discrete_dim=v_conf["bbox_discrete_dim"],
                coor_discrete_dim=v_conf["coor_discrete_dim"],
                )

    def inference(self, v_face_embeddings):
        face_mask = (v_face_embeddings != 0).all(dim=-1)
        B, L = face_mask.shape
        idx = torch.combinations(torch.arange(v_face_embeddings.shape[1]), 2)
        gathered_features = v_face_embeddings[:, idx]

        attened_features = self.intersector.inference(rearrange(gathered_features, 'b n c d -> (b n) c d'))
        intersected_edge_features = attened_features.view(B, -1, v_face_embeddings.shape[2])
        intersected_edge_mask = gathered_features.all(dim=-1).all(dim=-1)

        true_intersection = self.intersector.inference_label(intersected_edge_features[intersected_edge_mask])
        intersected_mask = intersected_edge_mask.new_zeros(intersected_edge_mask.shape).masked_scatter(
                intersected_edge_mask, true_intersection)

        recon_data = self.decoder(v_face_embeddings.view(-1, v_face_embeddings.shape[-1]),
                                  intersected_edge_features.view(-1, intersected_edge_features.shape[-1]))
        recon_faces, recon_edges = self.decoder.inference(recon_data)

        recon_edges = recon_edges.view(B, -1, 20, 3)
        recon_faces = recon_faces.view(B, -1, 20, 20, 3)
        recon_edges[~intersected_mask] = -1
        recon_faces[~face_mask] = -1
        return recon_edges, recon_faces

    def forward(self, v_data, only_return_recon=False, only_return_loss=True, is_inference=False, **kwargs):
        # 1. Encode the edge and face points
        face_embeddings, edge_embeddings, face_mask, edge_mask = self.encoder(v_data)

        # 2. GCN on edges and self-attention
        v_vertex_edge_connectivity = v_data["vertex_edge_connectivity"]

        # Solve the edge_face_connectivity: last two dimension (id_edge)
        edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
        edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)

        vertex_edge_connectivity_valid = (v_vertex_edge_connectivity != -1).all(dim=-1)
        vertex_edge_connectivity = v_vertex_edge_connectivity.clone()
        vertex_edge_connectivity[..., 1:] += edge_index_offsets[:, None, None]
        vertex_edge_connectivity = vertex_edge_connectivity[vertex_edge_connectivity_valid]

        edge_embeddings_gcn = self.gcn_on_edges(edge_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_mask)

        # 3. fuse edges features to the corresponding faces
        face_edge_loop = v_data["face_edge_loop"].clone()
        original_1 = face_edge_loop == -1
        original_2 = face_edge_loop == -2
        face_edge_loop += edge_index_offsets[:, None, None]
        face_edge_loop[original_1] = -1
        face_edge_loop[original_2] = -2
        face_edge_embeddings = self.fuser_edges_to_faces(
                v_face_edge_loop=face_edge_loop,
                v_face_mask=face_mask,
                v_edge_embedding=atten_edge_embeddings,
                v_face_embedding=face_embeddings
                )

        # 4. GCN on faces and self-attention
        edge_face_connectivity = v_data["edge_face_connectivity"].clone()
        edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
        edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
        edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]

        # Solve the edge_face_connectivity: last two dimension (id_face)
        face_index_offsets = reduce(face_mask.long(), 'b ne -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        edge_face_connectivity[~edge_face_connectivity_valid] = -1
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]

        face_edge_embeddings_gcn = self.gcn_on_faces(face_edge_embeddings,
                                                     edge_face_connectivity[..., 1:].permute(1, 0),
                                                     edge_attr=atten_edge_embeddings[edge_face_connectivity[..., 0]])

        atten_face_edge_embeddings = self.face_fuser(face_edge_embeddings_gcn, face_mask)

        # Intersection
        face_adj = v_data["face_adj"]
        intersected_edge_features, null_features = self.intersector(
                atten_face_edge_embeddings,
                edge_face_connectivity,
                face_adj,
                face_mask
                )

        # Decode the edge and face points
        edge_data = self.decoder.decode_edge(atten_edge_embeddings)
        recon_data = self.decoder(atten_face_edge_embeddings, intersected_edge_features)

        # Return
        if only_return_recon:
            return recon_data

        # Compute loss
        used_edge_indexes = edge_face_connectivity[..., 0]
        loss = self.decoder.loss(recon_data, v_data, face_mask, edge_mask, used_edge_indexes)

        loss_intersection = self.intersector.loss(intersected_edge_features, null_features)
        loss.update({"intersection": loss_intersection})
        loss["total_loss"] += loss_intersection

        loss_edge = self.decoder.loss_edge(
                edge_data, v_data, edge_mask,
                torch.arange(atten_edge_embeddings.shape[0]))
        for key in loss_edge:
            loss[key + "1"] = loss_edge[key]
            loss["total_loss"] += loss_edge[key]

        # Return
        if only_return_loss:
            return loss

        # Construct the full points using the mask
        # recon_data["face_coords_logits"] = nn.functional.one_hot(
        #     v_data["discrete_face_points"][face_mask], self.decoder.cd)
        # recon_data["face_bbox_logits"] = nn.functional.one_hot(
        #     v_data["discrete_face_bboxes"][face_mask], self.decoder.bd)
        # recon_data["edge_coords_logits"] = nn.functional.one_hot(
        #     v_data["discrete_edge_points"][edge_mask], self.decoder.cd)
        # recon_data["edge_bbox_logits"] = nn.functional.one_hot(
        #     v_data["discrete_edge_bboxes"][edge_mask], self.decoder.bd)

        recon_face, recon_edges = self.decoder.inference(recon_data)
        recon_face_full = recon_face.new_zeros(v_data["face_points"].shape)
        recon_face_full = recon_face_full.masked_scatter(rearrange(face_mask, '... -> ... 1 1 1'), recon_face)
        recon_face_full[~face_mask] = -1

        recon_edge_full = -torch.ones_like(v_data["edge_points"])
        bbb = recon_edge_full[edge_mask].clone()
        bbb[used_edge_indexes] = recon_edges
        recon_edge_full[edge_mask] = bbb

        face_embeddings_return = atten_face_edge_embeddings.new_zeros(
                (*face_mask.shape, atten_face_edge_embeddings.shape[-1]))
        face_embeddings_return = face_embeddings_return.masked_scatter(rearrange(face_mask, '... -> ... 1'),
                                                                       atten_face_edge_embeddings)

        data = {
            "recon_faces"    : recon_face_full,
            "recon_edges"    : recon_edge_full,
            "face_embeddings": face_embeddings_return,
            }
        # Compute the true loss with the continuous points
        true_recon_face_loss = nn.functional.mse_loss(recon_face_full, v_data["face_points"], reduction='mean')
        loss["true_recon_face"] = true_recon_face_loss
        true_recon_edge_loss = nn.functional.mse_loss(
                recon_edges, v_data["edge_points"][edge_mask][used_edge_indexes], reduction='mean')
        loss["true_recon_edge"] = true_recon_edge_loss
        return loss, data
