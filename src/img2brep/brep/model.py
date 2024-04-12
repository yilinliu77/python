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
from torch_geometric.nn import SAGEConv

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


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def first(it):
    return it[0]


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def is_empty(l):
    return len(l) == 0


def is_tensor_empty(t: Tensor):
    return t.numel() == 0


def set_module_requires_grad_(
        module: Module,
        requires_grad: bool
        ):
    for param in module.parameters():
        param.requires_grad = requires_grad


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim=dim)


def pad_at_dim(t, padding, dim=-1, value=0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value=value)


def pad_to_length(t, length, dim=-1, value=0, right=True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim=dim, value=value)


# resnet block

class PixelNorm(Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])


class SqueezeExcite(Module):
    def __init__(
            self,
            dim,
            reduction_factor=4,
            min_dim=16
            ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
                nn.Linear(dim, dim_inner),
                nn.SiLU(),
                nn.Linear(dim_inner, dim),
                nn.Sigmoid(),
                Rearrange('b c -> b c 1')
                )

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min=1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)


class Block(Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            dropout=0.
            ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = PixelNorm(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            *,
            dropout=0.
            ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
            self,
            x,
            mask=None
            ):
        res = self.residual_conv(x)
        h = self.block1(x, mask=mask)
        h = self.block2(h, mask=mask)
        h = self.excite(h, mask=mask)
        return h + res


class Intersector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_embeddings, v_null_embeddings, num_max_items=500):
        return


class DotIntersector(Intersector):
    def __init__(self):
        super().__init__()

    def forward(self, v_embeddings, v_null_embeddings, num_max_items=500):
        return


class SinalAttenBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.model = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layers_norm = nn.LayerNorm(dim)

    def forward(self, v_embeddings):
        v_embeddings_atten_output, _ = self.model(v_embeddings, v_embeddings, v_embeddings, need_weights=True)
        v_embeddings_atten_output = v_embeddings + self.layers_norm(v_embeddings_atten_output)

        return v_embeddings_atten_output


class AttnIntersector(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SinalAttenBlock(dim=dim, num_heads=num_heads, dropout=dropout))

    def forward(self, v_embeddings, v_null_embeddings, num_max_items=200):
        if v_null_embeddings.shape[0] > num_max_items:
            indices = torch.randperm(v_null_embeddings.shape[0])[:num_max_items]
            v_null_embeddings = v_null_embeddings[indices]

        # share a AttnIntersector model
        for layer in self.layers:
            v_embeddings = layer(v_embeddings)
            v_null_embeddings = layer(v_null_embeddings)

        return v_embeddings, v_null_embeddings


class AttnFaceEmbedding(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SinalAttenBlock(dim=dim, num_heads=num_heads, dropout=dropout))

    def forward(self, face_embeddings):
        for layer in self.layers:
            face_embeddings = layer(face_embeddings)

        return face_embeddings


class AutoEncoder(nn.Module):
    def __init__(self,
                 max_length=100,
                 dim_codebook_edge=256,
                 dim_codebook_face=256,
                 encoder_dims_through_depth: Tuple[int, ...] = (
                         64, 128, 256, 256
                         ),
                 decoder_dims_through_depth: Tuple[int, ...] = (
                         128, 128, 128, 128,
                         192, 192, 192, 192,
                         256, 256, 256, 256, 256, 256,
                         384, 384, 384
                         ),
                 init_decoder_conv_kernel=7,
                 resnet_dropout=0,
                 ):
        super(AutoEncoder, self).__init__()
        self.max_length = max_length
        self.dim_codebook_edge = dim_codebook_edge
        self.dim_codebook_face = dim_codebook_face
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # 1. Convolutional encoder
        # Map from (B*N, 20) to (B, dim_codebook(196))
        self.edge_encoder = nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, dim_codebook_edge)
                )
        self.face_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, dim_codebook_face)
                )

        # 2. GCN
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook_edge, init_encoder_dim, normalize=True, project=True)

        self.init_encoder_act_and_norm = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(init_encoder_dim)
                )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                    curr_dim,
                    dim_layer,
                    normalize=True,
                    project=True
                    )

            self.encoders.append(sage_conv)
            curr_dim = dim_layer

        # 3. Decoder
        # Map from (B*N, 256) to (B*N, 256)
        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        # For edges
        self.edge_decoder_init = nn.Sequential(
                nn.Conv1d(dim_codebook_edge, init_decoder_dim,
                          kernel_size=init_decoder_conv_kernel, padding=init_decoder_conv_kernel // 2),
                nn.SiLU(),
                Rearrange('b c n -> b n c'),
                nn.LayerNorm(init_decoder_dim),
                Rearrange('b n c -> b c n')
                )

        self.edge_decoder = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)

            self.edge_decoder.append(resnet_block)
            curr_dim = dim_layer

        self.to_edge = nn.Sequential(
                nn.Linear(curr_dim, 20 * 3),
                Rearrange('... (v c) -> ... v c', v=20)
                )

        # For faces
        self.face_decoder_init = nn.Sequential(
                nn.Conv1d(dim_codebook_face, init_decoder_dim,
                          kernel_size=init_decoder_conv_kernel, padding=init_decoder_conv_kernel // 2),
                nn.SiLU(),
                Rearrange('b c n -> b n c'),
                nn.LayerNorm(init_decoder_dim),
                Rearrange('b n c -> b c n')
                )

        self.face_decoder = ModuleList([])

        curr_dim = init_decoder_dim
        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)

            self.face_decoder.append(resnet_block)
            curr_dim = dim_layer

        self.to_face = nn.Sequential(
                nn.Linear(curr_dim, 20 * 20 * 3),
                Rearrange('... (v w c) -> ... v w c', v=20, w=20)
                )

        self.null_intersection = nn.Parameter(torch.rand(dim_codebook_face))

        self.intersector = AttnIntersector(dim=dim_codebook_face, num_heads=4, num_layers=6)

        self.face_embed_atten = AttnFaceEmbedding(dim=dim_codebook_face, num_heads=4, num_layers=6)

    # edge: (B, N, 20, 3)
    # edge_mask: (B, N)
    # edge_adj: (B, M, 2)
    def encode(self, edge, edge_mask, edge_adj):
        B, N, _, _ = edge.size()

        edge = edge.masked_fill(~repeat(edge_mask, 'b n -> b n v k', v=20, k=3), 0.)

        edge = rearrange(edge, 'b n e v -> (b n) v e')

        # 1. project in (B, N, dim_codebook)
        edge_embed = self.edge_encoder(edge)
        edge_embed = rearrange(edge_embed, '(b n) d -> b n d', b=B, n=N)

        # 2. GCN
        # first handle edges
        # needs to be offset by number of faces for each batch
        edge_adj_mask = (edge_adj != -1).all(dim=-1)
        edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
        edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_index_offsets = rearrange(edge_index_offsets, 'b -> b 1 1')

        edge_adj += edge_index_offsets
        edge_adj = edge_adj[edge_adj_mask]
        edge_adj = rearrange(edge_adj, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter

        orig_face_embed_shape = edge_embed.shape[:2]

        edge_embed = edge_embed[edge_mask]

        edge_embed = self.init_sage_conv(edge_embed, edge_adj)
        edge_embed = self.init_encoder_act_and_norm(edge_embed)

        for conv in self.encoders:
            edge_embed = conv(edge_embed, edge_adj)

        shape = (*orig_face_embed_shape, edge_embed.shape[-1])

        edge_embed = edge_embed.new_zeros(shape).masked_scatter(rearrange(edge_mask, '... -> ... 1'), edge_embed)

        return edge_embed

    def decode(self, edge_embeddings, edge_mask, face_embeddings, face_mask):
        B, N, _ = edge_embeddings.size()

        # Decode edges
        edge_mask = rearrange(edge_mask, 'b n -> b 1 n')
        x = edge_embeddings

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~edge_mask, 0.)
        x = self.edge_decoder_init(x)
        for resnet_block in self.edge_decoder:
            x = resnet_block(x, mask=edge_mask)

        recon_edges = x * edge_mask
        recon_edges = rearrange(recon_edges, 'b d n -> b n d')
        recon_edges = self.to_edge(recon_edges)

        # Decode faces
        face_mask = rearrange(face_mask, 'b n -> b 1 n')
        x = face_embeddings

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~face_mask, 0.)
        x = self.face_decoder_init(x)
        for resnet_block in self.face_decoder:
            x = resnet_block(x, mask=face_mask)

        # Mask out invalide points
        recon_faces = x * face_mask
        recon_faces = rearrange(recon_faces, 'b d n -> b n d')
        recon_faces = self.to_face(recon_faces)

        return recon_edges, recon_faces

    def decode_edge(self, edge_embeddings):
        x = self.edge_decoder_init(edge_embeddings[:, :, None])
        for resnet_block in self.edge_decoder:
            x = resnet_block(x)
        recon_edges = self.to_edge(x[..., 0])
        return recon_edges

    def decode_face(self, face_embeddings):
        # Decode faces
        x = self.face_decoder_init(face_embeddings[:, :, None])
        for resnet_block in self.face_decoder:
            x = resnet_block(x)
        recon_faces = self.to_face(x[..., 0])
        return recon_faces

    def encode_edge_coords(self, edge):
        # Project in
        edge_embed = self.edge_encoder(edge.permute(0, 2, 1))
        edge_embed = edge_embed
        return edge_embed

    def encode_face_coords(self, face):
        # Project in
        face_embed = self.face_encoder(face.permute(0, 3, 1, 2))
        return face_embed

    def gcn_on_edges(self, v_edge_embeddings, edge_adj):
        edge_embeddings = self.init_sage_conv(v_edge_embeddings, edge_adj)

        edge_embeddings = self.init_encoder_act_and_norm(edge_embeddings)

        for conv in self.encoders:
            edge_embeddings = conv(edge_embeddings, edge_adj)
        return edge_embeddings

    def intersection(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        # get the intersection_embedding of the two faces
        intersection_embedding = v_face_embeddings[v_edge_face_connectivity[:, 1:]]

        # # Build face adj from edge_face_connectivity
        # num_valid_face = reduce(face_mask.long(), 'b ne -> b', 'sum')
        # face_adj = torch.zeros(
        #     (face_embeddings.shape[0], num_valid_face.max(), num_valid_face.max()),
        #     device=face_embeddings.device,
        #     dtype=torch.bool
        # )
        # true_position = edge_face_connectivity[:, :, 1:].clone()
        # true_position[~face_adj_mask] = 0
        # # Mask out invalids
        # face_adj[torch.arange(face_adj.shape[0])[:, None], true_position[:, :, 0], true_position[:, :, 1]] = True
        # face_adj = torch.logical_or(face_adj, face_adj.transpose(1, 2))
        # face_adj[:, torch.arange(face_adj.shape[1]), torch.arange(face_adj.shape[2])] = False

        face_adj = v_face_adj.clone()
        face_adj[v_face_adj == 0] = 1
        face_adj[v_face_adj == 1] = 0
        torch.diagonal(face_adj, dim1=1, dim2=2).fill_(0)

        face_embeddings = v_face_embeddings.new_zeros((*v_face_mask.shape, v_face_embeddings.shape[-1]))
        face_embeddings = face_embeddings.masked_scatter(rearrange(v_face_mask, '... -> ... 1'), v_face_embeddings)

        zero_positions = (face_adj == 1).nonzero()
        face_embeddings1_idx = zero_positions[:, [0, 1]]
        face_embeddings2_idx = zero_positions[:, [0, 2]]

        face_embeddings1 = face_embeddings[face_embeddings1_idx[:, 0], face_embeddings1_idx[:, 1], :]
        face_embeddings2 = face_embeddings[face_embeddings2_idx[:, 0], face_embeddings2_idx[:, 1], :]
        null_intersection_embedding = torch.stack([face_embeddings1, face_embeddings2], dim=1)

        true_attened_features, null_features = self.intersector(intersection_embedding, null_intersection_embedding)

        edge_features = true_attened_features.mean(dim=1)
        null_features = null_features.mean(dim=1)

        return edge_features, null_features

    def forward(self, v_data, only_return_recon=False, only_return_loss=True, **kwargs):
        sample_points_faces = v_data["sample_points_faces"]
        sample_points_edges = v_data["sample_points_lines"]
        sample_points_vertices = v_data["sample_points_vertices"]

        v_face_edge_loop = v_data["face_edge_loop"]
        face_adj = v_data["face_adj"]
        v_edge_face_connectivity = v_data["edge_face_connectivity"]
        v_vertex_edge_connectivity = v_data["vertex_edge_connectivity"]

        timer = record_time()

        # GT
        gt_edges = sample_points_edges.clone()

        # Flatten all the features to accelerate computation
        edge_mask = (sample_points_edges != -1).all(dim=-1).all(dim=-1)
        face_mask = (sample_points_faces != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        # Face
        flatten_faces = sample_points_faces[face_mask]
        gt_faces = flatten_faces.clone()

        # Solve the edge_face_connectivity: first dimension (id_edge)
        edge_face_connectivity = v_edge_face_connectivity.clone()
        edge_face_connectivity_valid = (v_edge_face_connectivity != -1).all(dim=-1)
        edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
        edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]

        # Solve the edge_face_connectivity: last two dimension (id_face)
        face_index_offsets = reduce(face_mask.long(), 'b ne -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        edge_face_connectivity[~edge_face_connectivity_valid] = -1
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]

        face_edge_loop = v_face_edge_loop.clone()
        original_1 = v_face_edge_loop == -1
        original_2 = v_face_edge_loop == -2
        face_edge_loop += edge_index_offsets[:, None, None]
        face_edge_loop[original_1] = -1
        face_edge_loop[original_2] = -2

        # Edges
        flatten_edges = sample_points_edges[edge_mask]
        # Solve the vertex_edge_connectivity: first dimension (id_vertex)
        pass

        # Solve the edge_face_connectivity: last two dimension (id_edge)
        vertex_edge_connectivity_valid = (v_vertex_edge_connectivity != -1).all(dim=-1)
        vertex_edge_connectivity = v_vertex_edge_connectivity.clone()
        vertex_edge_connectivity[..., 1:] += edge_index_offsets[:, None, None]
        vertex_edge_connectivity = vertex_edge_connectivity[vertex_edge_connectivity_valid]

        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[0] += delta_time

        # 1. Encode the edge and face points
        edge_embeddings = self.encode_edge_coords(flatten_edges)
        face_embeddings = self.encode_face_coords(flatten_faces)

        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[1] += delta_time

        # 2. GCN on edges
        edge_embeddings_plus = self.gcn_on_edges(edge_embeddings, vertex_edge_connectivity[..., 1:].permute(1, 0))
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[2] += delta_time

        # aggregate the egde embeddings to face embeddings plus
        face_edge_relations = face_edge_loop[face_mask].clone()
        face_edge_relations_mask = torch.logical_and(face_edge_relations != -1, face_edge_relations != -2)
        face_edge_relations[~face_edge_relations_mask] = 0
        face_embeddings_plus = edge_embeddings_plus[face_edge_relations]
        # mask out invalids to 0
        face_embeddings_plus[~face_edge_relations_mask] = 0
        face_embeddings_plus = face_embeddings_plus.sum(dim=1) / face_edge_relations_mask.long().sum(
                dim=1, keepdim=True).clamp(min=1e-5)

        # fusion the face_embedding and face_embedding_puls
        face_embeddings = torch.stack([face_embeddings, face_embeddings_plus], dim=1)
        face_embeddings = self.face_embed_atten(face_embeddings)
        face_embeddings = face_embeddings.mean(dim=1)

        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[3] += delta_time

        # 3. Reconstruct the edge and face points
        recon_faces = self.decode_face(face_embeddings)
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[4] += delta_time
        intersected_edge_features, null_features = self.intersection(
                face_embeddings,
                edge_face_connectivity,
                face_adj,
                face_mask
                )
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[5] += delta_time
        recon_edges = self.decode_edge(intersected_edge_features)
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[6] += delta_time

        if only_return_recon:
            return recon_edges, recon_faces

        # recon loss
        used_edges = gt_edges[edge_mask][edge_face_connectivity[..., 0]]
        loss_edge = F.mse_loss(recon_edges, used_edges, reduction='mean')
        loss_face = F.mse_loss(recon_faces, gt_faces, reduction='mean')

        # Intersection
        normalized_intersection = l2norm(intersected_edge_features)
        normalized_null = l2norm(null_features)
        normalized_token = l2norm(self.null_intersection)

        loss_null_intersection = ((normalized_intersection * normalized_token).sum(dim=-1).abs().mean() +
                                  (1 - (normalized_null * normalized_token).sum(dim=-1)).abs().mean())

        total_loss = loss_edge + loss_face + loss_null_intersection

        loss = {
            "total_loss"       : total_loss,
            "edge"             : loss_edge,
            "face"             : loss_face,
            "null_intersection": loss_null_intersection
            }

        edge_mask_flatten = edge_mask.flatten()
        recon_edges_idx = edge_mask_flatten.nonzero().squeeze()[edge_face_connectivity[..., 0]]
        recon_edges_mask = edge_mask_flatten.new_zeros(edge_mask_flatten.shape)
        recon_edges_mask[recon_edges_idx] = True
        recon_edges_mask = rearrange(recon_edges_mask, '(b n) -> b n', b=edge_mask.shape[0], n=edge_mask.shape[1])
        recon_edges = sample_points_edges.new_zeros(sample_points_edges.shape).masked_scatter(
                recon_edges_mask[:, :, None, None].repeat(1, 1, 20, 3), recon_edges)
        recon_edges[~recon_edges_mask] = -1

        recon_faces = sample_points_faces.new_zeros(sample_points_faces.shape).masked_scatter(
                face_mask[:, :, None, None, None].repeat(1, 1, 20, 20, 3), recon_faces)
        recon_faces[~face_mask] = -1

        data = {
            "recon_edges": recon_edges,
            "recon_faces": recon_faces
            }

        if only_return_loss:
            return loss
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[7] += delta_time
        return loss, data
