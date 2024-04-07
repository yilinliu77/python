from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from torch_geometric.nn import SAGEConv

from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange


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

        self.null_intersection = nn.Parameter(torch.rand(dim_codebook_edge))

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

    def decode_edge(self, edge_embeddings, edge_mask):
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

        return recon_edges

    def decode_face(self, face_embeddings, face_mask):
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

        return recon_faces

    def encode_edge_coords(self, edge, edge_mask):
        B, N, _, _ = edge.size()

        edge = edge.masked_fill(~repeat(edge_mask, 'b n -> b n v k', v=20, k=3), 0.)

        edge = rearrange(edge, 'b n e v -> (b n) v e')

        # 1. project in (B, N, dim_codebook)
        edge_embed = self.edge_encoder(edge)

        edge_embed = rearrange(edge_embed, '(b n) v -> b n v', b=B)

        return edge_embed

    def encode_face_coords(self, face, face_mask):
        B, N, _, _, _ = face.size()

        face = face.masked_fill(~repeat(face_mask, 'b n -> b n w h k', w=20, h=20, k=3), 0.)

        face = rearrange(face, 'b n w h v -> (b n) v w h')

        # 1. project in (B, N, dim_codebook)
        face_embed = self.face_encoder(face)
        face_embed = rearrange(face_embed, '(b n) v -> b n v', b=B)

        return face_embed

    def intersection(self, face_embeddings, edge_face_idx):
        edge_face_idx_mask = edge_face_idx != -1
        edge_face_idx[~edge_face_idx_mask] = 0

        edge_face_embeddings = torch.gather(face_embeddings[:, None].repeat((1, edge_face_idx.shape[1], 1, 1)), dim=2,
                                            index=edge_face_idx[:, ..., None].repeat(
                                                    (1, 1, 1, face_embeddings.shape[-1])))

        face_embeddings1 = edge_face_embeddings[:, :, 0, :]
        face_embeddings2 = edge_face_embeddings[:, :, 1, :]

        intersection_embedding = face_embeddings1 * face_embeddings2
        intersection_embedding[~edge_face_idx_mask.all(dim=-1)] = 0

        return intersection_embedding

    def forward(self, v_data, only_return_recon=False, only_return_loss=True, **kwargs):
        face_edge_idx = v_data["face_edge_idx"]
        sample_points_faces = v_data["sample_points_faces"]
        sample_points_edges = v_data["sample_points_lines"]
        edge_adj = v_data["edge_adj"]
        face_adj = v_data["face_adj"]
        edge_face_idx = v_data["edge_face_idx"]  # N * 2  (face_idx1, face_idx2)

        # GT
        gt_edges = sample_points_edges.clone()
        gt_faces = sample_points_faces.clone()

        # Masks
        edge_mask = (sample_points_edges != -1).all(dim=-1).all(dim=-1)
        face_mask = (sample_points_faces != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        # 1. Encode the edge and face points
        edge_embeddings = self.encode_edge_coords(sample_points_edges, edge_mask)
        face_embeddings = self.encode_face_coords(sample_points_faces, face_mask)

        # 2. GCN
        # needs to be offset by number of faces for each batch
        edge_adj_mask = (edge_adj != -1).all(dim=-1)
        edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
        edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
        edge_index_offsets = rearrange(edge_index_offsets, 'b -> b 1 1')

        edge_adj += edge_index_offsets
        edge_adj = edge_adj[edge_adj_mask]
        edge_adj = rearrange(edge_adj, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter
        orig_face_embed_shape = edge_embeddings.shape[:2]

        edge_embeddings = edge_embeddings[edge_mask]

        edge_embeddings = self.init_sage_conv(edge_embeddings, edge_adj)
        edge_embeddings = self.init_encoder_act_and_norm(edge_embeddings)

        for conv in self.encoders:
            edge_embeddings = conv(edge_embeddings, edge_adj)

        shape = (*orig_face_embed_shape, edge_embeddings.shape[-1])

        edge_embeddings = edge_embeddings.new_zeros(shape).masked_scatter(rearrange(edge_mask, '... -> ... 1'),
                                                                          edge_embeddings)

        # aggregate the egde embeddings to face embeddings plus
        face_edge_idx_mask = face_edge_idx != -1
        face_edge_idx[~face_edge_idx_mask] = 0
        face_embeddings_plus = torch.gather(edge_embeddings[:, None].repeat((1, face_edge_idx.shape[1], 1, 1)), dim=2,
                                            index=face_edge_idx[:, ..., None].repeat(
                                                    (1, 1, 1, edge_embeddings.shape[-1])))
        face_edge_idx[~face_edge_idx_mask] = -1

        # mask out invalids to 0
        face_embeddings_plus[~face_edge_idx_mask] = 0
        face_embeddings_plus = face_embeddings_plus.sum(dim=2) / face_edge_idx_mask.sum(dim=2, keepdim=True).clamp(
                min=1e-5)
        face_embeddings += face_embeddings_plus

        # 3. Reconstruct the edge and face points
        recon_faces = self.decode_face(face_embeddings, face_mask)
        intersection_edge_embeddings = self.intersection(face_embeddings, edge_face_idx)
        recon_edges = self.decode_edge(intersection_edge_embeddings, edge_mask)
        # null_intersection_edge_embeddings = self.null_intersection(face_embeddings, face_edge_idx)

        # recon_edges, recon_faces = self.decode(edge_embeddings, edge_mask, face_embeddings, face_mask)

        if only_return_recon:
            return recon_edges, recon_faces

        recon_edges = recon_edges.masked_fill(~repeat(edge_mask, 'b n -> b n v k', v=20, k=3), 0.)
        recon_faces = recon_faces.masked_fill(~repeat(face_mask, 'b n -> b n w h k', w=20, h=20, k=3), 0.)
        gt_edges = gt_edges.masked_fill(~repeat(edge_mask, 'b n -> b n v k', v=20, k=3), 0.)
        gt_faces = gt_faces.masked_fill(~repeat(face_mask, 'b n -> b n w h k', w=20, h=20, k=3), 0.)
        loss_edge = F.mse_loss(recon_edges, gt_edges, reduction='mean')
        loss_face = F.mse_loss(recon_faces, gt_faces, reduction='mean')
        total_loss = loss_edge + loss_face
        if only_return_loss:
            return total_loss, loss_edge, loss_face

        return total_loss, loss_edge, loss_face, recon_edges, recon_faces


def test():
    max_length = 10
    model = AutoEncoder(max_length)

    B = 3
    N = max_length
    sample_input = torch.randn(B, N, 20, 3)  # 随机生成输入数据

    sample_mask = torch.tensor([
        [1] * 10 + [0] * (N - 10),
        [1] * 5 + [0] * (N - 5),
        [1] * 7 + [0] * (N - 7)
        ]).bool()

    edge_adj = torch.full((B, max_length, 2), -1)

    for i in range(B):
        valid_length = sample_mask[i].sum().item()
        edges = torch.tensor([(j, (j + 1) % valid_length) for j in range(valid_length)])
        edge_adj[i, :valid_length, :] = edges

    sample_input[~sample_mask] = -1

    output = model(sample_input, edge_adj, return_loss=True)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    test()
