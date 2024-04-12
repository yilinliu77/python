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


def discretize(
        t: Tensor,
        continuous_range: Tuple[float, float],
        num_discrete: int = 128
        ) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min=0, max=num_discrete - 1)


def undiscretize(
        t: Tensor,
        continuous_range=Tuple[float, float],
        num_discrete: int = 128
        ) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()
    t += 0.5
    t /= num_discrete

    return t * (hi - lo) + lo


def gaussian_blur_1d(
        t: Tensor,
        sigma: float = 1.
        ) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c=channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding=half_width, groups=channels)

    return rearrange(out, 'b c n -> b n c')


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
        self.linear = nn.Linear(dim, dim)

    def forward(self, v_embeddings):
        v_embeddings_atten_output, _ = self.model(v_embeddings, v_embeddings, v_embeddings, need_weights=True)
        v_embeddings_atten_output = v_embeddings + self.linear(self.layers_norm(v_embeddings_atten_output))

        return v_embeddings_atten_output


class AttnIntersector(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SinalAttenBlock(dim=dim, num_heads=num_heads, dropout=dropout))

    def forward(self, v_embeddings, num_max_items=None):
        if num_max_items is not None and v_embeddings.shape[0] > num_max_items:
            indices = torch.randperm(v_embeddings.shape[0])[:num_max_items]
            v_embeddings = v_embeddings[indices]

        # share a AttnIntersector model
        for layer in self.layers:
            v_embeddings = layer(v_embeddings)

        return v_embeddings


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


class Decoder(nn.Module):
    def __init__(self,
                 decoder_dims_through_depth,
                 init_decoder_conv_kernel,
                 init_decoder_dim,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout
                 ):
        super(Decoder, self).__init__()
        # For edges
        self.edge_decoder_init = nn.Sequential(
                nn.Linear(dim_codebook_edge, init_decoder_dim),
                nn.SiLU(),
                nn.LayerNorm(init_decoder_dim),
                )

        self.edge_decoder = ModuleList([])
        curr_dim = init_decoder_dim
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

    def forward(self, v_edge_embeddings, v_face_embeddings):
        x = self.edge_decoder_init(v_edge_embeddings[:, :, None])
        for resnet_block in self.edge_decoder:
            x = resnet_block(x)
        recon_edges = self.to_edge(x[..., 0])

        x = self.face_decoder_init(v_face_embeddings[:, :, None])
        for resnet_block in self.face_decoder:
            x = resnet_block(x)
        recon_faces = self.to_face(x[..., 0])
        return recon_edges, recon_faces


class Small_decoder(nn.Module):
    def __init__(self,
                 dim_codebook_edge,
                 dim_codebook_face,
                 resnet_dropout
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

    def forward(self, v_edge_embeddings, v_face_embeddings):
        recon_edges = self.edge_decoder(v_edge_embeddings)
        recon_faces = self.face_decoder(v_face_embeddings)
        return recon_edges, recon_faces


class Fuser(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, v_face_edge_loop, v_face_mask,
                v_edge_embedding, v_face_embedding):
        return


class Simple_fuser(Fuser):
    def __init__(self, dim_codebook_face=256):
        super().__init__()
        self.face_embed_atten = AttnFaceEmbedding(dim=dim_codebook_face, num_heads=4, num_layers=6)
        pass

    def forward(self, v_face_edge_loop, v_face_mask,
                v_edge_embedding, v_face_embedding):
        face_edge_relations = v_face_edge_loop[v_face_mask].clone()
        face_edge_relations_mask = torch.logical_and(face_edge_relations != -1, face_edge_relations != -2)
        face_edge_relations[~face_edge_relations_mask] = 0
        face_embeddings_plus = v_edge_embedding[face_edge_relations]
        # mask out invalids to 0
        face_embeddings_plus[~face_edge_relations_mask] = 0
        face_embeddings_plus = face_embeddings_plus.sum(dim=1) / face_edge_relations_mask.long().sum(
                dim=1, keepdim=True).clamp(min=1e-5)

        # fusion the face_embedding and face_embedding_puls
        face_embeddings = torch.stack([v_face_embedding, face_embeddings_plus], dim=1)
        face_embeddings = self.face_embed_atten(face_embeddings)
        face_embeddings = face_embeddings.mean(dim=1)
        return face_embeddings


class Attn_fuser(Fuser):
    def __init__(self):
        super().__init__()
        self.atten = nn.ModuleList([
            nn.MultiheadAttention(
                    embed_dim=256,
                    num_heads=2,
                    dropout=0.1,
                    batch_first=True
                    ),
            nn.MultiheadAttention(
                    embed_dim=256,
                    num_heads=2,
                    dropout=0.1,
                    batch_first=True
                    ),
            nn.MultiheadAttention(
                    embed_dim=256,
                    num_heads=2,
                    dropout=0.1,
                    batch_first=True
                    ),
            nn.MultiheadAttention(
                    embed_dim=256,
                    num_heads=2,
                    dropout=0.1,
                    batch_first=True
                    )
            ])
        pass

    def forward(self, v_face_edge_loop, v_face_mask,
                v_edge_embedding, v_face_embedding):
        L, _ = v_face_embedding.shape
        S, _ = v_edge_embedding.shape
        face_edge_attn_mask = torch.zeros(
                L, S + 1, device=v_face_embedding.device, dtype=torch.bool
                )
        face_edge_relations = v_face_edge_loop[v_face_mask].clone()
        valid_relation_mask = torch.logical_and(face_edge_relations != -1, face_edge_relations != -2)
        face_edge_relations[~valid_relation_mask] = S
        face_edge_attn_mask = face_edge_attn_mask.scatter(1, face_edge_relations, True)
        face_edge_attn_mask = face_edge_attn_mask[:, :S]

        x = v_face_embedding
        for layer in self.atten:
            x = layer(
                    query=x,
                    key=v_edge_embedding,
                    value=v_edge_embedding,
                    attn_mask=face_edge_attn_mask,
                    )[0]

        return x


class AutoEncoder(nn.Module):
    def __init__(self,
                 max_length=100,
                 dim_codebook_edge=256,
                 dim_codebook_face=256,
                 encoder_dims_through_depth: Tuple[int, ...] = (
                         64, 128, 256, 256
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

        # 2. GCN to distribute edge features to the nearby edges
        # Out: curr_dim
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        gcn_out_dims = init_encoder_dim
        self.init_sage_conv = SAGEConv(dim_codebook_edge, init_encoder_dim, normalize=True, project=True)
        self.init_encoder_act_and_norm = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(init_encoder_dim)
                )

        self.gcn_layers = ModuleList([])
        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                    gcn_out_dims,
                    dim_layer,
                    normalize=True,
                    project=True
                    )
            self.gcn_layers.append(sage_conv)
            gcn_out_dims = dim_layer

        # 3. Fuser
        # Inject edge features to the corresponding face
        # This is the true latent code we want to obtain during the generation
        # self.fuser = Simple_fuser()
        self.fuser = Attn_fuser()

        # 4. Intersection
        # Use face features and connectivity to obtain the edge latent
        self.intersector = AttnIntersector(dim=dim_codebook_face, num_heads=4, num_layers=6)
        self.null_intersection = nn.Parameter(torch.rand(dim_codebook_face))

        # 5. Decoder
        # Get BSpline surfaces and edges based on the true latent code
        pass
        # self.decoder = Decoder(
        #     decoder_dims_through_depth=(
        #         128, 128, 128, 128,
        #         192, 192, 192, 192,
        #         256, 256, 256, 256, 256, 256,
        #         384, 384, 384
        #     ),
        #     init_decoder_conv_kernel=7,
        #     init_decoder_dim=256,
        #     dim_codebook_edge=dim_codebook_edge,
        #     dim_codebook_face=dim_codebook_face,
        #     resnet_dropout=0,
        # )
        self.decoder = Small_decoder(
                dim_codebook_edge=dim_codebook_edge,
                dim_codebook_face=dim_codebook_face,
                resnet_dropout=0.1,
                )

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

        for conv in self.gcn_layers:
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

        true_attened_features = self.intersector(intersection_embedding)
        null_features = self.intersector(null_intersection_embedding)

        edge_features = true_attened_features.mean(dim=1)
        null_features = null_features.mean(dim=1)

        return edge_features, null_features


    def inference(self, v_face_embeddings):
        face_mask = (v_face_embeddings != 0).all(dim=-1)
        B, L = face_mask.shape
        idx = torch.combinations(torch.arange(v_face_embeddings.shape[1]), 2)
        gathered_features = v_face_embeddings[:, idx]

        attened_features = self.intersector(rearrange(gathered_features, 'b n c d -> (b n) c d'))
        intersected_edge_features = attened_features.mean(dim=1).view(B, -1, v_face_embeddings.shape[2])
        intersected_edge_mask = gathered_features.all(dim=[-1, -2])

        cos_simarility = torch.cosine_similarity(intersected_edge_features[intersected_edge_mask], self.null_intersection, dim=-1)
        intersected_mask = intersected_edge_mask.new_zeros(intersected_edge_mask.shape).masked_scatter(intersected_edge_mask, cos_simarility < 0.5)

        recon_edges, recon_faces = self.decoder(intersected_edge_features, v_face_embeddings)
        recon_edges[~intersected_mask] = -1
        recon_faces[~face_mask] = -1
        return recon_edges, recon_faces

    def forward(self, v_data, only_return_recon=False, only_return_loss=True, is_inference=False, **kwargs):
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
        face_embeddings = self.fuser(
                v_face_edge_loop=face_edge_loop,
                v_face_mask=face_mask,
                v_edge_embedding=edge_embeddings,
                v_face_embedding=face_embeddings
                )

        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[3] += delta_time

        # 3. Reconstruct the edge and face points
        intersected_edge_features, null_features = self.intersection(
                face_embeddings,
                edge_face_connectivity,
                face_adj,
                face_mask
                )
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[4] += delta_time

        recon_edges, recon_faces = self.decoder(intersected_edge_features, face_embeddings)
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[5] += delta_time

        if only_return_recon:
            return recon_edges, recon_faces

        # recon loss
        used_edges = gt_edges[edge_mask][edge_face_connectivity[..., 0]]
        loss_edge = F.mse_loss(recon_edges, used_edges, reduction='mean')
        loss_face = F.mse_loss(recon_faces, gt_faces, reduction='mean')

        intersection_feature = torch.cat([intersected_edge_features, null_features])
        gt_label = torch.cat(
                [-torch.ones_like(intersected_edge_features[:, 0]), torch.ones_like(null_features[:, 0])])
        loss_intersection = F.cosine_embedding_loss(intersection_feature, self.null_intersection[None,:], gt_label,
                                                    margin=0.5)

        total_loss = loss_edge + loss_face + loss_intersection

        loss = {
            "total_loss"       : total_loss,
            "edge"             : loss_edge,
            "face"             : loss_face,
            "null_intersection": loss_intersection,
            }

        recon_edges_full = -torch.ones_like(sample_points_edges)
        bbb = torch.zeros_like(recon_edges_full[edge_mask])
        bbb[edge_face_connectivity[..., 0]] = recon_edges
        recon_edges_full[edge_mask] = bbb

        recon_faces_full = sample_points_faces.new_zeros(sample_points_faces.shape).masked_scatter(
                face_mask[:, :, None, None, None].repeat(1, 1, 20, 20, 3), recon_faces)
        recon_faces_full[~face_mask] = -1

        recovered_face_embeddings = face_embeddings.new_zeros(sample_points_faces.shape[0], sample_points_faces.shape[1],
                                                                face_embeddings.shape[-1])
        recovered_face_embeddings = recovered_face_embeddings.masked_scatter(
            face_mask[:, :, None].repeat(1, 1, face_embeddings.shape[-1]), face_embeddings)

        data = {
            "recon_edges": recon_edges_full,
            "recon_faces": recon_faces_full,
            "face_embeddings": recovered_face_embeddings
        }

        if only_return_loss:
            return loss
        delta_time, timer = profile_time(timer, v_print=False)
        self.time_statics[6] += delta_time
        return loss, data
