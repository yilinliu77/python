# Copyright (c) 2025, Biao Zhang.

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

import math

import copy

# from flash_attn import flash_attn_kvpacked_func


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GEGLU(),
                nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, window_size=-1):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        kv = self.to_kv(context)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        kv = rearrange(kv, 'b m (p h d) -> p b h m d', h=h, p=2)
        k, v = kv[0], kv[1]

        with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(query=q, key=k, value=v)
        out = self.to_out(rearrange(out, 'b h n d -> b n (h d)'))
        return out

class SelfAttentionBlocks(nn.Module):
    def __init__(self, dim, num_heads=16, depth=1):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self_attn_block = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=num_heads, dim_head=dim // num_heads)),
            PreNorm(dim, FeedForward(dim))
        ])
        self.self_attn_block_list = nn.ModuleList([copy.deepcopy(self_attn_block) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.self_attn_block_list:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossAttentionBlocks(nn.Module):
    def __init__(self, dim, context_dim, num_heads=16, depth=1):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        cross_attn_block = nn.ModuleList([
            PreNorm(dim, Attention(dim, context_dim=context_dim, heads=num_heads, dim_head=dim // num_heads)),
            PreNorm(dim, FeedForward(dim))
        ])
        self.cross_attn_block_list = nn.ModuleList([copy.deepcopy(cross_attn_block) for _ in range(depth)])

    def forward(self, x, context):
        for attn, ff in self.cross_attn_block_list:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x

# class AttentionPool(nn.Module):
#     def __init__(self, query_dim, spatial_dim, context_dim=None, heads=8, dim_head=64, output_dim=None):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)
#         self.heads = heads
#         self.dim_head = dim_head
#         self.spatial_dim = spatial_dim  # 输入序列/空间长度
#
#         # 位置编码：适配 (spatial_dim + 1) 个Token（全局Token + 原始Token）
#         self.positional_embedding = nn.Parameter(
#             torch.randn(spatial_dim + 1, query_dim) / query_dim ** 0.5
#         )
#
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_out = nn.Linear(inner_dim, output_dim or query_dim)
#
#     def forward(self, x, context=None, mask=None):
#         """
#         核心逻辑对齐原AttentionPool：
#         1. 生成全局Token → 2. 拼接全局Token → 3. 加位置编码 → 4. 全局Token做Query聚合所有特征
#         Args:
#             x: 输入张量，shape=(N, L, C) （N=batch, L=spatial_dim, C=query_dim）
#             mask: 注意力掩码，shape=(N, L)，用于加权生成全局Token
#             context: 兼容原参数，默认=None（使用x自身）
#         Returns:
#             池化输出，shape=(N, output_dim) （全局聚合特征）
#         """
#         h = self.heads
#         context = default(context, x)  # 兼容原context参数逻辑
#
#         # ========== 步骤1：维度调整 + 生成全局Token（核心池化前置逻辑） ==========
#         x = x.permute(1, 0, 2)  # NLC → LNC (L=spatial_dim, N=batch, C=query_dim)
#         if mask is not None:
#             # 带mask的加权平均生成全局Token（避免无效区域干扰）
#             mask = mask.unsqueeze(-1).permute(1, 0, 2)  # (N,L) → (L,N,1)
#             global_emb = (x * mask).sum(dim=0) / mask.sum(dim=0)
#         else:
#             # 无mask时直接均值生成全局Token
#             global_emb = x.mean(dim=0, keepdim=True)  # (1, N, C)
#         # 拼接全局Token到最前端：(L+1, N, C)
#         x = torch.cat([global_emb, x], dim=0)
#
#         # ========== 步骤2：叠加位置编码（对齐原AttentionPool） ==========
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1, N, C)
#         x = x.permute(1, 0, 2)  # 恢复 NLC 格式：(N, L+1, C)
#
#         # ========== 步骤3：构建Q/K/V（Q仅用全局Token，K/V用全部Token） ==========
#         # Query：仅取第0位的全局Token → (N, 1, C)
#         q = self.to_q(x[:, :1, :])
#         # Key/Value：用所有Token（全局+原始）→ (N, L+1, C)
#         k = self.to_k(x)
#         v = self.to_v(x)
#
#         # ========== 步骤4：维度重排（保留原Attention的rearrange逻辑） ==========
#         q = rearrange(q, 'b n (h d) -> b h n d', h=h, d=self.dim_head)  # (N, h, 1, d)
#         k = rearrange(k, 'b m (h d) -> b h m d', h=h, d=self.dim_head)  # (N, h, L+1, d)
#         v = rearrange(v, 'b m (h d) -> b h m d', h=h, d=self.dim_head)  # (N, h, L+1, d)
#
#         # ========== 步骤5：FlashAttention加速计算（保留原逻辑） ==========
#         with torch.backends.cuda.sdp_kernel(
#                 enable_flash=True,
#                 enable_math=False,
#                 enable_mem_efficient=True
#         ):
#             out = F.scaled_dot_product_attention(query=q, key=k, value=v)
#
#         # ========== 步骤6：维度还原 + 输出投影（池化结果） ==========
#         out = rearrange(out, 'b h n d -> b n (h d)')  # (N, 1, inner_dim)
#         out = self.to_out(out).squeeze(1)  # 挤压维度 → (N, output_dim)
#
#         return out

class AttentionPool(nn.Module):
    def __init__(
            self,
            query_dim,
            spatial_dim,
            context_dim=None,
            num_queries=1,  # 要初始化的N个mean Query数量
            heads=8,
            dim_head=64,
            output_dim=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dim_head = dim_head
        self.spatial_dim = spatial_dim  # 原始长度L（无需整除N）
        self.num_queries = num_queries  # N个Query的数量

        # 位置编码：适配 (N + L) 个Token（N个mean Query + 原始L个Token）
        self.positional_embedding = nn.Parameter(
                torch.randn(self.num_queries + spatial_dim, query_dim) / query_dim ** 0.5
        )

        # ========== 可学习偏移参数（给每个Query加差异化偏移） ==========
        self.query_offset = nn.Parameter(
                torch.randn(self.num_queries, query_dim) / query_dim ** 0.5
        )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, output_dim or query_dim)

    def forward(self, x, context=None, mask=None):
        """
        核心：无分块 + 全局mean初始化 + 可学习偏移（N个Query差异化）
        输入：x (N, L, C) → 输出：(N, num_queries, output_dim)
        """
        h = self.heads
        context = default(context, x)
        B, L, C = x.shape  # B=batch, L=原始长度, C=query_dim
        N = self.num_queries  # 要的N个Query数量

        # ========== 步骤1：无分块！生成1个全局mean + 可学习偏移得到N个Query ==========
        x = x.permute(1, 0, 2)  # NLC → LNC (L, B, C)

        if mask is not None:
            # 带mask的全局加权mean
            mask = mask.unsqueeze(-1).permute(1, 0, 2)  # (B,L) → (L,B,1)
            global_sum = (x * mask).sum(dim=0)  # (B, C)
            mask_sum = mask.sum(dim=0).clamp(min=1e-8)  # (B, 1)
            global_emb = global_sum / mask_sum  # (B, C)
        else:
            # 无mask的全局mean
            global_emb = x.mean(dim=0)  # (B, C) → 1个全局mean

        # 核心：全局mean复制N份 + 可学习偏移（保留mean初始化，同时让Query差异化）
        query_means = global_emb.unsqueeze(0).repeat(N, 1, 1)  # (N, B, C) 复制N份mean
        query_means = query_means + self.query_offset[:, None, :].to(x.dtype)  # 加可学习偏移

        # ========== 步骤2：拼接N个mean+偏移的Query + 原始Token ==========
        x = torch.cat([query_means, x], dim=0)  # (N+L, B, C)

        # ========== 步骤3：原逻辑完全复用 ==========
        # 叠加位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (N+L, B, C)
        x = x.permute(1, 0, 2)  # 恢复 NLC 格式：(B, N+L, C)

        # 构建Q/K/V：Q取前N个mean+偏移的Query
        q = self.to_q(x[:, :N, :])  # (B, N, inner_dim)
        k = self.to_k(x)  # (B, N+L, inner_dim)
        v = self.to_v(x)  # (B, N+L, inner_dim)

        # 维度重排
        q = rearrange(q, 'b n (h d) -> b h n d', h=h, d=self.dim_head)  # (B, h, N, d)
        k = rearrange(k, 'b m (h d) -> b h m d', h=h, d=self.dim_head)  # (B, h, N+L, d)
        v = rearrange(v, 'b m (h d) -> b h m d', h=h, d=self.dim_head)  # (B, h, N+L, d)

        # FlashAttention加速
        with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(query=q, key=k, value=v)

        # 维度还原：保留N个Query的长度
        out = rearrange(out, 'b h n d -> b n (h d)')  # (B, N, inner_dim)
        out = self.to_out(out)  # (B, N, output_dim)

        return out

class PointEmbed3D(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
                'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


class PointEmbedXD(nn.Module):
    """
    适配原PointEmbed3D设计逻辑的通用维度点云编码器
    hidden_dim = sin+cos拼接后的最终位置编码维度
    """

    def __init__(self, hidden_dim=48, dim=128, coord_dim=3):
        super().__init__()
        self.coord_dim = coord_dim

        # 核心校验：对齐原3D的 %6==0 → 通用化为 % (2*coord_dim) == 0
        assert hidden_dim % (2 * self.coord_dim) == 0, \
            f"hidden_dim must be divisible by {2 * self.coord_dim} for {coord_dim}D (got {hidden_dim})"

        self.embedding_dim = hidden_dim
        # 频率数：和原3D一致 → 48//6=8 → 通用化为 hidden_dim // (2*coord_dim)
        freq_num = self.embedding_dim // (2 * self.coord_dim)
        # 生成频率基（和原3D完全一致的2^k×π逻辑）
        e = torch.pow(2, torch.arange(freq_num)).float() * np.pi

        # 动态构造basis矩阵（对齐原3D的拼接逻辑，适配任意coord_dim）
        basis_list = []
        for i in range(self.coord_dim):
            # 原3D逻辑：拼接3段（e+0+0 / 0+e+0 / 0+0+e）→ 通用化为拼接coord_dim段
            basis_row_segments = []
            for j in range(self.coord_dim):
                if j == i:
                    basis_row_segments.append(e)  # 当前轴的段填充频率基
                else:
                    basis_row_segments.append(torch.zeros(freq_num))  # 其他轴的段填充0
            # 拼接所有段 → 每行长度 = coord_dim * freq_num = 2*coord_dim*freq_num / 2 = hidden_dim / 2
            # （和原3D的basis每行长度=24=48/2 完全一致）
            basis_row = torch.cat(basis_row_segments)
            basis_list.append(basis_row)
        self.register_buffer('basis', torch.stack(basis_list))  # shape: [coord_dim, hidden_dim//2]

        # MLP输入维度：和原3D一致 → hidden_dim + 3 → 通用化为 hidden_dim + coord_dim
        self.mlp = nn.Linear(self.embedding_dim + self.coord_dim, dim)

    @staticmethod
    def embed(input, basis):
        # 完全复用原3D的embed逻辑（einsum + sin/cos拼接）
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x coord_dim（支持1/2/3D）
        # 完全复用原3D的forward逻辑
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))
        return embed


class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 hidden_dim: int = 128,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

        self.mlp = nn.Linear(self.out_dim, hidden_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return self.mlp(torch.cat((x, embed.sin(), embed.cos()), dim=-1))
            else:
                return self.mlp(torch.cat((embed.sin(), embed.cos()), dim=-1))
        else:
            return self.mlp(x)


def generate_2d_grid_coords(N, normalize=True):
    """
    生成 N×N×2 的 2D 归一化坐标矩阵（0~1 插值）
    :param N: 网格边长（整数），最终生成 N×N 个 2D 坐标点
    :param normalize: 是否归一化到 0~1（False 则为 0~N-1 的原始索引）
    :return: shape=(N, N, 2) 的 torch 张量，[:, :, 0] 是 x 坐标，[:, :, 1] 是 y 坐标
    """
    # 1. 生成 0~N-1 的整数网格索引（基础坐标）
    # indexing="xy"：笛卡尔坐标（x=列，y=行），符合 2D 坐标直觉
    x_indices = torch.arange(N)  # [0, 1, ..., N-1]
    y_indices = torch.arange(N)
    xx, yy = torch.meshgrid(x_indices, y_indices, indexing="xy")  # 均为 N×N 张量

    # 2. 归一化到 0~1 范围（核心插值步骤）
    if normalize:
        xx = xx / (N - 1) if N > 1 else xx  # 避免 N=1 时除以 0
        yy = yy / (N - 1) if N > 1 else yy

    # 3. 拼接为 N×N×2 的坐标矩阵（x 在前，y 在后）
    # 先扩展维度（N×N → N×N×1），再沿最后一维拼接
    grid_coords = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], dim=-1)

    return grid_coords


def generate_1d_grid(N):
    """
    生成 1×N 的一维插值矩阵（0→1）
    """
    # 1. 生成一维插值序列
    interp_1d = torch.linspace(0, 1, N)  # [0, ..., 1] (长度 N)

    return interp_1d.unsqueeze(1)  # 转为 1×N 矩阵


# class PointEmbed(nn.Module):
#     def __init__(self, hidden_dim=48, dim=128):
#         super().__init__()

#         assert hidden_dim % 6 == 0

#         self.embedding_dim = hidden_dim
#         e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
#         e = torch.stack([
#             torch.cat([e, torch.zeros(self.embedding_dim // 6),
#                         torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6), e,
#                         torch.zeros(self.embedding_dim // 6)]),
#             torch.cat([torch.zeros(self.embedding_dim // 6),
#                         torch.zeros(self.embedding_dim // 6), e]),
#         ])
#         self.register_buffer('basis', e)

#         self.mlp = nn.Linear(self.embedding_dim, dim)

#     @staticmethod
#     def embed(input, basis):
#         projections = torch.einsum('bnd,de->bne', input, basis)
#         embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
#         return embeddings

#     def forward(self, input):
#         # input: B x N x 3
#         embed = self.mlp(self.embed(input, self.basis)) # B x N x C
#         return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
                logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
                dim=dims)

    def mode(self):
        return self.mean


def fps_subsample(pc, N, M):
    # pc: B x N x 3
    B, N0, D = pc.shape
    assert N == N0

    ###### fps
    flattened = pc.view(B * N, D)

    batch = torch.arange(B).to(pc.device)
    batch = torch.repeat_interleave(batch, N)

    pos = flattened

    ratio = 1.0 * M / N

    idx = fps(pos, batch, ratio=ratio)

    sampled_pc = pos[idx]
    sampled_pc = sampled_pc.view(B, -1, 3)
    ######

    return sampled_pc