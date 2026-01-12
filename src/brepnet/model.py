import copy
import time
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
import torchvision

import numpy as np
from torch_cluster import fps

from src.brepnet.dataset import denormalize_coord1112

def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


def profile_time(time_dict, key, v_timer):
    torch.cuda.synchronize()
    cur = time.time()
    time_dict[key] += cur - v_timer
    return cur


class res_linear(nn.Module):
    def __init__(self, dim_in, dim_out, norm="none"):
        super(res_linear, self).__init__()
        self.conv = nn.Linear(dim_in, dim_out)
        self.act = nn.LeakyReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.LayerNorm(dim_out)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_1D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h -> ... h c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h c -> ... c h'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_2D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h w -> ... h w c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h w c -> ... c h w'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_xd(nn.Module):
    def __init__(self, dim, dim_in, dim_out, kernel_size=3, stride=1, padding=1, v_norm=None, v_norm_shape=None):
        super(res_block_xd, self).__init__()
        self.downsample = None
        if v_norm is None or  v_norm == "none":
            norm = nn.Identity()
        elif v_norm == "layer":
            norm = nn.LayerNorm(v_norm_shape)

        if dim == 0:
            self.conv1 = nn.Linear(dim_in, dim_out)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Linear(dim_out, dim_out)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    copy.deepcopy(norm),
                )
        if dim == 1:
            self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 2:
            self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )
        elif dim == 3:
            self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm1 = copy.deepcopy(norm)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm2 = copy.deepcopy(norm)
            if dim_in != dim_out:
                self.downsample = nn.Sequential(
                    nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                    copy.deepcopy(norm),
                )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def inv_sigmoid(x):
    return torch.log(x / (1 - x))

def compute_cos_loss(normal1, normal2, reduction='mean'):
    # normalize normal
    normal1 = F.normalize(normal1, dim=-1, eps=1e-8)
    normal2 = F.normalize(normal2, dim=-1, eps=1e-8)
    cos_sim = torch.sum(normal1 * normal2, dim=-1)  # 形状: (M,)
    loss = 1.0 - torch.abs(cos_sim)
    loss = loss.clamp(0, 1)

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    return loss

def bbox_loss(pred, target, loss_fun,
              center_loss_weight, scale_loss_weight, reduction='mean'):
    """
    计算边界框损失

    Args:
        pred: 预测tensor，形状 (N, 4)，包含中心坐标和尺度 (cx, cy, cz, s)
        target: 目标tensor，形状 (N, 4)，包含中心坐标和尺度 (cx, cy, cz, s)
        loss_fun: 用于计算损失的函数（如 nn.L1Loss 或 nn.MSELoss）
        center_loss_weight: 中心坐标损失的权重
        scale_loss_weight: 尺度损失的权重
        reduction: 损失的归约方式 ('mean' 或 'sum')

    Returns:
        总损失值
    """
    assert pred.shape[-1] == 4 and target.shape[-1] == 4, "pred and target must have last dimension of size 4"

    pred_center = pred[..., :3]
    pred_scale = pred[..., 3:]
    target_center = target[..., :3]
    target_scale = target[..., 3:]

    center_loss = loss_fun(pred_center, target_center)
    scale_loss = loss_fun(pred_scale, target_scale)

    if reduction == 'mean':
        center_loss = torch.mean(center_loss)
        scale_loss = torch.mean(scale_loss)
    elif reduction == 'sum':
        center_loss = torch.sum(center_loss)
        scale_loss = torch.sum(scale_loss)

    total_loss = center_loss_weight * center_loss + scale_loss_weight * scale_loss
    return total_loss

def geom_combined_loss(pred, target,
                       xyz_loss_fun=F.l1_loss,
                       xyz_weight=1.0,
                       normal_loss_fun=compute_cos_loss,
                       normal_weight=1e-1):
    """
    组合损失函数

    Args:
        pred: 预测tensor，形状 (N, 16, 16, 6)
        target: 目标tensor，形状 (N, 16, 16, 6)
        xyz_weight: 坐标损失的权重
        normal_weight: 法向量余弦损失的权重

    Returns:
        总损失值
    """
    assert pred.shape[-1] == 6 and target.shape[-1] == 6, "pred and target must have last dimension of size 6"

    # 分割坐标和法向量部分
    pred_xyz = pred[..., :3]
    pred_normal = pred[..., 3:]
    pred_normal = F.normalize(pred_normal, dim=-1, eps=1e-8)

    target_xyz = target[..., :3]
    target_normal = target[..., 3:]
    target_normal = F.normalize(target_normal, dim=-1, eps=1e-8)

    # 计算MSE损失
    mse_per_point = xyz_loss_fun(pred_xyz, target_xyz)

    # 计算余弦损失
    cos_loss_per_point = normal_loss_fun(pred_normal, target_normal)

    # 计算加权总损失
    return xyz_weight * mse_per_point + normal_weight * cos_loss_per_point

def compute_topology_loss(points_normals, topology_pairs, topology_type, reduction='mean'):
    assert topology_type in ['parallel', 'vertical'], "topology_type must be 'parallel' or 'vertical'"
    normals = points_normals[..., 3:]
    avg_normals = torch.mean(normals, dim=(1, 2))  # 形状: (B, 3)
    avg_normals = F.normalize(avg_normals, dim=-1, eps=1e-8)
    normals1 = avg_normals[topology_pairs[:, 0]]  # 形状: (M, 3)
    normals2 = avg_normals[topology_pairs[:, 1]]  # 形状: (M, 3)

    cos_sim = torch.sum(normals1 * normals2, dim=-1)  # 形状: (M,)

    if topology_type == 'vertical':
        loss = torch.abs(cos_sim)
    elif topology_type == 'parallel':
        loss = 1.0 - torch.abs(cos_sim)
    else:
        raise ValueError("topology_type must be 'parallel' or 'vertical'")

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)

    # 角度误差是arccos(|cos_sim|)
    # angle_rad = torch.acos(torch.abs(cos_sim))
    # angle_deg = torch.rad2deg(angle_rad)  # 形状: (M,)
    # Test:
    # compute_topology_loss(v_data["face_points"][..., :self.in_channels], v_data['pos_parallel_face_pair'], 'parallel')
    # compute_topology_loss(v_data["face_points"][..., :self.in_channels], v_data['pos_vertical_face_pair'], 'vertical')
    return loss

class AttnIntersection3(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_latent,
        num_layers,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)
        layer = nn.TransformerDecoderLayer(
            dim_latent, 8, dim_feedforward=2048, dropout=0,
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.attn_proj_out = nn.Linear(dim_latent, dim_in * 2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, dim_latent))


    def forward(
            self,
            src) -> Tensor:
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:,0:1]
        mem = output[:,1:2]

        for mod in self.layers:
            tgt = mod(tgt, mem)

        output = self.attn_proj_out(tgt)[:,0]
        return output

class PointEmbed(nn.Module):
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

class KLBottleneck(nn.Module):
    def __init__(self, dim, latent_dim, kl_weight):
        super().__init__()

        self.kl_weight = kl_weight

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def pre(self, x):
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return {'x': x, 'kl': self.kl_weight * kl}

    def post(self, x):
        x = self.proj(x)
        return x

class AutoEncoder_1119_light(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = v_conf["in_channels"]
        self.with_intersection = v_conf["with_intersection"]

        bd = 768
        self.point_embed = PointEmbed(dim=bd)

        # encoder
        cross_layer = nn.TransformerDecoderLayer(
                bd, 16, dim_feedforward=2048, dropout=0,
                batch_first=True, norm_first=True)
        attn_layer = nn.TransformerEncoderLayer(
                bd, 16, dim_feedforward=2048, dropout=0,
                batch_first=True, norm_first=True)

        self.face_coords_cross_query = ModuleList([copy.deepcopy(cross_layer) for i in range(2)])
        self.face_coords_attn = nn.TransformerEncoder(attn_layer, 8, nn.LayerNorm(bd))
        self.edge_coords_cross_query = ModuleList([copy.deepcopy(cross_layer) for i in range(2)])
        self.edge_coords_attn = nn.TransformerEncoder(attn_layer, 6, nn.LayerNorm(bd))

        self.graph_face_edge = nn.ModuleList()
        for i in range(2):
            self.graph_face_edge.append(GATv2Conv(
                    bd, bd,
                    heads=1, edge_dim=bd,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        # post gnn attn
        layer = nn.TransformerEncoderLayer(
                bd, 16, dim_feedforward=2048, dropout=0,
                batch_first=True, norm_first=True)
        self.post_gnn_attn = nn.TransformerEncoder(layer, 4, nn.LayerNorm(bd))
        self.post_gnn_attn_proj_out = nn.Linear(bd, df)

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        self.inter = AttnIntersection3(df, 512, 12)
        self.classifier = nn.Linear(df * 2, 1)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            res_block_xd(2, dl, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2, 2)),
            nn.ConvTranspose2d(ds // 1, ds // 2, kernel_size=2, stride=2),
            res_block_xd(2, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4, 4)),
            nn.ConvTranspose2d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(2, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8, 8)),
            nn.ConvTranspose2d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(2, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16, 16)),
            nn.Conv2d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c', c=self.in_channels),
        )
        self.face_center_scale_decoder = nn.Sequential(
            res_block_xd(0, dl * 2 * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            res_block_xd(1, df, ds, 3, 1, 1, v_norm=norm, v_norm_shape=(ds, 2,)),
            nn.ConvTranspose1d(ds, ds // 2, kernel_size=2, stride=2),
            res_block_xd(1, ds // 2, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 2, 4,)),
            nn.ConvTranspose1d(ds // 2, ds // 4, kernel_size=2, stride=2),
            res_block_xd(1, ds // 4, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 4, 8,)),
            nn.ConvTranspose1d(ds // 4, ds // 8, kernel_size=2, stride=2),
            res_block_xd(1, ds // 8, ds // 8, 3, 1, 1, v_norm=norm, v_norm_shape=(ds // 8, 16,)),
            nn.Conv1d(ds // 8, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c', c=self.in_channels),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            res_block_xd(0, df * 2, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            res_block_xd(0, ds, ds, v_norm=norm, v_norm_shape=(ds,)),
            nn.Linear(ds, 4),
        )

        self.sigmoid = v_conf["sigmoid"]
        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.Linear(self.df*2, self.df*2),
            )
        else:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.LeakyReLU(),
                nn.Linear(self.df, self.df),
                nn.Identity() if not self.sigmoid else nn.Sigmoid(),
            )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()
        self.mse_loss_fn = nn.MSELoss()
        # self.face_geom_loss_fn = lambda pred, target: geom_combined_loss(
        #         pred, target,
        #         xyz_loss_fun=self.loss_fn,
        #         xyz_weight=1.0,
        #         normal_loss_fun=compute_cos_loss,
        #         normal_weight=0.1,
        # )
        # self.edge_geom_loss_fn = lambda pred, target: geom_combined_loss(
        #         pred, target,
        #         xyz_loss_fun=self.loss_fn,
        #         xyz_weight=1.0,
        #         normal_loss_fun=compute_cos_loss,
        #         normal_weight=0.1,
        # )
        self.face_geom_loss_fn = self.loss_fn
        self.edge_geom_loss_fn = self.loss_fn
        self.bbox_loss_fn = lambda pred, target: bbox_loss(
                pred, target,
                loss_fun=self.loss_fn,
                center_loss_weight=1.0,
                scale_loss_weight=1.0,
        )

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return self.gaussian_proj(v_fused_face_features), torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        if v_is_test:
            fused_face_features = mean
        # return fused_face_features, kl_loss
        return fused_face_features, kl_loss, torch.cat([mean, std], dim=1)

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :3], 'b h w n -> b (h w) n').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :3], 'b h n -> b h n').contiguous()

        face_query_points = fps_subsample(face_points, face_points.shape[1], face_points.shape[1] / 64)
        edge_query_points = fps_subsample(edge_points, edge_points.shape[1], edge_points.shape[1] / 64)

        face_points_embed, face_query_points_embed = self.point_embed(face_points), self.point_embed(face_query_points)
        edge_points_embed, edge_query_points_embed = self.point_embed(edge_points), self.point_embed(edge_query_points)

        # face
        for cross_layer in self.face_coords_cross_query:
            face_latent = cross_layer(tgt=face_query_points_embed, memory=face_points_embed)
        face_latent = self.face_coords_attn(face_latent)

        # edge
        for cross_layer in self.edge_coords_cross_query:
            edge_latent = cross_layer(tgt=edge_query_points_embed, memory=edge_points_embed)
        edge_latent = self.edge_coords_attn(edge_latent)


        # Face attn
        # attn_x = self.face_attn_proj_in(face_features)
        # attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        # attn_x = self.face_attn_proj_out(attn_x)
        # fused_face_features = face_features + attn_x

        # # Face graph
        edge_face_connectivity = v_data["edge_face_connectivity"]
        x = fused_face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = x + fused_face_features

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=fused_face_features.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z) + fused_face_features

        return {
            "face_features": face_z,
            "edge_features": edge_features,
        }

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False):
        face_z = v_encoding_result["face_z"]
        face_feature = self.face_attn_proj_in2(face_z)
        if v_data is None:
            num_faces = face_z.shape[0]
            attn_mask = torch.zeros((num_faces, num_faces), dtype=bool, device=face_z.device)
        else:
            attn_mask = v_data["attn_mask"]
        face_feature = self.face_attn2(face_feature, attn_mask)
        face_z = self.face_attn_proj_out2(face_feature)

        decoding_results = {}
        decoding_results["face_points_local"] = self.face_points_decoder(face_z)
        decoding_results["face_center_scale"] = self.face_center_scale_decoder(face_z)

        if v_deduplicated: # Deduplicate
            face_points_local = decoding_results["face_points_local"]
            face_center_scale = decoding_results["face_center_scale"]
            pred_face_points = denormalize_coord1112(face_points_local, face_center_scale)
            num_faces = face_z.shape[0]

            deduplicate_face_id = []
            for i in range(num_faces):
                is_duplicate = False
                for j in deduplicate_face_id:
                    if torch.sqrt(((pred_face_points[i]-pred_face_points[j])**2).sum(dim=-1)).mean() < 1e-3:
                        is_duplicate=True
                        break
                if not is_duplicate:
                    deduplicate_face_id.append(i)
            face_z = face_z[deduplicate_face_id]

        if v_data is None:
            num_faces = face_z.shape[0]
            device = face_z.device
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = face_z[indexes]

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = feature_pair[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat((torch.arange(intersected_edge_feature.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)

            feature_pair = self.inter(feature_pair)
            pred = self.classifier(feature_pair)

            gt_labels = torch.ones_like(pred) * 0.95
            gt_labels[id_false_start:] = 0.05
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.mse_loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            decoding_results["loss_edge"] = loss_edge

            # parallel
            pos_parallel_face_pair = v_data["pos_parallel_face_pair"]
            neg_parallel_face_pair = v_data["neg_parallel_face_pair"]
            parallel_face_pair = torch.cat((pos_parallel_face_pair, neg_parallel_face_pair))
            valid_mask = (parallel_face_pair != -1).sum(-1).bool()

            true_parallel_embedding = face_z[pos_parallel_face_pair]
            false_parallel_embedding = face_z[neg_parallel_face_pair]
            id_false_start = true_parallel_embedding.shape[0]
            feature_pair = torch.cat((true_parallel_embedding, false_parallel_embedding), dim=0)

            feature_pair = self.inter_p(feature_pair)
            pred = self.classifier_p(feature_pair)

            gt_labels = torch.ones_like(pred) * 0.95
            gt_labels[id_false_start:] = 0.05
            loss_parallel = F.binary_cross_entropy_with_logits(pred, gt_labels, reduction='none')
            loss_parallel = loss_parallel[valid_mask].mean() if valid_mask.sum() != 0 else loss_parallel.mean() * 0

            decoding_results["loss_parallel"] = loss_parallel

            # vertical
            pos_vertical_face_pair = v_data["pos_vertical_face_pair"]
            neg_vertical_face_pair = v_data["neg_vertical_face_pair"]
            vertical_face_pair = torch.cat((pos_vertical_face_pair, neg_vertical_face_pair))
            valid_mask = (vertical_face_pair != -1).sum(-1).bool()

            true_vertical_embedding = face_z[pos_vertical_face_pair]
            false_vertical_embedding = face_z[neg_vertical_face_pair]
            id_false_start = true_vertical_embedding.shape[0]
            feature_pair = torch.cat((true_vertical_embedding, false_vertical_embedding), dim=0)

            feature_pair = self.inter_v(feature_pair)
            pred = self.classifier_v(feature_pair)

            gt_labels = torch.ones_like(pred) * 0.95
            gt_labels[id_false_start:] = 0.05
            loss_vertical = F.binary_cross_entropy_with_logits(pred, gt_labels, reduction='none')
            loss_vertical = loss_vertical[valid_mask].mean() if valid_mask.sum() != 0 else loss_vertical.mean() * 0

            decoding_results["loss_vertical"] = loss_vertical

            # GT loss = 0
            # compute_topology_loss(v_data["face_points"][..., :self.in_channels], v_data['pos_parallel_face_pair'], 'parallel')
            # compute_topology_loss(v_data["face_points"][..., :self.in_channels], v_data['pos_vertical_face_pair'], 'vertical')

            loss_normal_feature_parallel = compute_topology_loss(
                    points_normals=decoding_results["face_points_local"],
                    topology_pairs=v_data['pos_parallel_face_pair'],
                    topology_type='parallel')
            loss_normal_feature_vertical = compute_topology_loss(
                    points_normals=decoding_results["face_points_local"],
                    topology_pairs=v_data['pos_vertical_face_pair'],
                    topology_type='vertical')
            decoding_results['loss_normal_feature_parallel_reg'] = loss_normal_feature_parallel
            decoding_results['loss_normal_feature_vertical_reg'] = loss_normal_feature_vertical


        decoding_results["edge_points_local"] = self.edge_points_decoder(intersected_edge_feature)
        decoding_results["edge_center_scale"] = self.edge_center_scale_decoder(intersected_edge_feature)
        if "edge_features" in v_encoding_result:
            decoding_results["edge_points_local1"] = self.edge_points_decoder(v_encoding_result["edge_features"])
            decoding_results["edge_center_scale1"] = self.edge_center_scale_decoder(v_encoding_result["edge_features"])

        decoding_results["face_features"] = v_encoding_result["face_z"]
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        # Loss
        loss={}
        loss["face_norm"] = self.face_geom_loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.bbox_loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )

        loss["edge_norm1"] = self.edge_geom_loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.bbox_loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.01
        loss["parallel_classification"] = v_decoding_result["loss_parallel"] * 0.01
        loss["vertical_classification"] = v_decoding_result["loss_vertical"] * 0.01
        loss['normal_feature_parallel_reg'] = v_decoding_result['loss_normal_feature_parallel_reg'] * 0.01
        loss['normal_feature_vertical_reg'] = v_decoding_result['loss_normal_feature_vertical_reg'] * 0.01

        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.edge_geom_loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.bbox_loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )

        # gobal features
        # recon_face_points = denormalize_coord1112(v_decoding_result["face_points_local"], v_decoding_result["face_center_scale"])
        # recon_edge_points1 = denormalize_coord1112(v_decoding_result["edge_points_local1"], v_decoding_result["edge_center_scale1"])
        # recon_edge_points = denormalize_coord1112(v_decoding_result["edge_points_local"], v_decoding_result["edge_center_scale"])

        # loss["face_geom"] = self.face_geom_loss_fn(recon_face_points, v_data['face_points'])
        # loss["edge_geom1"] = self.edge_geom_loss_fn(recon_edge_points1, v_data['edge_points'])
        # loss["edge_geom"] = self.edge_geom_loss_fn(recon_edge_points, v_data['edge_points'][edge_face_connectivity[:, 0]])

        if self.gaussian_weights > 0:
            loss["kl_loss"] = v_decoding_result["kl_loss"]

        return loss

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        face_z, kl_loss, features = self.sample(encoding_result["face_features"], v_is_test=v_test)
        encoding_result["face_z"] = face_z
        decoding_result = self.decode(encoding_result, v_data)
        decoding_result["kl_loss"] = kl_loss
        loss = self.loss(decoding_result, v_data)
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            pred_data = self.decode(encoding_result)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            data["face_features"] = features.cpu().numpy()
            data["gt_face_adj"] = face_adj.reshape(-1)
            data["pred_face_adj"] = pred_data["pred_face_adj"].reshape(-1)
            data["gt_edge"] = v_data["edge_points"].detach().cpu().numpy()
            data["gt_edge_face_connectivity"] = v_data["edge_face_connectivity"].detach().cpu().numpy()
            pred_edge = denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])
            data["pred_edge"] = pred_edge.detach().cpu().numpy()
            data["pred_edge_face_connectivity"] = pred_data["pred_edge_face_connectivity"].detach().cpu().numpy()
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local"], decoding_result["edge_center_scale"])[..., :3],
                v_data["edge_points"][v_data["edge_face_connectivity"][:, 0]][..., :3]
            )

            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord1112(decoding_result["edge_points_local1"], decoding_result["edge_center_scale1"])[..., :3],
                v_data["edge_points"][..., :3]
            )
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()
            pred_face = denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])
            data["pred_face"] = pred_face.detach().cpu().numpy()
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face[..., :3],
                v_data["face_points"][..., :3]
            )

        return loss, data

    def inference(self, v_face_features):
        pred_data = self.decode({"face_z": v_face_features}, v_deduplicated=True)
        return {
            "face_features": v_face_features.to(torch.float32).cpu().numpy(),
            "pred_face_adj": pred_data["pred_face_adj"],
            "pred_face_adj_prob": pred_data["pred_edge_face_connectivity"].reshape(-1).to(torch.float32).cpu().numpy(),
            "pred_edge_face_connectivity": pred_data["pred_edge_face_connectivity"].to(torch.float32).cpu().numpy(),
            "pred_face": denormalize_coord1112(pred_data["face_points_local"], pred_data["face_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
            "pred_edge": denormalize_coord1112(pred_data["edge_points_local"], pred_data["edge_center_scale"])[...,:3].to(torch.float32).cpu().numpy(),
        }
