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


class AttnIntersection3(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_latent,
        num_layers,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)
        layer = nn.TransformerDecoderLayer(
            dim_latent, 16, dim_feedforward=2048, dropout=0,
            batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.attn_proj_out = nn.Linear(dim_latent, dim_out)

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


class MoEAttnIntersection3(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            dim_latent: int,
            num_layers: int,
            num_experts: int = 4,  # 专家数量，对应图中的 Expert-1 到 Expert-N
            expert_hidden_ratio: float = 2.0,  # 专家MLP隐藏层维度比例
            k: int = 4,  # Top-K专家选择
            dropout: float = 0.1,
            noisy_gating: bool = False,
            leaky_relu_negative_slope: float = 0.01,  # LeakyReLU参数
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k = k
        self.dim_latent = dim_latent
        self.dim_out = dim_out

        self.attn_proj_in = nn.Linear(dim_in, dim_latent)

        layer = nn.TransformerDecoderLayer(
                dim_latent, 8, dim_feedforward=2048, dropout=0.1,
                batch_first=True, norm_first=True)
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])

        self.pos_encoding = nn.Parameter(torch.randn(1, 2, dim_latent))

        # ========== MoE 输出层替换 ==========
        # 路由器（Router）- 对应图中的 Router
        self.router = nn.Sequential(
                nn.Linear(dim_latent, dim_latent // 2),
                nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),  # 使用LeakyReLU
                nn.Linear(dim_latent // 2, num_experts)
        )

        # 噪声门控（可选）
        if noisy_gating:
            self.noise_std = 0.1
            self.w_noise = nn.Linear(dim_latent, num_experts)

        # 共享专家（Shared Expert）
        self.shared_expert = self._build_expert_mlp(
                dim_latent, dim_out,
                int(dim_latent * expert_hidden_ratio),
                leaky_relu_negative_slope
        )

        # 任务专家网络（Experts）
        self.task_experts = ModuleList([
            self._build_expert_mlp(
                    dim_latent, dim_out,
                    int(dim_latent * expert_hidden_ratio),
                    leaky_relu_negative_slope
            )
            for _ in range(num_experts)
        ])

        self.attn_proj_out = nn.Linear(self.dim_out * (num_experts+1), self.dim_out)

    def _build_expert_mlp(self, dim_in: int, dim_out: int, hidden_dim: int,
                          negative_slope: float) -> nn.Module:
        """构建专家MLP，使用LeakyReLU激活函数"""
        return nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim // 2, dim_out)
        )

    def noisy_top_k_gating(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-K门控机制，对应图中的Learned Weights"""
        # 计算基础门控logits
        logits = self.router(x)

        # 添加噪声（训练时）
        if self.training and hasattr(self, 'w_noise'):
            noise_weights = F.softplus(self.w_noise(x))
            noise = torch.randn_like(logits) * noise_weights
            logits = logits + noise

        # Top-K选择
        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=-1)

        # 创建稀疏门控矩阵
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, top_indices, top_logits)
        gate_outputs = F.softmax(sparse_logits, dim=-1)

        return gate_outputs, top_indices

    def moe_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MoE前向传播，对应图中的数据流"""
        batch_size, hidden_dim = x.shape

        # 1. 共享专家处理（所有输入都经过，对应图中的 Shared Expert）
        shared_output = self.shared_expert(x)

        aux_loss = 0
        # 2. 任务专家处理（对应图中的 Expert-1 到 Expert-N）
        if self.num_experts > 0:
            # 门控计算（对应图中的 Router → Learned Weights）
            gate_outputs, indices = self.noisy_top_k_gating(x)

            # 计算辅助损失（负载均衡）
            if self.training:
                aux_loss = self._compute_auxiliary_loss(gate_outputs, indices, batch_size)

            # 使用堆叠而不是直接相加
            expert_outputs_list = []
            for i, expert in enumerate(self.task_experts):
                expert_mask = (indices == i).any(dim=-1)
                if expert_mask.any():
                    expert_input = x[expert_mask]
                    expert_out = expert(expert_input)
                    # 创建一个全零张量用于存储当前专家的输出
                    expert_full_output = torch.zeros(batch_size, self.dim_out, device=x.device)
                    # 将当前专家的输出放置到对应位置
                    expert_full_output[expert_mask] = expert_out
                    # 将当前专家的输出添加到列表中
                    expert_outputs_list.append(expert_full_output)
                else:
                    # 如果没有样本选择当前专家，添加全零输出
                    expert_outputs_list.append(torch.zeros(batch_size, self.dim_out, device=x.device))
            # 堆叠所有专家输出
            stacked_expert_outputs = torch.stack(expert_outputs_list, dim=1)  # (batch_size, num_experts, dim_out)
            # 应用门控权重
            gate_weights = gate_outputs.unsqueeze(-1)  # (batch_size, num_experts, 1)
            weighted_expert_outputs = stacked_expert_outputs * gate_weights

            # 使用融合层处理堆叠的专家输出
            flattened_outputs = weighted_expert_outputs.view(batch_size, -1)
            flattened_outputs = torch.cat((shared_output, flattened_outputs), dim=-1)
            # 3. 合并共享专家和任务专家输出
            final_output = self.attn_proj_out(flattened_outputs)
        else:
            # 如果没有任务专家，只使用共享专家
            final_output = shared_output

        return final_output, aux_loss

    def _compute_auxiliary_loss(self, gates: torch.Tensor, indices: torch.Tensor, batch_size: int):
        """计算负载均衡辅助损失"""
        # 创建专家选择掩码
        expert_mask = F.one_hot(indices, self.num_experts).float()
        # 计算每个专家的负载（被选择的次数）
        load = expert_mask.sum(dim=1).sum(dim=0)  # (num_experts,)
        # 计算门控概率的均值
        gate_mean = gates.sum(dim=0) / batch_size  # (num_experts,)
        # 负载均衡损失
        load_balance_loss = (load * gate_mean).sum() * self.num_experts
        return load_balance_loss

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入投影和位置编码（对应图中的 Input Hidden）
        output = self.attn_proj_in(src) + self.pos_encoding
        tgt = output[:, 0:1]  # (batch, 1, dim_latent)
        mem = output[:, 1:2]  # (batch, 1, dim_latent)

        # Transformer层处理
        for mod in self.layers:
            tgt = mod(tgt, mem)

        # 提取序列的第一个token，形状为(batch, dim_latent)
        tgt_flat = tgt[:, 0]

        # MoE输出层替换原来的线性层
        output, aux_loss = self.moe_forward(tgt_flat)  # (batch, dim_out)

        return output, aux_loss

    def get_expert_usage(self, x: torch.Tensor = None) -> dict:
        """获取专家使用情况统计"""
        if x is None:
            return {"message": "需要提供输入数据来计算专家使用情况"}

        with torch.no_grad():
            if x.dim() == 3:  # 如果有序列维度
                x = x.mean(dim=1)  # 取平均

            gate_outputs, indices = self.noisy_top_k_gating(x)

            # 计算使用统计
            batch_size = x.size(0)
            expert_usage = F.one_hot(indices, self.num_experts).float().sum(dim=1)
            usage_ratio = expert_usage / batch_size

            return {
                'gate_weights_mean' : gate_outputs.mean(dim=0).cpu().numpy(),
                'expert_usage_ratio': usage_ratio.cpu().numpy(),
                'top_experts'       : indices.cpu().numpy()
            }

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

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm , v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds // 1, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

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

        self.inter = AttnIntersection3(df, df * 2, 512, 12)
        self.classifier = nn.Linear(df * 2, 1)
        self.inter_p = AttnIntersection3(df, df * 2, 512, 2)
        self.classifier_p = nn.Linear(df * 2, 1)
        self.inter_v = AttnIntersection3(df, df * 2, 512, 2)
        self.classifier_v = nn.Linear(df * 2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8)

        # self.edge_attn_proj_in = nn.Linear(df * 2, bd)
        # self.edge_attn_proj_out = nn.Linear(bd, df * 2)
        # layer3 = nn.TransformerEncoderLayer(
        #         bd, 16, dim_feedforward=2048, dropout=0.1,
        #         batch_first=True, norm_first=True)
        # self.edge_attn = nn.TransformerEncoder(layer3, 8)

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
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
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
            Rearrange('... c w -> ... w c',c=self.in_channels),
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
            self.face_gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df * 2),
                nn.LeakyReLU(),
                nn.Linear(self.df * 2, self.df * 2),
            )
            self.edge_gaussian_proj = nn.Sequential(
                    nn.Linear(self.df * 2, self.df * 2 * 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.df * 2 * 2, self.df * 2 * 2),
            )
        else:
            self.face_gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.LeakyReLU(),
                nn.Linear(self.df, self.df),
                nn.Identity() if not self.sigmoid else nn.Sigmoid(),
            )
            self.edge_gaussian_proj = nn.Sequential(
                    nn.Linear(self.df * 2, self.df * 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.df * 2, self.df * 2),
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

    def face_sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return self.face_gaussian_proj(v_fused_face_features), torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.face_gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        face_kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        if v_is_test:
            fused_face_features = mean
        return fused_face_features, face_kl_loss, torch.cat([mean, std], dim=1)

    def edge_sample(self, v_edge_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return self.edge_gaussian_proj(v_edge_features), torch.zeros_like(v_edge_features[0,0])

        fused_edge_features_gau = self.edge_gaussian_proj(v_edge_features)
        fused_edge_features_gau = fused_edge_features_gau.reshape(-1, self.df * 2, 2)
        mean = fused_edge_features_gau[:, :, 0]
        logvar = fused_edge_features_gau[:, :, 1]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_edge_features = eps.mul(std).add_(mean)
        edge_kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        if v_is_test:
            fused_edge_features = mean
        return fused_edge_features, edge_kl_loss, torch.cat([mean, std], dim=1)

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Face attn
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

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

    def decode(self, v_encoding_result, v_data=None, v_deduplicated=False, v_is_test=False):
        face_z = v_encoding_result["face_z"]
        edge_z = v_encoding_result["edge_z"]

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

            indexes = indexes.reshape(-1, 2).to(device)
            feature_pair_pre = face_z[indexes]

            # intersection
            feature_pair = self.inter(feature_pair_pre)
            pred = self.classifier(feature_pair)[...,0]
            pred_labels = torch.sigmoid(pred) > 0.5
            intersected_edge_feature = feature_pair[pred_labels]
            intersected_edge_feature, _, edge_features_gau = self.edge_sample(intersected_edge_feature, v_is_test=v_is_test)
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat((torch.arange(intersected_edge_feature.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)

            # parallel
            feature_pair_p = self.inter_p(feature_pair_pre)
            pred_p = self.classifier_p(feature_pair_p)[..., 0]
            pred_labels_p = torch.sigmoid(pred_p) > 0.5
            decoding_results["pred_face_parallel_mat"] = pred_labels_p.reshape(-1)
            decoding_results["pred_face_parallel"] = indexes[pred_labels_p]

            # vertical
            feature_pair_v = self.inter_v(feature_pair_pre)
            pred_v = self.classifier_v(feature_pair_v)[..., 0]
            pred_labels_v = torch.sigmoid(pred_v) > 0.5
            decoding_results["pred_face_vertical_mat"] = pred_labels_v.reshape(-1)
            decoding_results["pred_face_vertical"] = indexes[pred_labels_v]

        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)

            feature_pair = self.inter(feature_pair)
            # decoding_results["loss_aux_loss"] = aux_loss
            pred = self.classifier(feature_pair)

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            intersected_edge_feature, _, edge_features_gau = self.edge_sample(intersected_edge_feature, v_is_test=v_is_test)
            decoding_results["loss_edge_features_gau"] = self.loss_fn(
                    edge_features_gau,
                    v_encoding_result["edge_features_gau"][edge_face_connectivity[:, 0]].detach(),
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

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_parallel = F.binary_cross_entropy_with_logits(pred, gt_labels, reduction='none')
            loss_parallel = loss_parallel[valid_mask].mean() if valid_mask.sum() > 0 \
                else torch.tensor(0., device=loss_parallel.device)

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

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_vertical = F.binary_cross_entropy_with_logits(pred, gt_labels, reduction='none')
            loss_vertical = loss_vertical[valid_mask].mean() if valid_mask.sum() > 0 \
                else torch.tensor(0., device=loss_vertical.device)

            decoding_results["loss_vertical"] = loss_vertical

        decoding_results["edge_points_local"] = self.edge_points_decoder(intersected_edge_feature)
        decoding_results["edge_center_scale"] = self.edge_center_scale_decoder(intersected_edge_feature)
        if "edge_features" in v_encoding_result:
            decoding_results["edge_points_local1"] = self.edge_points_decoder(v_encoding_result["edge_z"])
            decoding_results["edge_center_scale1"] = self.edge_center_scale_decoder(v_encoding_result["edge_z"])

        decoding_results["face_features"] = v_encoding_result["face_z"]
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        # Loss
        loss={}
        loss["face_norm"] = self.loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )
        loss["edge_norm1"] = self.loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_features_gau"] = v_decoding_result["loss_edge_features_gau"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
        loss["parallel_classification"] = v_decoding_result["loss_parallel"] * 0.1
        loss["vertical_classification"] = v_decoding_result["loss_vertical"] * 0.1

        # loss["loss_aux_loss"] = v_decoding_result["loss_aux_loss"] * 0.001

        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["face_kl_loss"] = v_decoding_result["face_kl_loss"]
            loss["edge_kl_loss"] = v_decoding_result["edge_kl_loss"]

        return loss

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        face_z, face_kl_loss, face_features_gau = self.face_sample(encoding_result["face_features"], v_is_test=v_test)
        edge_z, edge_kl_loss, edge_features_gau = self.edge_sample(encoding_result["edge_features"], v_is_test=v_test)
        encoding_result["face_z"], encoding_result["face_features_gau"] = face_z, face_features_gau
        encoding_result["edge_z"], encoding_result["edge_features_gau"] = edge_z, edge_features_gau
        decoding_result = self.decode(encoding_result, v_data, v_is_test=v_test)
        decoding_result["face_kl_loss"] = face_kl_loss
        decoding_result["edge_kl_loss"] = edge_kl_loss
        loss = self.loss(decoding_result, v_data)
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            pred_data = self.decode(encoding_result)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            data["face_features"] = face_features_gau.cpu().numpy()
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

class AutoEncoder_1225(nn.Module):
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

        self.face_coords = nn.Sequential(
            nn.Conv2d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,16)),
            nn.LeakyReLU(),
            res_block_xd(2, ds // 8, ds // 4, 3, 1, 1, v_norm=norm , v_norm_shape = (ds // 4, 16, 16)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_xd(2, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8, 8)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_xd(2, ds // 2, ds // 1, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4, 4)),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_xd(2, ds // 1, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2, 2)),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_coords = nn.Sequential(
            nn.Conv1d(self.in_channels, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((ds // 8, 16,)),
            nn.LeakyReLU(),
            res_block_xd(1, ds // 8, ds // 4, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 4, 16,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_xd(1, ds // 4, ds // 2, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 2, 8,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_xd(1, ds // 2, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 4,)),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_xd(1, ds, ds, 3, 1, 1, v_norm=norm, v_norm_shape = (ds // 1, 2,)),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768 # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

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

        self.inter = AttnIntersection3(df, 512, 8)
        self.classifier = nn.Linear(df*2, 1)

        self.face_attn_proj_in2 = nn.Linear(df, bd)
        self.face_attn_proj_out2 = nn.Linear(bd, df)
        layer2 = nn.TransformerEncoderLayer(
            bd, 16, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8)

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
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
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
            Rearrange('... c w -> ... w c',c=self.in_channels),
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
        face_points = rearrange(v_data["face_points"][..., :self.in_channels], 'b h w n -> b n h w').contiguous()
        edge_points = rearrange(v_data["edge_points"][..., :self.in_channels], 'b h n -> b n h').contiguous()
        face_features = self.face_coords(face_points)
        edge_features = self.edge_coords(edge_points)

        # Face attn
        attn_x = self.face_attn_proj_in(face_features)
        attn_x = self.face_attn(attn_x, v_data["attn_mask"])
        attn_x = self.face_attn_proj_out(attn_x)
        fused_face_features = face_features + attn_x

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

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = feature_pair[:id_false_start]

            decoding_results["loss_edge_feature"] = self.loss_fn(
                intersected_edge_feature,
                v_encoding_result["edge_features"][edge_face_connectivity[:, 0]].detach(),
            )

            decoding_results["loss_edge"] = loss_edge

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
        loss["face_norm"] = self.loss_fn(
            v_decoding_result["face_points_local"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = self.loss_fn(
            v_decoding_result["face_center_scale"],
            v_data["face_bbox"]
        )

        loss["edge_norm1"] = self.loss_fn(
            v_decoding_result["edge_points_local1"],
            v_data["edge_norm"]
        )
        loss["edge_bbox1"] = self.loss_fn(
            v_decoding_result["edge_center_scale1"],
            v_data["edge_bbox"]
        )
        loss["edge_feature"] = v_decoding_result["loss_edge_feature"]
        loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss["edge_norm"] = self.loss_fn(
            v_decoding_result["edge_points_local"],
            v_data["edge_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_bbox"] = self.loss_fn(
            v_decoding_result["edge_center_scale"],
            v_data["edge_bbox"][edge_face_connectivity[:, 0]]
        )
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
