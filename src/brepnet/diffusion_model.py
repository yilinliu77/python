import math
import time
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from diffusers import DDPMScheduler
from tqdm import tqdm

def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()

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


class Separate_encoder(nn.Module):
    def __init__(self, dim_shape=256, dim_latent=8, v_conf=None, **kwargs):
        super().__init__()
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, dim_shape, kernel_size=5, stride=1, padding=2),
            res_block_2D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 1 -> b c'),
        )

        self.face_fuser = Attn_fuser_single(dim_shape)
        self.face_proj = nn.Linear(dim_shape, dim_latent * 2)

        self.edge_coords = nn.Sequential(
            Rearrange('b h n -> b n h'),
            nn.Conv1d(3, dim_shape, kernel_size=5, stride=1, padding=2),
            res_block_1D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 -> b c'),
        )

        self.edge_fuser = Attn_fuser_single(dim_shape)

        self.vertex_coords = nn.Sequential(
            Rearrange('b c-> b c 1'),
            nn.Conv1d(3, dim_shape, kernel_size=1, stride=1, padding=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 -> b c'),
        )

        self.vertex_fuser = Attn_fuser_single(dim_shape)

    def forward(self, v_data):
        face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        face_features = self.face_coords(v_data["face_points"][face_mask])

        face_attn_mask = get_attn_mask(face_mask)
        face_features = self.face_fuser(face_features, face_attn_mask)
        face_features = self.face_proj(face_features)

        # Edge
        edge_mask = (v_data["edge_points"] != -1).all(dim=-1).all(dim=-1)
        edge_features = self.edge_coords(v_data["edge_points"][edge_mask])

        edge_attn_mask = get_attn_mask(edge_mask)
        edge_features = self.edge_fuser(edge_features, edge_attn_mask)

        # Vertex
        vertex_mask = (v_data["vertex_points"] != -1).all(dim=-1)
        vertex_features = self.vertex_coords(v_data["vertex_points"][vertex_mask])

        vertex_attn_mask = get_attn_mask(vertex_mask)
        vertex_features = self.vertex_fuser(vertex_features, vertex_attn_mask)

        return {
            "face_features": face_features,
            "edge_features": edge_features,
            "vertex_features": vertex_features,
            "face_mask": face_mask,
            "edge_mask": edge_mask,
            "vertex_mask": vertex_mask,
        }


def prepare_connectivity(v_data, ):
    face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
    edge_mask = (v_data["edge_points"] != -1).all(dim=-1).all(dim=-1)
    vertex_mask = (v_data["vertex_points"] != -1).all(dim=-1)
    # ================== Prepare data for flattened features ==================
    edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
    edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
    face_index_offsets = reduce(face_mask.long(), 'b ne -> b', 'sum')
    face_index_offsets = F.pad(face_index_offsets.cumsum(dim=0), (1, -1), value=0)
    vertex_index_offsets = reduce(vertex_mask.long(), 'b ne -> b', 'sum')
    vertex_index_offsets = F.pad(vertex_index_offsets.cumsum(dim=0), (1, -1), value=0)

    vertex_edge_connectivity = v_data["vertex_edge_connectivity"].clone()
    vertex_edge_connectivity_valid = (vertex_edge_connectivity != -1).all(dim=-1)
    # Solve the vertex_edge_connectivity: last two dimension (id_edge)
    vertex_edge_connectivity[..., 1:] += edge_index_offsets[:, None, None]
    # Solve the edge_face_connectivity: first (id_vertex)
    vertex_edge_connectivity[..., 0:1] += vertex_index_offsets[:, None, None]
    vertex_edge_connectivity = vertex_edge_connectivity[vertex_edge_connectivity_valid]
    v_data["vertex_edge_connectivity"] = vertex_edge_connectivity

    edge_face_connectivity = v_data["edge_face_connectivity"].clone()
    edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
    # Solve the edge_face_connectivity: last two dimension (id_face)
    edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
    # Solve the edge_face_connectivity: first dimension (id_edge)
    edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
    edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]
    v_data["edge_face_connectivity"] = edge_face_connectivity
    return


def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# Full continuous VAE
class Diffusion_base(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(Diffusion_base, self).__init__()
        self.dim_input = 8*2*2
        self.dim_latent = 768
        self.time_statics = [0 for _ in range(10)]

        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        ) 

        layer = nn.TransformerEncoderLayer(
            d_model=self.dim_latent, nhead=12, norm_first=True, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.net = nn.TransformerEncoder(layer, 12, nn.LayerNorm(self.dim_latent))

        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_input),
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='sample',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

    def inference(self, bs, device, **kwargs):
        face_features = torch.randn((bs, 64, 32)).to(device)  

        for t in tqdm(self.noise_scheduler.timesteps):   
            timesteps = t.reshape(-1).to(device)
            input_features = self.p_embed(face_features)
            time_embeds = self.time_embed(sincos_embedding(timesteps, self.dim_latent)).unsqueeze(1)  
            noise_input = time_embeds + input_features
            pred = self.net(src=noise_input)
            pred = self.fc_out(pred)
            face_features = self.noise_scheduler.step(pred, t, face_features).prev_sample

        return face_features

    def forward(self, v_data, v_test=False, **kwargs):
        face_features = v_data["face_features"]
        face_features = rearrange(face_features, 'b n c h w-> b n (c h w)')
        device = face_features.device
        bsz = face_features.size(0)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        
        # Prepare noise
        noise = torch.randn(face_features.shape).to(device)  
        noise_input = self.noise_scheduler.add_noise(face_features, noise, timesteps)

        noise_features = self.p_embed(noise_input)
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.dim_latent)).unsqueeze(1)  
        noise_features = noise_features + time_embeds

        # Predict noise
        pred = self.net(src=noise_features)
        pred = self.fc_out(pred)

        loss = {}
        # Loss (predict x0)
        loss["total_loss"] = nn.functional.mse_loss(pred, face_features)
    
        if v_test:
            pass

        return loss
