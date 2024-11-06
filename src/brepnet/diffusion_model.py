import importlib
import math
import time
import torch
from torch import isnan, nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from diffusers import DDPMScheduler
from tqdm import tqdm

from src.brepnet.model import AutoEncoder_0925
from thirdparty.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG, \
    PointnetFPModule


# from thirdparty.PointTransformerV3.model import *


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
            "face_features"  : face_features,
            "edge_features"  : edge_features,
            "vertex_features": vertex_features,
            "face_mask"      : face_mask,
            "edge_mask"      : edge_mask,
            "vertex_mask"    : vertex_mask,
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
    half = dim // 2
    freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=input.dtype, device=input.device) / half
    )
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def inv_sigmoid(x):
    return torch.log(x / (1 - x + 1e-6))


# Full continuous VAE
class Diffusion_base_bak(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(Diffusion_base, self).__init__()
        self.dim_input = 8 * 2 * 2
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


# Full continuous VAE
class Diffusion_base(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(Diffusion_base, self).__init__()
        self.dim_input = 8 * 2 * 2
        self.dim_latent = 768
        self.time_statics = [0 for _ in range(10)]

        # self.ae_model = AutoEncoder_0925(v_conf)
        model_mod = importlib.import_module("src.brepnet.model")
        model_mod = getattr(model_mod, v_conf["autoencoder"])
        self.ae_model = model_mod(v_conf)

        self.p_embed = nn.Sequential(
                nn.Linear(self.dim_input, self.dim_latent),
                nn.LayerNorm(self.dim_latent),
                nn.SiLU(),
                nn.Linear(self.dim_latent, self.dim_latent),
        )

        layer = nn.TransformerEncoderLayer(
                d_model=self.dim_latent, nhead=12, dim_feedforward=1024, norm_first=True, dropout=0.1, batch_first=True)
        self.net = nn.TransformerEncoder(layer, 24, nn.LayerNorm(self.dim_latent))

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

        self.num_max_faces = 64
        self.is_pretrained = v_conf["autoencoder_weights"] is not None
        self.is_stored_z = v_conf["stored_z"]
        self.is_train_decoder = v_conf["train_decoder"]
        if self.is_pretrained:
            checkpoint = torch.load(v_conf["autoencoder_weights"], weights_only=False)["state_dict"]
            weights = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            self.ae_model.load_state_dict(weights)
        if not self.is_train_decoder:
            for param in self.ae_model.parameters():
                param.requires_grad = False
            self.ae_model.eval()

    def inference(self, bs, device, v_data=None, **kwargs):
        face_features = torch.randn((bs, self.num_max_faces, 32)).to(device)

        errors = []
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).to(device)
            pred_x0 = self.diffuse(face_features, timesteps)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample
            errors.append((v_data["face_features"] - pred_x0).abs().mean(dim=[1, 2]))

        face_z = face_features
        recon_data = []
        for i in range(bs):
            mask = (face_z[i:i + 1] > 1e-2).any(dim=-1)
            face_z_item = face_z[i:i + 1][mask]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data

    def get_z(self, v_data, v_test):
        data = {}
        if self.is_stored_z:
            face_features = v_data["face_features"]
            bs = face_features.shape[0]
            num_face = face_features.shape[1]
            data["padded_face_z"] = face_features.reshape(bs, num_face, -1)
        else:
            encoding_result = self.ae_model.encode(v_data, v_test)
            data.update(encoding_result)
            face_features = encoding_result["face_z"]
            dim_latent = face_features.shape[-1]
            num_faces = v_data["num_face_record"]
            bs = num_faces.shape[0]
            padded_face_z = torch.zeros(
                    (bs, self.num_max_faces, dim_latent), device=face_features.device, dtype=face_features.dtype)
            # Fill the face_z to the padded_face_z without forloop
            mask = num_faces[:, None] > torch.arange(self.num_max_faces, device=num_faces.device)
            padded_face_z[mask] = face_features
            data["padded_face_z"] = padded_face_z
            data["mask"] = mask
        return data

    def diffuse(self, v_feature, v_timesteps, v_condition=None):
        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_latent)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)
        noise_features = noise_features + time_embeds
        pred_x0 = self.net(src=noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def forward(self, v_data, v_test=False, **kwargs):
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model
        pred_x0 = self.diffuse(noise_input, timesteps)

        if False:
            final_pred = self.noise_scheduler.step(pred_x0, timesteps, noise).pred_original_sample
            loss = nn.functional.mse_loss(final_pred, face_z)

        # Loss (predict x0)
        loss = {}
        if not self.is_train_decoder:
            loss["total_loss"] = nn.functional.l1_loss(pred_x0, face_z)
        else:
            pred_face_z = pred_x0[encoding_result["mask"]]
            encoding_result["face_z"] = pred_face_z
            loss, recon_data = self.ae_model.loss(v_data, encoding_result)
            loss["l2"] = F.l1_loss(pred_x0, face_z)
            loss["total_loss"] += loss["l2"]
        if torch.isnan(loss["total_loss"]).any():
            print("NaN detected in loss")
        return loss


class Diffusion_condition(nn.Module):
    def __init__(self, v_conf, ):
        super().__init__()
        self.dim_input = 8 * 2 * 2
        self.dim_latent = v_conf["diffusion_latent"]
        self.dim_condition = 256
        self.dim_total = self.dim_latent + self.dim_condition
        self.time_statics = [0 for _ in range(10)]

        self.addition_tag = False
        if "addition_tag" in v_conf:
            self.addition_tag = v_conf["addition_tag"]
        if self.addition_tag:
            self.dim_input += 1 

        self.p_embed = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_latent),
            nn.LayerNorm(self.dim_latent),
            nn.SiLU(),
            nn.Linear(self.dim_latent, self.dim_latent),
        )

        layer1 = nn.TransformerEncoderLayer(
                d_model=self.dim_total,
                nhead=self.dim_total // 64, norm_first=True, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.net1 = nn.TransformerEncoder(layer1, 24, nn.LayerNorm(self.dim_total))
        self.fc_out = nn.Sequential(
                nn.Linear(self.dim_total, self.dim_total),
                nn.LayerNorm(self.dim_total),
                nn.SiLU(),
                nn.Linear(self.dim_total, self.dim_input),
        )

        self.with_img = False
        self.with_pc = False
        if v_conf["condition"] == "single_img" or v_conf["condition"] == "multi_img":
            self.with_img = True
            self.img_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            for param in self.img_model.parameters():
                param.requires_grad = False
            self.img_model.eval()

            self.img_fc = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                    nn.Linear(1024, self.dim_condition),
            )
            self.camera_embedding = nn.Sequential(
                    nn.Embedding(24, 256),
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Linear(256, self.dim_condition),
            )
        elif v_conf["condition"] == "pc":
            self.with_pc = True
            self.SA_modules = nn.ModuleList()
            # PointNet2
            c_in = 6
            with_bn = False
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=1024,
                            radii=[0.05, 0.1],
                            nsamples=[16, 32],
                            mlps=[[c_in, 32], [c_in, 64]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_0 = 32 + 64

            c_in = c_out_0
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=256,
                            radii=[0.1, 0.2],
                            nsamples=[16, 32],
                            mlps=[[c_in, 64], [c_in, 128]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_1 = 64 + 128
            c_in = c_out_1
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=64,
                            radii=[0.2, 0.4],
                            nsamples=[16, 32],
                            mlps=[[c_in, 128], [c_in, 256]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            c_out_2 = 128 + 256

            c_in = c_out_2
            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint=16,
                            radii=[0.4, 0.8],
                            nsamples=[16, 32],
                            mlps=[[c_in, 512], [c_in, 512]],
                            use_xyz=True,
                            bn=with_bn
                    )
            )
            self.fc_lyaer = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                    nn.Linear(1024, self.dim_condition),
            )

        self.classifier = nn.Sequential(
                nn.Linear(self.dim_input, self.dim_input),
                nn.LayerNorm(self.dim_input),
                nn.SiLU(),
                nn.Linear(self.dim_input, 1),
        )    
        beta_schedule = "squaredcos_cap_v2"
        if "beta_schedule" in v_conf:
            beta_schedule = v_conf["beta_schedule"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=beta_schedule,
            prediction_type=v_conf["diffusion_type"],
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.LayerNorm(self.dim_total),
            nn.SiLU(),
            nn.Linear(self.dim_total, self.dim_total),
        )

        self.num_max_faces = v_conf["num_max_faces"]
        self.loss = nn.functional.l1_loss if v_conf["loss"] == "l1" else nn.functional.mse_loss
        self.diffusion_type = v_conf["diffusion_type"]
        self.pad_method = v_conf["pad_method"]

        # self.ae_model = AutoEncoder_0925(v_conf)
        model_mod = importlib.import_module("src.brepnet.model")
        model_mod = getattr(model_mod, v_conf["autoencoder"])
        self.ae_model = model_mod(v_conf)

        self.is_pretrained = v_conf["autoencoder_weights"] is not None
        self.is_stored_z = v_conf["stored_z"]
        self.is_train_decoder = v_conf["train_decoder"]
        if self.is_pretrained:
            checkpoint = torch.load(v_conf["autoencoder_weights"], weights_only=False)["state_dict"]
            weights = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            self.ae_model.load_state_dict(weights)
        if not self.is_train_decoder:
            for param in self.ae_model.parameters():
                param.requires_grad = False
            self.ae_model.eval()

    def inference(self, bs, device, v_data=None, **kwargs):
        face_features = torch.randn((bs, self.num_max_faces, self.dim_input)).to(device)
        condition = None
        if self.with_img or self.with_pc:
            condition = self.extract_condition(v_data)[:bs]
            # face_features = face_features[:condition.shape[0]]
        # error = []
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).to(device)
            pred_x0 = self.diffuse(face_features, timesteps, v_condition=condition)
            face_features = self.noise_scheduler.step(pred_x0, t, face_features).prev_sample
            # error.append((v_data["face_features"] - face_features).abs().mean(dim=[1,2]))

        face_z = face_features
        if self.pad_method == "zero":
            label = torch.sigmoid(self.classifier(face_features))[..., 0]
            mask = label > 0.5
        else:
            mask = torch.ones_like(face_z[:, :, 0]).to(bool)
        
        recon_data = []
        for i in range(bs):
            face_z_item = face_z[i:i + 1][mask[i:i + 1]]
            if self.addition_tag: # Deduplicate
                flag = face_z_item[...,-1] > 0
                face_z_item = face_z_item[flag][:, :-1]
            if self.pad_method == "random": # Deduplicate
                threshold = 1e-2
                max_faces = face_z_item.shape[0]
                index = torch.stack(torch.meshgrid(torch.arange(max_faces),torch.arange(max_faces), indexing="ij"), dim=2)
                features = face_z_item[index]
                distance = (features[:,:,0]-features[:,:,1]).abs().mean(dim=-1)
                final_face_z = []
                for j in range(max_faces):
                    valid = True
                    for k in final_face_z:
                        if distance[j,k] < threshold:
                            valid = False
                            break
                    if valid:
                        final_face_z.append(j)
                face_z_item = face_z_item[final_face_z]
            data_item = self.ae_model.inference(face_z_item)
            recon_data.append(data_item)
        return recon_data

    def get_z(self, v_data, v_test):
        data = {}
        if self.is_stored_z:
            face_features = v_data["face_features"]
            bs = face_features.shape[0]
            num_face = face_features.shape[1]
            data["padded_face_z"] = face_features.reshape(bs, num_face, -1)
        else:
            encoding_result = self.ae_model.encode(v_data, True)
            data.update(encoding_result)
            face_features = encoding_result["face_z"]
            dim_latent = face_features.shape[-1]
            num_faces = v_data["num_face_record"]
            bs = num_faces.shape[0]
            padded_face_z = torch.zeros(
                    (bs, self.num_max_faces, dim_latent), device=face_features.device, dtype=face_features.dtype)
            # Fill the face_z to the padded_face_z without forloop
            mask = num_faces[:, None] > torch.arange(self.num_max_faces, device=num_faces.device)
            padded_face_z[mask] = face_features
            data["padded_face_z"] = padded_face_z
            data["mask"] = mask
        return data

    def diffuse(self, v_feature, v_timesteps, v_condition=None):
        bs = v_feature.size(0)
        de = v_feature.device
        dt = v_feature.dtype
        time_embeds = self.time_embed(sincos_embedding(v_timesteps, self.dim_total)).unsqueeze(1)
        noise_features = self.p_embed(v_feature)
        v_condition = torch.zeros((bs, 1, self.dim_condition), device=de, dtype=dt) if v_condition is None else v_condition
        v_condition = v_condition.repeat(1, v_feature.shape[1], 1)
        noise_features = torch.cat([noise_features, v_condition], dim=-1)
        noise_features = noise_features + time_embeds

        pred_x0 = self.net1(noise_features)
        pred_x0 = self.fc_out(pred_x0)
        return pred_x0

    def extract_condition(self, v_data):
        condition = None
        if self.with_img:
            if "img_features" in v_data["conditions"]:
                img_feature = v_data["conditions"]["img_features"]
                num_imgs = img_feature.shape[1]
            else:
                imgs = v_data["conditions"]["imgs"]
                num_imgs = imgs.shape[1]
                imgs = imgs.reshape(-1, 3, 224, 224)
                img_feature = self.img_model(imgs)
            img_idx = v_data["conditions"]["img_id"]
            camera_embedding = self.camera_embedding(img_idx)
            img_feature = self.img_fc(img_feature)
            img_feature = (img_feature.reshape(-1, num_imgs, self.dim_condition) + camera_embedding).mean(dim=1)
            condition = img_feature[:, None]
        elif self.with_pc:
            pc = v_data["conditions"]["points"]
            l_xyz, l_features = [pc[:, 0, :, :3].contiguous()], [pc[:, 0, ].permute(0, 2, 1).contiguous()]

            with torch.autocast(device_type=pc.device.type, dtype=torch.float32):
                for i in range(len(self.SA_modules)):
                    li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
                features = self.fc_lyaer(l_features[-1].mean(dim=-1))
                condition = features[:, None]

        return condition

    def forward(self, v_data, v_test=False, **kwargs):
        encoding_result = self.get_z(v_data, v_test)
        face_z = encoding_result["padded_face_z"]
        device = face_z.device
        bs = face_z.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        condition = self.extract_condition(v_data)
        noise = torch.randn(face_z.shape, device=device)
        noise_input = self.noise_scheduler.add_noise(face_z, noise, timesteps)

        # Model
        pred = self.diffuse(noise_input, timesteps, condition)
        
        loss = {}
        loss_item = self.loss(pred, face_z if self.diffusion_type == "sample" else noise, reduction="none")
        loss["diffusion_loss"] = loss_item.mean()
        if self.pad_method == "zero":
            mask = torch.logical_not((face_z.abs() < 1e-4).all(dim=-1))
            label = self.classifier(pred)
            classification_loss = nn.functional.binary_cross_entropy_with_logits(label[..., 0], mask.float())
            if self.loss == nn.functional.l1_loss:
                classification_loss = classification_loss * 1e-1
            else:
                classification_loss = classification_loss * 1e-4
            loss["classification"] = classification_loss
        loss["total_loss"] = sum(loss.values())
        loss["t"] = torch.stack((timesteps, loss_item.mean(dim=1).mean(dim=1)), dim=1)

        if self.is_train_decoder:
            raise
            pred_face_z = pred[encoding_result["mask"]]
            encoding_result["face_z"] = pred_face_z
            loss, recon_data = self.ae_model.loss(v_data, encoding_result)
            loss["l2"] = self.loss(pred, face_z)
            loss["diffusion_loss"] += loss["l2"]
        return loss
