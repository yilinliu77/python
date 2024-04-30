import os
from pathlib import Path

import torch
from denoising_diffusion_pytorch import GaussianDiffusion, GaussianDiffusion1D, Unet1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import SinusoidalPosEmb, default
from diffusers import AutoencoderKL, DDPMScheduler, DDPMPipeline
from einops import rearrange

from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.img2brep.brep.dataset import Face_feature_dataset
from src.img2brep.brep.model import AutoEncoder
import pytorch_lightning as pl
import open3d as o3d


class TrainDiffusionModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainDiffusionModel, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.model = DiffusionModel(
        # self.model = DiffuserModel(
            self.hydra_conf["model"]
        )

        self.viz_recon = {}
        self.viz_gen = {}

    def train_dataloader(self):
        self.train_dataset = Face_feature_dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=False,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Face_feature_dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=1,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=False,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-8, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=20, threshold=1e-5, min_lr=1e-5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor'  : 'Training_Loss',
            #     }
        }

    def training_step(self, batch, batch_idx):
        data = batch

        loss, _ = self.model(data, only_return_loss=True)
        total_loss = loss["total_loss"]
        # for key in loss:
        #     if key == "total_loss":
        #         continue
        #     self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
        #              sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=False, batch_size=self.batch_size)
        # if torch.isnan(total_loss).any():
        #     print("NAN Loss")
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, only_return_loss=False)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == 0:
            recon_vertices, recon_edges, recon_faces, mse_loss = self.model.generate(batch_size=1, v_gt_feature=data[0])
            # self.viz_recon["face_points"] = data["face_points"].cpu().numpy()
            # self.viz_recon["line_points"] = data["edge_points"].cpu().numpy()
            self.viz_recon["recon_edges"] = recon_edges.cpu().numpy()
            self.viz_recon["recon_faces"] = recon_faces.cpu().numpy()
            self.log("Feature_MSE_Loss", mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        # return
        def vis(viz_data, subname):
            assert subname in ["recon", "gen"]

            recon_edges = viz_data["recon_edges"]
            recon_faces = viz_data["recon_faces"]

            valid_flag = (recon_edges != -1).all(axis=-1).all(axis=-1)
            recon_edges = recon_edges[valid_flag]
            valid_flag = (recon_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)
            recon_faces = recon_faces[valid_flag]

            edge_points = recon_edges.reshape(-1, 3)

            face_points = recon_faces.reshape(-1, 3)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(edge_points)
            o3d.io.write_point_cloud(
                str(self.log_root / f"{self.trainer.current_epoch:05}_viz_edges_{subname}.ply"),
                pc)

            pc.points = o3d.utility.Vector3dVector(face_points)
            o3d.io.write_point_cloud(
                str(self.log_root / f"{self.trainer.current_epoch:05}_viz_faces_{subname}.ply"),
                pc)
        vis(self.viz_recon, "recon")
        return


class Diff_transformer(nn.Module):
    def __init__(
            self,
            dim,
            channels=3,
            self_condition=False,
            sinusoidal_pos_emb_theta=10000,
            is_causal=False
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        self.is_causal = is_causal
        # determine dimensions

        dim = 768
        if self.self_condition:
            self.project_in = nn.Linear(channels * 2, dim)
        else:
            self.project_in = nn.Linear(channels, dim)

        # self.atten = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        #     # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        # ])

        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=12, norm_first=True,
                                                   dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerEncoder(layer, 12, nn.LayerNorm(dim))

        # time embeddings
        time_dim = dim
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, channels),
        )

    def forward(self, x, time, x_self_cond=None, is_causal=False):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        mask = None
        if self.is_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[2], device=x.device)

        x = self.project_in(x.permute(0, 2, 1))
        # x = x.permute(0,2,1)
        r = x.clone()
        t = self.time_mlp(time)
        x = x + t[:,None]
        # for layer in self.atten:
        #     x = layer(x, is_causal=self.is_causal, src_mask=mask)
        x = self.net(x)
        x = self.final_mlp(x)
        x = x.permute(0, 2, 1)
        return x


class DiffusionModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=256,
                 ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        if v_conf["checkpoint_autoencoder"] is not None:
            state_dict = torch.load(v_conf["checkpoint_autoencoder"])["state_dict"]
            state_dict_ = {k[6:]: v for k, v in state_dict.items()}
            self.autoencoder.load_state_dict(
                state_dict_, strict=True)
        self.autoencoder.eval()

        self.num_max_faces = v_conf["num_max_faces"]

        self.is_causal = v_conf["diffusion_causal"]
        self.model = Diff_transformer(
            dim=dim,
            channels=dim,
            self_condition=False,
            is_causal=self.is_causal,
        )

        self.diffusion = GaussianDiffusion1D(
            self.model,
            seq_length=self.num_max_faces,
            timesteps=1000,
            auto_normalize=True,
            beta_schedule='linear',
            # objective = 'pred_v',
            objective=v_conf["diffusion_objective"],
        )
        # self.diffusion = AutoencoderKL(in_channels=256,
        #     out_channels=256,
        #     down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        #     up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        #     block_out_channels=[256, 256, 256, 256],
        #     layers_per_block=2,
        #     act_fn='silu',
        #     latent_channels=256,
        #     norm_num_groups=32,
        #     sample_size=256,
        # )

        self.padding_token = nn.Parameter(torch.rand(dim), requires_grad=True)
        pass

    def forward_on_embedding1(self, v_face_embeddings, only_return_loss=False, only_return_recon=False):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings == 0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        face_embeddings = v_face_embeddings
        # face_embeddings[zero_flag] = 0
        # face_embeddings[zero_flag] = self.padding_token

        padded_face_embeddings = torch.cat((
            face_embeddings,
            self.padding_token.repeat((B, self.num_max_faces - face_embeddings.shape[1], 1)),
        ), dim=1)

        posterior = self.diffusion.encode(padded_face_embeddings.permute(0, 2, 1)).latent_dist
        z = posterior.sample()
        dec = self.model.decode(z).sample

        kl_loss = posterior.kl().mean()
        mse_loss = ((dec - padded_face_embeddings) ** 2).mean()
        total_loss = mse_loss + 1e-6 * kl_loss

        loss = {
            "total_loss": total_loss,
        }

        return loss, {}

    @torch.no_grad()
    def prepare_face_embedding(self, v_data):
        loss, recon_data = self.autoencoder(v_data, return_face_features=True)
        face_embeddings = recon_data["face_embeddings"]
        return face_embeddings

    def forward(self, v_face_embeddings, **kwargs):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings == 0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        idx = torch.arange(self.num_max_faces, device=v_face_embeddings.device)[None,:].repeat(B,1) % num_valid[:,None]

        idx = idx[:, torch.randperm(self.num_max_faces)]

        padded_face_embeddings = torch.gather(
            v_face_embeddings, dim=1, index=idx[:, :, None].expand(B, self.num_max_faces, dim))

        loss = self.diffusion(padded_face_embeddings.permute(0, 2, 1))

        loss = {
            "total_loss": loss,
        }

        return loss, {}

    @torch.no_grad()
    def generate(self, batch_size=1, v_gt_feature=None):
        samples = self.diffusion.sample(batch_size).permute(0, 2, 1)
        # distance = nn.functional.mse_loss(samples, self.padding_token[None, None, :], reduction='none').mean(dim=-1)
        # valid_flag = distance > 1e-2
        face_embeddings = samples
        # face_embeddings[~valid_flag] = 0
        recon_vertices, recon_edges, recon_faces = self.autoencoder.inference(face_embeddings)
        mse_loss = v_gt_feature.new_zeros(1)
        # mse_loss = nn.functional.mse_loss(face_embeddings[0,:3], v_gt_feature)
        return recon_vertices, recon_edges, recon_faces, mse_loss



class DiffuserModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=256,
                 ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        if v_conf["checkpoint_autoencoder"] is not None:
            state_dict = torch.load(v_conf["checkpoint_autoencoder"])["state_dict"]
            state_dict_ = {k[6:]: v for k, v in state_dict.items()}
            self.autoencoder.load_state_dict(
                state_dict_, strict=True)
        self.autoencoder.eval()

        self.num_max_faces = v_conf["num_max_faces"]

        self.is_causal = v_conf["diffusion_causal"]
        self.model = Diff_transformer(
            dim=dim,
            channels=dim,
            self_condition=False,
            is_causal=self.is_causal,
        )
        self.padding_token = nn.Parameter(torch.rand(dim), requires_grad=True)

        self.schedular = DDPMScheduler(
            num_train_timesteps=1000,
            # prediction_type=v_conf["diffusion_objective"],
        )
        self.pipeline = DDPMPipeline(unet=self.model, scheduler=self.schedular)
        pass


    @torch.no_grad()
    def prepare_face_embedding(self, v_data):
        loss, recon_data = self.autoencoder(v_data, return_face_features=True)
        face_embeddings = recon_data["face_embeddings"]
        return face_embeddings

    def forward(self, v_face_embeddings, **kwargs):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings == 0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        face_embeddings = v_face_embeddings
        # face_embeddings[zero_flag] = 0
        face_embeddings[zero_flag] = self.padding_token

        padded_face_embeddings = torch.cat((
            face_embeddings,
            self.padding_token.repeat((B, self.num_max_faces - face_embeddings.shape[1], 1)),
        ), dim=1).permute(0, 2, 1)
        padded_face_embeddings = padded_face_embeddings * 2 - 1

        # Diffusion
        noise = torch.randn(padded_face_embeddings.shape, device=padded_face_embeddings.device)
        timesteps = torch.randint(
            0, self.schedular.config.num_train_timesteps, (B,), device=padded_face_embeddings.device,
            dtype=torch.int64
        )
        noisy_images = self.schedular.add_noise(padded_face_embeddings, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = nn.functional.mse_loss(noise_pred, noise)

        loss = {
            "total_loss": loss,
        }

        return loss, {}

    @torch.no_grad()
    def generate(self, batch_size=1, v_gt_feature=None):
        generated_features = torch.randn((batch_size, 256, self.num_max_faces)).cuda()
        for t in tqdm(self.schedular.timesteps):
            timesteps = t.reshape(-1).cuda()
            pred = self.model(generated_features, timesteps)
            generated_features = self.schedular.step(pred, t, generated_features).prev_sample
        samples = generated_features.permute(0,2,1) / 2 + 0.5
        distance = nn.functional.mse_loss(samples, self.padding_token[None, None, :], reduction='none').mean(dim=-1)
        valid_flag = distance > 1e-2
        face_embeddings = samples
        face_embeddings[~valid_flag] = 0
        recon_vertices, recon_edges, recon_faces = self.autoencoder.inference(face_embeddings)
        mse_loss = v_gt_feature.new_zeros(1)
        # mse_loss = nn.functional.mse_loss(face_embeddings[0,:3], v_gt_feature)
        return recon_vertices, recon_edges, recon_faces, mse_loss
