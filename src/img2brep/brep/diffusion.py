import os
from pathlib import Path

import torch
from denoising_diffusion_pytorch import GaussianDiffusion, GaussianDiffusion1D, Unet1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import SinusoidalPosEmb, default
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
        self.dataset_name = self.hydra_conf["dataset"]["dataset_name"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.model = DiffusionModel(
                self.hydra_conf["model"]
                )

        self.viz_recon = {}
        self.viz_gen = {}

    def train_dataloader(self):
        self.train_dataset = Face_feature_dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Face_feature_dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
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
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        # determine dimensions

        if self.self_condition:
            self.project_in = nn.Linear(channels * 2, dim)
        else:
            self.project_in = nn.Linear(channels, dim)

        self.atten = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        ])


        # time embeddings
        time_dim = dim
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.final_mlp = nn.Linear(dim, channels)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.project_in(x.permute(0,2,1))
        # x = x.permute(0,2,1)
        r = x.clone()
        t = self.time_mlp(time)
        for layer in self.atten:
            x = layer(x + t[:,None,:])
        x = self.final_mlp(x + r)
        x = x.permute(0,2,1)
        return x


class DiffusionModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=768,
                 ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        state_dict = torch.load(v_conf["checkpoint_autoencoder"])["state_dict"]
        state_dict_ = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k and 'quantizer' not in k}
        self.autoencoder.load_state_dict(
            state_dict_, strict=False)
        self.autoencoder.eval()

        self.num_max_faces = v_conf["num_max_faces"]

        # model = Unet1D(
        #     dim=256,
        #     channels=256,
        #     dim_mults=(1, 1, 2, 2),
        # )
        self.model = Diff_transformer(
            dim=dim,
            channels=dim,
            self_condition=False,
        )

        # from diffusers.schedulers import DDPMScheduler, DDIMScheduler
        # self.diffusion = DDIMScheduler()

        self.diffusion = GaussianDiffusion1D(
            self.model,
            seq_length=self.num_max_faces,
            timesteps=1000,
            auto_normalize=False,
            # objective = 'pred_v',
        )

        self.padding_token = nn.Parameter(torch.randn(dim), requires_grad=True)
        pass

    def forward_on_embedding1(self, v_face_embeddings, only_return_loss=False, only_return_recon=False):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings==0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        # face_embeddings = self.mapper(v_face_embeddings)
        face_embeddings = torch.sigmoid(v_face_embeddings)
        # face_embeddings[zero_flag] = 0

        idx = torch.randperm(self.num_max_faces, device=face_embeddings.device)[None,:].repeat(B,1)
        idx = idx % num_valid[:,None]

        padded_face_embeddings = torch.gather(face_embeddings, 1, idx[:,:,None].repeat(1,1,dim))
        # padded_face_embeddings = face_embeddings
        # noise = torch.randn(padded_face_embeddings.shape, device=padded_face_embeddings.device)
        # timesteps = torch.randint(
        #     0, 1000, (B,), device=padded_face_embeddings.device,
        #     dtype=torch.int64
        # )
        # noisy_images = self.diffusion.add_noise(padded_face_embeddings, noise, timesteps)
        # noise_pred = self.model(noisy_images.permute(0,2,1), timesteps).permute(0,2,1)
        # loss = nn.functional.mse_loss(noise_pred, noise)

        loss = self.diffusion(padded_face_embeddings.permute(0,2,1))

        loss = {
            "total_loss"         : loss,
            }

        return loss, {}

    def forward_on_embedding(self, v_face_embeddings, only_return_loss=False, only_return_recon=False):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings==0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        # face_embeddings = torch.sigmoid(v_face_embeddings)
        face_embeddings = v_face_embeddings
        # face_embeddings[zero_flag] = 0
        face_embeddings[zero_flag] = self.padding_token

        padded_face_embeddings = torch.cat((
            face_embeddings,
            self.padding_token.repeat((B, self.num_max_faces-face_embeddings.shape[1], 1)),
        ), dim=1)

        loss = self.diffusion(padded_face_embeddings.permute(0,2,1))

        loss = {
            "total_loss"         : loss,
            }

        return loss, {}

    @torch.no_grad()
    def prepare_face_embedding(self, v_data):
        loss, recon_data = self.autoencoder(v_data, return_face_features=True)
        face_embeddings = recon_data["face_embeddings"]
        return face_embeddings

    def forward(self, v_data, **kwargs):
        face_embeddings = v_data
        # face_embeddings = self.prepare_face_embedding(v_data)
        return self.forward_on_embedding(face_embeddings, **kwargs)

    @torch.no_grad()
    def generate(self, batch_size=1, v_gt_feature=None):
        samples = self.diffusion.sample(batch_size).permute(0,2,1)
        distance = nn.functional.mse_loss(samples, self.padding_token[None,None,:], reduction='none').mean(dim=-1)
        valid_flag = distance > 1e-2
        face_embeddings = samples
        # face_embeddings = -torch.log(torch.clamp_min(1 / torch.clamp_min(samples, 1e-7) - 1, 1e-7))
        face_embeddings[~valid_flag] = 0
        recon_vertices, recon_edges, recon_faces = self.autoencoder.inference(face_embeddings)
        mse_loss = nn.functional.mse_loss(face_embeddings[0], v_gt_feature)
        return recon_vertices, recon_edges, recon_faces, mse_loss
