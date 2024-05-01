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
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from src.img2brep.brep.dataset import Face_feature_dataset
from src.img2brep.brep.model import AutoEncoder
import pytorch_lightning as pl
import open3d as o3d


class TrainVaeModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainVaeModel, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.model = VaeModel(
            self.hydra_conf["model"]
        )

        self.viz_recon = {}
        self.viz_gen = {}

    def train_dataloader(self):
        self.train_dataset = Face_feature_dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Face_feature_dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=1,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
        }

    def training_step(self, batch, batch_idx):
        data = batch

        loss, _ = self.model(data)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=False, batch_size=self.batch_size)
        # if torch.isnan(total_loss).any():
        #     print("NAN Loss")
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == -1:
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
        return
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


class VaeModel(nn.Module):
    def __init__(
            self,
            v_conf,
    ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        if v_conf["checkpoint_autoencoder"] is not None:
            state_dict = torch.load(v_conf["checkpoint_autoencoder"])["state_dict"]
            state_dict_ = {k[6:]: v for k, v in state_dict.items()}
            self.autoencoder.load_state_dict(
                state_dict_, strict=True)
        self.autoencoder.eval()

        dim = v_conf["dim_latent"]
        self.quantizer = ResidualVQ(
            dim=dim,
            codebook_dim=32,
            num_quantizers=8,
            quantize_dropout=False,

            # separate_codebook_per_head=True,
            codebook_size=16384,
        )
        # self.quantizer = VectorQuantize(
        #     dim=dim,
        #     codebook_dim=32,  # a number of papers have shown smaller codebook dimension to be acceptable
        #     heads=8,  # number of heads to vector quantize, codebook shared across all heads
        #     separate_codebook_per_head=True,
        #     # whether to have a separate codebook per head. False would mean 1 shared codebook
        #     codebook_size=8196,
        #     accept_image_fmap=False
        # )
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=512,
                                           batch_first=True, dropout=0.1)
        self.quantizer_proj = nn.Sequential(
            nn.TransformerEncoder(layer, 8, norm=nn.LayerNorm(dim)),
            nn.Linear(dim, dim),
        )


    def forward(self, x,):
        quantized_features, indices, quantization_loss = self.quantizer(x)
        quantized_features = self.quantizer_proj(quantized_features)
        quantized_features = torch.sigmoid(quantized_features)
        loss = nn.functional.mse_loss(quantized_features, x)

        return {
            "total_loss": loss + quantization_loss.mean(),
            "feature_l2": loss,
            "quantization_loss": quantization_loss.mean(),
        }, {}

