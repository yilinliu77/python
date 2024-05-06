import os
from pathlib import Path

import torch
from denoising_diffusion_pytorch import GaussianDiffusion, GaussianDiffusion1D, Unet1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import SinusoidalPosEmb, default
from diffusers import AutoencoderKL, DDPMScheduler, DDPMPipeline
from einops import rearrange, reduce

from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from src.img2brep.brep.dataset import Face_feature_dataset, AutoEncoder_dataset
import pytorch_lightning as pl
import open3d as o3d

import torch.nn.functional as F


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
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
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

class FineTuningVaeModel(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTuningVaeModel, self).__init__()
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
        self.train_dataset = Autoencoder_Dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Autoencoder_Dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def test_dataloader(self):
        self.test_dataset = Autoencoder_Dataset("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=False,
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
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
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


    def test_step(self, batch, batch_idx):
        data = batch

        loss, _ = self.model.test(data)
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


