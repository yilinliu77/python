from pathlib import Path
import sys
import numpy as np
import open3d as o3d

from src.brepnet.dataset import AutoEncoder_dataset
from src.brepnet.model import AutoEncoder_base

sys.path.append('../../../')
import os.path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric import seed_everything

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score
from torchmetrics import MetricCollection


class TrainAutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAutoEncoder, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.dataset_mod = AutoEncoder_dataset

        self.model = AutoEncoder_base(self.hydra_conf["model"])
        self.viz = {}
        pr_computer = {
            "P_5": BinaryPrecision(threshold=0.5),
            "P_7": BinaryPrecision(threshold=0.7),
            "R_5": BinaryRecall(threshold=0.5),
            "R_7": BinaryRecall(threshold=0.7),
            "F1": BinaryF1Score(threshold=0.5),
        }
        self.pr_computer = MetricCollection(pr_computer)

    def train_dataloader(self):
        self.train_dataset = self.dataset_mod("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_mod("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_mod.collate_fn,
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

        loss, data = self.model(data, return_loss=True,
                                return_recon=False, return_face_features=False, return_true_loss=False)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)
        if torch.isnan(total_loss).any():
            print("NAN Loss")
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, v_test=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == 0:
            # recon_edges, recon_faces = self.model.inference(recon_data["face_embeddings"])
            self.viz["face_points"] = data["face_points"].cpu().numpy()
            self.viz["recon_faces"] = recon_data["recon_faces"].cpu().numpy()
            if "recon_edges" in recon_data:
                self.viz["edge_points"] = data["edge_points"].cpu().numpy()
                self.viz["vertex_points"] = data["vertex_points"].cpu().numpy()
                self.viz["recon_edges"] = recon_data["recon_edges"].cpu().numpy()
                self.viz["recon_vertices"] = recon_data["recon_vertices"].cpu().numpy()

        pred = torch.cat(recon_data["pred"], dim=0)[:,0]
        gt = torch.cat(recon_data["gt"], dim=0)
        self.pr_computer.update(pred, gt)

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        #     return

        self.log_dict(self.pr_computer.compute(), prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True)
        self.pr_computer.reset()

        v_recon_faces = self.viz["recon_faces"]
        v_gt_faces = self.viz["face_points"]

        for idx in range(min(v_gt_faces.shape[0], 4)):
            gt_faces = v_gt_faces[idx]
            recon_faces = v_recon_faces[idx]
            gt_faces = gt_faces[(gt_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)]
            recon_faces = recon_faces[(recon_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)]

            num_face_points = gt_faces.shape[1] ** 2
            face_points = np.concatenate((gt_faces, recon_faces), axis=0).reshape(-1, 3)
            face_colors = np.concatenate(
                (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_faces.shape[0] * num_face_points, axis=0),
                    np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_faces.shape[0] * num_face_points, axis=0)), axis=0)

            pc = o3d.geometry.PointCloud()
            
            pc.points = o3d.utility.Vector3dVector(face_points)
            pc.colors = o3d.utility.Vector3dVector(face_colors / 255.0)
            o3d.io.write_point_cloud(
                str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_faces.ply"), pc)

            if "recon_edges" in self.viz:
                v_gt_vertices = self.viz["vertex_points"]
                v_gt_edges = self.viz["edge_points"]
                v_recon_vertices = self.viz["recon_vertices"]
                v_recon_edges = self.viz["recon_edges"]

                recon_vertices = v_recon_vertices[idx]
                recon_edges = v_recon_edges[idx]
                gt_vertices = v_gt_vertices[idx]
                gt_edges = v_gt_edges[idx]

                gt_edges = gt_edges[(gt_edges != -1).all(axis=-1).all(axis=-1)]
                recon_edges = recon_edges[(recon_edges != -1).all(axis=-1).all(axis=-1)]
                gt_vertices = gt_vertices[(gt_vertices != -1).all(axis=-1)]
                recon_vertices = recon_vertices[(recon_vertices != -1).all(axis=-1)]
                
                vertex_points = np.concatenate((gt_vertices, recon_vertices), axis=0).reshape(-1, 3)
                vertex_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_vertices.shape[0], axis=0),
                        np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_vertices.shape[0], axis=0)), axis=0)
                
                edge_points = np.concatenate((gt_edges, recon_edges), axis=0).reshape(-1, 3)
                edge_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_edges.shape[0] * 32, axis=0),
                        np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_edges.shape[0] * 32, axis=0)), axis=0)
                
                pc.points = o3d.utility.Vector3dVector(vertex_points)
                pc.colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
                o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_vertices.ply"), pc)
                
                pc.points = o3d.utility.Vector3dVector(edge_points)
                pc.colors = o3d.utility.Vector3dVector(edge_colors / 255.0)
                o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_edges.ply"), pc)


        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_mod("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          )

    def test_step(self, batch, batch_idx):
        data = batch
        loss, recon_data = self.model(data, v_test=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Test_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Test_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        pred = torch.cat(recon_data["pred"], dim=0)[:,0]
        gt = torch.cat(recon_data["gt"], dim=0)
        self.pr_computer.update(pred, gt)

    def on_test_epoch_end(self):
        self.log_dict(self.pr_computer.compute(), prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True)
        metrics = self.pr_computer.compute()
        for key in metrics:
            print("{:3}: {:.3f}".format(key, metrics[key].cpu().item()))

        for loss in self.trainer.callback_metrics:
            print("{}: {:.3f}".format(loss, self.trainer.callback_metrics[loss].cpu().item()))
        self.pr_computer.reset()
        return

@hydra.main(config_name="train_brepnet.yaml", config_path="../../configs/brepnet/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = TrainAutoEncoder(v_cfg)
    exp_name = v_cfg["trainer"]["exp_name"]
    logger = TensorBoardLogger(
        log_dir,
        name="autoencoder" if exp_name is None else exp_name)

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=logger,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true" if v_cfg["trainer"].gpu > 1 else "auto",
        devices=v_cfg["trainer"].gpu,
        enable_model_summary=False,
        callbacks=[mc, lr_monitor],
        max_epochs=int(v_cfg["trainer"]["max_epochs"]),
        # max_epochs=2,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        model = TrainAutoEncoder.load_from_checkpoint(v_cfg["trainer"].resume_from_checkpoint)
        model.hydra_conf = v_cfg

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
