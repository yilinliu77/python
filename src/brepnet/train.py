import importlib
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

from src.brepnet.dataset import AutoEncoder_dataset, AutoEncoder_dataset2

sys.path.append('../../../')
import os.path

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric import seed_everything

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, AdamW
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score
from torchmetrics import MetricCollection

import wandb
from pytorch_lightning.loggers import WandbLogger

os.environ["HTTP_PROXY"] = "http://172.31.178.126:7890"
os.environ["HTTPS_PROXY"] = "http://172.31.178.126:7890"

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

        dataset_mod = importlib.import_module("src.brepnet.dataset")
        self.dataset_mod = getattr(dataset_mod, self.hydra_conf["dataset"]["dataset_name"])

        model_mod = importlib.import_module("src.brepnet.model")
        model_mod = getattr(model_mod, self.hydra_conf["model"]["name"])
        self.model = model_mod(self.hydra_conf["model"])
        self.viz = {}
        pr_computer = {
            "P_5": BinaryPrecision(threshold=0.5),
            "P_7": BinaryPrecision(threshold=0.7),
            "R_5": BinaryRecall(threshold=0.5),
            "R_7": BinaryRecall(threshold=0.7),
            "F1": BinaryF1Score(threshold=0.5),
        }
        self.pr_computer = MetricCollection(pr_computer)

        if "compile" in hparams["trainer"] and hparams["trainer"]["compile"]:
            self.model = torch.compile(self.model, dynamic=True)

    def train_dataloader(self):
        self.train_dataset = self.dataset_mod("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          drop_last=True,
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=4 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_mod("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=1,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=4 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
        }

    def training_step(self, batch, batch_idx):
        data = batch
        loss, data = self.model(data)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Training/{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True ,batch_size=self.batch_size)
        self.log("Training/Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, v_test=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation/{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation/Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if "gt_face_adj" in recon_data:
            self.pr_computer.update(recon_data["pred_face_adj"], recon_data["gt_face_adj"])

        # Rank zero only
        if self.global_rank != 0:
            return total_loss
        
        if batch_idx == 0:
            if "pred_face" in recon_data:
                self.viz["gt_face"] = recon_data["gt_face"]
                self.viz["pred_face"] = recon_data["pred_face"]
            if "pred_edge" in recon_data:
                self.viz["gt_edge"] = recon_data["gt_edge"]
                self.viz["pred_edge"] = recon_data["pred_edge"]
        
        return total_loss

    def on_validation_epoch_end(self):
        self.log_dict(self.pr_computer.compute(), prog_bar=False, logger=True, on_step=False, on_epoch=True,
                    sync_dist=True)
        self.pr_computer.reset()

        if self.global_rank != 0:
            return

        if "pred_face" in self.viz:
            gt_faces = self.viz["gt_face"][..., :3]
            recon_faces = self.viz["pred_face"][..., :3]
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
                str(self.log_root / f"{self.trainer.current_epoch:05}_viz_faces.ply"), pc)

        if "pred_edge" in self.viz and self.viz["pred_edge"].shape[0]>0:
            recon_edges = self.viz["pred_edge"][..., :3]
            gt_edges = self.viz["gt_edge"][..., :3]

            gt_edges = gt_edges[(gt_edges != -1).all(axis=-1).all(axis=-1)]
            recon_edges = recon_edges[(recon_edges != -1).all(axis=-1).all(axis=-1)]

            edge_points = np.concatenate((gt_edges, recon_edges), axis=0).reshape(-1, 3)
            edge_colors = np.concatenate(
                (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_edges.shape[0] * gt_edges.shape[1], axis=0),
                    np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_edges.shape[0] * recon_edges.shape[1], axis=0)), axis=0)
            
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(edge_points)
            pc.colors = o3d.utility.Vector3dVector(edge_colors / 255.0)
            o3d.io.write_point_cloud(
                str(self.log_root / f"{self.trainer.current_epoch:05}_viz_edges.ply"), pc)
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_mod("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=1,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          )

    def test_step(self, batch, batch_idx):
        log_root = Path(self.hydra_conf["trainer"]["test_output_dir"])
        log_root.mkdir(exist_ok=True, parents=True)
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

        if "pred_face_adj" in recon_data:
            self.pr_computer.update(recon_data["pred_face_adj"].reshape(-1), recon_data["gt_face_adj"].reshape(-1))
        if True:
            local_root = log_root / f"{data['v_prefix'][0]}"
            local_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(local_root / f"data.npz"),
                                # pred_face_adj_prob=recon_data["pred_face_adj_prob"],
                                pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
                                pred_face=recon_data["pred_face"],
                                pred_edge=recon_data["pred_edge"],
                                pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],

                                gt_face_adj=recon_data["gt_face_adj"].cpu().numpy(),
                                gt_face=recon_data["gt_face"],
                                gt_edge=recon_data["gt_edge"],
                                gt_edge_face_connectivity=recon_data["gt_edge_face_connectivity"],

                                face_loss=loss["face_coords"].cpu().item(),
                                edge_loss=loss["edge_coords"].cpu().item(),
                                edge_loss_ori=loss["edge_coords1"].cpu().item(),
                                )
            np.save(str(local_root / "features"),
                                recon_data["face_features"],
                                )
            
        if False:
            gt_faces = recon_data["gt_face"][..., :-3]
            recon_faces = recon_data["pred_face"][..., :-3]
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
            o3d.io.write_point_cloud(str(f"{self.trainer.current_epoch:05}_viz_faces.ply"), pc)
            exit()

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
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    use_wandb = v_cfg["trainer"]["wandb"] if "wandb" in v_cfg["trainer"] else False
    exp_name = v_cfg["trainer"]["exp_name"]
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir'] + "/" + exp_name
    v_cfg["trainer"]["output"] = log_dir
    print("Log dir: ", log_dir)
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    mc = ModelCheckpoint(monitor="Validation/Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = TrainAutoEncoder(v_cfg)

    if v_cfg["trainer"]["evaluate"] is not True and exp_name!="test" and use_wandb:
        logger = WandbLogger(
            project='BRepNet++',
            save_dir=log_dir,
            name=exp_name,
        )
        logger.watch(model)
    else:
        logger = TensorBoardLogger(log_dir)

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=logger,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true" if v_cfg["trainer"].gpu > 1 else "auto",
        # strategy="auto",
        devices=v_cfg["trainer"].gpu,
        enable_model_summary=True,
        callbacks=[mc, lr_monitor],
        max_epochs=int(v_cfg["trainer"]["max_epochs"]),
        # max_epochs=2,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
        # profiler="advanced",
        # max_steps=100
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        model = TrainAutoEncoder.load_from_checkpoint(v_cfg["trainer"].resume_from_checkpoint, map_location="cpu")
        model.hydra_conf = v_cfg
    # model = torch.compile(model)
    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)
    

if __name__ == '__main__':
    main()
