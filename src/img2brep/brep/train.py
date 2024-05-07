import sys
from tqdm import tqdm

from shared.common_utils import check_dir, safe_check_dir, export_point_cloud
from src.img2brep.brep.autoregressive import AutoregressiveModel
from src.img2brep.brep.diffusion import DiffusionModel, TrainDiffusionModel
from src.img2brep.brep.vae import TrainVaeModel, FineTuningVaeModel

sys.path.append('../../../')
import os.path
from pathlib import Path

import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from lightning_fabric import seed_everything

from src.img2brep.brep.dataset import AutoEncoder_dataset, Face_feature_dataset
from src.img2brep.brep.autoencoder import TrainAutoEncoder


class TrainAutoregressiveModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAutoregressiveModel, self).__init__()
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
            recon_vertices, recon_edges, recon_faces = self.model.generate(batch_size=1)
            # self.viz_recon["face_points"] = data["face_points"].cpu().numpy()
            # self.viz_recon["line_points"] = data["edge_points"].cpu().numpy()
            self.viz_recon["recon_edges"] = recon_edges.cpu().numpy()
            self.viz_recon["recon_faces"] = recon_faces.cpu().numpy()

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

@hydra.main(config_name="train_brepgen.yaml", config_path="../../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    train_mode = v_cfg["trainer"]["train_mode"]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if train_mode == 0:
        modelTraining = TrainAutoEncoder(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="autoencoder")
    elif train_mode == 2:
        modelTraining = TrainAutoregressiveModel(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="transformer")
    elif train_mode == 3:
        modelTraining = TrainDiffusionModel(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="diffusion")
    elif train_mode == 4:
        modelTraining = TrainVaeModel(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="vae")
    elif train_mode == 5:
        modelTraining = FineTuningVaeModel(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="vae")

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=logger,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true" if v_cfg["trainer"].gpu > 1 else "auto",
        # strategy="auto",
        devices=v_cfg["trainer"].gpu,
        log_every_n_steps=25,
        enable_model_summary=False,
        callbacks=[mc, lr_monitor],
        max_epochs=int(v_cfg["trainer"]["max_epochs"]),
        # max_epochs=2,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
        # accumulate_grad_batches=1,
        # profiler=SimpleProfiler(dirpath=log_dir, filename="profiler.txt"),
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        # ckpt = torch.load(v_cfg["trainer"].resume_from_checkpoint)
        # state_dict = {k: v for k, v in ckpt["state_dict"].items() if "autoencoder" not in k and "quantizer" not in k}
        # ckpt["state_dict"]=state_dict
        # torch.save(ckpt, v_cfg["trainer"].resume_from_checkpoint)
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        if train_mode == 1:
            print(modelTraining.model.load_state_dict(state_dict, strict=False))
        else:
            modelTraining.model.load_state_dict(state_dict, strict=True)

    if v_cfg["trainer"].evaluate:
        trainer.test(modelTraining)
    else:
        trainer.fit(modelTraining)


if __name__ == '__main__':
    main()
