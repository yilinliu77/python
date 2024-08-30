import importlib
from pathlib import Path
import sys
from einops import rearrange
import numpy as np
import open3d as o3d

from src.brepnet.dataset import Diffusion_dataset

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

import trimesh

class TrainDiffusion(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainDiffusion, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.dataset_mod = Diffusion_dataset

        model_mod = importlib.import_module("src.brepnet.diffusion_model")
        model_mod = getattr(model_mod, self.hydra_conf["model"]["name"])
        # self.model = AutoEncoder_base(self.hydra_conf["model"])
        self.model = model_mod(self.hydra_conf["model"])
        # self.model = torch.compile(self.model)

        model_mod = importlib.import_module("src.brepnet.model")
        model_mod = getattr(model_mod, self.hydra_conf["model"]["autoencoder"])
        self.autoencoder = model_mod(hparams["model"])
        weights = torch.load(self.hydra_conf["model"]["autoencoder_weights"])["state_dict"]
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        self.autoencoder.load_state_dict(weights)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.viz = {}

    def train_dataloader(self):
        self.train_dataset = self.dataset_mod("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          drop_last=True,
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

        loss = self.model(data)
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
        bs = data["face_features"].shape[0]
        loss = self.model(data, v_test=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == 0:
            result = self.inference()
            self.viz = {}
            self.viz["recon_faces"] = result["recon_faces"]
            self.viz["recon_mask"] = result["recon_mask"]
            self.viz["gt_mask"] = torch.logical_not((data["face_features"]==0).all(dim=-1).all(dim=-1).all(dim=-1))
            gt_feature = rearrange(data["face_features"], "b n c h w -> (b n) c h w")
            result = rearrange(self.autoencoder.decode(gt_feature), "(b n) h w c-> b n h w c", b=bs)
            self.viz["gt_faces"] = result

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        #     return

        if "recon_faces" in self.viz:
            recon_faces = self.viz["recon_faces"].cpu().numpy()
            recon_masks = self.viz["recon_mask"].cpu().numpy()
            gt_faces = self.viz["gt_faces"].cpu().numpy()
            gt_mask = self.viz["gt_mask"].cpu().numpy()
            for i in range(self.batch_size):
                local_face = recon_faces[i][recon_masks[i]]
                trimesh.PointCloud(local_face.reshape(-1,3)).export(str(self.log_root / "{}_{}_recon_faces.ply".format(self.current_epoch,i)))
                local_face = gt_faces[i][gt_mask[i]]
                trimesh.PointCloud(local_face.reshape(-1,3)).export(str(self.log_root / "{}_{}_gt_faces.ply".format(self.current_epoch,i)))
        self.viz={}
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_mod("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=1,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          )

    def inference(self):
        bs = self.batch_size
        face_feature = self.model.inference(bs, self.device)
        result = rearrange(face_feature, "b n (c h w) -> (b n) c h w", c=8, h=4, w=4)
        decoded_faces = self.autoencoder.decode(result)
        result = rearrange(decoded_faces, "(b n) h w c-> b n h w c", b=bs)
        data = {
            "recon_faces": result,
            "recon_mask": (face_feature.abs()>1e-4).all(dim=-1)
        }
        return data

    def test_step(self, batch, batch_idx):
        inference_result = self.inference()

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

        self.pr_computer.update(recon_data["pred"], recon_data["gt"])
        if True:
            os.makedirs(self.log_root / "test", exist_ok=True)
            recon_faces = recon_data["recon_faces"].cpu().numpy()
            recon_edges = recon_data["recon_edges"].cpu().numpy()
            np.savez_compressed(str(self.log_root / "test" / f"{data['v_prefix'][0]}.npz"), 
                                recon_faces=recon_faces,
                                recon_edges=recon_edges,
                                pred=recon_data["pred"].cpu().numpy(),
                                gt=recon_data["gt"].cpu().numpy(),
                                face_features=recon_data["face_features"],
                                edge_loss=recon_data["edge_loss"],
                                face_loss=recon_data["face_loss"],
                                )
            np.savez_compressed(str(self.log_root / "test" / f"{data['v_prefix'][0]}_feature.npz"), 
                                face_features=recon_data["face_features"],
                                )

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

@hydra.main(config_name="train_diffusion.yaml", config_path="../../configs/brepnet/", version_base="1.1")
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

    model = TrainDiffusion(v_cfg)
    exp_name = v_cfg["trainer"]["exp_name"]
    logger = TensorBoardLogger(
        log_dir,
        name="diffusion" if exp_name is None else exp_name)

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
        model = TrainDiffusion.load_from_checkpoint(v_cfg["trainer"].resume_from_checkpoint)
        model.hydra_conf = v_cfg

        # weights = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        # weights = {k.replace("model.", ""): v for k, v in weights.items()}
        # model.model.load_state_dict(weights)
    # model = torch.compile(model)
    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
