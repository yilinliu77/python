import importlib
import os.path
from pathlib import Path

import h5py
import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics import MetricCollection

from shared.common_utils import export_point_cloud, sigmoid
import torch.distributed as dist

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score

from src.img2brep.meshgpt.dataset import Single_obj_dataset
from src.neural_bsp.my_dataloader import MyDataLoader

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

class MeshGPTTraining(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(MeshGPTTraining, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.autoencoder = MeshAutoencoder()
        self.transformer = MeshTransformer()


    def train_dataloader(self):
        self.train_dataset = Single_obj_dataset(
            "training",
            self.hydra_conf["dataset"],
        )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            collate_fn=self.dataset_name.collate_fn,
                            num_workers=self.hydra_conf["trainer"]["num_worker"],
                            pin_memory=True,
                            persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                            prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                            )

    def val_dataloader(self):
        self.valid_dataset = Single_obj_dataset(
            "validation",
            self.hydra_conf["dataset"],
        )
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch
        outputs = self.autoencoder(data, True)
        loss = self.autoencoder.loss(outputs, data)
        self.log("Training_Loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)
        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):
        data = batch[:2]
        outputs = self.autoencoder(data, False)
        loss = self.autoencoder.loss(outputs, data)
        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=1)

        return

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        return


@hydra.main(config_name="train_meshgpt.yaml", config_path="../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")
    model = MeshGPTTraining(v_cfg, v_cfg["dataset"]["root"])

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)

    trainer = Trainer(
        default_root_dir=log_dir,

        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false" if v_cfg["trainer"].gpu > 1 else "auto",
        devices=v_cfg["trainer"].gpu,

        enable_model_summary=False,
        callbacks=[mc],
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
        # gradient_clip_val=0.5,
    )
    torch.find_unused_parameters = False
    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
