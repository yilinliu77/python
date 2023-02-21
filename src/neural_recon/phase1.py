import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import List

import mcubes
import tinycudann as tcnn

import PIL.Image
import numpy as np
import open3d
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tqdm import tqdm

import math
import platform
import shutil
import sys, os

from typing import Tuple

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from shared.fast_dataloader import FastDataLoader
from scipy import stats

from shared.common_utils import debug_imgs
from shared.img_torch_tools import get_img_from_tensor

import cv2

from src.neural_recon.Image_dataset import Image_dataset
from src.neural_recon.colmap_io import read_dataset


class Phase1(pl.LightningModule):
    def __init__(self, hparams, v_img_path):
        super(Phase1, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.img = cv2.cvtColor(cv2.imread(v_img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.img_name = os.path.basename(v_img_path).split(".")[0]
        self.log_root = os.path.join(self.hydra_conf["trainer"]["output"], self.img_name)
        os.makedirs(self.log_root,exist_ok=True)
        os.makedirs(os.path.join(self.log_root, "imgs"),exist_ok=True)

        # Define models
        self.model1 = tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
        })
        self.model2 = tcnn.Network(n_input_dims=self.model1.n_output_dims, n_output_dims=3, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        })

    def forward(self, batch):
        features = self.model1(batch[0][0])  # (batch_size, feature_dim)
        predicted_pixel = self.model2(features)  # (batch_size, feature_dim)
        return predicted_pixel

    def train_dataloader(self):
        self.train_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "training"
        )
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "validation"
        )
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def test_dataloader(self):
        self.test_dataset = Geometric_dataset_inference(
            self.hydra_conf["model"]["marching_cube_resolution"],
            self.hydra_conf["trainer"]["batch_size"],
        )
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          # collate_fn=Geometric_dataset.collate_fn,
                          )

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        predicted_pixel = self.forward(batch)
        loss = F.mse_loss(predicted_pixel, batch[1][0])

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        predicted_pixel = self.forward(batch)
        loss = F.mse_loss(predicted_pixel, batch[1][0])

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch[0].shape[0])

        return predicted_pixel

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return

        predicted_pixel = torch.cat(result, dim=0).cpu().numpy()
        predicted_pixel = np.clip(predicted_pixel, 0, 1)
        predicted_pixel = predicted_pixel.reshape([self.img.shape[0], self.img.shape[1], 3])
        predicted_pixel = (predicted_pixel * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.log_root, "imgs/{}.png").format(self.trainer.current_epoch),
                    cv2.cvtColor(predicted_pixel, cv2.COLOR_RGB2BGR))

    def test_step(self, batch, batch_idx):
        batch = [batch[0][0], batch[1][0]]
        predicted_sdf = self.forward(batch)
        return predicted_sdf

    def test_epoch_end(self, result):
        predicted_sdf = -torch.cat(result, dim=0).cpu().numpy().astype(np.float32)
        resolution = self.hydra_conf["model"]["marching_cube_resolution"]
        predicted_sdf = predicted_sdf.reshape([resolution, resolution, resolution])
        vertices, triangles = mcubes.marching_cubes(predicted_sdf, 0)
        mcubes.export_obj(vertices, triangles,
                          os.path.join("outputs", "model_of_test.obj"))


@hydra.main(config_name="phase1_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    # Read dataset and bounds
    bounds_min = np.array(v_cfg["dataset"]["scene_boundary"][:3], dtype=np.float32)
    bounds_max = np.array(v_cfg["dataset"]["scene_boundary"][3:], dtype=np.float32)
    scene_bounds = np.array([bounds_min, bounds_max])
    imgs, world_points = read_dataset(v_cfg["dataset"]["colmap_dir"], scene_bounds)

    for id_img in range(len(imgs)):
        model = Phase1(v_cfg, imgs[id_img].img_path)
        img_name = os.path.basename(imgs[id_img].img_path).split(".")[0]
        if img_name not in v_cfg["dataset"]["target_img"]:
            continue
        from pytorch_lightning import loggers as pl_loggers
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="output/neural_recon/img_nif_log", name=img_name)
        checkpoint_callback = ModelCheckpoint(dirpath="output/neural_recon/img_nif_log/{}".format(img_name),
                                              save_top_k=1, monitor="Validation_Loss")
        trainer = Trainer(
            logger=tb_logger,
            accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
            devices=v_cfg["trainer"].gpu, enable_model_summary=False,
            max_epochs=200,
            num_sanity_val_steps=2,
            precision=16,
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            reload_dataloaders_every_n_epochs=v_cfg["dataset"]["resample_after_n_epoches"],
            callbacks=[checkpoint_callback]
        )
        if v_cfg["trainer"].evaluate:
            trainer.test(model)
        else:
            trainer.fit(model)


if __name__ == '__main__':
    main()
