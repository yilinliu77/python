import mcubes
import tinycudann as tcnn
import numpy as np
import PIL.Image
import open3d as o3d
import torch
from torch import nn
import cv2

import math
import platform
import shutil
import sys, os

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.Image_dataset import Image_dataset
from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.phase1 import NGPModel


class NGPModel1(nn.Module):
    def __init__(self):
        super(NGPModel1, self).__init__()

        # Define models
        self.n_frequencies = 12
        self.n_layer = 4
        self.model1 = tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "Frequency",
            "n_frequencies": self.n_frequencies
        }, dtype=torch.float32)
        assert self.model1.n_output_dims % self.n_layer == 0
        self.num_freq_per_layer = self.n_frequencies // self.n_layer

        self.model2 = []
        for i in range(self.n_layer):
            self.model2.append(tcnn.Network(
                n_input_dims=self.model1.n_output_dims // self.n_layer * (i + 1),
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 6,
                }))

    def forward(self, v_pixel_pos):
        bs = v_pixel_pos.shape[0]
        nf = self.num_freq_per_layer
        features = self.model1(v_pixel_pos)  # (batch_size, 2 * n_frequencies * 2)
        features = features.reshape(bs, 2, -1, 2)
        predicted_grays = []
        accumulated_pos_encoding = None
        for i_layer in range(self.n_layer):
            feature_layer = features[:, :, nf * i_layer:nf * i_layer + nf, :].reshape(bs, -1)
            if accumulated_pos_encoding is not None:
                feature_layer = torch.cat((accumulated_pos_encoding, feature_layer), dim=1)
            accumulated_pos_encoding = feature_layer
            predicted_gray = self.model2[i_layer](accumulated_pos_encoding)
            predicted_grays.append(predicted_gray)
        return predicted_grays


class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class NGPModel2(nn.Module):
    def __init__(self):
        super(NGPModel2, self).__init__()

        # Define models
        self.n_frequencies = 12
        self.n_layer = 4
        self.num_freq_per_layer = self.n_frequencies // self.n_layer
        v_hidden_dim = 256

        self.model2 = nn.ModuleList()
        for i in range(self.n_layer):
            self.model2.append(
                nn.Sequential(
                    torch.nn.Linear(2 * 2 * self.num_freq_per_layer * (i+1), v_hidden_dim),
                    torch.nn.ReLU(),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    torch.nn.Linear(v_hidden_dim, 1)
                )
            )


    def positional_encoding(self, input, L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc

    def forward(self, v_pixel_pos):
        bs = v_pixel_pos.shape[0]
        nf = self.num_freq_per_layer
        features = self.positional_encoding(v_pixel_pos, self.n_frequencies)  # (batch_size, 2 * n_frequencies * 2)
        features = features.reshape(bs, 2, 2, -1)
        predicted_grays = []
        accumulated_pos_encoding = None
        for i_layer in range(self.n_layer):
            feature_layer = features[:, :, :, nf * i_layer:nf * i_layer + nf].reshape(bs, -1)
            if accumulated_pos_encoding is not None:
                feature_layer = torch.cat((accumulated_pos_encoding, feature_layer), dim=1)
            accumulated_pos_encoding = feature_layer
            predicted_gray = self.model2[i_layer](accumulated_pos_encoding)
            predicted_grays.append(predicted_gray)
        return predicted_grays

class Phase11(pl.LightningModule):
    def __init__(self, hparams, v_img_path):
        super(Phase11, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.img = cv2.cvtColor(cv2.imread(v_img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
        self.img_size = self.hydra_conf["dataset"]["img_size"]
        self.img = cv2.resize(self.img, self.img_size, interpolation=cv2.INTER_AREA)[:, :, None]
        self.img_name = os.path.basename(v_img_path).split(".")[0]
        self.log_root = os.path.join(self.hydra_conf["trainer"]["output"], self.img_name)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(os.path.join(self.log_root, "imgs"), exist_ok=True)

        f = getattr(sys.modules[__name__], self.hydra_conf["model"]["model_name"])
        self.model = f()

    def train_dataloader(self):
        self.train_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "training",
            self.hydra_conf["dataset"]["sampling_strategy"],
            self.hydra_conf["dataset"]["query_strategy"],
        )
        return DataLoader(self.train_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=True,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "validation",
            self.hydra_conf["dataset"]["sampling_strategy"],
            self.hydra_conf["dataset"]["query_strategy"],
        )
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            # "scheduler": StepLR(optimizer, 100, 0.5),
            # "scheduler": ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5),
            # "frequency": self.hydra_conf["trainer"]["check_val_every_n_epoch"]
            # "monitor": "Validation_Loss",
            # },
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        pixel_pos = batch[0][0]
        gt_gray = batch[1][0]
        batch_size = gt_gray.shape[0]
        predicted_gray = self.model(pixel_pos)
        loss = torch.stack([F.l1_loss(prediction, gt_gray) for prediction in predicted_gray]).mean()

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_pos = batch[0][0]
        gt_gray = batch[1][0]
        batch_size = gt_gray.shape[0]
        predicted_gray = self.model(pixel_pos)
        loss = torch.stack([F.l1_loss(prediction, gt_gray) for prediction in predicted_gray]).mean()

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)
        return predicted_gray

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        n_layer = len(result[0])
        predicted_imgs = [torch.cat([item[i] for item in result]).cpu().numpy() for i in range(n_layer)]
        predicted_imgs = [
            np.clip(item, 0, 1).reshape([self.img.shape[0], self.img.shape[1], 1]) for item in predicted_imgs]
        predicted_imgs = [(item * 255).astype(np.uint8) for item in predicted_imgs]
        for i in range(n_layer):
            img = cv2.cvtColor(predicted_imgs[i], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(self.log_root, "imgs/{}_{}.png".format(self.trainer.current_epoch, i)),
                        img)
            self.trainer.logger.experiment.add_image("Image/{}".format(i),
                                                     img, self.trainer.current_epoch, dataformats="HWC")


@hydra.main(config_name="phase11_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # Read dataset and bounds
    bounds_min = np.array((-50, -50, -10), dtype=np.float32)
    bounds_max = np.array((250, 200, 60), dtype=np.float32)
    scene_bounds = np.array([bounds_min, bounds_max])
    imgs, world_points = read_dataset(v_cfg["dataset"]["colmap_dir"],
                                      scene_bounds)

    for id_img in range(len(imgs)):
        img_name = os.path.basename(imgs[id_img].img_path).split(".")[0]
        if img_name not in v_cfg["dataset"]["target_img"]:
            continue
        model = Phase11(v_cfg, imgs[id_img].img_path)
        trainer = Trainer(
            accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
            devices=v_cfg["trainer"].gpu, enable_model_summary=False,
            max_epochs=v_cfg["trainer"]["max_epoch"],
            # num_sanity_val_steps=2,
            # precision=16,
            reload_dataloaders_every_n_epochs=v_cfg["trainer"]["reload_dataloaders_every_n_epochs"],
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            default_root_dir=log_dir,
        )
        trainer.fit(model)


if __name__ == '__main__':
    main()
