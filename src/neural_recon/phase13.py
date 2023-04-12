from collections import OrderedDict

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

from shared.img_torch_tools import print_model_size
from src.neural_recon.Image_dataset import Image_dataset, Images_dataset
from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.phase1 import NGPModel


class NGPModel1(nn.Module):
    def __init__(self):
        super(NGPModel1, self).__init__()

        self.model1 = tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
        }, dtype=torch.float32)
        self.model2 = tcnn.Network(n_input_dims=self.model1.n_output_dims, n_output_dims=1, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        })

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



class Phase12(pl.LightningModule):
    def __init__(self, hparams, v_imgs):
        super(Phase12, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]

        self.imgs = {}
        for img_name in v_imgs:
            img = cv2.cvtColor(cv2.imread(v_imgs[img_name], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
            os.makedirs(os.path.join(self.log_root, img_name), exist_ok=True)
            cv2.imwrite(os.path.join(self.log_root, img_name, "gt.png"), img)
            self.imgs[img_name] = img

        f = getattr(sys.modules[__name__], self.hydra_conf["model"]["model_name"])
        self.model = f([item for item in self.imgs])
        print_model_size(self.model)
        # torch.set_float32_matmul_precision('medium')

    def train_dataloader(self):
        self.train_dataset = Images_dataset(
            self.imgs,
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
        self.valid_dataset = Images_dataset(
            self.imgs,
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
        img_name = batch[0][0]
        pixel_pos = batch[1][0]
        gt_gray = batch[2][0]
        batch_size = gt_gray.shape[0]

        predicted_gray = self.model(img_name, pixel_pos)

        loss = F.l1_loss(predicted_gray[:,0], gt_gray)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        img_name = batch[0][0]
        pixel_pos = batch[1][0]
        gt_gray = batch[2][0]
        batch_size = gt_gray.shape[0]

        predicted_gray = self.model(img_name, pixel_pos)

        loss = F.l1_loss(predicted_gray[:,0], gt_gray)

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)
        return img_name, predicted_gray

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        for img_name in self.imgs:
            predicted_imgs = torch.cat([item[1] for item in result if item[0] == img_name], dim=0)
            predicted_imgs = np.clip(predicted_imgs.cpu().numpy(), 0, 1).reshape(
                [self.imgs[img_name].shape[0], self.imgs[img_name].shape[1], 1])
            predicted_imgs = (predicted_imgs * 255).astype(np.uint8)
            img = cv2.cvtColor(predicted_imgs, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(self.log_root, img_name,"{}.png".format(self.trainer.current_epoch)),
                        img)
            self.trainer.logger.experiment.add_image("Image/{}".format(img_name),
                                                     img, self.trainer.current_epoch, dataformats="HWC")


@hydra.main(config_name="phase11_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # Read dataset and bounds
    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name):
        print("Found cache ", img_cache_name)
        imgs, points_3d = np.load(img_cache_name, allow_pickle=True)
    else:
        print("Dosen't find cache, read raw img data")
        bound_min = np.array((-40, -40, -5))
        bound_max = np.array((130, 150, 60))
        bounds_center = (bound_min + bound_max) / 2
        bounds_size = (bound_max - bound_min).max()
        imgs, points_3d = read_dataset(v_cfg["dataset"]["colmap_dir"],
                                       [bound_min,
                                        bound_max]
                                       )
        np.save(img_cache_name[:-4], np.asarray([imgs, points_3d], dtype=object))
        print("Save cache to ", img_cache_name)

    imgs = {img.img_name:img.img_path  for img in imgs[1:3]}

    model = Phase12(v_cfg, imgs)
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
