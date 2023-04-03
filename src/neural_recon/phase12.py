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


class NGPModelInv1(nn.Module):
    def __init__(self):
        super(NGPModelInv1, self).__init__()

        # Define models
        self.model1 = tcnn.Encoding(n_input_dims=32, encoding_config={
            "otype": "Frequency",
            "n_frequencies": 12
        }, dtype=torch.float32)
        self.model2 = tcnn.Network(n_input_dims=self.model1.n_output_dims, n_output_dims=2, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        })

    def forward(self, v_features):
        features = self.model1(v_features)  # (batch_size, feature_dim)
        predicted_pos = self.model2(features)  # (batch_size, 2)
        return predicted_pos


class Phase12(pl.LightningModule):
    def __init__(self, hparams, v_img_path):
        super(Phase12, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.img = cv2.cvtColor(cv2.imread(v_img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.img_size = self.hydra_conf["dataset"]["img_size"]
        self.img = cv2.resize(self.img, self.img_size, interpolation=cv2.INTER_AREA)
        self.img_name = os.path.basename(v_img_path).split(".")[0]
        self.log_root = os.path.join(self.hydra_conf["trainer"]["output"], self.img_name)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(os.path.join(self.log_root, "imgs_inv"), exist_ok=True)

        self.model_forward = NGPModel()
        checkpoint_name = [item for item in os.listdir(
            os.path.join(self.hydra_conf["model"]["input_model"], self.img_name)) if item[-4:] == "ckpt"]
        assert len(checkpoint_name) == 1
        # state_dict = torch.load(os.path.join(img_model_root_dir, img_name, checkpoint_name[0]))
        state_dict = torch.load(
            os.path.join(self.hydra_conf["model"]["input_model"], self.img_name, checkpoint_name[0]))["state_dict"]
        self.model_forward.load_state_dict({item[6:]: state_dict[item] for item in state_dict}, strict=True)
        self.model_forward.eval()
        for p in self.model_forward.parameters():
            p.requires_grad=False
        f = getattr(sys.modules[__name__], self.hydra_conf["model"]["model_name"])
        self.model_inverse = f()

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
        gt_rgb = batch[1][0]
        batch_size = gt_rgb.shape[0]
        predicted_features = self.model_forward.model1(pixel_pos)
        predicted_pos = self.model_inverse(predicted_features)
        loss = F.mse_loss(pixel_pos, predicted_pos)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_pos = batch[0][0]
        gt_rgb = batch[1][0]
        batch_size = gt_rgb.shape[0]
        predicted_features = self.model_forward.model1(pixel_pos)
        predicted_pos = self.model_inverse(predicted_features)
        loss = F.mse_loss(pixel_pos, predicted_pos)

        rgb = self.model_forward(predicted_pos)

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)
        return rgb

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        predicted_pixel = torch.cat(result, dim=0).cpu().numpy()
        predicted_pixel = np.clip(predicted_pixel, 0, 1)
        predicted_pixel = predicted_pixel.reshape([self.img.shape[0], self.img.shape[1], 3])
        predicted_pixel = (predicted_pixel * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.log_root, "imgs_inv/{}.png").format(self.trainer.current_epoch),
                    cv2.cvtColor(predicted_pixel, cv2.COLOR_RGB2BGR))
        self.trainer.logger.experiment.add_image("Image".format(self.trainer.current_epoch),
                                                  predicted_pixel, self.trainer.current_epoch, dataformats="HWC")


@hydra.main(config_name="phase12_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
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
        model = Phase12(v_cfg, imgs[id_img].img_path)
        trainer = Trainer(
            accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
            devices=v_cfg["trainer"].gpu, enable_model_summary=False,
            max_epochs=v_cfg["trainer"]["max_epoch"],
            num_sanity_val_steps=2,
            precision=16,
            reload_dataloaders_every_n_epochs=v_cfg["trainer"]["reload_dataloaders_every_n_epochs"],
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            default_root_dir=log_dir,
        )
        trainer.fit(model)


if __name__ == '__main__':
    main()
