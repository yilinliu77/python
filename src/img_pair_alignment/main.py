import math
import platform
import shutil
import sys, os

from typing import Tuple

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
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

from shared.img_torch_tools import get_img_from_tensor
from src.img_pair_alignment.dataset import Single_img_dataset
from src.img_pair_alignment.model import NeuralImageFunction

import cv2

from src.img_pair_alignment.original_warp import warp_corners


class img_pair_alignment(pl.LightningModule):
    def __init__(self, hparams):
        super(img_pair_alignment, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        # self.log(...,batch_size=self.batch_size)
        self.save_hyperparameters(hparams)

        if not os.path.exists(self.hydra_conf["trainer"]["output"]):
            os.makedirs(self.hydra_conf["trainer"]["output"])

        self.model = NeuralImageFunction(self.hydra_conf["dataset"]["img_height"],
                                         self.hydra_conf["dataset"]["img_width"],
                                         self.hydra_conf["dataset"]["img_crop_size"],
                                         self.hydra_conf["dataset"]["num_sample"],
                                         self.hydra_conf["model"]["pos_encoding"],
                                         )

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def train_dataloader(self):
        self.train_dataset = Single_img_dataset(
            self.hydra_conf["dataset"]["img_path"],
            self.hydra_conf["dataset"]["img_width"],
            self.hydra_conf["dataset"]["img_height"],
            self.hydra_conf["dataset"]["img_crop_size"],
            "training"
        )
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Single_img_dataset(
            self.hydra_conf["dataset"]["img_path"],
            self.hydra_conf["dataset"]["img_width"],
            self.hydra_conf["dataset"]["img_height"],
            self.hydra_conf["dataset"]["img_crop_size"],
            "validation")
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation Loss'
        }

    def training_step(self, batch, batch_idx):
        gt_img, raw_img = batch
        predict_img, homography_params = self.forward(gt_img)

        loss = self.model.loss(predict_img, gt_img)

        self.log("Training Loss", loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)

        if batch_idx % 100 == 0 and batch_idx !=0:
            self.model.visualize(predict_img, gt_img, raw_img, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        gt_img, raw_img = batch
        predict_img, homography_params = self.forward(gt_img)

        loss = self.model.loss(predict_img, gt_img)

        self.log("Validation Loss", loss, prog_bar=True, logger=True)
        self.model.visualize(predict_img,gt_img,raw_img,-1)
        return

    def validation_epoch_end(self, outputs) -> None:
        # mean_spearman, log_str = self._calculate_spearman(outputs)
        if self.trainer.sanity_checking:
            return


        return


@hydra.main(config_name="test.yaml", config_path="../../configs/img_pair_alignment/")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, enable_model_summary=False,
                      max_epochs=1,
                      num_sanity_val_steps = 1,
                      check_val_every_n_epoch=99999
                      )

    model = img_pair_alignment(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        # for item in list(state_dict.keys()):
        #     if "point_feature_extractor" in item:
        #         state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    if platform.system() == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    main()
