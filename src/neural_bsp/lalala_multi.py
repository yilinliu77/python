import itertools
import sys, os
import time
from typing import List

import h5py
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

# sys.path.append("thirdparty/sdf_computer/build/")
# import pysdf

import math

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F

import mcubes
import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm, trange
import ray
import platform
import shutil
# import torchsort

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import mcubes


class BSP_dataset_single_object(torch.utils.data.Dataset):
    def __init__(self, v_data, v_training_mode):
        super(BSP_dataset_single_object, self).__init__()
        data0 = []
        for id_item, item in enumerate(v_data['0']):
            data0.append(np.concatenate((item, id_item * np.ones_like(item[:,0:1])), axis=1))
        # for id_item, item in enumerate(v_data['1']):
        #     v_data['1'][id_item] = np.concatenate((item, id_item * np.ones_like(item[:,0:1])), axis=1)

        data = torch.from_numpy(np.concatenate(data0,axis=0))
        self.points = data

        self.mode = v_training_mode
        self.batch_size = 4096
        pass

    def __len__(self):
        return 100 if self.mode == "training" else math.ceil(self.points.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        if self.mode == "training":
            which_point = torch.randint(0, self.points.shape[0], (self.batch_size,))
            return self.points[which_point]
        else:
            id_start = idx * self.batch_size
            id_end = (idx + 1) * self.batch_size
            return self.points[id_start:id_end]


class Base_model(nn.Module):
    def __init__(self, v_num_plane=8):
        super(Base_model, self).__init__()
        self.z_vector = nn.Parameter(torch.rand(1, 128))

        self.plane_features_extractor = nn.Sequential(
            nn.Linear(128, v_num_plane * 32),
            nn.ReLU(),
            nn.Linear(v_num_plane * 32, v_num_plane * 32),
            nn.ReLU(),
        )
        self.to_parameters = nn.Linear(32, 2)

        self.v_num_plane = v_num_plane

    def forward(self, v_data, v_training=False, v_is_mask=True):
        batch_size = v_data.shape[0]
        num_queries = v_data.shape[1]

        plane_features = self.plane_features_extractor(self.z_vector).reshape(1, self.v_num_plane, 32)
        # Parameters
        plane_m_polar = self.to_parameters(plane_features)
        rho = plane_m_polar[:, :, 0] * 2 * torch.pi
        # convert rho, theta to a,b,c
        a = torch.cos(rho)
        b = torch.sin(rho)
        c = -plane_m_polar[:, :, 1]
        plane_m = torch.stack((a, b, c), dim=1)

        points = torch.cat((v_data[:,:,:2],torch.ones_like(v_data[:,:,0:1])),dim=-1)
        id = v_data[:,:,3].to(torch.long)

        # Calculated nearest distance to all planes
        d_n = torch.abs(torch.matmul(points, plane_m))

        distance = torch.gather(d_n, dim=2, index=id.unsqueeze(2))[:,:,0]

        return distance, plane_m

    def loss(self, v_predictions, v_input):
        sdfs = v_input[:,:,2]
        pred_udf, plane_m, = v_predictions
        udf_loss = F.l1_loss(pred_udf, sdfs)
        return udf_loss


class Base_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Base_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.data = v_data
        v_num_plane = len(v_data["0"])
        self.phase = self.hydra_conf["model"]["phase"]
        self.model = Base_model(v_num_plane)

        # Used for visualizing during the training
        self.viz_data = {}

    def train_dataloader(self):
        self.train_dataset = BSP_dataset_single_object(
            self.data,
            "training",
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        self.valid_dataset = BSP_dataset_single_object(
            self.data,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=1, num_workers=0)

    def configure_optimizers(self):
        # grouped_parameters = [
        #     {"params": [self.model.seg_distance], 'lr': self.learning_rate},
        #     {"params": [self.model.v_up], 'lr': 1e-2},
        # ]

        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        # optimizer = SGD(grouped_parameters, lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, True, v_is_mask=batch_idx%2==0)
        total_loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", total_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, False, v_is_mask=True)
        total_loss = self.model.loss(outputs, batch)

        self.log("Validation_Loss", total_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])

        return batch, outputs

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        if self.trainer.sanity_checking:
            return

        query_points = []
        gt_dis = []
        plane_m = None
        d1 = []
        for item in result:
            query_points.append(item[0][0,:,:2].cpu().numpy())
            gt_dis.append(item[0][0,:,2].cpu().numpy())
            plane_m = item[1][1][0].cpu().numpy()
            d1.append(item[1][0][0].cpu().numpy())

        query_points = np.concatenate(query_points, axis=0)
        gt_dis = np.concatenate(gt_dis, axis=0)
        d1 = np.concatenate(d1, axis=0)  # udf_distance

        num_plane = plane_m.shape[1]
        x1 = np.ones(num_plane) * -1
        y1 = (-plane_m[2] - x1 * plane_m[0]) / plane_m[1]
        x2 = np.ones(plane_m.shape[1]) * 1
        y2 = (-plane_m[2] - x2 * plane_m[0]) / plane_m[1]
        coords = np.stack((x1, y1, x2, y2), axis=1).reshape((num_plane, 2, 2))

        # Draw gt distance
        matplotlib.use('agg')
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        for i in range(plane_m.shape[1]):
            plt.plot(coords[i, :, 0], coords[i, :, 1], '-')
        plt.scatter(query_points[:, 0], query_points[:, 1], c=gt_dis, vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)

        plt.subplot(2, 2, 2)
        for i in range(plane_m.shape[1]):
            plt.plot(coords[i, :, 0], coords[i, :, 1], '-')
        plt.scatter(query_points[:, 0], query_points[:, 1], c=np.abs(d1), vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_root, "{}.jpg".format(self.current_epoch)))
        plt.close()
        return


def prepare_dataset():
    print("Start to construct dataset")

    query_points = np.load("output/1.npy", allow_pickle=True).item()

    print("Done")
    print("{} vertices and {} planes;".format(
        len(query_points["0"]),
        len(query_points["1"]),
    ))

    return query_points


@hydra.main(config_name="lalala.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))
    data = prepare_dataset()

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Base_phase(v_cfg, data)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        default_root_dir=log_dir,
        # precision=16,
        # gradient_clip_val=0.5
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
