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

from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.dataset import Single_img_dataset, Image, Point_3d, Single_img_dataset_with_kdtree_index, \
    Geometric_dataset, Geometric_dataset_inference


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

        imgs, world_points = read_dataset(self.hydra_conf)
        self.imgs, self.world_points = imgs, world_points

        bounds_min = np.array(self.hydra_conf["dataset"]["scene_boundary"][:3], dtype=np.float32)
        bounds_max = np.array(self.hydra_conf["dataset"]["scene_boundary"][3:], dtype=np.float32)
        bounds_center = (bounds_max + bounds_min) / 2
        bounds_size = bounds_max - bounds_min

        self.model = tcnn.NetworkWithInputEncoding(
            3, 1,
            {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2.0,
            },
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,

            }
        )
        self.model.to("cuda")

    def forward_test(self, v_data):
        # While true:
        ## For each keypoint:
        ### For each visible img:
        ### Project and get 2d coordinate
        ### Compute the point loss
        # If stable:
        ## Duplicate each keypoints

        batch_size = v_data["id_points"].shape[0]
        candidate_points = self.candidate_points[v_data["id_points"]]
        candidate_points = torch.cat([candidate_points, torch.ones_like(candidate_points[:, 0:1])], dim=-1)
        losses = []
        for id_batch in range(batch_size):
            projected_points = torch.matmul(v_data["projection_matrix"][id_batch], candidate_points[id_batch])
            projected_points = projected_points[:, :2] / projected_points[:, 2:3]
            valid_projected_points = projected_points[v_data["valid_views"][id_batch]]
            # If it is accelerated
            if True:
                loss = []
                for i_view in range(valid_projected_points.shape[0]):
                    query_point = valid_projected_points[i_view:i_view + 1]
                    distance, i_matched_point = self.kdtrees[v_data["id_imgs"][id_batch, i_view]].search(query_point, 1)
                    matched_point = torch.from_numpy(self.imgs[v_data["id_imgs"][id_batch, i_view]].line_field[i_matched_point][:2]).to(query_point.device).unsqueeze(0)
                    loss.append(F.mse_loss(query_point, matched_point, reduction='sum'))

                    # Debug
                    if False:
                        with torch.no_grad():
                            print("{}/{}".format(i_view, valid_projected_points.shape[0]))
                            print(query_point.detach().cpu().numpy())
                            img = cv2.imread(self.imgs[v_data["id_imgs"][id_batch, i_view]].img_path)
                            qp = (query_point[0].detach().cpu().numpy() * np.array([6000, 4000])).astype(np.int32)
                            mp = (self.imgs[v_data["id_imgs"][id_batch, i_view]].detected_points[
                                      i_matched_point] * np.array([6000, 4000])).astype(np.int32)
                            img = cv2.circle(img, mp, 10, (0, 255, 255), 10)
                            img = cv2.circle(img, qp, 10, (0, 0, 255), 10)
                            debug_imgs([img])
                    continue
                losses.append(torch.stack(loss).mean())
            else:
                valid_keypoints = v_data["keypoints"][id_batch, v_data["valid_views"][id_batch]]
                valid_keypoints = valid_keypoints[valid_keypoints[:, :, 2].bool(), :2]

            continue

        return torch.stack(losses).mean()

    def forward(self, v_data):
        batch_size = v_data[0].shape[0]
        predicted_sdf = self.model(v_data[0])

        return predicted_sdf

    def train_dataloader(self):
        self.train_dataset = Geometric_dataset(
            self.hydra_conf["dataset"]["mesh_dir"],
            self.hydra_conf["dataset"]["num_sample"],
            "training"
        )
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=True,
                          pin_memory=True,
                          # collate_fn=Geometric_dataset.collate_fn,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Geometric_dataset_inference(
            self.hydra_conf["model"]["marching_cube_resolution"],
            )
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          # collate_fn=Geometric_dataset.collate_fn,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def test_dataloader(self):
        self.test_dataset = Geometric_dataset_inference(
            self.hydra_conf["model"]["marching_cube_resolution"],
        )
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
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
        predicted_sdf = self.forward(batch)

        loss = F.mse_loss(predicted_sdf, batch[1])

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        predicted_sdf = self.forward(batch)

        return predicted_sdf

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        predicted_sdf = -torch.cat(result,dim=0).cpu().numpy().astype(np.float32)
        resolution = self.hydra_conf["model"]["marching_cube_resolution"]
        predicted_sdf = predicted_sdf.reshape([resolution,resolution,resolution])
        vertices, triangles = mcubes.marching_cubes(predicted_sdf, 0)
        mcubes.export_obj(vertices, triangles, os.path.join("outputs", "model_of_epoch_{}.obj".format(self.trainer.current_epoch)))

    def test_step(self, batch, batch_idx):
        predicted_sdf = self.forward(batch)
        return predicted_sdf

    def test_epoch_end(self, result):
        predicted_sdf = -torch.cat(result, dim=0).cpu().numpy().astype(np.float32)
        resolution = self.hydra_conf["model"]["marching_cube_resolution"]
        predicted_sdf = predicted_sdf.reshape([resolution, resolution, resolution])
        vertices, triangles = mcubes.marching_cubes(predicted_sdf, 0)
        mcubes.export_obj(vertices, triangles,
                          os.path.join("outputs", "model_of_test.obj"))

@hydra.main(config_name="test_3d_reconstruction.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=10000,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=100,
        precision=16,
    )

    model = img_pair_alignment(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()

    # While true:
    # For each keypoint:
    # For each visible img:
    # Project and get 2d coordinate
    # Compute the point loss
    # If stable:
    # Duplicate each keypoints

    # For each keypoint:
    # For each visible img:
    # Project and get 2d coordinate
    # For each nearest keypoints
    # Compute the line loss

    exit()
