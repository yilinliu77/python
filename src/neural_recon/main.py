import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import List

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
import faiss
import faiss.contrib.torch_utils

from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.dataset import Single_img_dataset, Image, Point_3d, Single_img_dataset_with_kdtree_index

def prepare_kdtree_data(v_img):
    line_field = cv2.imread(v_img.line_field_path, cv2.IMREAD_UNCHANGED)
    points_coordinates_yx = np.stack(np.where(np.all(line_field!=0, axis=-1)),axis=1)
    points_coordinates_xy_normalized = points_coordinates_yx[:,[1,0]].astype(np.float32) / np.array([[line_field.shape[1],line_field.shape[0]]],dtype=np.float32)
    v_img.line_field = np.concatenate([points_coordinates_xy_normalized[:,0:1],points_coordinates_xy_normalized[:,1:2],
                                          line_field[points_coordinates_yx[:,0], points_coordinates_yx[:,1], :2]],axis=1)
    return v_img


def build_cuda_kdtree(v_gpu, v_img):
    index_flat = faiss.GpuIndexFlat(v_gpu, 2, faiss.METRIC_L2)
    detected_points_cuda = torch.from_numpy(v_img.line_field[:,:2]).contiguous().cuda()
    index_flat.add(detected_points_cuda)
    return index_flat


def save_pointcloud(v_item, v_id, v_bound_center, v_bound_size):
    pc = open3d.geometry.PointCloud()
    points = v_item
    pc.points = open3d.utility.Vector3dVector(points * v_bound_size / 2 + v_bound_center)
    open3d.io.write_point_cloud("output/neural_recon/{}.ply".format(v_id), pc)
    return


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
        self.candidate_points = torch.nn.Parameter(
            torch.tensor(np.array(list(map(lambda item: item.pos, self.world_points))), dtype=torch.float32))

        print("Prepare kdtree")
        pool = Pool(12)
        gpu_resource = faiss.StandardGpuResources()
        self.imgs = list(pool.map(prepare_kdtree_data, imgs))
        pool.close()
        print("Build kdtree")
        self.kdtrees = list(map(partial(build_cuda_kdtree, gpu_resource), self.imgs))

        bounds_min = np.array(self.hydra_conf["dataset"]["scene_boundary"][:3], dtype=np.float32)
        bounds_max = np.array(self.hydra_conf["dataset"]["scene_boundary"][3:], dtype=np.float32)
        bounds_center = (bounds_max + bounds_min) / 2
        bounds_size = bounds_max - bounds_min
        save_pointcloud(self.candidate_points.data.detach().cpu().numpy(), -1, bounds_center,
                        bounds_size)

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
        batch_size = v_data["id_points"].shape[0]
        id_batch=0
        candidate_points = self.candidate_points[v_data["id_points"]][id_batch:id_batch+1]
        candidate_points = torch.cat([candidate_points, torch.ones_like(candidate_points[:, 0:1])], dim=-1)
        losses = []
        projected_points = torch.matmul(v_data["projection_matrix"][id_batch], candidate_points[id_batch])
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]
        valid_projected_points = projected_points[v_data["valid_views"][id_batch]]
        loss = []
        for i_view in range(valid_projected_points.shape[0]):
            query_point = valid_projected_points[i_view:i_view + 1]
            distance, i_matched_point = self.kdtrees[v_data["id_imgs"][id_batch, i_view]].search(query_point, 1)
            matched_point = torch.from_numpy(self.imgs[v_data["id_imgs"][id_batch, i_view]].line_field[i_matched_point][:2]).to(query_point.device).unsqueeze(0)
            loss.append(F.mse_loss(query_point, matched_point, reduction='sum'))

            # Debug
            if True:
                with torch.no_grad():
                    print("{}/{}".format(i_view, valid_projected_points.shape[0]))
                    print(query_point.detach().cpu().numpy())
                    img = cv2.imread(self.imgs[v_data["id_imgs"][id_batch, i_view]].img_path)
                    qp = (query_point[0].detach().cpu().numpy() * np.array([6000, 4000])).astype(np.int32)
                    mp = (self.imgs[v_data["id_imgs"][id_batch, i_view]].line_field[i_matched_point][:2] * np.array([6000, 4000])).astype(np.int32)
                    img = cv2.circle(img, mp, 10, (0, 255, 0), 5)
                    img = cv2.circle(img, qp, 10, (0, 0, 255), 5)
                    debug_imgs([img])
            continue
        losses.append(torch.stack(loss).mean())

        return torch.stack(losses).mean()

    def train_dataloader(self):
        self.train_dataset = Single_img_dataset_with_kdtree_index(
            self.imgs, self.world_points,
            "training"
        )
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=Single_img_dataset_with_kdtree_index.collate_fn,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Single_img_dataset_with_kdtree_index(
            self.imgs, self.world_points,
            "validation")
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=Single_img_dataset_with_kdtree_index.collate_fn,
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
        loss = self.forward(batch)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch["id_points"].shape[0])

        # if batch_idx % 100 == 0 and batch_idx != 0:

        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        bounds_min = np.array(self.hydra_conf["dataset"]["scene_boundary"][:3], dtype=np.float32)
        bounds_max = np.array(self.hydra_conf["dataset"]["scene_boundary"][3:], dtype=np.float32)
        bounds_center = (bounds_max + bounds_min) / 2
        bounds_size = bounds_max - bounds_min
        save_pointcloud(self.candidate_points.data.detach().cpu().numpy(), self.trainer.current_epoch, bounds_center,
                        bounds_size)

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)

        self.log("Validation_Loss", loss, prog_bar=True, logger=True,
                 batch_size=batch["id_points"].shape[0])
        return

    def validation_epoch_end(self, outputs) -> None:
        if self.trainer.sanity_checking:
            return

        return


@hydra.main(config_name="test.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=10000,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=99999
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
