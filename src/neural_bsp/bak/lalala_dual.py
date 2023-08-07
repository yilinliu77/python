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
        self.points = torch.from_numpy(v_data["points"])
        self.points = torch.cat((self.points, torch.ones_like(self.points[:, 0:1])), dim=1)
        self.udfs = torch.from_numpy(v_data["nudfs"])

        self.mode = v_training_mode
        self.batch_size = 4096
        pass

    def __len__(self):
        return 100 if self.mode == "training" else math.ceil(self.points.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        if self.mode == "training":
            which_point = torch.randint(0, self.points.shape[0], (self.batch_size,))
            return self.points[which_point], self.udfs[which_point]
        else:
            id_start = idx * self.batch_size
            id_end = (idx + 1) * self.batch_size
            return self.points[id_start:id_end], self.udfs[id_start:id_end]


class Base_model1(nn.Module):
    def __init__(self, v_num_plane=7):
        super(Base_model1, self).__init__()
        self.z_vector = nn.Parameter(torch.rand(1, 128))

        self.to_parameters = nn.Sequential(
            nn.Linear(128, v_num_plane * 32),
            nn.ReLU(),
            nn.Linear(v_num_plane * 32, v_num_plane * 2),
        )
        self.to_interest_region = nn.ModuleList()
        for i in range(v_num_plane):
            self.to_interest_region.append(nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ))

        self.to_plane_belongings = nn.Sequential(
            nn.Linear(2, v_num_plane * 32),
            nn.ReLU(),
            nn.Linear(v_num_plane * 32, v_num_plane * 1),
            nn.Softmax(dim=2),
        )

        self.v_num_plane = v_num_plane

    def project_points(self, v_points, v_plane_m):
        a_expand = v_plane_m[:, 0, :, None]
        b_expand = v_plane_m[:, 1, :, None]
        c_expand = v_plane_m[:, 2, :, None]

        x_proj = (b_expand * (b_expand * v_points[:, None, :, 0]
                              - a_expand * v_points[:, None, :, 1]) - (a_expand * c_expand))
        y_proj = (a_expand * (-b_expand * v_points[:, None, :, 0]
                              + a_expand * v_points[:, None, :, 1]) - b_expand * c_expand)

        proj = torch.stack((x_proj, y_proj), dim=-1)
        return proj

    def forward1(self, v_data, v_training=False, v_is_mask=True):
        points, sdfs = v_data
        batch_size = points.shape[0]
        num_queries = points.shape[1]

        # plane_m_polar = self.to_parameters(self.sa(self.plane_m,self.plane_m,self.plane_m)[0])
        plane_m_polar = self.to_parameters(self.z_vector).reshape(1, self.v_num_plane, 2)
        rho = plane_m_polar[:, :, 0] * 2 * torch.pi
        # convert rho, theta to a,b,c
        a = torch.cos(rho)
        b = torch.sin(rho)
        c = -plane_m_polar[:, :, 1]
        plane_m = torch.stack((a, b, c), dim=1)

        # batch_size, num_queries, num_plane
        distance_from_point_to_all_plane = torch.abs(torch.matmul(points, plane_m))

        # batch_size, num_queries, num_plane
        projection = self.project_points(points, plane_m)
        distance_to_roi = []
        for i in range(self.v_num_plane):
            distance_to_roi.append(self.to_interest_region[i](projection[:,i]))
        distance_to_roi = torch.cat(distance_to_roi,dim=2)

        true_distance = torch.sqrt(distance_to_roi**2+distance_from_point_to_all_plane**2)

        # batch_size, num_queries
        if v_is_mask:
            plane_belongs = self.to_plane_belongings(points[:, :, :2])
            argmax = plane_belongs.argmax(dim=2)
            distance_from_point_to_one_plane = torch.gather(true_distance, 2, argmax[:,:,None])[:,:,0]
        else:
            plane_belongs = self.to_plane_belongings(points[:, :, :2])

            distance_from_point_to_one_plane = torch.sum(true_distance * plane_belongs, dim=2)

        return distance_from_point_to_one_plane, plane_belongs, plane_m, projection, distance_to_roi

    def forward(self, v_data, v_training=False, v_is_mask=True):
        points, sdfs = v_data
        batch_size = points.shape[0]
        num_queries = points.shape[1]

        # plane_m_polar = self.to_parameters(self.sa(self.plane_m,self.plane_m,self.plane_m)[0])
        plane_m_polar = self.to_parameters(self.z_vector).reshape(1, self.v_num_plane, 2)
        rho = plane_m_polar[:, :, 0] * 2 * torch.pi
        # convert rho, theta to a,b,c
        a = torch.cos(rho)
        b = torch.sin(rho)
        c = -plane_m_polar[:, :, 1]
        plane_m = torch.stack((a, b, c), dim=1)

        # batch_size, num_queries, num_plane
        distance_from_point_to_all_plane = torch.abs(torch.matmul(points, plane_m))

        # batch_size, num_queries, num_plane
        projection = self.project_points(points, plane_m)
        distance_to_roi = []
        for i in range(self.v_num_plane):
            distance_to_roi.append(self.to_interest_region[i](projection[:,i]))
        distance_to_roi = torch.cat(distance_to_roi,dim=2)

        true_distance = torch.sqrt(distance_to_roi**2+distance_from_point_to_all_plane**2)

        # batch_size, num_queries
        if v_is_mask:
            plane_belongs = self.to_plane_belongings(points[:, :, :2])
            argmax = true_distance.argmax(dim=2)
            distance_from_point_to_one_plane = torch.gather(true_distance, 2, argmax[:,:,None])[:,:,0]
        else:
            plane_belongs = self.to_plane_belongings(points[:, :, :2])

            distance_from_point_to_one_plane = torch.sum(true_distance * plane_belongs, dim=2)

        return distance_from_point_to_one_plane, plane_belongs, plane_m, projection, distance_to_roi

    def duplicated_plane_loss(self, v_plane_m):
        plane_pairs = v_plane_m[:,:,torch.combinations(torch.arange(v_plane_m.shape[2],device=v_plane_m.device))]
        plane_pairs = plane_pairs.permute(0,2,3,1)
        angle = (plane_pairs[:,:,0,:2] * plane_pairs[:,:,1,:2]).sum(dim=-1)

        ratio = plane_pairs[:,:,0]/(plane_pairs[:,:,1]+1e-9)
        loss = (torch.abs(
            ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,1]-ratio[:,:,2]))/3
        loss = 0.1 - loss.clamp(0,0.1)
        return loss.mean()

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf, distance_weights, plane_m, _, _ = v_predictions
        # udf_loss = torch.mean((sdfs - pred_udf) ** 2)
        udf_loss = F.l1_loss(pred_udf, sdfs)

        # 21
        # sparsity_loss  = (distance_weights.pow(2).sum(dim=2)-1).pow(2).mean() * 0.0001
        # regularization_loss = torch.linalg.norm(distance_weights, ord=1, dim=2)

        # 22
        sparsity_loss = (-torch.max(distance_weights, dim=2)[0]
                         + torch.sum(torch.pow(distance_weights, 2),dim=2)).mean() * 0.01

        # 3
        plane_distance = self.duplicated_plane_loss(plane_m) * 0.01

        # 4
        plane_belongs_loss = (1-distance_weights.max(dim=1)[0].mean(dim=0)).mean(dim=0) * 0.01
        return udf_loss + sparsity_loss + plane_distance + plane_belongs_loss, udf_loss, sparsity_loss

class Base_model2(nn.Module):
    def __init__(self, v_num_plane=7):
        super(Base_model2, self).__init__()
        self.z_vector = nn.Parameter(torch.rand(1, 256))

        self.plane_features_extractor = nn.Sequential(
            nn.Linear(256, v_num_plane * 32),
            nn.ReLU(),
            nn.Linear(v_num_plane * 32, v_num_plane * 32),
            nn.ReLU(),
        )

        self.to_belongings_soft = nn.Linear(32, 96)
        self.to_parameters = nn.Linear(32, 2)
        self.to_interest_region = nn.Linear(32, 96)

        self.v_num_plane = v_num_plane

    def project_points(self, v_points, v_plane_m):
        a_expand = v_plane_m[:, 0, :, None]
        b_expand = v_plane_m[:, 1, :, None]
        c_expand = v_plane_m[:, 2, :, None]

        x_proj = (b_expand * (b_expand * v_points[:, None, :, 0]
                              - a_expand * v_points[:, None, :, 1]) - (a_expand * c_expand))
        y_proj = (a_expand * (-b_expand * v_points[:, None, :, 0]
                              + a_expand * v_points[:, None, :, 1]) - b_expand * c_expand)

        proj = torch.stack((x_proj, y_proj), dim=-1)
        return proj

    def forward(self, v_data, v_training=False, v_is_mask=True):
        points, sdfs = v_data
        batch_size = points.shape[0]
        num_queries = points.shape[1]

        plane_features = self.plane_features_extractor(self.z_vector).reshape(1, self.v_num_plane, 32)
        # Parameters
        plane_m_polar = self.to_parameters(plane_features)
        rho = plane_m_polar[:, :, 0] * 2 * torch.pi
        # convert rho, theta to a,b,c
        a = torch.cos(rho)
        b = torch.sin(rho)
        c = -plane_m_polar[:, :, 1]
        plane_m = torch.stack((a, b, c), dim=1)

        # Calculated nearest distance to all planes
        d_n = torch.abs(torch.matmul(points, plane_m))

        # Calculated distance to interest regions
        roi = self.to_interest_region(plane_features).reshape(1, self.v_num_plane, 3, 32)
        projection = self.project_points(points[:,:,:2], plane_m)
        d_roi = (projection @ roi[:, :, 0:2] @ roi[:, :, 2:3].permute(0,1,3,2))
        d_roi = d_roi.permute(0,2,1,3)[:,:,:,0]

        # Calculated final distance to all planes
        true_distance = torch.sqrt(d_n**2+d_roi**2)

        # Predict distance
        predict_belongs_field = self.to_belongings_soft(plane_features).reshape(1, self.v_num_plane, 3, 32)
        predict_belongs = points[:,None,:,:2] @ predict_belongs_field[:, :, 0:2] \
                 @ predict_belongs_field[:, :, 2:3].permute(0,1,3,2)
        predict_belongs = torch.softmax(predict_belongs.permute(0,2,1,3)[:,:,:,0], dim=-1)

        soft_distance = (predict_belongs * true_distance).sum(dim=-1)
        hard_distance = true_distance.min(dim=-1)[0]

        return soft_distance, predict_belongs, plane_m, projection, d_roi

    def duplicated_plane_loss(self, v_plane_m):
        plane_pairs = v_plane_m[:,:,torch.combinations(torch.arange(v_plane_m.shape[2],device=v_plane_m.device))]
        plane_pairs = plane_pairs.permute(0,2,3,1)
        angle = (plane_pairs[:,:,0,:2] * plane_pairs[:,:,1,:2]).sum(dim=-1)

        ratio = plane_pairs[:,:,0]/(plane_pairs[:,:,1]+1e-9)
        loss = (torch.abs(
            ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,1]-ratio[:,:,2]))/3
        loss = 0.1 - loss.clamp(0,0.1)
        return loss.mean()

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf, distance_weights, plane_m, _, _ = v_predictions
        # udf_loss = torch.mean((sdfs - pred_udf) ** 2)
        udf_loss = F.l1_loss(pred_udf, sdfs)

        # 21
        # sparsity_loss  = (distance_weights.pow(2).sum(dim=2)-1).pow(2).mean() * 0.0001
        # regularization_loss = torch.linalg.norm(distance_weights, ord=1, dim=2)

        # 22
        sparsity_loss = (-torch.max(distance_weights, dim=2)[0]
                         + torch.sum(torch.pow(distance_weights, 2),dim=2)).mean() * 0.01

        # 3
        plane_distance = self.duplicated_plane_loss(plane_m) * 0.01

        # 4
        plane_belongs_loss = (1-distance_weights.max(dim=1)[0].mean(dim=0)).mean(dim=0) * 0.01
        return udf_loss, udf_loss, sparsity_loss

class Base_model(nn.Module):
    def __init__(self, v_num_plane=7):
        super(Base_model, self).__init__()
        self.z_vector = nn.Parameter(torch.rand(1, 128))

        self.plane_features_extractor = nn.Sequential(
            nn.Linear(128, v_num_plane * 32),
            nn.ReLU(),
            nn.Linear(v_num_plane * 32, v_num_plane * 32),
            nn.ReLU(),
        )
        self.to_parameters = nn.Linear(32, 2)
        self.to_distance_soft = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.to_belongings_soft = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )
        self.to_interest_region = nn.Linear(32, 96)

        self.soft_belongs = nn.Parameter(torch.rand(7,1))

        self.v_num_plane = v_num_plane

    def project_points(self, v_points, v_plane_m):
        a_expand = v_plane_m[:, 0, :, None]
        b_expand = v_plane_m[:, 1, :, None]
        c_expand = v_plane_m[:, 2, :, None]

        x_proj = (b_expand * (b_expand * v_points[:, None, :, 0]
                              - a_expand * v_points[:, None, :, 1]) - (a_expand * c_expand))
        y_proj = (a_expand * (-b_expand * v_points[:, None, :, 0]
                              + a_expand * v_points[:, None, :, 1]) - b_expand * c_expand)

        proj = torch.stack((x_proj, y_proj), dim=-1)
        return proj

    def forward(self, v_data, v_training=False, v_is_mask=True):
        points, sdfs = v_data
        batch_size = points.shape[0]
        num_queries = points.shape[1]

        plane_features = self.plane_features_extractor(self.z_vector).reshape(1, self.v_num_plane, 32)
        # Parameters
        plane_m_polar = self.to_parameters(plane_features)
        rho = plane_m_polar[:, :, 0] * 2 * torch.pi
        # convert rho, theta to a,b,c
        a = torch.cos(rho)
        b = torch.sin(rho)
        c = -plane_m_polar[:, :, 1]
        plane_m = torch.stack((a, b, c), dim=1)

        # Calculated nearest distance to all planes
        d_n = torch.abs(torch.matmul(points, plane_m))

        packed_data = torch.cat((
            points[:, :, None, :2].tile(1, 1, self.v_num_plane, 1),
            plane_m_polar[:, None, :, :].tile(batch_size, num_queries, 1, 1)
        ),dim=-1)
        packed_data = packed_data.reshape(-1,4)
        all_soft_distance = self.to_distance_soft(packed_data)
        all_soft_distance = all_soft_distance.reshape(batch_size,num_queries,self.v_num_plane)

        soft_belongs = torch.softmax(self.to_belongings_soft(points[:, :, :2]),dim=-1)

        # soft_distance = all_soft_distance.min(dim=2)[0]
        # soft_distance = (all_soft_distance @ self.soft_belongs)[:,:,0]
        soft_distance = (all_soft_distance * soft_belongs).sum(-1)

        return soft_distance, soft_belongs, plane_m, None, None

    def duplicated_plane_loss(self, v_plane_m):
        plane_pairs = v_plane_m[:,:,torch.combinations(torch.arange(v_plane_m.shape[2],device=v_plane_m.device))]
        plane_pairs = plane_pairs.permute(0,2,3,1)
        angle = (plane_pairs[:,:,0,:2] * plane_pairs[:,:,1,:2]).sum(dim=-1)

        ratio = plane_pairs[:,:,0]/(plane_pairs[:,:,1]+1e-9)
        loss = (torch.abs(
            ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,0]-ratio[:,:,1])+torch.abs(ratio[:,:,1]-ratio[:,:,2]))/3
        loss = 0.1 - loss.clamp(0,0.1)
        return loss.mean()

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf, distance_weights, plane_m, _, _ = v_predictions
        # udf_loss = torch.mean((sdfs - pred_udf) ** 2)
        udf_loss = F.l1_loss(pred_udf, sdfs)

        # 21
        # sparsity_loss  = (distance_weights.pow(2).sum(dim=2)-1).pow(2).mean() * 0.0001
        # regularization_loss = torch.linalg.norm(distance_weights, ord=1, dim=2)

        # 22
        sparsity_loss = (-torch.max(distance_weights, dim=2)[0]
                         + torch.sum(torch.pow(distance_weights, 2),dim=2)).mean() * 0.1

        # 3
        # plane_distance = self.duplicated_plane_loss(plane_m) * 0.01

        # 4
        plane_belongs_loss = (1-distance_weights.max(dim=1)[0].mean(dim=0)).mean(dim=0) * 0.1
        return udf_loss + sparsity_loss + plane_belongs_loss, udf_loss, plane_belongs_loss + sparsity_loss


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
        self.phase = self.hydra_conf["model"]["phase"]
        self.model = Base_model()

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
        total_loss, udf_loss, regularization_loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", total_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])
        self.log("Training_udf_loss", udf_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])
        self.log("Training_regularization_loss", regularization_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, False, v_is_mask=True)
        total_loss, udf_loss, regularization_loss = self.model.loss(outputs, batch)

        self.log("Validation_Loss", total_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])
        self.log("Validation_udf_loss", udf_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])
        self.log("Validation_regularization_loss", regularization_loss.detach(),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])

        return batch, outputs

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        if self.trainer.sanity_checking:
            return
        # return
        query_points = []
        gt_dis = []
        plane_m = None
        d1 = []
        weights = []
        d3 = []
        projs = []
        proj_dis = []
        for item in result:
            query_points.append(item[0][0].cpu().numpy())
            gt_dis.append(item[0][1].cpu().numpy())
            plane_m = item[1][2][0].cpu().numpy()
            d1.append(item[1][0].cpu().numpy())
            weights.append(item[1][1].cpu().numpy())
            # projs.append(item[1][3].cpu().numpy())
            # proj_dis.append(item[1][4].cpu().numpy())

        query_points = np.concatenate(query_points, axis=1)
        gt_dis = np.concatenate(gt_dis, axis=1)
        d1 = np.concatenate(d1, axis=1)  # udf_distance
        weights = np.concatenate(weights, axis=1)  # norm_distance
        # projs = np.concatenate(projs, axis=2)
        # proj_dis = np.concatenate(proj_dis, axis=1)

        num_plane = plane_m.shape[1]
        x1 = np.ones(num_plane) * -1
        y1 = (-plane_m[2] - x1 * plane_m[0]) / plane_m[1]
        x2 = np.ones(plane_m.shape[1]) * 1
        y2 = (-plane_m[2] - x2 * plane_m[0]) / plane_m[1]
        coords = np.stack((x1, y1, x2, y2), axis=1).reshape((num_plane, 2, 2))

        # Draw gt distance
        matplotlib.use('agg')
        plt.figure(figsize=(10, 8))
        line_colors = np.array((
            (56, 12, 77),
            (70, 48, 120),
            (57, 92, 134),
            (37, 129, 140),
            (27, 164, 136),
            (71, 195, 11),
            (158, 217, 67),
            (243, 232, 52)
        )) / 255.
        plt.subplot(2, 2, 1)
        for i in range(plane_m.shape[1]):
            plt.plot(coords[i, :, 0], coords[i, :, 1], '-', c=line_colors[i])
        plt.scatter(query_points[0, :, 0], query_points[0, :, 1], c=gt_dis[0], vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.subplot(2, 2, 2)
        for i in range(plane_m.shape[1]):
            plt.plot(coords[i, :, 0], coords[i, :, 1], '-', c=line_colors[i])
        plt.scatter(query_points[0, :, 0], query_points[0, :, 1], c=np.abs(d1[0]), vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        #
        plt.subplot(2, 2, 3)
        for i in range(plane_m.shape[1]):
            plt.plot(coords[i, :, 0], coords[i, :, 1], '-', c=line_colors[i])
        plt.scatter(query_points[0, :, 0], query_points[0, :, 1], c=weights[0, :].argmax(axis=1), vmin=0, vmax=7)
        plt.colorbar()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # plt.subplot(2, 2, 4)
        # for i in range(plane_m.shape[1]):
        #     plt.scatter(projs[0, i, :, 0], projs[0, i, :, 1], c=np.abs(proj_dis[0, :, i]), vmin=0, vmax=0.3)
        # plt.colorbar()
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_root, "{}.jpg".format(self.current_epoch)))
        plt.close()
        return


def prepare_dataset():
    print("Start to construct dataset")

    query_points_x = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    query_points = np.stack(np.meshgrid(query_points_x, query_points_x), axis=2)
    query_points = query_points.reshape(-1, 2)

    # Concave shape
    vertices = np.array([
        [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25], [0.25, 0.25],
        [0.1, 0.25], [0.1, 0.1], [-0.1, 0.1], [-0.1, 0.25]
    ], dtype=np.float32)

    distances = np.linalg.norm(query_points[:, None, :] - vertices[None, :], axis=2)
    fudf = distances.max(axis=1)

    edge_vecs = np.roll(vertices, 1, axis=0) - vertices
    point_vecs = query_points[:, np.newaxis, :] - vertices
    # Calculate the lengths of the edges and the projections of the point onto the edges
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    edge_unit_vecs = edge_vecs / edge_lengths[:, np.newaxis]
    t = np.einsum('ij,aij->ai', edge_unit_vecs, point_vecs)

    # Find the closest points on the edges to the point
    t = t.clip(0, edge_lengths)
    closest_points = vertices + (t[:, :, np.newaxis] * edge_unit_vecs)

    # Calculate the distances from the point to the closest points and find the minimum distance
    distances = np.linalg.norm(query_points[:, np.newaxis, :] - closest_points, axis=2)
    nudf = distances.min(axis=1)

    print("Done")
    print("{} points and values;".format(
        query_points.shape[0]
    ))

    return {
        "points": query_points,
        "fudfs": fudf,
        "nudfs": nudf,
    }


@hydra.main(config_name="lalala.yaml", config_path="../../../configs/neural_bsp/", version_base="1.1")
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
