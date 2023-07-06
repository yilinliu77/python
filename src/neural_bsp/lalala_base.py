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
        self.sdfs = torch.from_numpy(v_data["udfs"])

        self.mode = v_training_mode
        self.batch_size = 4096
        pass

    def __len__(self):
        return 100 if self.mode == "training" else math.ceil(self.points.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        if self.mode == "training":
            which_point = torch.randint(0, self.points.shape[0], (self.batch_size,))
            return self.points[which_point], self.sdfs[which_point]
        else:
            id_start = idx * self.batch_size
            id_end = (idx + 1) * self.batch_size
            return self.points[id_start:id_end], self.sdfs[id_start:id_end]


class Base_model(nn.Module):
    def __init__(self, v_num_plane=7):
        super(Base_model, self).__init__()
        self.num_plane = v_num_plane
        self.z_vector = nn.Parameter(torch.rand(1,256))
        self.to_distance = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.num_freq = 8
        self.pos_linear = nn.Linear(self.num_freq * 2 * 2, 256)

    def pos_encoding(self, v_points):
        freq = 2 ** torch.arange(self.num_freq, dtype=torch.float32, device=v_points.device) * torch.pi  # [L]
        spectrum = v_points[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*v_points.shape[:-1], -1)  # [B,...,2NL]
        return self.pos_linear(input_enc)

    def forward(self, v_data, v_training=False):
        points, sdfs = v_data
        pos_encoding = self.pos_encoding(points)

        pred_udf = self.to_distance(pos_encoding+self.z_vector[:,None])[:,:,0]
        return pred_udf, None

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf = v_predictions[0]
        loss = F.l1_loss(pred_udf, sdfs)
        return loss


class Base_model_plane(Base_model):
    def __init__(self, v_num_plane=7):
        super(Base_model_plane, self).__init__(v_num_plane)
        self.num_plane = v_num_plane
        self.z_vector = nn.Parameter(torch.rand(v_num_plane, 256) * 2 - 1)
        self.to_parameters = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, v_data, v_training=False):
        points, sdfs = v_data

        plane_parameters = self.to_parameters(self.z_vector).unsqueeze(0).permute(0,2,1)

        plane_parameters2 = plane_parameters**2
        norm = torch.sqrt(plane_parameters2[:,0]+plane_parameters2[:,1])[:,None]

        all_distance = torch.matmul(points, plane_parameters)
        all_distance = all_distance / norm
        single_distance = all_distance.min(dim=2)[0]

        return single_distance, plane_parameters.permute(0,2,1)

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf = v_predictions[0]
        loss = F.l1_loss(pred_udf, sdfs)
        return loss


class Base_model_plane_roi(Base_model):
    def __init__(self, v_num_plane=7):
        super(Base_model_plane_roi, self).__init__(v_num_plane)
        self.num_plane = v_num_plane
        self.z_vector = nn.Parameter(torch.rand(v_num_plane, 256) * 2 - 1)
        self.to_parameters = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.interest_region = nn.ModuleList()
        for i in range(v_num_plane):
            self.interest_region.append(
                nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.ReLU(),
                )
            )

    def forward(self, v_data, v_training=False):
        points, sdfs = v_data

        plane_parameters = self.to_parameters(self.z_vector).unsqueeze(0).permute(0,2,1)

        plane_parameters2 = plane_parameters**2
        norm = torch.sqrt(plane_parameters2[:,0]+plane_parameters2[:,1] + 1e-16)[:,None]

        all_distance = torch.matmul(points, plane_parameters)
        all_distance = all_distance / norm

        interest_distances=[]
        for i_plane in range(self.num_plane):
            interest_distance = self.interest_region[i_plane](points[:,:,:2])
            interest_distances.append(interest_distance)
        interest_distances = torch.stack(interest_distances, dim=1).permute(0,2,1,3)[:,:,:,0]

        true_distance = torch.sqrt(all_distance**2+interest_distances**2 + 1e-16)
        single_distance = true_distance.min(dim=2)[0]

        return single_distance, plane_parameters.permute(0,2,1)

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf = v_predictions[0]
        loss = F.l1_loss(pred_udf, sdfs)
        return loss


class Base_model_plane_roi_soft(Base_model):
    def __init__(self, v_num_plane=7):
        super(Base_model_plane_roi_soft, self).__init__(v_num_plane)
        self.num_plane = v_num_plane
        self.z_vector = nn.Parameter(torch.rand(v_num_plane, 256) * 2 - 1)
        self.to_parameters = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.interest_region = nn.ModuleList()
        for i in range(v_num_plane):
            self.interest_region.append(
                nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.ReLU(),
                )
            )

        self.concave_layer = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_plane),
        )

    def forward(self, v_data, v_training=False):
        points, sdfs = v_data

        plane_parameters = self.to_parameters(self.z_vector).unsqueeze(0).permute(0,2,1)

        plane_parameters2 = plane_parameters**2
        norm = torch.sqrt(plane_parameters2[:,0]+plane_parameters2[:,1] + 1e-16)[:,None]

        all_distance = torch.matmul(points, plane_parameters)
        all_distance = all_distance / norm

        interest_distances=[]
        for i_plane in range(self.num_plane):
            interest_distance = self.interest_region[i_plane](points[:,:,:2])
            interest_distances.append(interest_distance)
        interest_distances = torch.stack(interest_distances, dim=1).permute(0,2,1,3)[:,:,:,0]

        true_distance = torch.sqrt(all_distance**2+interest_distances**2 + 1e-16)
        # single_distance = true_distance.min(dim=2)[0]
        distance_weights = torch.softmax(self.concave_layer(points[:,:,:2]),dim=-1)
        single_distance = torch.sum(true_distance*distance_weights,dim=2)

        return single_distance, plane_parameters.permute(0,2,1), distance_weights

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf = v_predictions[0]
        distance_weights = v_predictions[2]
        loss = F.l1_loss(pred_udf, sdfs)

        sparsity_loss = (-torch.max(distance_weights, dim=2)[0]
                         + torch.sum(torch.pow(distance_weights, 2), dim=2)).mean() * 0.1

        return loss + sparsity_loss

class Model1(Base_model):
    def __init__(self, v_num_plane=8):
        super(Model1, self).__init__(v_num_plane)
        self.num_plane = v_num_plane
        self.z_vector = nn.Parameter(torch.rand(v_num_plane, 256))
        self.to_distance = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.to_region = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


    def forward(self, v_data, v_training=False):
        points, sdfs = v_data

        pos_encoding = self.pos_encoding(points[0, :,:2])

        fused_features = self.z_vector[None, :] + pos_encoding[:,None]
        fused_distances = self.to_distance(fused_features)
        fused_regions = self.to_region(fused_features)

        return single_distance, plane_parameters.permute(0,2,1), distance_weights

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        pred_udf = v_predictions[0]
        distance_weights = v_predictions[2]
        loss = F.l1_loss(pred_udf, sdfs)

        sparsity_loss = (-torch.max(distance_weights, dim=2)[0]
                         + torch.sum(torch.pow(distance_weights, 2), dim=2)).mean() * 0.1

        return loss + sparsity_loss


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
        # self.model = Base_model()
        # self.model = Base_model_plane()
        # self.model = Base_model_plane_roi()
        self.model = Model1()

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
        outputs = self.model(batch, True)
        total_loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])

        return total_loss

    def on_train_epoch_end(self, outputs) -> None:
        # self.model.sample_mask()
        pass

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, False)
        total_loss = self.model.loss(outputs, batch)

        self.log("Validation_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[1])

        return batch[0], batch[1], outputs[0], outputs[1]

    def on_validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        if self.trainer.sanity_checking:
            return

        data = list(zip(*result))
        query_points = torch.concatenate(data[0],dim=1).cpu().numpy()[0,:,:2]
        gt_udfs = torch.concatenate(data[1],dim=1).cpu().numpy()[0]
        pred_udfs = torch.concatenate(data[2],dim=1).cpu().numpy()[0]

        has_plane = result[0][3] is not None
        if has_plane:
            plane_m = data[3][0][0].cpu().numpy()
            num_plane = plane_m.shape[0]
            x1 = np.ones(num_plane) * -1
            y1 = (-plane_m[:,2] - x1 * plane_m[:,0]) / plane_m[:,1]
            x2 = np.ones(plane_m.shape[0]) * 1
            y2 = (-plane_m[:,2] - x2 * plane_m[:,0]) / plane_m[:,1]
            coords = np.stack((x1, y1, x2, y2), axis=1).reshape((num_plane, 2, 2))

        # Draw gt distance
        matplotlib.use('agg')
        plt.figure(figsize=(10, 8))

        ax1 = plt.subplot(2, 2, 1)
        if has_plane:
            for i in range(plane_m.shape[0]):
                plt.plot(coords[i, :, 0], coords[i, :, 1], 'b-')
        plt.scatter(query_points[:, 0], query_points[:, 1], c=gt_udfs, vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.subplot(2, 2, 2, sharex=ax1)
        if has_plane:
            for i in range(plane_m.shape[0]):
                plt.plot(coords[i, :, 0], coords[i, :, 1], 'b-')
        plt.scatter(query_points[:, 0], query_points[:, 1], c=np.abs(pred_udfs), vmin=0, vmax=0.3)
        plt.colorbar()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # plt.subplot(2, 2, 3, sharex=ax1)
        # for i in range(plane_m.shape[1]):
        #     plt.plot(coords[i, :, 0], coords[i, :, 1], 'b-')
        # plt.scatter(query_points[0, :, 0], query_points[0, :, 1], c=np.abs(d2[0]).min(axis=1), vmin=0, vmax=0.3)
        # plt.colorbar()
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        #
        # plt.subplot(2, 2, 4, sharex=ax1)
        # for i in range(plane_m.shape[1]):
        #     plt.scatter(projs[0, i, :, 0], projs[0, i, :, 1], c=np.abs(d3[0, :, i]), vmin=0, vmax=0.3)
        # plt.colorbar()
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_root, "{}.jpg".format(self.current_epoch)))
        plt.close()
        # self.model.reinitialize(np.unique(pred_index, return_counts=True))
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
    shortest_distances = distances.min(axis=1)

    # from matplotlib import pyplot as plt
    # plt.plot(query_points[shortest_distances>0.1,0],query_points[shortest_distances>0.1,1],'ro', markersize=2)
    # plt.plot(vertices[:, 0], vertices[:, 1], 'b-')
    # plt.plot(np.roll(vertices,1,axis=0)[:, 0], np.roll(vertices,1,axis=0)[:, 1], 'b-')
    # plt.show(block=True)

    print("Done")
    print("{} points and values;".format(
        query_points.shape[0]
    ))

    return {
        "points": query_points,
        "udfs": shortest_distances,
    }


@hydra.main(config_name="lalala.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(v_cfg["trainer"]["random"])
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
