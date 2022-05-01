import math
import platform
import shutil
import sys, os
import time
from datetime import timedelta
from itertools import groupby
from typing import Tuple

import cv2
from plyfile import PlyData, PlyElement
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats, optimize, interpolate
from argparse import ArgumentParser
import torch
import torch.distributed as dist
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
from src.regress_reconstructability_hyper_parameters.dataset import Regress_hyper_parameters_dataset, \
    Regress_hyper_parameters_dataset_with_imgs, Regress_hyper_parameters_dataset_with_imgs_with_truncated_error, \
    Regress_hyper_parameters_img_dataset, My_ddp_sampler, My_ddp_sampler2
from src.regress_reconstructability_hyper_parameters.model import Regress_hyper_parameters_Model, Brute_force_nn, Correlation_nn
from src.regress_reconstructability_hyper_parameters.model_train import Correlation_net

# import torchsort
# from torchsort import soft_rank

from scipy import stats

from src.regress_reconstructability_hyper_parameters.preprocess_view_features import calculate_transformation_matrix


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def write_views_to_txt_file(v_args):
    point_idx, views = v_args
    id_point = int(point_idx)
    filename = "temp/test_scene_output/{}.txt".format(int(id_point))
    if os.path.exists(filename):
        raise

    # Write views
    with open(filename, "w") as f:
        for item in views:
            if item[6] <= 1e-3:
                break
            item[3:6] /= np.linalg.norm(item[3:6])

            pitch = math.asin(item[5])
            yaw = math.atan2(item[4], item[3])

            f.write("xxx.png,{},{},{},{},{},{}\n".format(
                item[0], item[1], item[2],
                pitch / math.pi * 180, 0, yaw / math.pi * 180,
            ))


# v_data: Predict recon error, Predict gt error, smith 18 recon, avg recon error, avg gt error, inconsistency (0 is consistent), Point index, x, y, z
def output_test_with_pc_and_views(
        v_predict_acc,
        v_predict_com,
        v_gt_acc,
        v_gt_com,
        v_smith_error,
        v_num_total_points,
        v_views,
        v_points,  # (x,y,z)
):
    # Write views
    vertexes = v_points[:, :3]

    vertexes_describer = PlyElement.describe(np.array(
        [(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]) for item in
         np.concatenate([vertexes,
                         v_predict_acc[:, np.newaxis],
                         v_predict_com[:, np.newaxis],
                         v_gt_acc[:, np.newaxis],
                         v_gt_com[:, np.newaxis],
                         v_smith_error[:, np.newaxis],
                         ], axis=1)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
               ('Predict_Recon', 'f8'), ('Predict_Gt', 'f8'),
               ('GT_Recon', 'f8'), ('GT_Gt', 'f8'),
               ('Smith_Recon', 'f8'),
               ]), 'vertex')

    PlyData([vertexes_describer]).write('temp/test_scene_output/whole_point.ply')

    # thread_map(write_views_to_txt_file, zip(v_views, v_points[:,3]), max_workers=16)
    process_map(write_views_to_txt_file, enumerate(v_views), max_workers=8, chunksize=4096)

    return


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x), device=x.device)
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


class Regress_hyper_parameters(pl.LightningModule):
    def __init__(self, hparams):
        super(Regress_hyper_parameters, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        # self.log(...,batch_size=self.batch_size)
        self.save_hyperparameters(hparams)
        model_module = __import__("src")
        model_module = getattr(model_module, "regress_reconstructability_hyper_parameters")
        model_module = getattr(model_module, "model")
        f = getattr(model_module, self.hydra_conf["model"]["model_name"])

        dataset_module = __import__("src")
        dataset_module = getattr(dataset_module, "regress_reconstructability_hyper_parameters")
        dataset_module = getattr(dataset_module, "dataset")
        self.dataset_builder = getattr(dataset_module, self.hydra_conf["trainer"]["dataset_name"])
        self.model = f(hparams)

        self.img_worker = max(self.hydra_conf["trainer"]["num_worker"] // 4 * 3, 1)
        self.scene_worker = max(self.hydra_conf["trainer"]["num_worker"] // 4 * 1, 1)

        assert (self.hydra_conf["trainer"]["batch_size"] <= 1 or self.hydra_conf["model"]["num_points_per_batch"] <= 1)

        self.involved_imgs = self.hydra_conf["model"]["involve_img"]
        self.is_ddp = self.hydra_conf["trainer"]["gpu"] > 0
        self.dataset_name_dict = {

        }

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def setup_dataset(self, v_dataset_root, v_dataset_strs, v_mode="testing"):
        train_dataset = self.hydra_conf["trainer"]["train_dataset"].split("*")
        if len(train_dataset) > 1:
            v_mode = "testing"
        scene_datasets = []
        img_datasets = []
        for dataset_path in v_dataset_strs:
            scene_root = os.path.join(v_dataset_root, dataset_path)
            scene_datasets.append(self.dataset_builder(scene_root, self.hydra_conf, v_mode))
            self.dataset_name_dict[scene_datasets[-1].scene_name] = len(self.dataset_name_dict)
            img_datasets.append(Regress_hyper_parameters_img_dataset(
                os.path.join(scene_root, "../"),
                scene_datasets[-1].view_paths,
                scene_datasets[-1].used_index
            ))

        return scene_datasets, img_datasets

    def train_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        dataset_paths = self.hydra_conf["trainer"]["train_dataset"].split("*")

        scene_dataset, img_dataset = self.setup_dataset(dataset_root, dataset_paths)
        self.train_scene_dataset = torch.utils.data.ConcatDataset(scene_dataset)
        self.train_img_dataset = torch.utils.data.ConcatDataset(img_dataset)

        train_scene_sampler = My_ddp_sampler2(self.train_scene_dataset, self.batch_size,
                                             v_sample_mode="internal", shuffle=True)
        train_img_sampler = My_ddp_sampler2(self.train_img_dataset, self.batch_size,
                                           v_sample_mode="internal", shuffle=True)
        if self.involved_imgs:
            combined_dataset = {
                "scene": DataLoader(self.train_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    sampler=train_scene_sampler,
                                    persistent_workers=True
                                    ),
                "img": DataLoader(self.train_img_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.img_worker,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=self.dataset_builder.collate_fn,
                                  sampler=train_img_sampler,
                                  persistent_workers=True
                                  )}
            assert len(combined_dataset["scene"]) == len(combined_dataset["img"])
        else:
            combined_dataset = {
                "scene": DataLoader(self.train_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    sampler=train_scene_sampler,
                                    persistent_workers=True
                                    )
            }

        return CombinedLoader(combined_dataset, mode="min_size")

    def val_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        dataset_paths = self.hydra_conf["trainer"]["valid_dataset"].split("*")

        scene_dataset, img_dataset = self.setup_dataset(dataset_root, dataset_paths)
        self.valid_scene_dataset = torch.utils.data.ConcatDataset(scene_dataset)
        self.valid_img_dataset = torch.utils.data.ConcatDataset(img_dataset)

        valid_scene_sampler = My_ddp_sampler2(self.valid_scene_dataset, self.batch_size,
                                             v_sample_mode="internal", shuffle=False)
        valid_img_sampler = My_ddp_sampler2(self.valid_img_dataset, self.batch_size,
                                           v_sample_mode="internal", shuffle=False)
        if self.involved_imgs:
            combined_dataset = {
                "scene": DataLoader(self.valid_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    sampler=valid_scene_sampler,
                                    persistent_workers=True
                                    ),
                "img": DataLoader(self.valid_img_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.img_worker,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=self.dataset_builder.collate_fn,
                                  sampler=valid_img_sampler,
                                  persistent_workers=True
                                  )}
            assert len(combined_dataset["scene"]) == len(combined_dataset["img"])
        else:
            combined_dataset = {
                "scene": DataLoader(self.valid_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    sampler=valid_scene_sampler,
                                    persistent_workers=True
                                    )
            }

        return CombinedLoader(combined_dataset, mode="min_size")

    def test_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        dataset_paths = self.hydra_conf["trainer"]["test_dataset"].split("*")

        scene_dataset, img_dataset = self.setup_dataset(dataset_root, dataset_paths)

        self.test_scene_dataset = torch.utils.data.ConcatDataset(scene_dataset)
        self.test_img_dataset = torch.utils.data.ConcatDataset(img_dataset)
        if self.involved_imgs:
            combined_dataset = {
                "scene": DataLoader(self.test_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    ),
                "img": DataLoader(self.test_img_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.img_worker,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=self.dataset_builder.collate_fn,
                                  )}
            assert len(combined_dataset["scene"]) == len(combined_dataset["img"])
        else:
            combined_dataset = {
                "scene": DataLoader(self.test_scene_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.scene_worker,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=self.dataset_builder.collate_fn,
                                    ),
            }

        return CombinedLoader(combined_dataset, mode="min_size")

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )
        # optimizer = Adam(self.parameters(), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch
        batch_size = data["scene"]["views"].shape[0]
        num_points_per_item = data["scene"]["views"].shape[1]
        data["total"] = {}
        data["total"].update(data["scene"])
        if self.hparams["model"]["involve_img"]:
            data["total"]["point_features"] = data["img"]["point_features"].reshape((
                                                                                        batch_size,
                                                                                        num_points_per_item) +
                                                                                    data["img"]["point_features"].shape[
                                                                                    -2:])
            data["total"]["point_features_mask"] = data["img"]["point_features_mask"].reshape((
                                                                                                  batch_size,
                                                                                                  num_points_per_item) +
                                                                                              data["img"][
                                                                                                  "point_features_mask"].shape[
                                                                                              -1:])
        data = data["total"]
        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Training Recon Loss", recon_loss.detach(), prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)
        self.log("Training Gt Loss", gt_loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)
        self.log("Training Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            pass

        return total_loss

        # return {
        #     "loss":total_loss,
        #     "result":torch.cat([results, data["point_attribute"][:, :, [1,5]], data["points"][:,:,3:4]], dim=2).detach()
        # }

    def validation_step(self, batch, batch_idx):
        data = batch
        batch_size = data["scene"]["views"].shape[0]
        num_points_per_item = data["scene"]["views"].shape[1]
        data["total"] = {}
        data["total"].update(data["scene"])
        if self.hparams["model"]["involve_img"]:
            data["total"]["point_features"] = data["img"]["point_features"].reshape((
                                                                                        batch_size,
                                                                                        num_points_per_item) +
                                                                                    data["img"]["point_features"].shape[
                                                                                    -2:])
            data["total"]["point_features_mask"] = data["img"]["point_features_mask"].reshape((
                                                                                                  batch_size,
                                                                                                  num_points_per_item) +
                                                                                              data["img"][
                                                                                                  "point_features_mask"].shape[
                                                                                              -1:])
        data = data["total"]
        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", total_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)
        self.log("Validation Recon Loss", recon_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)
        self.log("Validation Gt Loss", gt_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)

        return [results,
                torch.cat([data["point_attribute"], data["views"][:, :, :, 0].sum(dim=-1, keepdim=True)], dim=-1),
                data["scene_name"],
                ]

    def _calculate_spearman(self, outputs: Tuple):
        error_mean_std = outputs[0][0].new(self.hydra_conf["model"]["error_mean_std"])

        prediction = torch.cat(list(map(lambda x: x[0].reshape((-1, x[0].shape[2])), outputs)), dim=0)
        if self.hparams["trainer"]["loss"] == "loss_l2_error":
            prediction = prediction * error_mean_std[2:] + error_mean_std[:2]
        point_attribute = torch.cat(list(map(lambda x: x[1].reshape((-1, x[1].shape[2])), outputs)), dim=0)
        acc_mask = point_attribute[:, 1] != -1
        com_mask = point_attribute[:, 2] != -1
        point_attribute[acc_mask, 1] = point_attribute[acc_mask, 1] * error_mean_std[2] + error_mean_std[0]
        point_attribute[com_mask, 2] = point_attribute[com_mask, 2] * error_mean_std[3] + error_mean_std[1]
        names = np.concatenate(
            list(map(lambda x: [self.dataset_name_dict[item] for batch in x[2] for item in batch], outputs)))
        names = error_mean_std.new(names)
        if len(self.dataset_name_dict) == 1 and self.hparams["trainer"]["evaluate"]:
            views = np.concatenate(list(map(lambda x: x[3].reshape((-1,) + (x[3].shape[2:])), outputs)), axis=0)
            points = np.concatenate(list(map(lambda x: x[4].reshape((-1,) + (x[4].shape[2:])), outputs)), axis=0)

        log_str = ""
        mean_spearman = 0
        spearman_dict = {}
        for scene_item in self.dataset_name_dict:
            predicted_acc = prediction[names == self.dataset_name_dict[scene_item]][:, 0]
            predicted_com = prediction[names == self.dataset_name_dict[scene_item]][:, 1]
            smith_error = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0]
            gt_acc = point_attribute[names == self.dataset_name_dict[scene_item]][:, 1]
            gt_com = point_attribute[names == self.dataset_name_dict[scene_item]][:, 2]
            baseline_number = point_attribute[names == self.dataset_name_dict[scene_item]][:, -1]
            if not self.hparams["model"]["involve_img"]:
                # spearmanr_factor = stats.spearmanr(
                #     predicted_acc[gt_acc != -1],
                #     gt_acc[gt_acc != -1]
                # )[0]
                # smith_spearmanr_factor = stats.spearmanr(
                #     smith_error[gt_acc != -1],
                #     gt_acc[gt_acc != -1]
                # )[0]

                spearmanr_factor = spearman_correlation(predicted_acc[gt_acc != -1],
                                                        gt_acc[gt_acc != -1])
                smith_spearmanr_factor = spearman_correlation(smith_error[gt_acc != -1],
                                                              gt_acc[gt_acc != -1])
                baseline_number_spearmanr_factor = spearman_correlation(baseline_number[gt_acc != -1],
                                                                        gt_acc[gt_acc != -1])
            else:
                spearmanr_factor = spearman_correlation(predicted_com[gt_com != -1],
                                                        gt_com[gt_com != -1])
                smith_spearmanr_factor = spearman_correlation(smith_error[gt_com != -1],
                                                              gt_com[gt_com != -1])
                baseline_number_spearmanr_factor = spearman_correlation(baseline_number[gt_com != -1],
                                                                        gt_com[gt_com != -1])
                # spearmanr_factor = stats.spearmanr(
                #     predicted_com[gt_com != -1],
                #     gt_com[gt_com != -1]
                # )[0]
                # smith_spearmanr_factor = stats.spearmanr(
                #     smith_error[gt_com != -1],
                #     gt_com[gt_com != -1]
                # )[0]

            spearman_dict[scene_item] = spearmanr_factor
            log_str += "{:<35}: {:.4f}    ".format(scene_item, spearmanr_factor.cpu().numpy())
            log_str += "Smith_{:<35}: {:.4f}    ".format(scene_item, smith_spearmanr_factor.cpu().numpy())
            log_str += "Baseline_{:<35}: {:.4f}    ".format(scene_item, baseline_number_spearmanr_factor.cpu().numpy())
            log_str += "Boost: {:.4f}  \n".format((spearmanr_factor - abs(smith_spearmanr_factor)).cpu().numpy())
            mean_spearman += spearmanr_factor - abs(smith_spearmanr_factor)

            if len(self.dataset_name_dict) == 1 and self.hparams["trainer"]["evaluate"]:
                views_item = views[(names == self.dataset_name_dict[scene_item]).cpu().numpy()]
                points_item = points[(names == self.dataset_name_dict[scene_item]).cpu().numpy()]
                output_test_with_pc_and_views(predicted_acc.cpu().numpy(),
                                              predicted_com.cpu().numpy(),
                                              gt_acc.cpu().numpy(),
                                              gt_com.cpu().numpy(),
                                              smith_error.cpu().numpy(),
                                              self.test_scene_dataset.datasets[
                                                  self.dataset_name_dict[scene_item]].point_attribute.shape[0],
                                              views_item,
                                              points_item
                                              )

        mean_spearman = mean_spearman / len(self.dataset_name_dict)

        return mean_spearman, log_str

    def validation_epoch_end(self, outputs) -> None:
        # mean_spearman, log_str = self._calculate_spearman(outputs)

        prediction = torch.cat(list(map(lambda x: x[0].reshape((-1, x[0].shape[2])), outputs)), dim=0)
        point_attribute = torch.cat(list(map(lambda x: x[1].reshape((-1, x[1].shape[2])), outputs)), dim=0)
        acc_mask = point_attribute[:, 1] != -1
        com_mask = point_attribute[:, 2] != -1

        if not self.involved_imgs:
            our_spearman = spearman_correlation(prediction[acc_mask][:, 0], point_attribute[acc_mask][:, 1])
            smith_spearman = spearman_correlation(point_attribute[acc_mask][:, 0], point_attribute[acc_mask][:, 1])
        else:
            our_spearman = spearman_correlation(prediction[com_mask][:, 1], point_attribute[com_mask][:, 2])
            smith_spearman = spearman_correlation(point_attribute[com_mask][:, 0], point_attribute[com_mask][:, 2])
        spearman_boost = our_spearman + smith_spearman
        self.log("Valid mean spearman boost", spearman_boost, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)
        # self.trainer.logger.experiment.add_text("Validation spearman", log_str, global_step=self.trainer.current_epoch)

        # if not self.trainer.sanity_checking:
        #     for dataset in self.train_dataset.datasets:
        #         dataset.sample_points_to_different_patches()

        return

    def on_test_epoch_start(self) -> None:
        self.data_mean_std = \
            np.load(os.path.join(self.test_scene_dataset.datasets[0].data_root, "data_centralize.npz"))[
                "arr_0"]
        if os.path.exists("temp/test_scene_output"):
            shutil.rmtree("temp/test_scene_output")
        os.mkdir("temp/test_scene_output")
        pass

    def test_step(self, batch, batch_idx):
        data = batch
        batch_size = data["scene"]["views"].shape[0]
        num_points_per_item = data["scene"]["views"].shape[1]
        data["total"] = {}
        data["total"].update(data["scene"])
        if self.hparams["model"]["involve_img"]:
            data["total"]["point_features"] = data["img"]["point_features"].reshape((
                                                                                        batch_size,
                                                                                        num_points_per_item) +
                                                                                    data["img"]["point_features"].shape[
                                                                                    -2:])
            data["total"]["point_features_mask"] = data["img"]["point_features_mask"].reshape((
                                                                                                  batch_size,
                                                                                                  num_points_per_item) +
                                                                                              data["img"][
                                                                                                  "point_features_mask"].shape[
                                                                                              -1:])
        data = data["total"]
        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)
        self.log("Test Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("Test Recon Loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("Test Gt Loss", gt_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)

        if len(self.dataset_name_dict) == 1:
            normals = data["point_attribute"][:, :, 7:10].cpu().numpy()
            magic_matrix = calculate_transformation_matrix(normals)
            magic_matrix = np.transpose(magic_matrix, (0, 1, 3, 2))

            view_mean_std = np.array(self.hydra_conf["model"]["view_mean_std"])

            views = data["views"].cpu().numpy()
            view_mask = views[:, :, :, 0] > 0
            views[view_mask, 1] = views[view_mask, 1] * view_mean_std[5] + view_mean_std[0]
            views[view_mask, 2] = views[view_mask, 2] * view_mean_std[6] + view_mean_std[1]

            dz = np.cos(views[:, :, :, 1])
            dx = np.sin(views[:, :, :, 1]) * np.cos(views[:, :, :, 2])
            dy = np.sin(views[:, :, :, 1]) * np.sin(views[:, :, :, 2])
            local_point_to_view = np.stack([dx, dy, dz], axis=-1)
            global_point_to_view = np.matmul(magic_matrix, local_point_to_view.transpose((0, 1, 3, 2))).transpose(
                (0, 1, 3, 2))

            view_dir = global_point_to_view * (
                    data["views"][:, :, :, 3:4].cpu().numpy() * view_mean_std[7] + view_mean_std[2]) * 60
            points = data["point_attribute"][:, :, 3:6].cpu().numpy()
            points = points * self.data_mean_std[3] + self.data_mean_std[:3]
            views = points[:, :, np.newaxis] + view_dir
            views = np.concatenate([views, -view_dir, data["views"][:, :, :, 0:1].cpu().numpy()], axis=-1)

            return [results,
                    torch.cat([data["point_attribute"], data["views"][:, :, :, 0].sum(dim=-1, keepdim=True)], dim=-1),
                    data["scene_name"],
                    views,  # num_points, num_views, 7
                    points
                    ]
        else:
            return [results,
                    torch.cat([data["point_attribute"], data["views"][:, :, :, 0].sum(dim=-1, keepdim=True)], dim=-1),
                    data["scene_name"],
                    ]

    def test_epoch_end(self, outputs) -> None:
        mean_spearman, log_str = self._calculate_spearman(outputs)

        self.log("Test mean spearman", mean_spearman, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)
        print(log_str)

        pass

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def predict_recon(self):
        pass


@hydra.main(config_name="test.yaml", config_path="../../configs/regress_hyper_parameters/")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)
    torch.autograd.set_detect_anomaly(True)
    early_stop_callback = EarlyStopping(
        patience=100,
        monitor="Validation Loss"
    )

    model_check_point = ModelCheckpoint(
        monitor='Valid mean spearman boost',
        save_top_k=1,
        save_last=True,
        mode="max",
        auto_insert_metric_name=True,
        train_time_interval=timedelta(seconds=60 * 60)
    )

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, enable_model_summary=False,
                      strategy=DDPStrategy(
                          process_group_backend="gloo" if platform.system() == "Windows" else "nccl",
                          find_unused_parameters=False
                      ) if not v_cfg["trainer"]["evaluate"] else None,
                      # early_stop_callback=early_stop_callback,
                      callbacks=[model_check_point],
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=3000,
                      gradient_clip_val=0.1,
                      check_val_every_n_epoch=1,
                      replace_sampler_ddp=False
                      )

    model = Regress_hyper_parameters(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        # for item in list(state_dict.keys()):
        #     if "point_feature_extractor" in item:
        #         state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].auto_lr_find:
        trainer.tune(model)
        print(model.learning_rate)
    # model.save('temp/model.pt')
    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    if platform.system() == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    main()
