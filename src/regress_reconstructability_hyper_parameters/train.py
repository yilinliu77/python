import math
import shutil
import sys, os
import time
from itertools import groupby
from typing import Tuple

import cv2
from plyfile import PlyData, PlyElement
from pytorch_lightning.strategies import DDPStrategy
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
    Regress_hyper_parameters_dataset_with_imgs, Regress_hyper_parameters_dataset_with_imgs_with_truncated_error
from src.regress_reconstructability_hyper_parameters.model import Regress_hyper_parameters_Model, Brute_force_nn, \
    Correlation_nn

# import torchsort
# from torchsort import soft_rank

from scipy import stats


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

        self.dataset_name_dict = {

        }

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def train_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        dataset_paths = self.hydra_conf["trainer"]["train_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(self.dataset_builder(os.path.join(dataset_root, dataset_path), self.hydra_conf,
                                                 "training" if len(
                                                     dataset_paths) == 1 else "testing", ))

        self.train_dataset = torch.utils.data.ConcatDataset(datasets)

        DataLoader_chosed = DataLoader if self.hydra_conf["trainer"]["gpu"] > 0 else FastDataLoader
        return DataLoader_chosed(self.train_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.hydra_conf["trainer"].num_worker,
                                 shuffle=True,
                                 # drop_last=True,
                                 pin_memory=True,
                                 collate_fn=self.dataset_builder.collate_fn,
                                 persistent_workers=True,
                                 # prefetch_factor=10
                                 )

    def val_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        use_part_dataset_to_validate = len(self.hydra_conf["trainer"]["train_dataset"].split("*")) == 1

        dataset_paths = self.hydra_conf["trainer"]["valid_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(self.dataset_builder(os.path.join(dataset_root, dataset_path), self.hydra_conf,
                                                 "validation" if use_part_dataset_to_validate else "testing", ))
            self.dataset_name_dict[datasets[-1].scene_name] = len(self.dataset_name_dict)

        self.valid_dataset = torch.utils.data.ConcatDataset(datasets)
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          # drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.dataset_builder.collate_fn,
                          persistent_workers=True,
                          # prefetch_factor=10
                          )

    def test_dataloader(self):
        dataset_root = self.hydra_conf["trainer"]["dataset_root"]
        dataset_paths = self.hydra_conf["trainer"]["test_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(
                self.dataset_builder(os.path.join(dataset_root, dataset_path), self.hydra_conf, "testing"))
            self.dataset_name_dict[datasets[-1].scene_name] = len(self.dataset_name_dict)

        self.test_dataset = torch.utils.data.ConcatDataset(datasets)

        return DataLoader(self.test_dataset,
                          batch_size=self.hydra_conf["trainer"]["batch_size"],
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          # drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.dataset_builder.collate_fn,
                          # persistent_workers=True,
                          # prefetch_factor=10
                          )

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
        batch_size = data["point_attribute"]
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
        batch_size = data["point_attribute"]
        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", total_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)
        self.log("Validation Recon Loss", recon_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)
        self.log("Validation Gt Loss", gt_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=1)

        return [results,
                data["point_attribute"],
                data["scene_name"],
                ]

    def _calculate_spearman(self, outputs: Tuple):
        error_mean_std = outputs[0][0].new(self.hydra_conf["model"]["error_mean_std"])

        prediction = torch.cat(list(map(lambda x: x[0], outputs))) * error_mean_std[2:] + error_mean_std[:2]
        prediction = prediction.reshape([-1, prediction.shape[2]])
        point_attribute = torch.cat(list(map(lambda x: x[1], outputs)))
        point_attribute = point_attribute.reshape([-1, point_attribute.shape[2]])
        acc_mask = point_attribute[:, 1] != -1
        com_mask = point_attribute[:, 2] != -1
        point_attribute[acc_mask, 1] = point_attribute[acc_mask, 1] * error_mean_std[2] + error_mean_std[0]
        point_attribute[com_mask, 2] = point_attribute[com_mask, 2] * error_mean_std[3] + error_mean_std[1]
        names = np.concatenate(
            list(map(lambda x: [self.dataset_name_dict[item] for batch in x[2] for item in batch], outputs)))
        names = error_mean_std.new(names)
        if len(self.dataset_name_dict) == 1:
            views = np.concatenate(list(map(lambda x: x[3], outputs)))
            views = views.reshape([-1, views.shape[2], views.shape[3]])
            points = np.concatenate(list(map(lambda x: x[4], outputs)))
            points = points.reshape([-1, points.shape[2]])

        log_str = ""
        mean_spearman = 0
        spearman_dict = {}
        for scene_item in self.dataset_name_dict:
            predicted_acc = prediction[names == self.dataset_name_dict[scene_item]][:, 0]
            predicted_com = prediction[names == self.dataset_name_dict[scene_item]][:, 1]
            smith_error = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0]
            gt_acc = point_attribute[names == self.dataset_name_dict[scene_item]][:, 1]
            gt_com = point_attribute[names == self.dataset_name_dict[scene_item]][:, 2]
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
            else:
                spearmanr_factor = spearman_correlation(predicted_com[gt_com != -1],
                                    gt_com[gt_com != -1])
                smith_spearmanr_factor = spearman_correlation(smith_error[gt_com != -1],
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
            log_str += "{:<35}: {:.2f}    ".format(scene_item, spearmanr_factor.cpu().numpy())
            log_str += "Smith_{:<35}: {:.2f}    ".format(scene_item, smith_spearmanr_factor.cpu().numpy())
            log_str += "Boost: {:.2f}  \n".format((spearmanr_factor - abs(smith_spearmanr_factor)).cpu().numpy())
            mean_spearman += spearmanr_factor - abs(smith_spearmanr_factor)

            if len(self.dataset_name_dict) == 1:
                views_item = views[(names == self.dataset_name_dict[scene_item]).cpu().numpy()]
                points_item = points[(names == self.dataset_name_dict[scene_item]).cpu().numpy()]
                output_test_with_pc_and_views(predicted_acc.cpu().numpy(),
                                              predicted_com.cpu().numpy(),
                                              gt_acc.cpu().numpy(),
                                              gt_com.cpu().numpy(),
                                              smith_error.cpu().numpy(),
                                              self.test_dataset.datasets[
                                                  self.dataset_name_dict[scene_item]].point_attribute.shape[0],
                                              views_item,
                                              points_item
                                              )

        mean_spearman = mean_spearman / len(self.dataset_name_dict)

        return mean_spearman, log_str

    def validation_epoch_end(self, outputs) -> None:
        mean_spearman, log_str = self._calculate_spearman(outputs)

        self.log("Valid mean spearman boost", mean_spearman, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 batch_size=1)
        self.trainer.logger.experiment.add_text("Validation spearman", log_str, global_step=self.trainer.current_epoch)

        if not self.trainer.sanity_checking:
            for dataset in self.train_dataset.datasets:
                dataset.sample_points_to_different_patches()

        return

    def on_test_epoch_start(self) -> None:
        self.data_mean_std = np.load(os.path.join(self.test_dataset.datasets[0].data_root, "data_centralize.npz"))[
            "arr_0"]
        if os.path.exists("temp/test_scene_output"):
            shutil.rmtree("temp/test_scene_output")
        os.mkdir("temp/test_scene_output")
        pass

    def test_step(self, batch, batch_idx):
        data = batch

        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)
        self.log("Test Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("Test Recon Loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("Test Gt Loss", gt_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=1)

        if len(self.dataset_name_dict) == 1:

            normals = data["point_attribute"][:, :, 7:10].cpu().numpy()
            normal_theta = np.arccos(normals[:, :, 2])
            normal_phi = np.arctan2(normals[:, :, 1], normals[:, :, 0])

            view_mean_std = np.array(self.hydra_conf["model"]["view_mean_std"])

            views = data["views"].cpu().numpy()
            view_mask = views[:, :, :, 0] > 0
            views[view_mask, 1] = views[view_mask, 1] * view_mean_std[5] + view_mean_std[0]
            views[view_mask, 2] = views[view_mask, 2] * view_mean_std[6] + view_mean_std[1]
            views[:, :, :, 1] = views[:, :, :, 1] + normal_theta[:, :, np.newaxis]
            views[:, :, :, 2] = views[:, :, :, 2] + normal_phi[:, :, np.newaxis]

            dz = np.cos(views[:, :, :, 1])
            dx = np.sin(views[:, :, :, 1]) * np.cos(views[:, :, :, 2])
            dy = np.sin(views[:, :, :, 1]) * np.sin(views[:, :, :, 2])

            view_dir = np.stack([dx, dy, dz], axis=-1) * (
                        data["views"][:, :, :, 3:4].cpu().numpy() * view_mean_std[7] + view_mean_std[2]) * 60
            points = data["point_attribute"][:, :, 3:6].cpu().numpy()
            points = points * self.data_mean_std[3] + self.data_mean_std[:3]
            views = points[:, :, np.newaxis] + view_dir
            views = np.concatenate([views, -view_dir, data["views"][:, :, :, 0:1].cpu().numpy()], axis=-1)

            return [results,
                    data["point_attribute"],
                    data["scene_name"],
                    views,  # num_points, num_views, 7
                    points
                    ]
        else:
            return [results,
                    data["point_attribute"],
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
        monitor='Validation Loss',
        save_top_k=3,
        save_last=True
    )

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, enable_model_summary=False,
                      # strategy=DDPStrategy() if v_cfg["trainer"].gpu > 1 else None,
                      strategy=DDPStrategy(find_unused_parameters=False) if v_cfg["trainer"].gpu > 1 else None,
                      # early_stop_callback=early_stop_callback,
                      callbacks=[model_check_point],
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=3000,
                      gradient_clip_val=0.1,
                      check_val_every_n_epoch=1,
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
    # import os
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    main()
