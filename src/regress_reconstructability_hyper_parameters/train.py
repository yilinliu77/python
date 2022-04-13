import math
import shutil
import sys, os
import time
from itertools import groupby

import cv2
from plyfile import PlyData, PlyElement
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats, optimize, interpolate
from argparse import ArgumentParser
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
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
    views, point_idx = v_args
    id_point = int(point_idx)
    filename = "temp/test_scene_output/{}.txt".format(int(id_point))
    if os.path.exists(filename):
        raise

    # Write views
    with open(filename, "w") as f:
        for item in views:
            if item[6] == 0:
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
        v_points,  # (x,y,z,idx)
):
    # Write views
    vertexes = np.zeros((v_num_total_points, 3), dtype=np.float32)  # x, y, z
    vertexes[v_points[:,3].astype(np.int32)] = v_points[:,:3]

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
    process_map(write_views_to_txt_file, zip(v_views, v_points[:,3]), max_workers=8, chunksize = 4096)

    return


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
        dataset_paths = self.hydra_conf["trainer"]["train_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(self.dataset_builder(dataset_path, self.hydra_conf,
                                                 "training" if len(
                                                     dataset_paths) == 1 else "testing", ))

        self.train_dataset = torch.utils.data.ConcatDataset(datasets)

        DataLoader_chosed = DataLoader if self.hydra_conf["trainer"]["gpu"] > 0 else FastDataLoader
        return DataLoader_chosed(self.train_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.hydra_conf["trainer"].num_worker,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 collate_fn=self.dataset_builder.collate_fn,
                                 persistent_workers=True,
                                 prefetch_factor=10
                                 )

    def val_dataloader(self):
        use_part_dataset_to_validate = len(self.hydra_conf["trainer"]["train_dataset"].split("*")) == 1

        dataset_paths = self.hydra_conf["trainer"]["valid_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(self.dataset_builder(dataset_path, self.hydra_conf,
                                                 "validation" if use_part_dataset_to_validate else "testing", ))
            self.dataset_name_dict[datasets[-1].scene_name] = len(self.dataset_name_dict)

        self.valid_dataset = torch.utils.data.ConcatDataset(datasets)
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.dataset_builder.collate_fn,
                          persistent_workers=True,
                          prefetch_factor=10
                          )

    def test_dataloader(self):
        dataset_paths = self.hydra_conf["trainer"]["test_dataset"].split("*")
        datasets = []
        for dataset_path in dataset_paths:
            datasets.append(
                self.dataset_builder(dataset_path, self.hydra_conf, "testing"))
            self.dataset_name_dict[datasets[-1].scene_name] = len(self.dataset_name_dict)

        self.test_dataset = torch.utils.data.ConcatDataset(datasets)

        return DataLoader(self.test_dataset,
                          batch_size=self.hydra_conf["trainer"]["batch_size"],
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.dataset_builder.collate_fn,
                          persistent_workers=True,
                          prefetch_factor=10
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

        self.log("Training Recon Loss", recon_loss.detach(), prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("Training Gt Loss", gt_loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("Training Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            pass

        return total_loss

        # return {
        #     "loss":total_loss,
        #     "result":torch.cat([results, data["point_attribute"][:, :, [1,5]], data["points"][:,:,3:4]], dim=2).detach()
        # }

    # def training_epoch_end(self, outputs) -> None:
    # spearmanr_factor,accuracy,whole_points_prediction_error  = output_test(
    #     torch.cat([item["result"] for item in outputs], dim=0).cpu().numpy(), self.train.datasets[0].point_attribute.shape[0])
    # self.log("Training spearman", spearmanr_factor, prog_bar=True, logger=True, on_step=False,
    #          on_epoch=True)

    def validation_step(self, batch, batch_idx):
        data = batch
        batch_size = data["point_attribute"]
        results, weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", total_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("Validation Recon Loss", recon_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("Validation Gt Loss", gt_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)

        return [results,
                data["point_attribute"],
                data["point_attribute"].new(list(map(lambda x: self.dataset_name_dict[x], data["scene_name"]))),
                ]

    def validation_epoch_end(self, outputs) -> None:
        if self.trainer.global_rank == 0:
            if self.hparams["trainer"].gpu > 1:
                outputs_world = self.all_gather(outputs)
                prediction = torch.flatten(torch.cat(list(map(lambda x: x[0], outputs_world)), dim=1),start_dim=0,end_dim=1).cpu().numpy()
                point_attribute = torch.flatten(torch.cat(list(map(lambda x: x[1], outputs_world)), dim=1),start_dim=0,end_dim=1).cpu().numpy()
                names = torch.flatten(torch.cat(list(map(lambda x: x[2], outputs_world)),dim=1),start_dim=0,end_dim=1).cpu().numpy()
            else:
                prediction = torch.cat(list(map(lambda x: x[0], outputs)),dim=0).cpu().numpy()
                point_attribute = torch.cat(list(map(lambda x: x[1], outputs)),dim=0).cpu().numpy()
                names = torch.cat(list(map(lambda x: x[2], outputs))).cpu().numpy()

            log_str = ""
            mean_spearman = 0
            spearman_dict = {}
            min_num_points = 9999999
            for scene_item in self.dataset_name_dict:
                predicted_acc = prediction[names == self.dataset_name_dict[scene_item]][:, 0, 0]
                predicted_com = prediction[names == self.dataset_name_dict[scene_item]][:, 0, 1]
                gt_acc = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0, 1]
                gt_com = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0, 1]
                if not self.hparams["model"]["involve_img"]:
                    spearmanr_factor = stats.spearmanr(
                        predicted_acc[gt_acc != -1],
                        gt_acc[gt_acc != -1]
                    )[0]

                else:
                    spearmanr_factor = stats.spearmanr(
                        predicted_com[gt_com != -1],
                        gt_com[gt_com != -1]
                    )[0]

                spearman_dict[scene_item] = spearmanr_factor
                mean_spearman += spearmanr_factor
                log_str += "{:<35}: {:.2f}  \n".format(scene_item, spearmanr_factor)
                min_num_points = min(min_num_points, predicted_acc[gt_acc != -1].shape[0])

            mean_spearman = mean_spearman / len(self.dataset_name_dict)
            self.log("Validation mean spearman", mean_spearman,
                     prog_bar=True, logger=True, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log("Validation min num points",min_num_points,
                     prog_bar=True, logger=True, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            pass

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

        normals = data["point_attribute"][:, :, 7:10].cpu().numpy()
        normal_theta = np.arccos(normals[:, :, 2])
        normal_phi = np.arctan2(normals[:, :, 1], normals[:, :, 0])

        view_mean_std = np.array(self.hydra_conf["model"]["view_mean_std"])

        theta = (data["views"][:, :, :, 1].cpu().numpy() + view_mean_std[0]) * view_mean_std[5] + normal_theta[:, :, np.newaxis]
        phi = (data["views"][:, :, :, 2].cpu().numpy() + view_mean_std[1]) * view_mean_std[6] + normal_phi[:, :, np.newaxis]

        dz = np.cos(theta)
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)

        view_dir = np.stack([dx, dy, dz], axis=3) * (data["views"][:, :, :, 3:4].cpu().numpy() + view_mean_std[3]) * view_mean_std[7] * 60
        centre_point_index = data["points"][:, :, 3].cpu().numpy()
        points = data["points"][:, :, :3].cpu().numpy()
        points = points + self.test_dataset.datasets[0].original_points[
            centre_point_index.reshape(-1).astype(np.int32)].reshape(points.shape)
        points = points * self.data_mean_std[3] + self.data_mean_std[:3]
        views = points[:, :, np.newaxis] + view_dir
        views = np.concatenate([views, -view_dir, data["views"][:, :, :, 0:1].cpu().numpy()], axis=-1)

        self.log("Test Loss", total_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Recon Loss", recon_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Gt Loss", gt_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)

        return [results,
                data["point_attribute"],
                np.array(list(map(lambda x: self.dataset_name_dict[x], data["scene_name"]))),
                views[:, 0],  # num_points, num_views, 7
                np.concatenate([points, data["points"][:, :, 3:4].cpu().numpy()], axis=2)[:, 0, :]  # (x,y,z,idx)
                ]

    def test_epoch_end(self, outputs) -> None:
        error_mean_std = np.array(self.hydra_conf["model"]["error_mean_std"])

        prediction = (torch.cat(list(map(lambda x: x[0], outputs))).cpu().numpy() + error_mean_std[:2])*error_mean_std[2:]
        point_attribute = torch.cat(list(map(lambda x: x[1], outputs))).cpu().numpy()
        names = np.concatenate(list(map(lambda x: x[2], outputs)))
        if len(self.dataset_name_dict) == 1:
            views = np.concatenate(list(map(lambda x: x[3], outputs)))
            points = np.concatenate(list(map(lambda x: x[4], outputs)))

        log_str = ""
        mean_spearman = 0
        mean_smith_spearmanr_factor = 0
        spearman_dict = {}
        for scene_item in self.dataset_name_dict:
            predicted_acc = prediction[names == self.dataset_name_dict[scene_item]][:, 0, 0]
            predicted_com = prediction[names == self.dataset_name_dict[scene_item]][:, 0, 1]
            smith_error = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0, 0]
            gt_acc = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0, 0]
            gt_com = point_attribute[names == self.dataset_name_dict[scene_item]][:, 0, 1]
            if not self.hparams["model"]["involve_img"]:
                spearmanr_factor = stats.spearmanr(
                    predicted_acc[gt_acc != -1],
                    gt_acc[gt_acc != -1]
                )[0]
                smith_spearmanr_factor = stats.spearmanr(
                    smith_error[gt_acc != -1],
                    gt_acc[gt_acc != -1]
                )[0]
            else:
                spearmanr_factor = stats.spearmanr(
                    predicted_com[gt_com != -1],
                    gt_com[gt_com != -1]
                )[0]
                smith_spearmanr_factor = stats.spearmanr(
                    smith_error[gt_com != -1],
                    gt_com[gt_com != -1]
                )[0]

            spearman_dict[scene_item] = spearmanr_factor
            mean_spearman += spearmanr_factor
            mean_smith_spearmanr_factor += smith_spearmanr_factor
            log_str += "{:<35}: {:.2f}  ".format(scene_item, spearmanr_factor)
            log_str += "Smith_{:<35}: {:.2f}  \n".format(scene_item, smith_spearmanr_factor)

            if len(self.dataset_name_dict) == 1:
                views_item = views[names == self.dataset_name_dict[scene_item]]
                points_item = points[names == self.dataset_name_dict[scene_item]]
                output_test_with_pc_and_views(predicted_acc,
                                              predicted_com,
                                              gt_acc,
                                              gt_com,
                                              smith_error,
                                              self.test_dataset.datasets[
                                                  self.dataset_name_dict[scene_item]].point_attribute.shape[0],
                                              views_item,
                                              points_item
                                              )

        mean_spearman = mean_spearman / len(self.dataset_name_dict)
        mean_smith_spearmanr_factor = mean_smith_spearmanr_factor / len(self.dataset_name_dict)
        self.log("Test mean spearman", mean_spearman, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("Test smith mean spearman", mean_smith_spearmanr_factor, prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)
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

    """
    v_points
    v_views
    v_visibility
    """

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
                      strategy=DDPStrategy(find_unused_parameters=False) if v_cfg["trainer"].gpu > 1 else None,
                      accelerator="gpu" if v_cfg["trainer"].gpu > 0 else "cpu",
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
        model.load_state_dict(state_dict, strict=True)

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
