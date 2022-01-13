import math
import shutil
import sys, os

import cv2
from plyfile import PlyData, PlyElement
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

from shared.fast_dataloader import FastDataLoader
from src.regress_reconstructability_hyper_parameters.dataset import Regress_hyper_parameters_dataset, \
    Regress_hyper_parameters_dataset_with_imgs
from src.regress_reconstructability_hyper_parameters.model import Regress_hyper_parameters_Model, Brute_force_nn, \
    Correlation_nn

# import torchsort
# from torchsort import soft_rank

from scipy import stats


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# v_data: Predict reconstructability, Predict inconsistency, valid_flag, GT error, inconsistency (0 is consistent), Point index, x, y, z
def output_test(v_data, v_num_total_points):
    predict_result = v_data.reshape(-1, v_data.shape[-1])
    invalid_mask = predict_result[:, 4]  # Filter out the point with gt reconstructability = 0
    print("{}/{} predictions are inconsistent thus do not consider to calculate the spearman correlation".format(
        invalid_mask.sum(), predict_result.shape[0]))
    sorted_index = np.argsort(predict_result[:, 5])  # Sort according to the point index
    predict_result = predict_result[sorted_index]
    sorted_group = np.split(predict_result, np.unique(predict_result[:, 5], return_index=True)[1][1:])
    whole_points_prediction_error = np.zeros((v_num_total_points, 4), dtype=np.float32)
    print("Merge the duplication and calculate the spearman")
    for id_item in tqdm(range(len(sorted_group))):
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 0] = np.mean(sorted_group[id_item][:, 0])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 1] = sorted_group[id_item][0, 3]
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 2] = np.mean(sorted_group[id_item][:, 1])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 3] = sorted_group[id_item][0, 4]

    # Calculate consistency accuracy
    predicted_good_point_mask = sigmoid(whole_points_prediction_error[:, 2]) > 0.5
    gt_good_point_mask = 1 - whole_points_prediction_error[:, 3]
    accuracy = (predicted_good_point_mask == gt_good_point_mask).sum() / gt_good_point_mask.shape[0]
    # Calculate spearman factor
    consistent_point_mask = np.logical_and(whole_points_prediction_error[:, 0] != 0,whole_points_prediction_error[:, 3] == 0)
    print("Consider {}/{} points to calculate the spearman".format(
        consistent_point_mask.sum(),
        whole_points_prediction_error.shape[0]))

    spearmanr_factor = stats.spearmanr(
        whole_points_prediction_error[consistent_point_mask][:, 0],
        whole_points_prediction_error[consistent_point_mask][:, 1]
    )[0]

    return spearmanr_factor,accuracy,whole_points_prediction_error # predict_recon, gt_recon, predict_consitency, gt_inconsistency

# v_data: Predict reconstructability, Predict inconsistency, valid_flag, GT error, inconsistency (0 is consistent), Point index, x, y, z
def output_test_with_pc_and_views(v_data, v_num_total_points):
    prediction_result = torch.cat([item[0] for item in v_data], dim=0)
    spearmanr_factor,accuracy,whole_points_prediction_error = output_test(prediction_result.cpu().numpy(),v_num_total_points)

    prediction_result_reshape = prediction_result.reshape([-1, prediction_result.shape[-1]])
    views = np.concatenate([item[1] for item in v_data], axis=0)
    views = views.reshape([-1, views.shape[2], views.shape[3]])
    # Write views
    vertexes = np.zeros((v_num_total_points, 3), dtype=np.float32) # x, y, z
    for id_view, view in enumerate(tqdm(views)):
        id_point = int(prediction_result_reshape[id_view,5])
        filename = "temp/test_scene_output/{}.txt".format(int(id_point))
        if os.path.exists(filename):
            continue

        vertexes[id_point][0:3] = prediction_result_reshape[id_view,6:9].cpu().numpy()

        # Write views
        with open(filename, "w") as f:
            for item in view:
                if item[6] == 0:
                    break
                item[3:6] /= np.linalg.norm(item[3:6])

                pitch = math.asin(item[5])
                yaw = math.atan2(item[4], item[3])

                f.write("xxx.png,{},{},{},{},{},{}\n".format(
                    item[0], item[1], item[2],
                    pitch / math.pi * 180, 0, yaw / math.pi * 180,
                ))


    vertexes_describer = PlyElement.describe(np.array(
        [(item[0], item[1], item[2], item[3], item[4], item[5], 1-item[6]) for item in
         np.concatenate([vertexes,whole_points_prediction_error],axis=1)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
               ('Predict_Recon', 'f8'), ('GT_Error', 'f8'),
               ('Predict_Consistency', 'f8'), ('GT_Consistency', 'f8')]), 'vertex')

    PlyData([vertexes_describer]).write('temp/test_scene_output/whole_point.ply')

    return spearmanr_factor, accuracy


class Regress_hyper_parameters(pl.LightningModule):
    def __init__(self, hparams):
        super(Regress_hyper_parameters, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        # self.log(...,batch_size=self.batch_size)

        model_module = __import__("src")
        model_module = getattr(model_module,"regress_reconstructability_hyper_parameters")
        model_module = getattr(model_module,"model")
        f = getattr(model_module,self.hydra_conf["model"]["model_name"])

        self.model = f(hparams)

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def train_dataloader(self):
        dataset_paths = self.hydra_conf["trainer"]["train_dataset"].split("*")
        datasets=[]
        for dataset_path in dataset_paths:
            datasets.append(Regress_hyper_parameters_dataset(dataset_path, self.hydra_conf,
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
                                 collate_fn=Regress_hyper_parameters_dataset.collate_fn,
                                 )

    def val_dataloader(self):
        use_part_dataset_to_validate = len(self.hydra_conf["trainer"]["train_dataset"].split("*")) == 1

        dataset_paths = self.hydra_conf["trainer"]["valid_dataset"].split("*")
        datasets=[]
        for dataset_path in dataset_paths:
            datasets.append(Regress_hyper_parameters_dataset(dataset_path, self.hydra_conf,
                                                                       "validation" if use_part_dataset_to_validate else "testing",))
        self.valid_dataset = torch.utils.data.ConcatDataset(datasets)
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=Regress_hyper_parameters_dataset.collate_fn,
                          )

    def test_dataloader(self):
        self.test_dataset = Regress_hyper_parameters_dataset(self.hydra_conf["trainer"]["test_dataset"],
                                                                       self.hydra_conf, "testing",)

        return DataLoader(self.test_dataset,
                          batch_size=self.hydra_conf["trainer"]["batch_size"],
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=Regress_hyper_parameters_dataset.collate_fn
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch
        results = self.forward(data)

        error_loss, inconsitency_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Training Error Loss",error_loss, prog_bar=False,logger=True,on_step=False,on_epoch=True)
        self.log("Training Inconsitency Loss",inconsitency_loss, prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log("Training Loss",total_loss, prog_bar=True,logger=True,on_step=False,on_epoch=True)

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            pass

        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch
        results = self.forward(data)

        error_loss, inconsitency_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", total_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation Error Loss", error_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation Inconsitency Loss", inconsitency_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return torch.cat([results, data["point_attribute"][:, :, [2,6]], data["points"][:,:,3:4]], dim=2)

    def validation_epoch_end(self, outputs) -> None:
        spearmanr_factor,accuracy,whole_points_prediction_error  = output_test(
            torch.cat([item for item in outputs], dim=0).cpu().numpy(), self.valid_dataset.datasets[0].point_attribute.shape[0])

        self.log("Validation spearman", spearmanr_factor, prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)
        self.log("Validation accuracy", accuracy, prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)

        pass

        if not self.trainer.sanity_checking:
            for dataset in self.train_dataset.datasets:
                dataset.sample_points_to_different_patches()

        pass

    def on_test_epoch_start(self) -> None:
        self.data_mean_std = np.load(os.path.join(self.test_dataset.data_root,"../data_centralize.npz"))["arr_0"]
        if os.path.exists("temp/test_scene_output"):
            shutil.rmtree("temp/test_scene_output")
        os.mkdir("temp/test_scene_output")
        pass

    def test_step(self, batch, batch_idx) -> None:
        data = batch

        results = self.forward(data)

        error_loss, inconsitency_loss, total_loss = self.model.loss(data["point_attribute"], results)

        normals = data["point_attribute"][:,:,7:10].cpu().numpy()
        normal_theta = np.arccos(normals[:,:,2])
        normal_phi = np.arctan2(normals[:,:,1], normals[:,:,0])

        theta = data["views"][:, :, :, 1].cpu().numpy() + normal_theta[:,:,np.newaxis]
        phi = data["views"][:, :, :, 2].cpu().numpy() + normal_phi[:,:,np.newaxis]

        dz = np.cos(theta)
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)

        view_dir = np.stack([dx,dy,dz],axis=3) * data["views"][:,:,:,3:4].cpu().numpy() * 60
        # centre_point_index = data["points"][:,:,4].cpu().numpy()
        points = data["points"][:,:,:3].cpu().numpy()
        # points = points + self.test_dataset.original_points[centre_point_index.reshape(-1).astype(np.int32)].reshape(points.shape)
        points = points * self.data_mean_std[3] + self.data_mean_std[:3]
        views = points[:,:,np.newaxis] + view_dir
        views=np.concatenate([views,-view_dir,data["views"][:,:,:,0:1].cpu().numpy()],axis=-1)

        self.log("Test Loss", total_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Recon Loss", error_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Inconsistency Loss", inconsitency_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)

        return torch.cat([
            results,  # Predict reconstructability and inconsistency
            data["point_attribute"][:, :, [2,6]], # Avg error and Is point with 0 reconstructability
            data["points"][:,:,3:4], # Point index, centre point index
            results.new(points)
        ], dim=2), views # x,y,z, dx,dy,dz, valid

    def test_epoch_end(self, outputs) -> None:
        spearmanr_factor, accuracy = output_test_with_pc_and_views(outputs,self.test_dataset.point_attribute.shape[0])
        self.log("Test spearman", spearmanr_factor, prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)
        self.log("Test accuracy", accuracy, prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)
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

@hydra.main(config_name="test.yaml")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

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
                      accelerator="ddp" if v_cfg["trainer"].gpu > 1 else None,
                      # early_stop_callback=early_stop_callback,
                      callbacks=[model_check_point],
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=3000,
                      gradient_clip_val=0.1,
                      check_val_every_n_epoch=1
                      )

    model = Regress_hyper_parameters(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        for item in list(state_dict.keys()):
            if "point_feature_extractor" in item:
                state_dict.pop(item)
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
    main()
