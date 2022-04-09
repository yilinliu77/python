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
from tqdm.contrib.concurrent import thread_map

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

# v_data: Predict recon error, Predict gt error, smith 18 recon, avg recon error, avg gt error, inconsistency (0 is consistent), Point index, x, y, z
def output_test(v_data, v_num_total_points):
    predict_result = v_data.reshape(-1, v_data.shape[-1])
    invalid_mask = predict_result[:, 5]  # Filter out the point with gt reconstructability = 0
    print("{}/{} predictions are inconsistent thus do not consider to calculate the spearman correlation".format(
        invalid_mask.sum(), predict_result.shape[0]))
    sorted_index = np.argsort(predict_result[:, 6])  # Sort according to the point index
    predict_result = predict_result[sorted_index]
    sorted_group = np.split(predict_result, np.unique(predict_result[:, 6], return_index=True)[1][1:])
    whole_points_prediction_error = np.zeros((v_num_total_points, 5), dtype=np.float32) # predict_recon, smith recon, gt_recon, predict_consitency, gt_inconsistency,
    print("Merge the duplication and calculate the spearman")
    for id_item in tqdm(range(len(sorted_group))):
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 0] = np.mean(sorted_group[id_item][:, 0])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 1] = np.mean(sorted_group[id_item][:, 2])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 2] = np.mean(sorted_group[id_item][:, 1])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 3] = np.mean(sorted_group[id_item][:, 3])
        whole_points_prediction_error[int(sorted_group[id_item][0][5]), 4] = np.mean(sorted_group[id_item][:, 4])

    spearmanr_factor = stats.spearmanr(
        whole_points_prediction_error[whole_points_prediction_error[:,1]!=-1][:, 0],
        whole_points_prediction_error[whole_points_prediction_error[:,1]!=-1][:, 2]
    )[0]

    smith_spearmanr_factor = stats.spearmanr(
        whole_points_prediction_error[whole_points_prediction_error[:,1]!=-1][:, 1],
        whole_points_prediction_error[whole_points_prediction_error[:,1]!=-1][:, 2]
    )[0]

    return spearmanr_factor,smith_spearmanr_factor,whole_points_prediction_error

# v_data: Predict recon error, Predict gt error, smith 18 recon, avg recon error, avg gt error, inconsistency (0 is consistent), Point index, x, y, z
def output_test_with_pc_and_views(v_data, v_num_total_points):
    prediction_result = torch.cat([item[0] for item in v_data], dim=0)
    # spearmanr, smith_spearmanr, whole_points_prediction_error = output_test(prediction_result.cpu().numpy(), v_num_total_points)

    prediction_result=prediction_result.cpu().numpy()
    predicted_recon = prediction_result[:,0,0]
    predicted_gt = prediction_result[:,0,1]
    smith_recon = prediction_result[:,0,2]
    gt_recon = prediction_result[:,0,3]
    gt_gt = prediction_result[:,0,4]

    recon_spearmanr = stats.spearmanr(
        predicted_recon[gt_recon != -1],
        gt_recon[gt_recon!= -1]
    )[0]
    gt_spearmanr = stats.spearmanr(
        predicted_gt[gt_gt != -1],
        gt_gt[gt_gt != -1]
    )[0]
    smith_recon_spearmanr = stats.spearmanr(
        smith_recon[gt_recon != -1],
        gt_recon[gt_recon != -1]
    )[0]
    smith_gt_spearmanr = stats.spearmanr(
        smith_recon[gt_gt != -1],
        gt_gt[gt_gt != -1]
    )[0]

    prediction_result_reshape = prediction_result.reshape([-1, prediction_result.shape[-1]])
    views = np.concatenate([item[1] for item in v_data], axis=0)
    views = views.reshape([-1, views.shape[2], views.shape[3]])
    # Write views
    vertexes = np.zeros((v_num_total_points, 3), dtype=np.float32) # x, y, z

    def write_views_to_txt_file(v_args):
        id_view, view = v_args
        id_point = int(prediction_result_reshape[id_view, 6])
        filename = "temp/test_scene_output/{}.txt".format(int(id_point))
        if os.path.exists(filename):
            return

        vertexes[id_point][0:3] = prediction_result_reshape[id_view, 7:10]

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

    thread_map(write_views_to_txt_file, enumerate(views),max_workers=32)

    vertexes_describer = PlyElement.describe(np.array(
        [(item[0], item[1], item[2], item[3], item[4], item[5], item[6]) for item in
         np.concatenate([vertexes,predicted_recon[:,np.newaxis],gt_recon[:,np.newaxis],predicted_gt[:,np.newaxis],gt_gt[:,np.newaxis]],axis=1)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
               ('Predict_Recon', 'f8'), ('GT_Recon', 'f8'),
               ('Predict_Gt', 'f8'), ('GT_Gt', 'f8')]), 'vertex')

    PlyData([vertexes_describer]).write('temp/test_scene_output/whole_point.ply')

    return recon_spearmanr, gt_spearmanr, smith_recon_spearmanr, smith_gt_spearmanr


class Regress_hyper_parameters(pl.LightningModule):
    def __init__(self, hparams):
        super(Regress_hyper_parameters, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        # self.log(...,batch_size=self.batch_size)
        self.save_hyperparameters(hparams)
        model_module = __import__("src")
        model_module = getattr(model_module,"regress_reconstructability_hyper_parameters")
        model_module = getattr(model_module,"model")
        f = getattr(model_module,self.hydra_conf["model"]["model_name"])

        dataset_module = __import__("src")
        dataset_module = getattr(dataset_module,"regress_reconstructability_hyper_parameters")
        dataset_module = getattr(dataset_module,"dataset")
        self.dataset_builder = getattr(dataset_module,self.hydra_conf["trainer"]["dataset_name"])
        self.model = f(hparams)

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def train_dataloader(self):
        dataset_paths = self.hydra_conf["trainer"]["train_dataset"].split("*")
        datasets=[]
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
        datasets=[]
        for dataset_path in dataset_paths:
            datasets.append(self.dataset_builder(dataset_path, self.hydra_conf,
                                                                       "validation" if use_part_dataset_to_validate else "testing",))
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
                self.dataset_builder(dataset_path, self.hydra_conf,"testing"))

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
        # optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch
        results,weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Training Recon Loss",recon_loss.detach(), prog_bar=False,logger=True,on_step=False,on_epoch=True)
        self.log("Training Gt Loss",gt_loss.detach(), prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log("Training Loss",total_loss.detach(), prog_bar=True,logger=True,on_step=False,on_epoch=True)

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
        results,weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", total_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation Recon Loss", recon_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation Gt Loss", gt_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {
            "predicted recon error": results[:,:,0],
            "predicted gt error": results[:,:,1],
            "gt recon error": data["point_attribute"][:, :, 1],
            "gt gt error": data["point_attribute"][:, :, 2],
            "index": data["points"][:,:,3:4],
            "scene_name":  data["scene_name"]
        }

    def validation_epoch_end(self, outputs) -> None:
        scene_dict = {}
        for prediction_result in outputs:
            if not self.hparams["model"]["involve_img"]:
                valid_mask = (prediction_result["gt recon error"]!=-1).bool()
            else:
                valid_mask = (prediction_result["gt gt error"]!=-1).bool()
            for scene in prediction_result["scene_name"]:
                if scene not in scene_dict:
                    scene_dict[scene] = [[],[]]
                if not self.hparams["model"]["involve_img"]:
                    scene_dict[scene][0].append(prediction_result["predicted recon error"][valid_mask])
                    scene_dict[scene][1].append(prediction_result["gt recon error"][valid_mask])
                else:
                    scene_dict[scene][0].append(prediction_result["predicted gt error"][valid_mask])
                    scene_dict[scene][1].append(prediction_result["gt gt error"][valid_mask])

        spearman_dict = {}

        log_str=""
        mean_spearman=0
        for scene_item in scene_dict:
            predicted_error = torch.cat(scene_dict[scene_item][0],dim=0).cpu().numpy()
            gt_error = torch.cat(scene_dict[scene_item][1],dim=0).cpu().numpy()
            spearmanr_factor = stats.spearmanr(
                predicted_error,
                gt_error
            )[0]
            spearman_dict[scene_item] = spearmanr_factor
            mean_spearman+=spearmanr_factor
            log_str += "{:<35}: {:.2f}  \n".format(scene_item,spearmanr_factor)
        mean_spearman = mean_spearman / len(scene_dict)
        self.trainer.logger.experiment.add_text("Validation spearman",log_str,global_step=self.trainer.global_step)
        # spearmanr_factor,accuracy,whole_points_prediction_error = output_test(
        #     torch.cat([item for item in outputs], dim=0).cpu().numpy(), self.valid_dataset.datasets[0].point_attribute.shape[0])

        self.log("Validation mean spearman", mean_spearman, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # if not self.trainer.sanity_checking:
        #     for dataset in self.train_dataset.datasets:
        #         dataset.sample_points_to_different_patches()

        pass

    def on_test_epoch_start(self) -> None:
        self.data_mean_std = np.load(os.path.join(self.test_dataset.datasets[0].data_root,"../data_centralize.npz"))["arr_0"]
        if os.path.exists("temp/test_scene_output"):
            shutil.rmtree("temp/test_scene_output")
        os.mkdir("temp/test_scene_output")
        pass

    def test_step(self, batch, batch_idx) -> None:
        data = batch

        results,weights = self.forward(data)

        recon_loss, gt_loss, total_loss = self.model.loss(data["point_attribute"], results)

        normals = data["point_attribute"][:,:,7:10].cpu().numpy()
        normal_theta = np.arccos(normals[:,:,2])
        normal_phi = np.arctan2(normals[:,:,1], normals[:,:,0])

        theta = data["views"][:, :, :, 1].cpu().numpy() + normal_theta[:,:,np.newaxis]
        phi = data["views"][:, :, :, 2].cpu().numpy() + normal_phi[:,:,np.newaxis]

        dz = np.cos(theta)
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)

        view_dir = np.stack([dx,dy,dz],axis=3) * data["views"][:,:,:,3:4].cpu().numpy() * 60
        centre_point_index = data["points"][:,:,3].cpu().numpy()
        points = data["points"][:,:,:3].cpu().numpy()
        points = points + self.test_dataset.datasets[0].original_points[centre_point_index.reshape(-1).astype(np.int32)].reshape(points.shape)
        points = points * self.data_mean_std[3] + self.data_mean_std[:3]
        views = points[:,:,np.newaxis] + view_dir
        views=np.concatenate([views,-view_dir,data["views"][:,:,:,0:1].cpu().numpy()],axis=-1)

        self.log("Test Loss", total_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Recon Loss", recon_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
        self.log("Test Gt Loss", gt_loss, prog_bar=True, logger=False, on_step=True, on_epoch=True)

        return torch.cat([
            results,  # Predict reconstructability and inconsistency
            data["point_attribute"][:, :, [0,1,2,5]], # smith18 recon, Avg recon error, Avg gt error and Is point with 0 reconstructability
            data["points"][:,:,3:4], # Point index, centre point index
            results.new(points)
        ], dim=2), views # x,y,z, dx,dy,dz, valid

    def test_epoch_end(self, outputs) -> None:
        recon_spearmanr, gt_spearmanr, smith_recon_spearmanr, smith_gt_spearmanr = \
            output_test_with_pc_and_views(outputs,self.test_dataset.datasets[0].point_attribute.shape[0])
        self.log("recon_spearmanr", recon_spearmanr, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True)
        self.log("gt_spearmanr", gt_spearmanr, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True)
        self.log("smith_recon_spearmanr", smith_recon_spearmanr, prog_bar=True, logger=False, on_step=False,
                 on_epoch=True)
        self.log("smith_gt_spearmanr", smith_gt_spearmanr, prog_bar=True, logger=False, on_step=False,
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
    main()
