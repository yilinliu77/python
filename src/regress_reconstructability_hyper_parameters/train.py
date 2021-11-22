import shutil
import sys, os

import cv2
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

import torchsort
# from torchsort import soft_rank

from scipy import stats

from src.regress_reconstructability_hyper_parameters.preprocess_data import preprocess_data


class Regress_hyper_parameters(pl.LightningModule):
    def __init__(self, hparams):
        super(Regress_hyper_parameters, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate

        model_module = __import__("src")
        model_module = getattr(model_module,"regress_reconstructability_hyper_parameters")
        model_module = getattr(model_module,"model")
        f = getattr(model_module,self.hydra_conf["model"]["model_name"])

        self.model = f(hparams)

    def forward(self, v_data):
        data = self.model(v_data)
        return data

    def train_dataloader(self):
        self.train_dataset = Regress_hyper_parameters_dataset_with_imgs(self.hydra_conf, "training",)

        DataLoader_chosed = DataLoader if self.hydra_conf["trainer"]["gpu"] > 0 else FastDataLoader
        return DataLoader_chosed(self.train_dataset,
                                 batch_size=1,
                                 num_workers=self.hydra_conf["trainer"].num_worker,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 collate_fn=self.train_dataset.collate_fn,
                                 )

    def val_dataloader(self):
        self.valid_dataset = Regress_hyper_parameters_dataset_with_imgs(self.hydra_conf, "validation",)

        return DataLoader(self.valid_dataset,
                          batch_size=self.hydra_conf["trainer"]["batch_size"],
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.valid_dataset.collate_fn,
                          )

    def test_dataloader(self):
        self.test_dataset = Regress_hyper_parameters_dataset_with_imgs(self.hydra_conf, "test",)

        return DataLoader(self.test_dataset,
                          batch_size=self.hydra_conf["trainer"]["batch_size"],
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.test_dataset.collate_fn
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

        loss, gt_spearman, num_valid_point = self.model.loss(data["point_attribute"], results)

        self.log("Training Predict spearman",loss, prog_bar=False,logger=True,on_step=False,on_epoch=True)
        self.log("Training num valid point",num_valid_point, prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log("Training spearman baseline",gt_spearman, prog_bar=False,logger=True,on_step=True,on_epoch=True)
        self.log("Training Loss",loss, prog_bar=True,logger=True,on_step=False,on_epoch=True)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            pass

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        results = self.forward(data)

        loss, gt_spearman, num_valid_point = self.model.loss(data["point_attribute"], results)

        self.log("Validation Loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation Predict spearman", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Validation num valid point", num_valid_point, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return torch.cat([results,data["point_attribute"][:,:,2:3]],dim=2)

    def validation_epoch_end(self, outputs) -> None:
        result = torch.cat(outputs,dim=0).cpu().detach().numpy()
        spearmanr_factors = []
        for id_item in range(result.shape[0]):
            spearmanr_factor = stats.spearmanr(result[id_item][:,0],result[id_item][:,1])[0]
            spearmanr_factors.append(spearmanr_factor)
        self.log("Validation spearman baseline",np.mean(spearmanr_factors),prog_bar=True, logger=True, on_step=False, on_epoch=True)
        pass

    def test_step(self, batch, batch_idx)->None:
        data = batch
        results = self.forward(data)

        loss, gt_spearman, num_valid_point = self.model.loss(data["point_attribute"], results)

        self.log("Test Loss", loss, prog_bar=True, logger=False, on_step=False, on_epoch=True)

        return torch.cat([results, data["point_attribute"][:, :, 2:3], data["points"][:,:,3:4]], dim=2)

    def test_epoch_end(self, outputs) -> None:
        result = torch.cat(outputs, dim=0).cpu().detach().numpy()
        result = result.reshape(-1,result.shape[-1])
        result = result[np.argsort(result[:,2])]
        sorted_group = np.split(result[:, :3], np.unique(result[:, 2], return_index=True)[1][1:])
        whole_points_prediction_error = np.zeros((self.test_dataset.point_attribute.shape[0],2),dtype=np.float32)
        print("Merge the duplication and calculate the spearman")
        for id_item in tqdm(range(len(sorted_group))):
            whole_points_prediction_error[int(sorted_group[id_item][0][2]),0] = np.mean(sorted_group[id_item][:,0])
            whole_points_prediction_error[int(sorted_group[id_item][0][2]),1] = sorted_group[id_item][0,1]
        print("{} points are not covered by the sampling".format(np.all(whole_points_prediction_error==0,axis=1).sum()))
        spearmanr_factor = stats.spearmanr(whole_points_prediction_error[:, 0], whole_points_prediction_error[:, 1])[0]
        self.log("Test spearman", spearmanr_factor, prog_bar=True, logger=True, on_step=False,
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

    if v_cfg["model"].is_preprocess:
        if not os.path.exists(v_cfg["model"].preprocess_path):
            os.mkdir(v_cfg["model"].preprocess_path)
        view, view_pair, point_attribute, view_paths = preprocess_data(
            v_cfg["model"].target_data_dir,
            v_cfg["model"].error_point_cloud_dir,
            v_cfg["model"].img_dir
        )
        np.savez_compressed(os.path.join(v_cfg["model"].preprocess_path, "views"), view)
        np.savez_compressed(os.path.join(v_cfg["model"].preprocess_path, "view_pairs"), view_pair)
        np.savez_compressed(os.path.join(v_cfg["model"].preprocess_path, "point_attribute"), point_attribute)
        np.savez_compressed(os.path.join(v_cfg["model"].preprocess_path, "view_paths"), view_paths)

        print("Pre-compute data done")

    early_stop_callback = EarlyStopping(
        patience=100,
        monitor="Validation Loss"
    )

    model_check_point = ModelCheckpoint(
        monitor='Validation Loss',
        save_top_k=3,
        save_last=True
    )

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, weights_summary=None,
                      accelerator="ddp" if v_cfg["trainer"].gpu > 1 else None,
                      # early_stop_callback=early_stop_callback,
                      callbacks=[model_check_point],
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=500,
                      gradient_clip_val=0.1,
                      check_val_every_n_epoch=5
                      )

    model = Regress_hyper_parameters(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        model.load_state_dict(torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"], strict=True)
    if v_cfg["trainer"].auto_lr_find:
        trainer.tune(model)
        print(model.learning_rate)
    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
