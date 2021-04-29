import sys,os
sys.path.append("./")
sys.path.append(os.path.join(os.getcwd(),"thirdparty/visualDet3D"))

from argparse import ArgumentParser
from model import Yolo3D
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import KittiMonoDataset

class Mono_det_3d(pl.LightningModule):
    def __init__(self, hparams):
        super(Mono_det_3d, self).__init__()
        self.hparams = hparams

        self.model = Yolo3D(self.hparams)
        self.dataset_builder = KittiMonoDataset

        # self.test_loss = pl.metrics.Accuracy()

    def forward(self, v_data):
        data = self.model(v_data)

        return data

    def train_dataloader(self):
        self.train_dataset = self.dataset_builder(self.hparams)

        # dataset_path = self.hparams.train_dataset.split(";")
        # dataset = []
        # for item in dataset_path:
        #     dataset.append(self.dataset_builder(item,self.hparams, True))
        # self.train_dataset = torch.utils.data.ConcatDataset(dataset)
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams["trainer"].batch_size,
                          num_workers=self.hparams["trainer"].num_worker,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          collate_fn=self.train_dataset.collate_fn
                          )

    def val_dataloader(self):
        # dataset_path = self.hparams.valid_dataset.split(";")
        # dataset = []
        # for item in dataset_path:
        #     dataset.append(self.dataset_builder(item,self.hparams, False))
        # self.valid_dataset = torch.utils.data.ConcatDataset(dataset)
        self.valid_dataset = self.dataset_builder(self.hparams,"validation")

        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams["trainer"].batch_size,
                          num_workers=self.hparams["trainer"].num_worker,
                          drop_last=True,
                          pin_memory=True,
                          collate_fn=self.valid_dataset.collate_fn
                          )

    def test_dataloader(self):
        self.test_dataset = self.dataset_builder(self.hparams.test_dataset)
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams["trainer"].batch_size,
                          num_workers=self.hparams["trainer"].num_worker,
                          pin_memory=True,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams["trainer"].learning_rate,)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': ReduceLROnPlateau(optimizer, patience=10, verbose=True, min_lr=1e-7, threshold=1e-6,
            #                                   factor=0.5),
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch

        cls_loss, reg_loss = self.forward(data)

        self.log("training_cls", cls_loss, prog_bar=True, on_epoch=True)
        self.log("training_reg", reg_loss, prog_bar=True, on_epoch=True)

        return {
            'loss': reg_loss + cls_loss,
        }

    def validation_step(self, batch, batch_idx):
        data = batch

        cls_loss, reg_loss = self.forward(data)
        return {
            'val_loss': reg_loss + cls_loss,
            'val_cls_loss': cls_loss,
            'val_reg_loss': reg_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss_total = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_cls = torch.stack([x['val_cls_loss'] for x in outputs]).mean()
        avg_loss_reg = torch.stack([x['val_reg_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss_total.item(), prog_bar=True, on_epoch=True)
        self.log("val_cls_loss", avg_loss_cls.item(), prog_bar=True, on_epoch=True)
        self.log("val_reg_loss", avg_loss_reg.item(), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data = batch

        prediction_training_data = self.forward(data)

        return {

        }

    def test_epoch_end(self, outputs):
        if self.hparams.save_points:
            np.save("temp/points.npy", self.model.test_coords.cpu().numpy())

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        chamfer_distance = np.stack([x['chamfer_distance'] for x in outputs]).mean()
        skip_nums = np.sum([x['skip'] for x in outputs])
        loss_svr = torch.stack([x['loss_svr'] for x in outputs]).mean()

        return {
            "chamfer_distance": chamfer_distance * 1000,
            "avg_err_points": loss,
            "skip": skip_nums,
            "loss_svr": loss_svr,
        }


@hydra.main(config_name=".")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)
    # set_start_method('spawn')

    early_stop_callback = EarlyStopping(
        patience=100,
        monitor="val_loss"
    )

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, weights_summary=None,
                      distributed_backend="ddp" if v_cfg["trainer"].gpu > 1 else None,
                      # early_stop_callback=early_stop_callback,
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=5000,
                      )

    model = Mono_det_3d(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        model.load_state_dict(torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"], strict=True)
    if v_cfg["trainer"].auto_lr_find:
        trainer.tune(model)
        print(model.hparams.learning_rate)
    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
