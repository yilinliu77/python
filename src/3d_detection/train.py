import shutil
import sys, os

import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append("./")
sys.path.append(os.path.join(os.getcwd(), "thirdparty/visualDet3D"))
sys.path.append(os.path.join(os.getcwd(), "thirdparty/kitti-object-eval-python"))

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

from visualDet3D.data.kitti.utils import write_result_to_file


class Mono_det_3d(pl.LightningModule):
    def __init__(self, hparams):
        super(Mono_det_3d, self).__init__()
        self.hparams = hparams

        self.model = Yolo3D(self.hparams)
        self.dataset_builder = KittiMonoDataset

        # self.test_loss = pl.metrics.Accuracy()
        self.evaluate_root = os.path.join(
            hydra.utils.get_original_cwd(),
            hparams["det_3d"]["preprocessed_path"],
            "output/")
        if os.path.exists(self.evaluate_root):
            shutil.rmtree(self.evaluate_root)
        os.makedirs(self.evaluate_root)
        self.evaluate_index = [item.strip() for item in open(os.path.join(hydra.utils.get_original_cwd(),
                                                                          self.hparams["trainer"]["valid_split"]
                                                                          )).readlines()]

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
        self.valid_dataset = self.dataset_builder(self.hparams, "validation")

        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.hparams["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.valid_dataset.collate_fn
                          )

    def test_dataloader(self):
        self.test_dataset = self.dataset_builder(self.hparams, "validation")
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=self.hparams["trainer"].num_worker,
                          drop_last=False,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.test_dataset.collate_fn
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams["trainer"].learning_rate, )
        return {
            'optimizer': optimizer,
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=30, eta_min=3e-5),
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

    def evaluate(self, v_data, v_results, v_id):
        bbox_2d=self.model.rectify_2d_box(v_results["bboxes"][:, :4],v_data['original_calib'][0],v_data['calib'][0])

        bbox_3d_state_3d = torch.cat([v_results["bboxes"][:, 6:12], v_results["bboxes"][:, 13:14]], dim=1)

        write_result_to_file(self.evaluate_root,
                             # int(self.evaluate_index[v_id]),
                             v_id,
                             v_results["scores"],
                             bbox_2d,
                             bbox_3d_state_3d,
                             v_results["bboxes"][:, 12],
                             ["Car" for _ in v_results["scores"]])

    def validation_step(self, batch, batch_idx):
        data = batch

        results = self.forward(data)
        self.validation_example = {
            "bbox": results["bboxes"].cpu().numpy(),
            "gt_bbox": data["bbox2d"][0].cpu().numpy(),
            "image": data["image"][0].cpu().permute(1, 2, 0).numpy()
        }
        if self.current_epoch % 10 == 9:
            self.evaluate(data, results, batch_idx)
        return {
            'val_loss': results["cls_loss"] + results["reg_loss"],
            'val_cls_loss': results["cls_loss"],
            'val_reg_loss': results["reg_loss"],
        }

    def validation_epoch_end(self, outputs):
        avg_loss_total = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_cls = torch.stack([x['val_cls_loss'] for x in outputs]).mean()
        avg_loss_reg = torch.stack([x['val_reg_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss_total.item(), prog_bar=True, on_epoch=True)
        self.log("val_cls_loss", avg_loss_cls.item(), prog_bar=True, on_epoch=True)
        self.log("val_reg_loss", avg_loss_reg.item(), prog_bar=True, on_epoch=True)

        # Visualize the example
        img = self.validation_example["image"]
        img = np.clip((img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).astype(
            np.uint8)
        viz_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
        # GT
        if self.validation_example["gt_bbox"].shape[0] > 0:
            for box in self.validation_example["gt_bbox"]:
                pts = list(map(int, box[0:4]))
                viz_img = cv2.rectangle(viz_img,
                                        (pts[0], pts[1]),
                                        (pts[2], pts[3]),
                                        (255, 0, 0),
                                        5
                                        )

        # Prediction
        if self.validation_example["bbox"].shape[0] > 0:
            for box in self.validation_example["bbox"]:
                pts = list(map(int, box[0:4]))
                viz_img = cv2.rectangle(viz_img,
                                        (pts[0], pts[1]),
                                        (pts[2], pts[3]),
                                        (0, 255, 0),
                                        3
                                        )
        self.logger.experiment.add_image("Validation_example", viz_img, dataformats='HWC', global_step=self.global_step)
        if self.current_epoch % 10 == 9 and not self.trainer.running_sanity_check:
            from visualDet3D.evaluator.kitti.evaluate import evaluate
            result_texts = evaluate(
                label_path=os.path.join(self.hparams["trainer"]["valid_dataset"], 'label_2'),
                result_path=self.evaluate_root,
                label_split_file=os.path.join(hydra.utils.get_original_cwd(),
                                              self.hparams["trainer"]["valid_split"]
                                              ),
                current_classes=[0],
            )
            self.logger.experiment.add_text("Validation_kitti", result_texts[0], global_step=self.global_step)
            if os.path.exists(self.evaluate_root):
                shutil.rmtree(self.evaluate_root)
            os.makedirs(self.evaluate_root)

    def test_step(self, batch, batch_idx):
        data = batch

        results = self.forward(data)

        self.evaluate(data, results, batch_idx)

        return {
            'val_loss': results["cls_loss"] + results["reg_loss"],
            'val_cls_loss': results["cls_loss"],
            'val_reg_loss': results["reg_loss"],
        }

    def test_epoch_end(self, outputs):
        # from visualDet3D.evaluator.kitti.evaluate import evaluate
        from evaluate import evaluate
        result_texts = evaluate(
            label_path=os.path.join(self.hparams["trainer"]["valid_dataset"], 'label_2'),
            result_path=self.evaluate_root,
            label_split_file=os.path.join(hydra.utils.get_original_cwd(),
                                          self.hparams["trainer"]["valid_split"]),
            current_class=0,
            coco=False
        )
        # print(result_texts[0])


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
                      # check_val_every_n_epoch=10
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
