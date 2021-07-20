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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import KittiMonoDataset, KittiMonoTestDataset

from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.networks.utils import BBox3dProjector, BackProjection


class Mono_det_3d(pl.LightningModule):
    def __init__(self, hparams):
        super(Mono_det_3d, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"].learning_rate

        self.model = Yolo3D(self.hydra_conf)

        ### modified
        if hparams['trainer'].evaluate:
            self.dataset_builder = KittiMonoTestDataset
        else:
            self.dataset_builder = KittiMonoDataset
        ###

        self.evaluate_root = os.path.join(
            hydra.utils.get_original_cwd(),
            hparams["det_3d"]["preprocessed_path"],
            "output/")
        if os.path.exists(self.evaluate_root):
            shutil.rmtree(self.evaluate_root)
        os.makedirs(self.evaluate_root)

        if hparams['trainer'].evaluate:
            self.evaluate_index = [item.strip() for item in open(os.path.join(hydra.utils.get_original_cwd(),
                                                                              self.hydra_conf["trainer"]["test_split"]
                                                                              )).readlines()]
        else:
            self.evaluate_index = [item.strip() for item in open(os.path.join(hydra.utils.get_original_cwd(),
                                                                              self.hydra_conf["trainer"]["valid_split"]
                                                                              )).readlines()]
        ### modified
        self.backprojector = BackProjection().cuda()
        self.projector = BBox3dProjector().cuda()
        ###

    def forward(self, v_data):
        data = self.model(v_data)

        return data

    def train_dataloader(self):
        self.train_dataset = self.dataset_builder(self.hydra_conf, "training", self.model.train_preprocess)

        # dataset_path = self.hparams.train_dataset.split(";")
        # dataset = []
        # for item in dataset_path:
        #     dataset.append(self.dataset_builder(item,self.hparams, True))
        # self.train_dataset = torch.utils.data.ConcatDataset(dataset)
        return DataLoader(self.train_dataset,
                          batch_size=self.hydra_conf["trainer"].batch_size,
                          num_workers=self.hydra_conf["trainer"].num_worker,
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
        self.valid_dataset = self.dataset_builder(self.hydra_conf, "validation", self.model.test_preprocess)

        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.hydra_conf["trainer"].num_worker,
                          drop_last=True,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=self.valid_dataset.collate_fn
                          )

    def test_dataloader(self):
        self.test_dataset = self.dataset_builder(self.hydra_conf, "testing", self.model.test_preprocess)
        return DataLoader(self.test_dataset,
                          batch_size=1,
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
            'lr_scheduler': CosineAnnealingLR(optimizer, T_max=50., eta_min=3e-5),
            'monitor': 'Validation Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch

        cls_loss, reg_loss, loss_sep = self.forward(data)

        self.log_dict({"Training Classification": cls_loss,
                       "Training Regression": reg_loss,
                       "Training 2d": loss_sep[0],
                       "Training 3d_xyz": loss_sep[1],
                       "Training 3d_sin_cos": loss_sep[2],
                       "Training 3d_whl": loss_sep[3],
                       }, on_epoch=True, on_step=False)

        return {
            'loss': reg_loss + cls_loss,
        }

    """
    v_results: [x1, y1, x2, y2, cx, cy, z, w, h, l, alpha]
    """
    def evaluate(self, v_data, v_results, v_id):
        bbox_2d = self.model.rectify_2d_box(v_results["bboxes"][:, :4], v_data['original_calib'][0], v_data['calib'][0])

        P2 = v_data['calib'][0]
        bbox_3d_state = v_results["bboxes"][:, 4:]
        bbox_3d_state_3d = self.backprojector(bbox_3d_state,
                                              P2)  # [x3d, y3d, z, w, h, l, alpha]
        _, _, thetas = self.projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))  # Calculate theta

        write_result_to_file(self.evaluate_root,
                             int(self.evaluate_index[v_id]),
                             v_results["scores"],
                             bbox_2d,
                             bbox_3d_state_3d,
                             thetas,
                             ["Car" for _ in v_results["scores"]])

    def validation_step(self, batch, batch_idx):
        data = batch
        results = self.forward(data)
        self.validation_example = {
            "bbox": results["bboxes"].cpu().numpy(),
            "gt_bbox": data["bbox2d"][0].cpu().numpy(),
            "image": data["image"][0].cpu().permute(1, 2, 0).numpy()
        }
        self.evaluate(data, results, batch_idx)
        return {
            'Validation Loss': results["cls_loss"] + results["reg_loss"],
            "Validation Classification": results["cls_loss"],
            "Validation Regression": results["reg_loss"],
            "Validation 2d": results["loss_sep"][0],
            "Validation 3d_xyz": results["loss_sep"][1],
            "Validation 3d_sin_cos": results["loss_sep"][2],
            "Validation 3d_whl": results["loss_sep"][3],
        }

    def validation_epoch_end(self, outputs):
        # Visualize the example
        log_dict = {item: 0 for item in outputs[0]}
        for item in outputs:
            for key in log_dict:
                log_dict[key] += item[key]
        for key in log_dict:
            log_dict[key] /= len(outputs)
        self.log_dict(log_dict)

        # Visualize imgs
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
        if not self.trainer.running_sanity_check:
            from visualDet3D.evaluator.kitti.evaluate import evaluate
            result_texts = evaluate(
                label_path=os.path.join(self.hydra_conf["trainer"]["valid_dataset"], 'label_2'),
                result_path=self.evaluate_root,
                label_split_file=os.path.join(hydra.utils.get_original_cwd(),
                                              self.hydra_conf["trainer"]["valid_split"]
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


"""
    def test_epoch_end(self, outputs):
        from visualDet3D.evaluator.kitti.evaluate import evaluate
        # from evaluate import evaluate
        result_texts = evaluate(
            label_path=os.path.join(self.hydra_conf["trainer"]["valid_dataset"], 'label_2'),
            result_path=self.evaluate_root,
            label_split_file=os.path.join(hydra.utils.get_original_cwd(),
                                          self.hydra_conf["trainer"]["valid_split"]),
            current_class=0,
            coco=False
        )
        # print(result_texts[0])
"""

@hydra.main(config_name="kitti_fpn.yaml")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)
    # set_start_method('spawn')

    early_stop_callback = EarlyStopping(
        patience=100,
        monitor="Validation Loss"
    )

    model_check_point =  ModelCheckpoint(
        #dirpath="C:\\Users\\zihan\\Desktop\\checkpoints\\",
        monitor='Validation Loss',
        save_top_k=3,
        save_last=True
    )

    trainer = Trainer(gpus=v_cfg["trainer"].gpu, weights_summary=None,
                      distributed_backend="ddp" if v_cfg["trainer"].gpu > 1 else None,
                      #early_stop_callback=early_stop_callback,
                      callbacks=[model_check_point],
                      auto_lr_find="learning_rate" if v_cfg["trainer"].auto_lr_find else False,
                      max_epochs=60,
                      gradient_clip_val=0.1,
                      check_val_every_n_epoch=3
                      )

    model = Mono_det_3d(v_cfg)
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


"""
# crate test split

import random
import numpy as np
import os

total_list = list(range(0, 7481))
val_list = random.sample(total_list, 500)
val_list = sorted(val_list)
train_list = [ele for ele in total_list if ele not in val_list]

path = os.path.join("C:\\Users\\zihan\\Desktop\\visualDet3D\\visualDet3D\\visualDet3D\data\kitti", "train_split")

with open(os.path.join(path, "train.txt"), 'w') as f:
    for i in train_list:
        f.write("0"*(6-len(str(i))) + str(i))
        f.write('\n')

with open(os.path.join(path, "val.txt"), 'w') as f:
    for i in val_list:
        f.write("0"*(6-len(str(i))) + str(i))
        f.write('\n')
"""