import importlib
import os.path
from pathlib import Path

import h5py
import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from shared.common_utils import export_point_cloud, sigmoid
import torch.distributed as dist

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score


class PC_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(PC_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.data = v_data
        mod = importlib.import_module('src.neural_bsp.model')
        self.model = getattr(mod, self.hydra_conf["model"]["model_name"])(
            self.hydra_conf["model"]
        )
        # Import module according to the dataset_name
        mod = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
        self.dataset_name = getattr(mod, self.hydra_conf["dataset"]["dataset_name"])

        # Used for visualizing during the training
        self.id_viz = 0
        self.viz_data = {
            "loss": [],
            "prediction": [],
            "gt": [],
            "id_patch": [],
        }

        # self.target_viz_name = "00015724"
        pr_computer = {
            "P_3": BinaryPrecision(threshold=0.3),
            "P_5": BinaryPrecision(threshold=0.5),
            "P_7": BinaryPrecision(threshold=0.7),
            "P_9": BinaryPrecision(threshold=0.9),
            "R_3": BinaryRecall(threshold=0.3),
            "R_5": BinaryRecall(threshold=0.5),
            "R_7": BinaryRecall(threshold=0.7),
            "R_9": BinaryRecall(threshold=0.9),
            "AP": BinaryAveragePrecision(thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]),
            "F1": BinaryF1Score(threshold=0.5),
        }
        self.pr_computer = MetricCollection(pr_computer)

    def train_dataloader(self):
        self.train_dataset = self.dataset_name(
            self.data,
            "training",
            self.hydra_conf["dataset"],
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=False,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=1 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_name(
            self.data,
            "validation",
            self.hydra_conf["dataset"],
        )
        self.target_viz_name = self.valid_dataset.names[self.id_viz + self.valid_dataset.validation_start]
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=False,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=1 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch[:2]
        name = batch[2]
        outputs = self.model(data, True)
        loss = self.model.loss(outputs, data)
        for loss_name in loss:
            if loss_name == "total_loss":
                self.log("Training_Loss", loss[loss_name], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=data[0].shape[0])
            else:
                self.log("Training_"+loss_name, loss[loss_name], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=data[0].shape[0])
        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):
        data = batch[:2]
        name = batch[2]
        id_patch = batch[3]

        outputs = self.model(data, False)
        loss = self.model.loss(outputs, data)
        for idx, name_item in enumerate(name):
            if name_item == self.target_viz_name:
                self.viz_data["loss"].append(loss["total_loss"].item())
                self.viz_data["id_patch"].append(id_patch[idx])
                self.viz_data["prediction"].append(outputs[idx])
                self.viz_data["gt"].append(data[1][idx])
        for loss_name in loss:
            if loss_name == "total_loss":
                self.log("Validation_Loss", loss[loss_name], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=data[0].shape[0])
            else:
                self.log("Validation_"+loss_name, loss[loss_name], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=data[0].shape[0])

        prob = torch.sigmoid(outputs[:,:,4])
        gt = data[1][:,:,7].to(torch.long)
        self.pr_computer.update(prob, gt)
        return

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.viz_data["gt"].clear()
            self.viz_data["prediction"].clear()
            self.viz_data["loss"].clear()
            self.viz_data["id_patch"].clear()
            self.pr_computer.reset()
            return

        self.log_dict(self.pr_computer.compute(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True)

        self.pr_computer.reset()

        if len(self.viz_data["id_patch"]) == 0:
            return

        id_patch = torch.stack(self.viz_data["id_patch"], dim=0)
        prediction = torch.cat(self.viz_data["prediction"], dim=0)
        gt = torch.cat(self.viz_data["gt"], dim=0)

        gathered_prediction = prediction.cpu().numpy()
        gathered_gt = gt.cpu().numpy()

        idx = self.trainer.current_epoch + 1 if not self.trainer.sanity_checking else 0

        assert gathered_prediction.shape[0] % 512 == 0
        predicted_labels = gathered_prediction.reshape((256, 256, 256, -1))
        gt_labels = gathered_gt.reshape((256, 256, 256, -1))

        query_points = gt_labels[:,:,:,:3]
        gt_gradient = gt_labels[:,:,:,3:6]
        gt_udf = gt_labels[:,:,:,6:7]
        gt_flag = gt_labels[:,:,:,7:8].astype(bool)

        pred_udf = predicted_labels[:,:,:,0:1]
        pred_gradient = predicted_labels[:,:,:,1:4]
        pred_flag = sigmoid(predicted_labels[:,:,:,4:5]) > 0.5

        gt_surface_points = (query_points + gt_gradient * gt_udf)
        export_point_cloud(
            str(self.log_root / "{}_{}_gt_p.ply".format(idx, self.target_viz_name)),
            gt_surface_points.reshape(-1,3)
        )

        pred_surface_points = (query_points + pred_gradient * pred_udf)
        export_point_cloud(
            str(self.log_root / "{}_{}_pred_p.ply".format(idx, self.target_viz_name)),
            pred_surface_points.reshape(-1,3)
        )

        gt_boundary = query_points[gt_flag[:,:,:,0]]
        export_point_cloud(
            str(self.log_root / "{}_{}_gt_b.ply".format(idx, self.target_viz_name)),
            gt_boundary
        )

        pred_boundary = query_points[pred_flag[:,:,:,0]]
        export_point_cloud(
            str(self.log_root / "{}_{}_pred_b.ply".format(idx, self.target_viz_name)),
            pred_boundary.reshape(-1,3)
        )

        self.viz_data["gt"].clear()
        self.viz_data["prediction"].clear()
        self.viz_data["loss"].clear()
        self.viz_data["id_patch"].clear()
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_name(
            self.data,
            "testing",
            self.validation_batch_size,
            self.hydra_conf["dataset"]["v_output_features"],
            self.hydra_conf["dataset"]["test_list"],
        )
        return DataLoader(self.test_dataset, batch_size=1,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          )

    def test_step(self, batch, batch_idx):
        # data = self.denormalize(batch[:2])
        data = batch[:2]
        name = batch[2][0]
        id_patch = batch[3]

        outputs = self.model(data, False)
        loss = self.model.loss(outputs, data)

        # PR
        self.pr_computer.update(torch.sigmoid(outputs), data[1].to(torch.long))

        self.log("Test_Loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])

        # Save
        if len(self.hydra_conf["dataset"]["test_list"]) > 0:
            threshold = self.hydra_conf["model"]["test_threshold"]
            predicted_labels = (torch.sigmoid(outputs[:, 0]) > threshold).cpu().numpy().astype(np.ubyte)
            predicted_labels = np.transpose(
                predicted_labels.reshape((8, 8, 8, 32, 32, 32)), (0, 3, 1, 4, 2, 5)).reshape((256, 256, 256))
            gt_labels = (torch.sigmoid(data[1][:, 0]) > threshold).cpu().numpy().astype(np.ubyte)
            gt_labels = np.transpose(
                gt_labels.reshape((8, 8, 8, 32, 32, 32)), (0, 3, 1, 4, 2, 5)).reshape((256, 256, 256))
            features = (data[0].permute(0, 2, 3, 4, 1).cpu().numpy()).astype(np.float32)
            features = np.transpose(
                features.reshape((8, 8, 8, 32, 32, 32, 4)), (0, 3, 1, 4, 2, 5, 6)).reshape((256, 256, 256, 4))
            features[:, :, :, 3] /= np.pi

            np.save(os.path.join(self.log_root, "{}_feat.npy".format(name)), features)
            np.save(os.path.join(self.log_root, "{}_pred.npy".format(name)), predicted_labels)
            np.save(os.path.join(self.log_root, "{}_gt.npy".format(name)), gt_labels)

            coords = np.stack(
                np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij"), axis=3).reshape(
                8,32,8,32,8,32,3).transpose(0,2,4,1,3,5,6).reshape(512,32,32,32,3)
            coords = (coords / 255 - 0.5) * 2
            pred_flags = (torch.sigmoid(outputs[:, 0]) > threshold).cpu().numpy()
            gt_flags = (torch.sigmoid(data[1][:, 0]) > threshold).cpu().numpy()
            for i_patch in range(outputs.shape[0]):
                feat = data[0][i_patch].permute(1,2,3,0).cpu().numpy()
                points = coords[i_patch] + feat[:,:,:,:3] * feat[:,:,:,3:4] / np.pi
                points = points.reshape(-1,3)
                export_point_cloud("output/mesh.ply", points)

                points = coords[i_patch][pred_flags[i_patch]]
                # points = coords[i_patch][gt_flags[i_patch]]
                export_point_cloud("output/original.ply", points)

                pass

        return

    def on_test_end(self):
        metrics = self.pr_computer.compute()
        for key in metrics:
            print("{:3}: {:.3f}".format(key, metrics[key].cpu().item()))
        print("Loss: {:.3f}".format(self.trainer.callback_metrics["Test_Loss_epoch"].cpu().item()))


@hydra.main(config_name="train_model_pc.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = PC_phase(v_cfg, v_cfg["dataset"]["root"])

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)

    trainer = Trainer(
        default_root_dir=log_dir,

        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false" if v_cfg["trainer"].gpu > 1 else "auto",
        devices=v_cfg["trainer"].gpu,

        enable_model_summary=False,
        callbacks=[mc],
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
        # gradient_clip_val=0.5,
    )
    torch.find_unused_parameters = False
    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=True)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
