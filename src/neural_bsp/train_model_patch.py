import importlib
import os.path

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

from torchmetrics.classification import BinaryPrecision, BinaryRecall

class Patch_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Patch_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.validation_batch_size = self.hydra_conf["trainer"]["validation_batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.data = v_data
        self.phase = self.hydra_conf["model"]["phase"]
        mod = importlib.import_module('src.neural_bsp.model')
        self.model = getattr(mod, self.hydra_conf["model"]["model_name"])(
            self.phase,
            self.hydra_conf["model"]["loss"],
            self.hydra_conf["model"]["loss_alpha"],
        )
        # Import module according to the dataset_name
        mod = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
        self.dataset_name = getattr(mod, self.hydra_conf["dataset"]["dataset_name"])

        # Used for visualizing during the training
        self.id_viz = 0
        resolution = 256
        source_coords = np.stack(np.meshgrid(
            np.arange(resolution), np.arange(resolution), np.arange(resolution), indexing="ij"),
            axis=3).reshape(-1, 3)
        source_coords = ((source_coords / (resolution - 1)) * 2 - 1).astype(np.float32)
        self.viz_data = {
            "query_points": source_coords,
            "loss": [],
            "prediction": [],
            "gt": [],
            "id_patch": [],
        }

        # self.target_viz_name = "00015724"
        pr_computer={
            "P_3": BinaryPrecision(threshold=0.3),
            "P_5": BinaryPrecision(threshold=0.5),
            "P_7": BinaryPrecision(threshold=0.7),
            "P_9": BinaryPrecision(threshold=0.9),
            "R_3": BinaryRecall(threshold=0.3),
            "R_5": BinaryRecall(threshold=0.5),
            "R_7": BinaryRecall(threshold=0.7),
            "R_9": BinaryRecall(threshold=0.9),
        }
        self.pr_computer = MetricCollection(pr_computer)

    def train_dataloader(self):
        self.train_dataset = self.dataset_name(
            self.data,
            "training",
            self.batch_size
        )
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=1 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_name(
            self.data,
            "validation",
            self.validation_batch_size
        )
        self.target_viz_name = self.valid_dataset.names[self.id_viz + self.valid_dataset.validation_start]
        return DataLoader(self.valid_dataset, batch_size=1,
                          collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=1 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def denormalize(self, v_batch):
        return (v_batch[0], torch.max_pool3d(v_batch[1], 4, 4))
        flags = v_batch[1]
        feature = v_batch[0]

        feature = (torch.from_numpy(feature.astype(np.float32)).to(flags.device) / 65535).permute(0, 4, 1, 2, 3)
        flags = torch.max_pool3d(flags.to(torch.float32)[:, None, :, :], 4, 4)

        return (feature, flags)

    def training_step(self, batch, batch_idx):
        # data = self.denormalize(batch[:2])
        data = batch[:2]
        # data = [batch[0][0],batch[1][0]]
        name = batch[2]
        # print(name[0])
        outputs = self.model(data, True)
        loss = self.model.loss(outputs, data)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        # data = self.denormalize(batch[:2])
        data = batch[:2]
        name = batch[2]
        id_patch = batch[3]

        outputs = self.model(data, False)
        loss = self.model.loss(outputs, data)
        for idx, name_item in enumerate(name):
            if name_item == self.target_viz_name:
                self.viz_data["loss"].append(loss.item())
                self.viz_data["id_patch"].append(id_patch[idx])
                self.viz_data["prediction"].append(outputs[idx])
                self.viz_data["gt"].append(data[1][idx])
        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])

        prob = torch.sigmoid(outputs)
        gt = data[1].to(torch.long)
        self.pr_computer.update(prob, gt)
        return

    def gather_data(self):
        id_patch = torch.stack(self.viz_data["id_patch"], dim=0)
        prediction = torch.cat(self.viz_data["prediction"], dim=0)
        gt = torch.cat(self.viz_data["gt"], dim=0)
        if self.trainer.world_size != 1:
            device = prediction.device
            dtype = prediction.dtype
            size = torch.tensor([prediction.shape[0]], device=device, dtype=torch.int64)
            gathered_size = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(self.trainer.world_size)]
            dist.all_gather(gathered_size, size)

            gathered_id_patch = [torch.zeros((item.item(),), dtype=torch.int64, device=device) for item in
                                 gathered_size]
            dist.all_gather(gathered_id_patch, id_patch)
            gathered_id_patch = torch.cat(gathered_id_patch, dim=0)

            gathered_prediction_list = [torch.zeros((item.item(), 32, 32, 32), dtype=dtype, device=device) for item in
                                        gathered_size]
            gathered_gt_list = [torch.zeros((item.item(), 32, 32, 32), dtype=dtype, device=device) for item in
                                gathered_size]

            dist.all_gather(gathered_prediction_list, prediction)
            dist.all_gather(gathered_gt_list, gt)
            gathered_prediction_list = torch.cat(gathered_prediction_list, dim=0)
            gathered_gt_list = torch.cat(gathered_gt_list, dim=0)

            total_size = sum(gathered_size).item()
            gathered_prediction = torch.zeros((total_size, 32, 32, 32), dtype=dtype, device=device)
            gathered_prediction[gathered_id_patch] = gathered_prediction_list
            gathered_gt = torch.zeros((total_size, 32, 32, 32), dtype=dtype, device=device)
            gathered_gt[gathered_id_patch] = gathered_gt_list
            gathered_prediction = gathered_prediction.cpu().numpy()[:512]
            gathered_gt = gathered_gt.cpu().numpy()[:512]
        else:
            gathered_prediction = prediction.cpu().numpy()
            gathered_gt = gt.cpu().numpy()

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
        query_points = self.viz_data["query_points"]

        predicted_labels = gathered_prediction.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(-1, 256, 256, 256)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(-1, 256, 256, 256)

        predicted_labels = sigmoid(predicted_labels[0]) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_{}_pred.ply".format(idx,self.target_viz_name)), query_points[mask])

        gt_labels = sigmoid(gt_labels[0]) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_{}_gt.ply".format(idx,self.target_viz_name)), query_points[mask])

        self.viz_data["gt"].clear()
        self.viz_data["prediction"].clear()
        self.viz_data["loss"].clear()
        self.viz_data["id_patch"].clear()
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_name(
            self.data,
            "testing",
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
            predicted_labels = (torch.sigmoid(outputs[:,0]) > threshold).cpu().numpy().astype(np.ubyte)
            gt_labels = (torch.sigmoid(data[1][:,0]) > threshold).cpu().numpy().astype(np.ubyte)
            features = (data[0].permute(0, 2, 3, 4, 1).cpu().numpy() * 65535).astype(np.uint16)

            np.save(os.path.join(self.log_root, "{}_feat.npy".format(name)), features)
            np.save(os.path.join(self.log_root, "{}_pred.npy".format(name)),predicted_labels)
            np.save(os.path.join(self.log_root, "{}_gt.npy".format(name)), gt_labels)

        return

    def on_test_end(self):
        print(self.pr_computer.compute())
        print("Loss: ", self.trainer.callback_metrics["Test_Loss_epoch"])

@hydra.main(config_name="train_model_patch.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Patch_phase(v_cfg, v_cfg["dataset"]["root"])

    mc = ModelCheckpoint(monitor="Validation_Loss", )

    # torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        default_root_dir=log_dir,

        accelerator='gpu',
        strategy = "ddp_find_unused_parameters_false" if v_cfg["trainer"].gpu > 1 else "auto",
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
        # trainer.validate(model)
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
