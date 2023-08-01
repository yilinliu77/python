import os.path
import random
import time

import h5py
import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ray
import scipy
import torch
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pytorch_lightning as pl
import faiss
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from shared.fast_dataloader import FastDataLoader
from src.neural_bsp.model import AttU_Net_3D, U_Net_3D
from shared.common_utils import export_point_cloud, sigmoid
from src.neural_bsp.train_model import ABC_dataset, Base_model
import torch.distributed as dist


class ABC_dataset_patch(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode):
        super(ABC_dataset_patch, self).__init__(v_data_root, v_training_mode)
        self.num_objects = self.num_items
        self.num_patches = self.num_objects * 512
        self.validation_start = self.num_objects // 4 * 3

    def __len__(self):
        if self.mode == "training":
            return self.num_items // 4 * 3 * 512
        elif self.mode == "validation":
            return self.num_items // 4 * 512
        elif self.mode == "testing":
            return self.num_items * 512
        raise

    def get_patch(self, v_id_item, v_id_patch):
        features = np.load(self.objects[v_id_item] + "_feat.npy", mmap_mode="r")
        features = features[v_id_patch]
        flags = np.load(self.objects[v_id_item] + "_flag.npy", mmap_mode="r")
        flags = flags[v_id_patch]
        return features, flags

    def __getitem__(self, idx):
        if self.mode == "training" or self.mode == "testing":
            id_dummy = 0
        else:
            id_dummy = self.validation_start * 512

        id_object = (idx+id_dummy) // 512
        id_patch = (idx+id_dummy) % 512

        times = [0] * 10
        cur_time = time.time()
        feat_data, flag_data = self.get_patch(id_object, id_patch)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (3, 0, 1, 2))
        flag_data = flag_data.astype(np.float32)[None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[id_object], id_patch


class ABC_dataset_patch_hdf5(ABC_dataset_patch):
    def __init__(self, v_data_root, v_training_mode):
        super(ABC_dataset_patch_hdf5, self).__init__(None,None)
        self.data_root = v_data_root
        with h5py.File(self.data_root, "r") as f:
            self.num_items = f["features"].shape[0]
            self.names = ["{:08d}".format(item) for item in np.asarray(f["names"])]
        self.mode = v_training_mode

    def get_patch(self, v_id_item, v_id_patch):
        with h5py.File(self.data_root, "r") as f:
            features = f["features"][v_id_item, v_id_patch]
            flags = f["flags"][v_id_item, v_id_patch]
        return features, flags


class Base_model_full(Base_model):
    def __init__(self, v_phase=0):
        super(Base_model_full, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1, v_pool_first=False, v_depth=4)
        # self.encoder = AttU_Net_3D(img_ch=4, output_ch=1)


class Patch_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Patch_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.data = v_data
        self.phase = self.hydra_conf["model"]["phase"]
        self.model = globals()[self.hydra_conf["model"]["model_name"]](self.phase)
        self.dataset_name = globals()[self.hydra_conf["dataset"]["dataset_name"]]

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

    def train_dataloader(self):
        self.train_dataset = self.dataset_name(
            self.data,
            "training",
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          # collate_fn=ABC_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=1 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_name(
            self.data,
            "validation"
        )
        self.target_viz_name = self.valid_dataset.names[self.id_viz + self.valid_dataset.validation_start]
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          # collate_fn=ABC_dataset.collate_fn,
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
        name = batch[2]

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
        return

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.viz_data["gt"].clear()
            self.viz_data["prediction"].clear()
            self.viz_data["loss"].clear()
            self.viz_data["id_patch"].clear()
            return

        id_patch = torch.stack(self.viz_data["id_patch"], dim=0)
        prediction = torch.cat(self.viz_data["prediction"], dim=0)
        gt = torch.cat(self.viz_data["gt"], dim=0)
        if self.trainer.world_size!=1:
            device = prediction.device
            dtype = prediction.dtype
            size = torch.tensor([prediction.shape[0]], device=device, dtype=torch.int64)
            gathered_size = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(self.trainer.world_size)]
            dist.all_gather(gathered_size, size)

            gathered_id_patch = [torch.zeros((item.item(), ), dtype=torch.int64, device=device) for item in gathered_size]
            dist.all_gather(gathered_id_patch, id_patch)
            gathered_id_patch = torch.cat(gathered_id_patch, dim=0)
            
            gathered_prediction_list = [torch.zeros((item.item(), 32, 32, 32), dtype=dtype, device=device) for item in gathered_size]
            gathered_gt_list = [torch.zeros((item.item(), 32, 32, 32), dtype=dtype, device=device) for item in gathered_size]

            dist.all_gather(gathered_prediction_list, prediction)
            dist.all_gather(gathered_gt_list, gt)
            gathered_prediction_list = torch.cat(gathered_prediction_list, dim=0)
            gathered_gt_list = torch.cat(gathered_gt_list, dim=0)
            
            total_size = sum(gathered_size).item()
            gathered_prediction = torch.zeros((total_size, 32, 32, 32), dtype=dtype, device=device)
            gathered_prediction[gathered_id_patch] = gathered_prediction_list
            gathered_gt = torch.zeros((total_size, 32, 32, 32), dtype=dtype, device=device)
            gathered_gt[gathered_id_patch] = gathered_gt_list
            gathered_prediction = gathered_prediction.cpu().numpy()
            gathered_gt = gathered_gt.cpu().numpy()
        else:
            gathered_prediction = prediction.cpu().numpy()
            gathered_gt = gt.cpu().numpy()
        # all_gather(gathered_gt, gt)

        if self.global_rank != 0:
            self.viz_data["gt"].clear()
            self.viz_data["prediction"].clear()
            self.viz_data["loss"].clear()
            self.viz_data["id_patch"].clear()
            return
        # Gather the "self.viz_data" along all the gpu
        idx = self.trainer.current_epoch + 1 if not self.trainer.sanity_checking else 0

        assert gathered_prediction.shape[0] % 512 == 0
        query_points = self.viz_data["query_points"]

        predicted_labels = gathered_prediction.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(-1, 256, 256, 256)
        gt_labels = gathered_gt.reshape(
            (-1, 8, 8, 8, 32, 32, 32)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(-1, 256, 256, 256)

        predicted_labels = sigmoid(predicted_labels[0]) > 0.5
        mask = predicted_labels.reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_pred.ply".format(idx)), query_points[mask])

        gt_labels = sigmoid(gt_labels[0]) > 0.5
        mask = gt_labels.reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_gt.ply".format(idx)), query_points[mask])

        self.viz_data["gt"].clear()
        self.viz_data["prediction"].clear()
        self.viz_data["loss"].clear()
        self.viz_data["id_patch"].clear()
        return

    def test_dataloader(self):
        self.test_dataset = self.dataset_name(
            self.data,
            "testing"
        )
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          # collate_fn=ABC_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          )

    def test_step(self, batch, batch_idx):
        # data = self.denormalize(batch[:2])
        data = batch[:2]
        name = batch[2]

        outputs = self.model(data, False)
        loss = self.model.loss(outputs, data)
        self.viz_data["loss"].append(loss.item())

        features = (data[0].permute(0, 2, 3, 4, 1).cpu().numpy() * 65535).astype(np.uint16)
        outputs = torch.nn.functional.interpolate(torch.sigmoid(outputs), scale_factor=4) > 0.5
        prediction = (outputs.cpu().permute(0, 2, 3, 4, 1).numpy()).astype(np.ubyte)
        gt = torch.nn.functional.interpolate(data[1], scale_factor=4) > 0.5
        gt = gt.cpu().permute(0, 2, 3, 4, 1).numpy().astype(np.ubyte)
        self.viz_data["prediction"].append(prediction)
        self.viz_data["gt"].append(gt)

        def wrap_data(v_data):
            resolution = v_data.shape[0]
            chunk = 32
            num_chunk = resolution // chunk
            t = v_data.reshape(num_chunk, chunk, num_chunk, chunk, num_chunk, chunk, v_data.shape[-1])
            t = t.transpose((0, 2, 4, 1, 3, 5, 6)).reshape(-1, chunk, chunk, chunk, v_data.shape[-1])
            return t

        for id_batch in range(data[0].shape[0]):
            np.save(os.path.join(self.log_root, "{}_feat.npy".format(name[id_batch])), wrap_data(features[id_batch]))
            np.save(os.path.join(self.log_root, "{}_pred.npy".format(name[id_batch])),
                    wrap_data(prediction[id_batch, ..., 0:1]))
            np.save(os.path.join(self.log_root, "{}_gt.npy".format(name[id_batch])), wrap_data(gt[id_batch, ..., 0:1]))
        self.log("Test_Loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])

    def on_test_end(self):
        pass


@hydra.main(config_name="train_model_patch.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Patch_phase(v_cfg, v_cfg["dataset"]["root"])

    mc = ModelCheckpoint(monitor="Validation_Loss", )


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
        # precision=16,
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
