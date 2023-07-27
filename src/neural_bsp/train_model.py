import os.path
import random
import time

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


class ABC_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data_root=None, v_training_mode=None):
        super(ABC_dataset, self).__init__()
        self.data_root = v_data_root
        if v_data_root is None:
            return
        self.names = list(set([item[:8] for item in os.listdir(v_data_root)]))
        self.names = sorted(self.names, key=lambda x: int(x))
        self.objects = [os.path.join(v_data_root, item) for item in self.names]

        self.num_items = len(self.objects)

        self.mode = v_training_mode

        pass

    def __len__(self):
        if self.mode == "training":
            return self.num_items // 4 * 3
        elif self.mode == "validation":
            return self.num_items // 4
        elif self.mode == "testing":
            return self.num_items
        raise

    def get_total(self, v_idx):
        features = np.load(self.objects[v_idx] + "_feat.npy")
        features = features.reshape(8, 8, 8, 32, 32, 32, 3).transpose((0, 3, 1, 4, 2, 5, 6)).reshape(256, 256, 256, 3)
        flags = np.load(self.objects[v_idx] + "_flag.npy")
        flags = flags.reshape(8, 8, 8, 32, 32, 32).transpose((0, 3, 1, 4, 2, 5)).reshape(256, 256, 256)
        return features, flags



    def __getitem__(self, idx):
        if self.mode == "training" or self.mode == "testing":
            id_dummy = 0
        else:
            id_dummy = self.num_items // 4 * 3
        times = [0] * 10
        cur_time = time.time()
        feat_data, flag_data = self.get_total(idx + id_dummy)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32) / 65535, (3, 0, 1, 2))
        flag_data = flag_data.astype(np.float32)[None, :, :, :]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[idx + id_dummy]

    @staticmethod
    def collate_fn(v_batches):
        input_features = []
        consistent_flags = []
        for item in v_batches:
            input_features.append(item[0])
            consistent_flags.append(item[1])

        input_features = np.stack(input_features, axis=0)
        # input_features = torch.from_numpy(np.stack(input_features,axis=0).astype(np.float32)).permute(0, 4, 1, 2, 3)
        consistent_flags = torch.from_numpy(np.stack(consistent_flags, axis=0))

        return input_features, consistent_flags


class ABC_dataset_patch(ABC_dataset):
    def __init__(self, v_data_root, v_training_mode):
        super(ABC_dataset_patch, self).__init__(v_data_root, v_training_mode)

    def __len__(self):
        if self.mode=="training":
            return self.num_items // 4 * 3 * 512
        elif self.mode=="validation":
            return self.num_items // 4 * 512
        elif self.mode=="testing":
            return self.num_items * 512
        raise

    def __getitem__(self, idx):
        if self.mode=="training" or self.mode=="testing":
            id_dummy = 0
        else:
            id_dummy = self.num_items // 4 * 3

        times = [0] * 10
        cur_time = time.time()
        feat_data, flag_data = self.get_patch(idx // 512, idx % 512)
        times[0] += time.time() - cur_time
        cur_time = time.time()
        feat_data = np.transpose(feat_data.astype(np.float32)/65535, (3,0,1,2))
        flag_data = flag_data.astype(np.float32)[None,:,:,:]
        times[1] += time.time() - cur_time
        return feat_data, flag_data, self.names[(idx+id_dummy)]


class Base_model(nn.Module):
    def __init__(self, v_phase=0):
        super(Base_model, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1)
        # self.encoder = AttU_Net_3D(img_ch=4, output_ch=1)

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = sigmoid_focal_loss(v_predictions, labels,
                                  alpha=0.75,
                                  reduction="mean"
                                  )

        return loss


class Atten_model(nn.Module):
    def __init__(self, v_phase=0):
        super(Atten_model, self).__init__()
        self.phase = v_phase
        self.encoder = AttU_Net_3D(img_ch=3, output_ch=1)

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = sigmoid_focal_loss(v_predictions, labels,
                                  alpha=0.75,
                                  reduction="mean"
                                  )

        return loss


class Base_model_full(Base_model):
    def __init__(self, v_phase=0):
        super(Base_model_full, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=3, output_ch=1, v_pool_first=False, v_depth=4)
        # self.encoder = AttU_Net_3D(img_ch=4, output_ch=1)

class Base_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Base_phase, self).__init__()
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
        resolution = 64
        source_coords = np.stack(np.meshgrid(
                np.arange(resolution), np.arange(resolution), np.arange(resolution), indexing="ij"),
                axis=3).reshape(-1, 3)
        source_coords = ((source_coords / (resolution-1)) * 2 - 1).astype(np.float32)
        self.viz_data = {
            "query_points": source_coords,
            "loss": [],
            "prediction": [],
            "gt": [],
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
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          # collate_fn=ABC_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
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

        feature = (torch.from_numpy(feature.astype(np.float32)).to(flags.device)/65535).permute(0,4,1,2,3)
        flags = torch.max_pool3d(flags.to(torch.float32)[:,None,:,:], 4, 4)

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

        outputs = self.model(data, False)
        loss = self.model.loss(outputs, data)
        self.viz_data["loss"].append(loss.item())
        self.viz_data["prediction"].append(outputs.cpu().numpy())
        self.viz_data["gt"].append(data[1].cpu().numpy())
        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])
        return

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        if self.trainer.sanity_checking:
            return

        idx = self.trainer.current_epoch + 1 if not self.trainer.sanity_checking else 0

        num_items = sum([item.shape[0] for item in self.viz_data["gt"]])
        query_points = self.viz_data["query_points"]
        # valid_flags = self.viz_data["valid_flags"]
        # valid_query_points = np.tile(query_points[:, None], (1, 26, 1))[valid_flags]
        # valid_target_points = query_points[self.viz_data["target_vertices"][valid_flags]]
        #
        # # 1 indicates non consistent
        # gt_labels = np.concatenate(self.viz_data["gt"], axis=0).transpose((0, 2, 3, 4, 1)).astype(bool)
        # num_items = gt_labels.shape[0]
        # valid_gt_labels = gt_labels.reshape(num_items, -1, 26)[:, valid_flags]

        id_viz = 0
        predicted_labels = sigmoid(self.viz_data["prediction"][id_viz][0].transpose((1,2,3,0))) > 0.5
        mask = predicted_labels.any(axis=3).reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_pred.ply".format(idx)), query_points[mask])

        gt_labels = sigmoid(self.viz_data["gt"][id_viz][0].transpose((1, 2, 3, 0))) > 0.5
        mask = gt_labels.any(axis=3).reshape(-1)
        export_point_cloud(os.path.join(self.log_root, "{}_gt.ply".format(idx)), query_points[mask])

        self.viz_data["gt"].clear()
        self.viz_data["prediction"].clear()
        self.viz_data["loss"].clear()
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

        features = (data[0].permute(0,2,3,4,1).cpu().numpy() * 65535).astype(np.uint16)
        outputs = torch.nn.functional.interpolate(torch.sigmoid(outputs), scale_factor=4) > 0.5
        prediction = (outputs.cpu().permute(0,2,3,4,1).numpy()).astype(np.ubyte)
        gt = torch.nn.functional.interpolate(data[1], scale_factor=4) > 0.5
        gt = gt.cpu().permute(0,2,3,4,1).numpy().astype(np.ubyte)
        self.viz_data["prediction"].append(prediction)
        self.viz_data["gt"].append(gt)
        def wrap_data(v_data):
            resolution = v_data.shape[0]
            chunk = 32
            num_chunk = resolution // chunk
            t = v_data.reshape(num_chunk, chunk, num_chunk, chunk, num_chunk, chunk, v_data.shape[-1])
            t = t.transpose((0,2,4,1,3,5,6)).reshape(-1, chunk, chunk, chunk, v_data.shape[-1])
            return t

        for id_batch in range(data[0].shape[0]):
            np.save(os.path.join(self.log_root, "{}_feat.npy".format(name[id_batch])), wrap_data(features[id_batch]))
            np.save(os.path.join(self.log_root, "{}_pred.npy".format(name[id_batch])), wrap_data(prediction[id_batch,...,0:1]))
            np.save(os.path.join(self.log_root, "{}_gt.npy".format(name[id_batch])), wrap_data(gt[id_batch,...,0:1]))
        self.log("Test_Loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=data[0].shape[0])

    def on_test_end(self):
        pass


@hydra.main(config_name="train_model_global.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Base_phase(v_cfg, v_cfg["dataset"]["root"])

    mc = ModelCheckpoint(monitor="Validation_Loss",)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        callbacks=[mc],
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        default_root_dir=log_dir,
        # precision=16,
        # gradient_clip_val=0.5
    )

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
