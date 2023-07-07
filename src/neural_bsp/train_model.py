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

from src.neural_bsp.model import AttU_Net_3D, U_Net_3D
from shared.common_utils import export_point_cloud, sigmoid

class ABC_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data_root, v_training_mode):
        super(ABC_dataset, self).__init__()
        self.data_root = v_data_root
        self.objects = os.listdir(v_data_root)
        self.num_items = len(self.objects)

        self.mode = v_training_mode

        pass

    def __len__(self):
        return self.num_items // 4 * 3 if self.mode == "training" else self.num_items // 4

    def __getitem__(self, idx):
        id_dummy = 0 if self.mode=="training" else self.num_items // 4 * 3
        times = [0] * 10
        cur_time = time.time()
        data = np.load(os.path.join(self.data_root, self.objects[idx+id_dummy], "data.npy"), allow_pickle=True).item()
        times[0] += time.time() - cur_time
        cur_time = time.time()
        resolution = data["resolution"]
        input_features = torch.from_numpy(data["input_features"])
        times[1] += time.time() - cur_time
        cur_time = time.time()
        consistent_flags = torch.from_numpy(np.unpackbits(
            data["consistent_flags"]).reshape(resolution, resolution, resolution, 1))
        times[2] += time.time() - cur_time
        cur_time = time.time()
        return input_features, consistent_flags

    @staticmethod
    def collate_fn(v_batches):
        input_features = []
        consistent_flags = []
        for item in v_batches:
            input_features.append(item[0])
            consistent_flags.append(item[1])

        input_features = torch.stack(input_features, dim=0).permute(0, 4, 1, 2, 3)
        consistent_flags = torch.stack(consistent_flags, dim=0).permute(0, 4, 1, 2, 3)

        return input_features.to(torch.float32), consistent_flags.to(torch.float32)


class Base_model(nn.Module):
    def __init__(self, v_phase=0):
        super(Base_model, self).__init__()
        self.phase = v_phase
        self.encoder = U_Net_3D(img_ch=4, output_ch=1)
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
        self.model = Base_model(self.phase)

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
        self.train_dataset = ABC_dataset(
            self.data,
            "training",
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=ABC_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        self.valid_dataset = ABC_dataset(
            self.data,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=ABC_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        batch = (batch[0] ,torch.max_pool3d(batch[1], 4, 4))

        outputs = self.model(batch, True)
        loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 # sync_dist=True,
                 batch_size=batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        batch = (batch[0] ,torch.max_pool3d(batch[1], 4, 4))

        outputs = self.model(batch, False)
        loss = self.model.loss(outputs, batch)
        self.viz_data["loss"].append(loss.item())
        self.viz_data["prediction"].append(outputs.cpu().numpy())
        self.viz_data["gt"].append(batch[1].cpu().numpy())
        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 # sync_dist=True,
                 batch_size=batch[0].shape[0])
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

        id_viz = 1
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


@hydra.main(config_name="grid_clustering_model_3d.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Base_phase(v_cfg, v_cfg["dataset"]["root"])

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
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
        trainer.validate(model)
        # trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
