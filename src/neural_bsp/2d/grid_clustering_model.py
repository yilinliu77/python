import os.path

import hydra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

from src.neural_bsp.attention_unet import AttU_Net


class BSP_dataset(torch.utils.data.Dataset):
    def __init__(self, v_data, v_training_mode):
        super(BSP_dataset, self).__init__()
        self.resolution = int(v_data["resolution"])
        self.input_features = torch.from_numpy(
            v_data["input_features"]).reshape(self.resolution, self.resolution, 3).permute(2,0,1)
        self.edges = torch.from_numpy(v_data["all_edges"])
        self.valid_flags = torch.from_numpy(v_data["valid_flags"]).reshape(self.resolution, self.resolution, 8)
        self.non_consistent_flags = ~torch.from_numpy(
            v_data["consistent_flags"]).reshape(self.resolution, self.resolution,8)

        self.training_flags = torch.logical_and(self.valid_flags, self.non_consistent_flags)
        self.training_flags = self.training_flags.permute(2,0,1).to(torch.float32)

        self.mode = v_training_mode

        pass

    def __len__(self):
        return 100 if self.mode=="training" else 1

    def __getitem__(self, idx):
        return self.input_features, self.training_flags


class Base_model(nn.Module):
    def __init__(self, v_phase=0):
        super(Base_model, self).__init__()
        self.phase = v_phase
        self.encoder = AttU_Net(img_ch=3, output_ch=8)

    def forward(self, v_data, v_training=False):
        features, labels = v_data
        prediction = self.encoder(features)

        return prediction

    def loss(self, v_predictions, v_input):
        features, labels = v_input

        loss = sigmoid_focal_loss(v_predictions, labels,
                                  alpha = 0.75,
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
        self.viz_data = {
            "loss": 0,
            "query_points": v_data["query_points"],
            "target_vertices": v_data["target_vertices"],
            "all_edges": v_data["all_edges"],
            "valid_flags": v_data["valid_flags"],
        }

    def train_dataloader(self):
        self.train_dataset = BSP_dataset(
            self.data,
            "training",
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        self.valid_dataset = BSP_dataset(
            self.data,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, True)
        loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, False)
        loss = self.model.loss(outputs, batch)
        self.viz_data["loss"]=loss
        self.viz_data["prediction"]=outputs
        self.viz_data["gt"]=batch[1]
        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[0])

        return

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        if self.trainer.sanity_checking:
            return

        idx = self.global_step+1 if not self.trainer.sanity_checking else 0

        query_points = self.viz_data["query_points"]
        target_vertices = self.viz_data["target_vertices"]
        all_edges = self.viz_data["all_edges"]
        valid_flags = self.viz_data["valid_flags"]
        gt_labels = self.viz_data["gt"][0].permute(1,2,0).cpu().numpy().reshape(-1).astype(bool) # 1 indicates non consistent

        predicted_labels = self.viz_data["prediction"][0].permute(1,2,0).cpu().numpy().reshape(-1) > 0.5
        # Mask out boundary
        predicted_labels[~valid_flags] = False

        pred_edges = all_edges[predicted_labels]
        gt_edges = all_edges[gt_labels]

        fig, axes = plt.subplots(1,2)
        fig.suptitle("Iter: {}; Loss: {}".format(
            idx,
            self.viz_data["loss"].cpu().item()
        ))
        # Fig 1: Prediction
        x_values = [query_points[pred_edges[:, 0], 0], query_points[pred_edges[:, 1], 0]]
        y_values = [query_points[pred_edges[:, 0], 1], query_points[pred_edges[:, 1], 1]]
        axes[0].plot(x_values, y_values, 'g-')
        axes[0].scatter(target_vertices[:,0], target_vertices[:,1], s=3, color=(1,0,0))
        axes[0].axis("scaled")
        axes[0].set_xlim(-0.5, 0.5)
        axes[0].set_ylim(-0.5, 0.5)
        axes[0].tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        axes[0].set_title("Prediction")

        # Fig 2: GT
        x_values = [query_points[gt_edges[:, 0], 0], query_points[gt_edges[:, 1], 0]]
        y_values = [query_points[gt_edges[:, 0], 1], query_points[gt_edges[:, 1], 1]]
        axes[1].plot(x_values, y_values, 'g-', linewidth=1)
        axes[1].scatter(target_vertices[:,0], target_vertices[:,1], s=3, color=(1,0,0))
        axes[1].axis("scaled")
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_ylim(-0.5, 0.5)
        axes[1].tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        axes[1].set_title("GT")

        # plt.show(block=True)
        plt.savefig(os.path.join(self.log_root,"{}".format(idx)), dpi=300)
        return


@hydra.main(config_name="grid_clustering_model.yaml", config_path="../../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    data = np.load(v_cfg["dataset"]["root"], allow_pickle=True).item()

    model = Base_phase(v_cfg, data)

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
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
