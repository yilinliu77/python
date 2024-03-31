import os.path
from pathlib import Path

import hydra
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score

from shared.common_utils import export_point_cloud, sigmoid

from src.img2brep.nerf2nvd_dataset import nerf2nvd_dataset
from src.neural_bsp.model import conv_block, up_conv, U_Net_3D
from diffusers import PNDMPipeline


class Autoencoder(nn.Module):
    def __init__(self, dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList([
            conv_block(2, dim, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
            conv_block(dim, dim * 2, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
            conv_block(dim * 2, dim * 4, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
        ])

        self.decoder = nn.Sequential(
            up_conv(dim * 4, dim * 2, 4),
            up_conv(dim * 2, dim, 4),
            up_conv(dim, dim, 4),
            nn.Conv3d(dim, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, is_training=True):
        x = torch.cat(x[:2], dim=1)
        # encoding
        features = [x]
        for module in self.encoder:
            features.append(module(features[-1]))

        # decoding
        recon = self.decoder(features[-1])
        return recon

    def loss(self, x, y, is_training=True):
        predicted_udf = x[:, 0:1]
        predicted_flag = x[:, 1:2]

        gt_udf = y[0]
        gt_voronoi_flag = y[1]

        udf_loss = nn.functional.mse_loss(predicted_udf, gt_udf)
        flag_loss = nn.functional.binary_cross_entropy_with_logits(
            predicted_flag,
            gt_voronoi_flag.to(torch.int32).to(torch.float32)
        )
        total_loss = udf_loss + flag_loss
        return {
            "udf_loss": udf_loss,
            "flag_loss": flag_loss,
            "total_loss": total_loss,
        }


class Autoencoder2(Autoencoder):
    def __init__(self, dim=16):
        super(Autoencoder2, self).__init__(dim)
        self.udf_decoder = nn.Conv3d(dim, 1, kernel_size=1, stride=1, padding=0)
        self.voronoi_decoder = nn.Conv3d(dim, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, is_training=True):
        x = torch.cat(x, dim=1)
        # encoding
        features = [x]
        for module in self.encoder:
            features.append(module(features[-1]))

        # decoding
        recon = self.decoder(features[-1])
        udf = self.udf_decoder(recon)
        voronoi = self.voronoi_decoder(recon)
        recon = torch.cat([udf, voronoi], dim=1)
        return recon


class Autoencoder3(Autoencoder):
    def __init__(self, dim=16):
        super(Autoencoder3, self).__init__(dim)
        self.encoder = nn.ModuleList([
            conv_block(2, dim, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
            conv_block(dim, dim * 2, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
            conv_block(dim * 2, dim * 4, ),
            nn.MaxPool3d(kernel_size=4, stride=4),
        ])

        self.decoder = nn.Sequential(
            nn.Conv3d(dim * 4, dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(dim * 2, dim * 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(dim * 1, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, is_training=True):
        query_points = x[2]
        num_total_points = query_points.shape[1]

        x = torch.cat(x[:2], dim=1)
        # encoding
        features = [x]
        for module in self.encoder:
            features.append(module(features[-1]))

        # decoding
        patch_size = 10000
        if is_training:
            selected_id = torch.randperm(num_total_points, device=query_points.device)[:10000]
            query_points = query_points[:, selected_id]
            recon = nn.functional.grid_sample(
                features[-1],
                query_points.unsqueeze(1).unsqueeze(1),
                mode="bilinear",
                align_corners=True,
            )
            recon = self.decoder(recon)
        else:
            selected_id = torch.arange(num_total_points, device=query_points.device)
            query_points = query_points[:, selected_id]

            num_patches = (num_total_points + patch_size - 1) // patch_size
            results = []
            for i in range(num_patches):
                start = i * patch_size
                end = min((i + 1) * patch_size, num_total_points)
                query_points_patch = query_points[:, start:end]
                recon_patch = nn.functional.grid_sample(
                    features[-1],
                    query_points_patch.unsqueeze(1).unsqueeze(1),
                    mode="bilinear",
                    align_corners=True,
                )
                recon_patch = self.decoder(recon_patch)
                results.append(recon_patch)
            recon = torch.cat(results, dim=-1)

        self.selected_id = selected_id
        return recon

    def loss(self, x, y, is_training=True):
        predicted_udf = x[:, 0:1, 0, 0]
        predicted_flag = x[:, 1:2, 0, 0]

        gt_udf = y[0]
        gt_voronoi_flag = y[1]

        gt_udf = gt_udf.reshape(-1, 1, 256 * 256 * 256)[:, :, self.selected_id]
        gt_voronoi_flag = gt_voronoi_flag.reshape(-1, 1, 256 * 256 * 256)[:, :, self.selected_id]

        udf_loss = nn.functional.mse_loss(predicted_udf, gt_udf)
        flag_loss = nn.functional.binary_cross_entropy_with_logits(
            predicted_flag,
            gt_voronoi_flag.to(torch.int32).to(torch.float32)
        )
        total_loss = udf_loss + flag_loss
        return {
            "udf_loss": udf_loss,
            "flag_loss": flag_loss,
            "total_loss": total_loss,
        }


class Autoencoder4(Autoencoder3):
    def __init__(self, dim=16):
        super(Autoencoder4, self).__init__(dim)
        self.encoder = nn.ModuleList([
            nn.Conv3d(2, dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * 2, dim * 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            conv_block(dim * 4, dim * 8, kernel_size=3, stride=1, padding=1, with_bn=False),
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_block(dim * 8, dim * 16, kernel_size=3, stride=1, padding=1, with_bn=False),
            nn.MaxPool3d(kernel_size=2, stride=2),
            conv_block(dim * 16, dim * 16, kernel_size=3, stride=1, padding=1, with_bn=False),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(dim * 16, dim * 16, kernel_size=3, stride=1, padding=1),
        ])

        self.decoder = nn.Sequential(
            nn.Conv3d(dim * 16, dim * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(dim * 4, dim * 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(dim * 1, 2, kernel_size=1, stride=1, padding=0),
        )

        self.id_features = [7, 9, 12]

    def forward(self, x, is_training=True):
        query_points = x[2]
        num_total_points = query_points.shape[1]

        x = torch.cat(x[:2], dim=1)
        # encoding
        features = []
        feature = x
        for idx, module in enumerate(self.encoder):
            feature = module(feature)
            if idx in self.id_features:
                features.append(feature)

        # decoding
        patch_size = 1000000
        if is_training:
            selected_id = torch.randperm(num_total_points, device=query_points.device)[:patch_size]
            query_points = query_points[:, selected_id]
            recon = nn.functional.grid_sample(
                features[-1],
                query_points.unsqueeze(1).unsqueeze(1),
                mode="bilinear",
                align_corners=True,
            )
            recon = self.decoder(recon)
        else:
            selected_id = torch.arange(num_total_points, device=query_points.device)
            query_points = query_points[:, selected_id]

            num_patches = (num_total_points + patch_size - 1) // patch_size
            results = []
            for i in range(num_patches):
                start = i * patch_size
                end = min((i + 1) * patch_size, num_total_points)
                query_points_patch = query_points[:, start:end]
                recon_patch = nn.functional.grid_sample(
                    features[-1],
                    query_points_patch.unsqueeze(1).unsqueeze(1),
                    mode="bilinear",
                    align_corners=True,
                )
                recon_patch = self.decoder(recon_patch)
                results.append(recon_patch)
            recon = torch.cat(results, dim=-1)

        self.selected_id = selected_id
        return recon

    def loss(self, x, y, is_training=True):
        predicted_udf = x[:, 0:1, 0, 0]
        predicted_flag = x[:, 1:2, 0, 0]

        gt_udf = y[0]
        gt_voronoi_flag = y[1]

        gt_udf = gt_udf.reshape(-1, 1, 256 * 256 * 256)[:, :, self.selected_id]
        gt_voronoi_flag = gt_voronoi_flag.reshape(-1, 1, 256 * 256 * 256)[:, :, self.selected_id]

        udf_loss = nn.functional.mse_loss(predicted_udf, gt_udf)
        flag_loss = nn.functional.binary_cross_entropy_with_logits(
            predicted_flag,
            gt_voronoi_flag.to(torch.int32).to(torch.float32)
        )
        total_loss = udf_loss + flag_loss
        return {
            "udf_loss": udf_loss,
            "flag_loss": flag_loss,
            "total_loss": total_loss,
        }


class Diffusion1(nn.Module):
    def __init__(self, dim=16):
        super(Diffusion1, self).__init__()
        self.diffuser = PNDMPipeline.from_pretrained(model_id)

    def forward(self, x):
        # diffusion
        latent = self.diffuser(x)
        return latent


class Diffusion_phase(pl.LightningModule):
    def __init__(self, hparams):
        super(Diffusion_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.encoder = Autoencoder4()
        self.model = self.encoder
        # self.diffuser = Diffusion1()
        # Import module according to the dataset_name
        # mod = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
        self.dataset_name = nerf2nvd_dataset

        # Used for visualizing during the training
        self.viz_data = {
            "loss": [],
        }

        pr_computer = {
            "F1": BinaryF1Score(threshold=0.5),
        }
        self.pr_computer = MetricCollection(pr_computer)

    def train_dataloader(self):
        self.train_dataset = self.dataset_name(
            "training",
            self.hydra_conf["dataset"],
        )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          # collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_name(
            "validation",
            self.hydra_conf["dataset"],
        )
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )
        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        bs = batch[0].shape[0]
        outputs = self.model(batch, True)
        loss = self.model.loss(outputs, batch)
        for loss_name in loss:
            if loss_name == "total_loss":
                self.log("Training_Loss", loss[loss_name], prog_bar=True, logger=True, on_step=True, on_epoch=True,
                         sync_dist=True,
                         batch_size=bs)
            else:
                self.log("Training_" + loss_name, loss[loss_name], prog_bar=True, logger=True, on_step=False,
                         on_epoch=True,
                         sync_dist=True,
                         batch_size=bs)
        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):
        bs = batch[0].shape[0]
        outputs = self.model(batch, False)
        loss = self.model.loss(outputs, batch)
        for loss_name in loss:
            if loss_name == "total_loss":
                self.log("Validation_Loss", loss[loss_name], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=bs)
            else:
                self.log("Validation_" + loss_name, loss[loss_name], prog_bar=True, logger=True, on_step=False,
                         on_epoch=True,
                         sync_dist=True,
                         batch_size=bs)

        pred_voronoi = outputs[:, 1]
        prob = torch.sigmoid(pred_voronoi).reshape(bs, -1)
        gt = batch[1].reshape(bs, -1).to(torch.long)
        self.pr_computer.update(prob, gt)
        return

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self.log_dict(self.pr_computer.compute(), prog_bar=True, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True)
        self.pr_computer.reset()
        return


@hydra.main(config_name="nerf2nvd.yaml", config_path="../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")
    model = Diffusion_phase(v_cfg)

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
