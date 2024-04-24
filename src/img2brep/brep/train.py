import sys
from tqdm import tqdm

from shared.common_utils import check_dir, safe_check_dir
from src.img2brep.brep.autoregressive import AutoregressiveModel
from src.img2brep.brep.diffusion import DiffusionModel

sys.path.append('../../../')
import os.path
from pathlib import Path

import hydra
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler
from lightning_fabric import seed_everything

from src.img2brep.brep.dataset import Autoencoder_Dataset, Face_feature_dataset
from src.img2brep.brep.model import AutoEncoder


class TrainAutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAutoEncoder, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.dataset_name = self.hydra_conf["dataset"]["dataset_name"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.autoencoder = AutoEncoder(self.hydra_conf["model"])
        self.model = self.autoencoder

        self.viz = {}

    def train_dataloader(self):
        self.train_dataset = Autoencoder_Dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Autoencoder_Dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor'  : 'Validation_Loss',
            #     }
            }

    def training_step(self, batch, batch_idx):
        data = batch

        loss, data = self.model(data, return_loss=True,
                                return_recon=False, return_face_features=False, return_true_loss=False)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)
        if torch.isnan(total_loss).any():
            print("NAN Loss")
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, return_loss=True,
                                      return_recon=True, return_face_features=False, return_true_loss=True)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == 0:
            # recon_edges, recon_faces = self.model.inference(recon_data["face_embeddings"])
            self.viz["face_points"] = data["face_points"].cpu().numpy()
            self.viz["edge_points"] = data["edge_points"].cpu().numpy()
            self.viz["vertex_points"] = data["vertex_points"].cpu().numpy()
            self.viz["recon_faces"] = recon_data["recon_faces"].cpu().numpy()
            self.viz["recon_edges"] = recon_data["recon_edges"].cpu().numpy()
            self.viz["recon_vertices"] = recon_data["recon_vertices"].cpu().numpy()

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        #     return
        v_gt_vertices = self.viz["vertex_points"]
        v_gt_edges = self.viz["edge_points"]
        v_gt_faces = self.viz["face_points"]
        v_recon_vertices = self.viz["recon_vertices"]
        v_recon_edges = self.viz["recon_edges"]
        v_recon_faces = self.viz["recon_faces"]

        for idx in range(min(v_gt_edges.shape[0], 4)):
            gt_vertices = v_gt_vertices[idx]
            gt_edges = v_gt_edges[idx]
            gt_faces = v_gt_faces[idx]
            recon_vertices = v_recon_vertices[idx]
            recon_edges = v_recon_edges[idx]
            recon_faces = v_recon_faces[idx]

            gt_edges = gt_edges[(gt_edges != -1).all(axis=-1).all(axis=-1)]
            recon_edges = recon_edges[(recon_edges != -1).all(axis=-1).all(axis=-1)]
            gt_faces = gt_faces[(gt_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)]
            recon_faces = recon_faces[(recon_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)]
            gt_vertices = gt_vertices[(gt_vertices != -1).all(axis=-1)]
            recon_vertices = recon_vertices[(recon_vertices != -1).all(axis=-1)]

            vertex_points = np.concatenate((gt_vertices, recon_vertices), axis=0).reshape(-1, 3)
            vertex_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_vertices.shape[0], axis=0),
                        np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_vertices.shape[0], axis=0)), axis=0)

            edge_points = np.concatenate((gt_edges, recon_edges), axis=0).reshape(-1, 3)
            edge_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_edges.shape[0] * 20, axis=0),
                     np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_edges.shape[0] * 20, axis=0)), axis=0)

            face_points = np.concatenate((gt_faces, recon_faces), axis=0).reshape(-1, 3)
            face_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_faces.shape[0] * 400, axis=0),
                     np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_faces.shape[0] * 400, axis=0)), axis=0)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vertex_points)
            pc.colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
            o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_vertices.ply"), pc)

            pc.points = o3d.utility.Vector3dVector(edge_points)
            pc.colors = o3d.utility.Vector3dVector(edge_colors / 255.0)
            o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_edges.ply"), pc)

            pc.points = o3d.utility.Vector3dVector(face_points)
            pc.colors = o3d.utility.Vector3dVector(face_colors / 255.0)
            o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_idx_{idx:02}_viz_faces.ply"), pc)
        return

    def test_dataloader(self):
        self.test_dataset = Autoencoder_Dataset("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=1,
                          collate_fn=Autoencoder_Dataset.collate_fn,
                          shuffle=False,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          )

    def test_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, return_loss=True,
                                      return_recon=True, return_face_features=True, return_true_loss=True)
        face_embeddings = recon_data["face_embeddings"]
        inferenced_vertices, inferenced_edges, inferenced_faces = self.model.inference(face_embeddings)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Test_{key}", loss[key], prog_bar=True, logger=False, on_step=False, on_epoch=True,
                     batch_size=self.batch_size)
        self.log("Test_Loss", total_loss, prog_bar=True, logger=False, on_step=False, on_epoch=True,
                 batch_size=self.batch_size)


        if batch_idx == 0:
            self.viz = {
                "prefixes": [],
                "gt_face_points": [],
                "gt_edge_points": [],
                "gt_vertex_points": [],

                "pred_face_points": [],
                "pred_edge_points": [],
                "pred_vertex_points": [],
            }

        prefix = data["v_prefix"][0]
        gt_face = data["face_points"].cpu().numpy()
        gt_edge_point = data["edge_points"].cpu().numpy()
        gt_vertex = data["vertex_points"].cpu().numpy()
        pred_face = inferenced_faces.cpu().numpy()
        pred_edge_point = inferenced_edges.cpu().numpy()
        pred_vertex = inferenced_vertices.cpu().numpy()


        root = Path(self.hydra_conf["trainer"]["test_output_dir"])
        check_dir(root / prefix)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.concatenate(
            (gt_face.reshape(-1, 3), pred_face.reshape(-1, 3)), axis=0))
        pc.colors = o3d.utility.Vector3dVector(np.concatenate((
            np.repeat(np.array([[255, 0, 0]], dtype=np.uint8),
                      gt_face.reshape(-1, 3).shape[0], axis=0),
            np.repeat(np.array([[0, 255, 0]], dtype=np.uint8),
                      pred_face.reshape(-1, 3).shape[0], axis=0)),
            axis=0) / 255.0)
        o3d.io.write_point_cloud(str(root / prefix / f"face.ply"), pc)

        pc.points = o3d.utility.Vector3dVector(np.concatenate(
            (gt_edge_point.reshape(-1, 3), pred_edge_point.reshape(-1, 3)), axis=0))
        pc.colors = o3d.utility.Vector3dVector(np.concatenate((
            np.repeat(np.array([[255, 0, 0]], dtype=np.uint8),
                      gt_edge_point.reshape(-1, 3).shape[0], axis=0),
            np.repeat(np.array([[0, 255, 0]], dtype=np.uint8),
                      pred_edge_point.reshape(-1, 3).shape[0], axis=0)),
            axis=0) / 255.0)
        o3d.io.write_point_cloud(str(root / prefix / f"edge.ply"), pc)

        pc.points = o3d.utility.Vector3dVector(np.concatenate(
            (gt_vertex.reshape(-1, 3), pred_vertex.reshape(-1, 3)), axis=0))
        pc.colors = o3d.utility.Vector3dVector(np.concatenate((
            np.repeat(np.array([[255, 0, 0]], dtype=np.uint8),
                      gt_vertex.reshape(-1, 3).shape[0], axis=0),
            np.repeat(np.array([[0, 255, 0]], dtype=np.uint8),
                      pred_vertex.reshape(-1, 3).shape[0], axis=0)),
            axis=0) / 255.0)
        o3d.io.write_point_cloud(str(root / prefix / f"vertex.ply"), pc)

        if self.hydra_conf["trainer"]["save_face_embedding"]:
            safe_check_dir(root / "face_embedding")
            np.save(root / "face_embedding" / f"{prefix}", face_embeddings.cpu().numpy()[0])
        return

    def on_test_end(self) -> None:
        for key in self.trainer.callback_metrics:
            print("{}: {:.5f}".format(key, self.trainer.callback_metrics[key].cpu().numpy().item()))



class TrainAutoregressiveModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAutoregressiveModel, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.dataset_name = self.hydra_conf["dataset"]["dataset_name"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.model = DiffusionModel(
                self.hydra_conf["model"]
                )

        self.viz_recon = {}
        self.viz_gen = {}

    def train_dataloader(self):
        self.train_dataset = Face_feature_dataset("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = Face_feature_dataset("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=Face_feature_dataset.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-8, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=20, threshold=1e-5, min_lr=1e-5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor'  : 'Training_Loss',
            #     }
            }

    def training_step(self, batch, batch_idx):
        data = batch

        loss, _ = self.model(data, only_return_loss=True)
        total_loss = loss["total_loss"]
        # for key in loss:
        #     if key == "total_loss":
        #         continue
        #     self.log(f"Training_{key}", loss[key], prog_bar=True, logger=True, on_step=False, on_epoch=True,
        #              sync_dist=True, batch_size=self.batch_size)
        self.log("Training_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=False, batch_size=self.batch_size)
        # if torch.isnan(total_loss).any():
        #     print("NAN Loss")
        return total_loss

    def validation_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, only_return_loss=False)
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Validation_{key}", loss[key], prog_bar=True, logger=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=self.batch_size)
        self.log("Validation_Loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=self.batch_size)

        if batch_idx == 0:
            recon_vertices, recon_edges, recon_faces = self.model.generate(batch_size=1)
            # self.viz_recon["face_points"] = data["face_points"].cpu().numpy()
            # self.viz_recon["line_points"] = data["edge_points"].cpu().numpy()
            self.viz_recon["recon_edges"] = recon_edges.cpu().numpy()
            self.viz_recon["recon_faces"] = recon_faces.cpu().numpy()

        return total_loss

    def on_validation_epoch_end(self):
        # if self.trainer.sanity_checking:
        # return

        def vis(viz_data, subname):
            assert subname in ["recon", "gen"]

            recon_edges = viz_data["recon_edges"]
            recon_faces = viz_data["recon_faces"]

            valid_flag = (recon_edges != -1).all(axis=-1).all(axis=-1)
            recon_edges = recon_edges[valid_flag]
            valid_flag = (recon_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)
            recon_faces = recon_faces[valid_flag]

            edge_points = recon_edges.reshape(-1, 3)

            face_points = recon_faces.reshape(-1, 3)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(edge_points)
            o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_viz_edges_{subname}.ply"),
                    pc)

            pc.points = o3d.utility.Vector3dVector(face_points)
            o3d.io.write_point_cloud(
                    str(self.log_root / f"{self.trainer.current_epoch:05}_viz_faces_{subname}.ply"),
                    pc)

        vis(self.viz_recon, "recon")

        return


@hydra.main(config_name="train_brepgen.yaml", config_path="../../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    train_autoregressive = v_cfg["trainer"]["train_autoregressive"]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if train_autoregressive:
        modelTraining = TrainAutoregressiveModel(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="transformer")
    else:
        modelTraining = TrainAutoEncoder(v_cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, "tb_logs_brepgen"), name="autoencoder")

    trainer = Trainer(
            default_root_dir=log_dir,
            logger=logger,
            accelerator='gpu',
            # strategy="ddp_find_unused_parameters_false" if v_cfg["trainer"].gpu > 1 else "auto",
            strategy="auto",
            devices=v_cfg["trainer"].gpu,
            log_every_n_steps=25,
            enable_model_summary=False,
            callbacks=[mc, lr_monitor],
            max_epochs=int(v_cfg["trainer"]["max_epochs"]),
            num_sanity_val_steps=2,
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            precision=v_cfg["trainer"]["accelerator"],
            # accumulate_grad_batches=1,
            profiler=AdvancedProfiler(dirpath=log_dir, filename="profiler.txt"),
            )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]

        if train_autoregressive:
            state_dict_ = {}
            for k, v in state_dict.items():
                if 'transformer.' in k:
                    state_dict_[k[12:]] = v
                elif 'model.' in k:
                    state_dict_[k[6:]] = v
        else:
            state_dict_ = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k}
        del state_dict

        modelTraining.model.load_state_dict(state_dict_, strict=True)

    if v_cfg["trainer"].evaluate:
        trainer.test(modelTraining)
    else:
        trainer.fit(modelTraining)


if __name__ == '__main__':
    main()
