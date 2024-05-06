import importlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv
from vector_quantize_pytorch import ResidualLFQ, VectorQuantize, ResidualVQ

import pytorch_lightning as pl

from shared.common_utils import *
from src.img2brep.brep.common import *
from src.img2brep.brep.model_encoder import GAT_GraphConv, SAGE_GraphConv
from src.img2brep.brep.model_fuser import Attn_fuser_cross, Attn_fuser_single

import open3d as o3d


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class TrainAutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAutoEncoder, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]

        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        dataset_mod = importlib.import_module('src.img2brep.brep.dataset')
        self.dataset_mod = getattr(dataset_mod, self.hydra_conf["dataset"]["dataset_name"])

        model_mod = importlib.import_module('src.img2brep.brep.autoencoder_model')
        self.model = getattr(model_mod, self.hydra_conf["model"]["model_name"])(self.hydra_conf["model"])
        self.viz = {}

    def train_dataloader(self):
        self.train_dataset = self.dataset_mod("training", self.hydra_conf["dataset"], )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.dataset_mod.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        self.valid_dataset = self.dataset_mod("validation", self.hydra_conf["dataset"], )

        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.dataset_mod.collate_fn,
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

        if batch_idx == 0 and self.hydra_conf["trainer"]["output_validation_model"]:
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
        if self.hydra_conf["trainer"]["output_validation_model"]:
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
        self.test_dataset = self.dataset_mod("testing", self.hydra_conf["dataset"], )

        return DataLoader(self.test_dataset, batch_size=1,
                          collate_fn=self.dataset_mod.collate_fn,
                          shuffle=False,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          )

    def test_step(self, batch, batch_idx):
        data = batch

        loss, recon_data = self.model(data, return_loss=True,
                                      return_recon=True, return_face_features=True, return_true_loss=True)
        face_embeddings = recon_data["face_embeddings"]
        inferenced_vertices, inferenced_edges, inferenced_faces, = self.model.inference(face_embeddings, False)
        # ============================= Loss stuffs =============================
        total_loss = loss["total_loss"]
        for key in loss:
            if key == "total_loss":
                continue
            self.log(f"Test_{key}", loss[key], prog_bar=True, logger=False, on_step=False, on_epoch=True,
                     batch_size=self.batch_size)
        self.log("Test_Loss", total_loss, prog_bar=True, logger=False, on_step=False, on_epoch=True,
                 batch_size=self.batch_size)

        # ============================= Loss stuffs =============================
        prefix = data["v_prefix"][0]
        gt_face = data["face_points"].cpu().numpy()[0]
        gt_edge_point = data["edge_points"].cpu().numpy()[0]
        gt_vertex = data["vertex_points"].cpu().numpy()[0]
        gt_edge_face_connectivity = data["edge_face_connectivity"].cpu().numpy()[0]
        gt_vertex_edge_connectivity = data["vertex_edge_connectivity"].cpu().numpy()[0]

        pred_face = inferenced_faces.cpu().numpy()
        pred_edge_point = inferenced_edges.cpu().numpy()
        pred_vertex = inferenced_vertices.cpu().numpy()
        # pred_edge_face_connectivity = face_edge_connectivity.cpu().numpy()[0]
        # pred_vertex_edge_connectivity = edge_vertex_connectivity.cpu().numpy()[0]

        root = Path(self.hydra_conf["trainer"]["test_output_dir"])
        check_dir(root / prefix)

        np.savez(str(root / prefix / f"data.npz"),
                 gt_face=gt_face,
                 gt_edge_point=gt_edge_point,
                 gt_vertex=gt_vertex,
                 gt_edge_face_connectivity=gt_edge_face_connectivity,
                 gt_vertex_edge_connectivity=gt_vertex_edge_connectivity,
                 pred_face=pred_face,
                 pred_edge_point=pred_edge_point,
                 pred_vertex=pred_vertex,
                 # pred_edge_face_connectivity=pred_edge_face_connectivity,
                 # pred_vertex_edge_connectivity=pred_vertex_edge_connectivity,
                 )

        gt_color = np.array([[1, 0, 0]], dtype=np.float32)
        pred_color = np.array([[0, 1, 0]], dtype=np.float32)

        points = np.concatenate(
            (gt_face.reshape(-1, 3), pred_face.reshape(-1, 3)), axis=0)
        colors = np.concatenate((
            np.repeat(gt_color, gt_face.reshape(-1, 3).shape[0], axis=0),
            np.repeat(pred_color, pred_face.reshape(-1, 3).shape[0], axis=0)), axis=0)
        export_point_cloud(str(root / prefix / f"face.ply"), points, colors)

        points = np.concatenate(
            (gt_edge_point.reshape(-1, 3), pred_edge_point.reshape(-1, 3)), axis=0)
        colors = np.concatenate((
            np.repeat(gt_color, gt_edge_point.reshape(-1, 3).shape[0], axis=0),
            np.repeat(pred_color, pred_edge_point.reshape(-1, 3).shape[0], axis=0)), axis=0)
        export_point_cloud(str(root / prefix / f"edge.ply"), points, colors)

        points = np.concatenate(
            (gt_vertex.reshape(-1, 3), pred_vertex.reshape(-1, 3)), axis=0)
        colors = np.concatenate((
            np.repeat(gt_color, gt_vertex.reshape(-1, 3).shape[0], axis=0),
            np.repeat(pred_color, pred_vertex.reshape(-1, 3).shape[0], axis=0)), axis=0)
        export_point_cloud(str(root / prefix / f"vertex.ply"), points, colors)

        if self.hydra_conf["trainer"]["save_face_embedding"]:
            safe_check_dir(root / "face_embedding")
            np.save(root / "face_embedding" / f"{prefix}", face_embeddings.cpu().numpy()[0])
        return

    def on_test_end(self) -> None:
        for key in self.trainer.callback_metrics:
            print("{}: {:.5f}".format(key, self.trainer.callback_metrics[key].cpu().numpy().item()))
