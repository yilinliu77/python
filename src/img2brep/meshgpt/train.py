import sys
sys.path.append('../../../')
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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics import MetricCollection

import torch.distributed as dist

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAveragePrecision, BinaryF1Score

from src.img2brep.meshgpt.dataset import Auotoencoder_Dataset, Transformer_Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
    )

import open3d as o3d
import tqdm
from einops import rearrange, repeat, reduce, pack, unpack


def export_recon_faces(recon_faces, path):
    recon_faces = recon_faces.detach().cpu().numpy()
    vertices = recon_faces.reshape(-1, 3)

    # 生成三角形顶点索引，例如: [[0, 1, 2], [3, 4, 5], ...]
    triangles_indices = np.arange(vertices.shape[0]).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles_indices)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(path), mesh)

    # print(f'Mesh saved to: {path}')


class MeshGPTTraining(pl.LightningModule):
    def __init__(self, hparams):
        super(MeshGPTTraining, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.dataset_name = self.hydra_conf["dataset"]["dataset_name"]
        self.dataset_path = self.hydra_conf["dataset"]["root"]
        
        self.vis_recon_faces = self.hydra_conf["trainer"]["vis_recon_faces"]
        self.is_train_transformer = self.hydra_conf["trainer"]["train_transformer"]
        self.condition_on_text = self.hydra_conf["trainer"]["condition_on_text"]
        
        self.save_hyperparameters(hparams)

        self.log_root = Path(self.hydra_conf["trainer"]["output"])
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.autoencoder = MeshAutoencoder(num_discrete_coors=128, pad_id=-1)
        self.model = self.autoencoder

        if self.is_train_transformer:
            if self.hydra_conf["trainer"].checkpoint_autoencoder is None:
                raise ValueError("checkpoint_autoencoder is None")
            
            checkpoint_autoencoder = torch.load(self.hydra_conf["trainer"].checkpoint_autoencoder)["state_dict"]
            checkpoint_autoencoder_ = {k[12:]: v for k, v in checkpoint_autoencoder.items() if 'autoencoder' in k}
            self.autoencoder.load_state_dict(checkpoint_autoencoder_, strict=True)
            self.transformer = MeshTransformer(self.autoencoder, 
                                               max_seq_len=768, 
                                               coarse_pre_gateloop_depth = 2, # Better performance using more gateloop layers
                                               fine_pre_gateloop_depth= 2,
                                               attn_depth = 12,
                                               condition_on_text=self.condition_on_text)
            self.model = self.transformer

    def train_dataloader(self):
        if not self.is_train_transformer:
            self.train_dataset = Auotoencoder_Dataset("training", self.hydra_conf["dataset"],)
        else:
            self.train_dataset = Transformer_Dataset("training", self.hydra_conf["dataset"], self.autoencoder)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          # collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          # pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def val_dataloader(self):
        if not self.is_train_transformer:
            self.valid_dataset = Auotoencoder_Dataset("validation", self.hydra_conf["dataset"],)
        else:
            self.valid_dataset = Transformer_Dataset("validation", self.hydra_conf["dataset"], self.autoencoder)
            
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          # collate_fn=self.dataset_name.collate_fn,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          # pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False,
                          prefetch_factor=2 if self.hydra_conf["trainer"]["num_worker"] > 0 else None,
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
        return {
            'optimizer'   : optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor'  : 'Validation_Loss',
                }
            }

    def training_step(self, batch, batch_idx):
        data = batch
        
        if not self.is_train_transformer:
            loss = self.model(vertices=data['vertices'], faces=data["faces"], face_edges=data['face_edges'],)
        else:
            if self.condition_on_text:
                loss = self.model(vertices=None, faces=None, codes=data['codes'], text_embeds=data['img_embed'])
            else:
                loss = self.model(vertices=None, faces=None, codes=data['codes'])
                
        self.log("Training_Loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch

        if not self.is_train_transformer:
            recon_faces, loss = self.model(vertices=data['vertices'], faces=data["faces"], 
                                           face_edges=data['face_edges'], return_recon_faces=True, )
            
            if self.vis_recon_faces and self.current_epoch % 100 == 0:
                vertices_batch = data["vertices"]
                faces_batch = data["faces"]
                
                face_mask = reduce(faces_batch != self.autoencoder.pad_id, 'b nf c -> b nf', 'all')
                mse_loss_sum = []
                mse_loss = nn.MSELoss()
                for i in tqdm.tqdm(range(vertices_batch.shape[0])):
                    triangles_c = vertices_batch[i][faces_batch[i]]
                    recon_triangles_c = recon_faces[i]

                    face_mask_c = face_mask[i]
                    triangles_c = triangles_c[face_mask_c]
                    recon_triangles_c = recon_triangles_c[face_mask_c]

                    mse_loss_sum.append(mse_loss(triangles_c, recon_triangles_c))
                    export_recon_faces(triangles_c,
                                       self.log_root / f"batch{batch_idx}_{i}_epoch{self.current_epoch}_gt.ply")
                    export_recon_faces(recon_triangles_c,
                                       self.log_root / f"batch{batch_idx}_{i}_epoch{self.current_epoch}_recon.ply")

                mse_loss_mean = torch.mean(torch.stack(mse_loss_sum))

                self.log("Mse_Loss", mse_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=self.batch_size)
        
        else:
            if self.condition_on_text:
                loss = self.model(vertices=None, faces=None, codes=data['codes'], text_embeds=data['img_embed'])
            else:
                loss = self.model(vertices=None, faces=None, codes=data['codes'])
                
            if self.vis_recon_faces and self.current_epoch != 0 and self.current_epoch % 100 == 0:
                img_embed = data['img_embed']
                mse_loss_sum = []
                mse_loss = nn.MSELoss()
                for i in tqdm.tqdm(range(vertices_batch.shape[0])):
                    triangles_c = vertices_batch[i][faces_batch[i]]
                    
                    recon_faces, face_mask = self.model.generate(text_embeds=img_embed[i].unsqueeze(0), temperature=0, cond_scale=1)
                    
                    recon_triangles_c = recon_faces[face_mask]
                    
                    N = triangles_c.shape[0]
                    M = recon_triangles_c.shape[0]
                    
                    if M < N:
                        zeros_to_pad = torch.zeros((N - M, 3, 3), device=recon_triangles_c.device, dtype=recon_triangles_c.dtype)
                        recon_triangles_c = torch.cat((recon_triangles_c, zeros_to_pad), dim=0)
                    elif M > N:
                        recon_triangles_c = recon_triangles_c[:N, :, :]

                    mse_loss_sum.append(mse_loss(triangles_c, recon_triangles_c))
                    export_recon_faces(triangles_c,
                                       self.log_root / f"batch{batch_idx}_{i}_epoch{self.current_epoch}_gt.ply")
                    export_recon_faces(recon_triangles_c,
                                       self.log_root / f"batch{batch_idx}_{i}_epoch{self.current_epoch}_recon.ply")

                mse_loss_mean = torch.mean(torch.stack(mse_loss_sum))
                
                self.log("Mse_Loss", mse_loss_mean, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                         sync_dist=True,
                         batch_size=self.batch_size)

        self.log("Validation_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=self.batch_size)

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        return


@hydra.main(config_name="train_meshgpt.yaml", config_path="../../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    is_train_transformer = v_cfg["trainer"]["train_transformer"]
    if is_train_transformer:
        logger = TensorBoardLogger("tb_logs", name="transformer")
    else:
        logger = TensorBoardLogger("tb_logs", name="autoencoder")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")
    meshgptTraining = MeshGPTTraining(v_cfg)

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
            default_root_dir=log_dir,
            logger=logger,
            accelerator='gpu',
            strategy="auto",
            devices=v_cfg["trainer"].gpu,
            log_every_n_steps=25,
            enable_model_summary=False,
            callbacks=[mc, lr_monitor],
            max_epochs=int(1e8),
            num_sanity_val_steps=2,
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            precision=v_cfg["trainer"]["accelerator"],
            accumulate_grad_batches = 1,
            )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        
        if is_train_transformer:
            state_dict_ = {}
            for k, v in state_dict.items():
                if 'transformer.' in k:
                    state_dict_[k[12:]] = v
                elif 'model.' in k:
                    state_dict_[k[6:]] = v
        else:
            state_dict_ = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k}
        del state_dict

        meshgptTraining.model.load_state_dict(state_dict_, strict=True)
        

    if v_cfg["trainer"].evaluate:
        trainer.test(meshgptTraining)
    else:
        trainer.fit(meshgptTraining)


if __name__ == '__main__':
    main()
