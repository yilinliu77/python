import sys, os

sys.path.append("thirdparty/sdf_computer/build/")

from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import List

import mcubes
import tinycudann as tcnn

import PIL.Image
import numpy as np
import open3d
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tqdm import tqdm

import math
import platform
import shutil

from typing import Tuple

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from shared.fast_dataloader import FastDataLoader
from scipy import stats

from shared.common_utils import debug_imgs, to_homogeneous
from shared.img_torch_tools import get_img_from_tensor

import cv2

from src.neural_recon.colmap_dataset import Colmap_dataset, Blender_Segment_dataset
from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
from src.neural_recon.dataset import Single_img_dataset, Single_img_dataset_with_kdtree_index, \
    Geometric_dataset, Geometric_dataset_inference

import pysdf
import open3d as o3d

from src.neural_recon.models import Explorer, Segment_explorer
from src.neural_recon.phase1 import Phase1


class Phase2(pl.LightningModule):
    def __init__(self, hparams):
        super(Phase2, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        # self.log(...,batch_size=self.batch_size)
        self.save_hyperparameters(hparams)

        if not os.path.exists(self.hydra_conf["trainer"]["output"]):
            os.makedirs(self.hydra_conf["trainer"]["output"])

        bounds_min = np.array(self.hydra_conf["dataset"]["scene_boundary"][:3], dtype=np.float32)
        bounds_max = np.array(self.hydra_conf["dataset"]["scene_boundary"][3:], dtype=np.float32)
        self.bounds_center = (bounds_min + bounds_max) / 2
        self.bounds_size = (bounds_max - bounds_min).max()
        self.scene_bounds = np.array([bounds_min, bounds_max])

        # self.data = self.prepare_dataset(self.hydra_conf["dataset"]["colmap_dir"], self.scene_bounds)
        # self.model = self.prepare_model1(self.data["img_database"], self.hydra_conf["dataset"]["img_nif_dir"])
        self.data = self.prepare_dataset_blender(self.hydra_conf["dataset"]["colmap_dir"], self.scene_bounds)
        self.model, self.imgs = self.prepare_model2(self.data["img_database"])

    def prepare_dataset(self, v_colmap_dir, v_bounds):
        print("Start to load colmap")
        # Read colmap and segments
        imgs, world_points = read_dataset(v_colmap_dir, v_bounds)

        data = {}
        data["img_database"]: List[Image] = imgs
        data["world_points"]: List[Point_3d] = world_points

        print("Start to load mesh")
        # Read mesh and normalize it
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(v_colmap_dir, "gt_mesh.ply"))
        data["gt_mesh_vertices"] = np.asarray(self.mesh.vertices)
        data["gt_mesh_faces"] = np.asarray(self.mesh.triangles)
        data["gt_mesh_vertices"] = (data["gt_mesh_vertices"] - self.bounds_center) / self.bounds_size + 0.5

        # Compute gt visibility
        if True:
            # Set -1 to skip the test
            print("Start to sample points and compute sdf")
            id_test_img = -1
            sdf_computer = pysdf.PYSDF_computer()
            # Fix the bounds
            sdf_computer.setup_bounds(
                np.append(self.bounds_center, self.bounds_size)
            )
            sdf_computer.setup_mesh(data["gt_mesh_vertices"][data["gt_mesh_faces"]],
                                    False)  # Do not automatically compute the bounds
            # Sample points and calculate sdf
            sdf = sdf_computer.compute_sdf(int(1e0), int(1e4), int(1e4), False)
            data["sample_points"] = sdf[:, :3]
            data["sample_distances"] = sdf[:, 3:]

            print("Start to check visibility")
            # Start to check visibility
            pool = Pool(16)
            check_visibility(np.zeros((4, 4), dtype=np.float32),
                             np.zeros((2, 3), np.float32))  # Dummy, just for compiling the function
            visibility_inside_frustum = pool.map(
                partial(check_visibility, v_points=data["sample_points"]),
                [item.projection for item in imgs], chunksize=10)
            visibility_inside_frustum = np.stack(visibility_inside_frustum, axis=0)

            visibility_intersection_free = sdf_computer.check_visibility(
                np.asarray([item.pos for item in imgs]),
                data["sample_points"]
            )
            visibility_intersection_free = visibility_intersection_free.reshape([len(imgs), -1]).astype(bool)

            data["final_visibility"] = np.logical_and(visibility_inside_frustum, visibility_intersection_free)

            if id_test_img != -1:
                tr = o3d.geometry.TriangleMesh()
                pc = o3d.geometry.PointCloud()
                # Store normalized model
                tr.vertices = o3d.utility.Vector3dVector(vertices)
                tr.triangles = o3d.utility.Vector3iVector(faces)
                o3d.io.write_triangle_mesh("output/model.ply", tr)
                # Store normalized cameras
                pc.points = o3d.utility.Vector3dVector(np.asarray([item.pos for item in imgs]))
                o3d.io.write_point_cloud("output/cameras.ply", pc)
                # Store normalized sample points
                pc.points = o3d.utility.Vector3dVector(self.sample_points)
                o3d.io.write_point_cloud("output/1.ply", pc)
                # Store points inside frustum
                pc.points = o3d.utility.Vector3dVector(self.sample_points[visibility_inside_frustum[id_test_img]])
                o3d.io.write_point_cloud("output/2.ply", pc)
                # Store points that are collision-free
                pc.points = o3d.utility.Vector3dVector(
                    self.sample_points[visibility_intersection_free[id_test_img] == 1])
                o3d.io.write_point_cloud("output/3.ply", pc)
                # Store points that are visible to both
                pc.points = o3d.utility.Vector3dVector(self.sample_points[self.final_visibility[id_test_img]])
                o3d.io.write_point_cloud("output/4.ply", pc)
                pc.clear()
            pass

        return data

    def prepare_dataset_blender(self, v_data_dir, v_bounds):
        print("Start to load blender")

        model_matrix = np.zeros((4, 4), dtype=np.float32)
        model_matrix[0, 0] = self.bounds_size
        model_matrix[1, 1] = self.bounds_size
        model_matrix[2, 2] = self.bounds_size
        model_matrix[0, 3] = self.bounds_center[0] - self.bounds_size / 2
        model_matrix[1, 3] = self.bounds_center[1] - self.bounds_size / 2
        model_matrix[2, 3] = self.bounds_center[2] - self.bounds_size / 2
        model_matrix[3, 3] = 1

        # Read intrinsic
        intrinsic = np.loadtxt(os.path.join(v_data_dir, "cameras.txt")).astype(np.float32)

        intrinsic[0, 0] /= 1920
        intrinsic[0, 2] /= 1920
        intrinsic[1, 1] /= 1080
        intrinsic[1, 2] /= 1080

        # Read poses
        extrinsics = []
        for i in range(0, 4):
            extrinsic = np.loadtxt(os.path.join(v_data_dir, "{}.txt".format(i))).astype(np.float32)
            extrinsics.append(extrinsic)

        data = {}
        data["img_database"]: List[Image] = [
            Image(
                id_img=i,
                img_name="{}.png".format(i),
                img_path=os.path.join(v_data_dir, "{}.png".format(i)),
                pos=(np.linalg.inv(to_homogeneous(extrinsics[i]))[:3, 3] - self.bounds_center) / self.bounds_size + 0.5,
                intrinsic=to_homogeneous(intrinsic),
                extrinsic=to_homogeneous(extrinsics[i]),
                projection=to_homogeneous(intrinsic @ extrinsics[i] @ model_matrix),
                detected_points=np.zeros((1, 1)),
                detected_lines=np.zeros((1, 1)),
                line_field=np.zeros((1, 1)),
                line_field_path="",
                img_size=(1920, 1080)
            ) for i in range(len(extrinsics))]

        # Read mesh and normalize it
        self.mesh = o3d.io.read_triangle_mesh(os.path.join(v_data_dir, "annoying.ply"))
        data["gt_mesh_vertices"] = np.asarray(self.mesh.vertices)
        data["gt_mesh_faces"] = np.asarray(self.mesh.triangles)
        data["gt_mesh_vertices"] = (data["gt_mesh_vertices"] - self.bounds_center) / self.bounds_size + 0.5

        # Compute gt visibility
        if True:
            # Set -1 to skip the test
            print("Start to sample points and compute sdf")
            id_test_img = -1
            sdf_computer = pysdf.PYSDF_computer()
            # Fix the bounds
            sdf_computer.setup_bounds(
                np.append(self.bounds_center, self.bounds_size)
            )
            sdf_computer.setup_mesh(data["gt_mesh_vertices"][data["gt_mesh_faces"]],
                                    False)  # Do not automatically compute the bounds
            # Sample points and calculate sdf
            num_point_near_surface = int(1e6)
            num_point_uniform = int(1e6)
            sdf = sdf_computer.compute_sdf(int(1e0), num_point_near_surface, num_point_uniform, False)
            data["sample_points"] = sdf[:, :3]
            data["sample_distances"] = sdf[:, 3:]

            print("Start to check visibility")
            # Start to check visibility
            pool = Pool(16)
            check_visibility(np.zeros((4, 4), dtype=np.float32),
                             np.zeros((2, 3), np.float32))  # Dummy, just for compiling the function
            visibility_inside_frustum = pool.map(
                partial(check_visibility, v_points=data["sample_points"]),
                [item.projection for item in data["img_database"]], chunksize=min(10, len(data["img_database"]) // 2))
            visibility_inside_frustum = np.stack(visibility_inside_frustum, axis=0)

            visibility_intersection_free = sdf_computer.check_visibility(
                np.asarray([item.pos for item in data["img_database"]]),
                data["sample_points"]
            )
            visibility_intersection_free = visibility_intersection_free.reshape([len(data["img_database"]), -1]).astype(
                bool)

            data["final_visibility"] = np.logical_and(visibility_inside_frustum, visibility_intersection_free)

            if id_test_img != -1:
                tr = o3d.geometry.TriangleMesh()
                pc = o3d.geometry.PointCloud()
                # Store normalized model
                tr.vertices = o3d.utility.Vector3dVector(data["gt_mesh_vertices"])
                tr.triangles = o3d.utility.Vector3iVector(data["gt_mesh_faces"])
                o3d.io.write_triangle_mesh("output/model.ply", tr)
                # Store normalized cameras
                pc.points = o3d.utility.Vector3dVector(np.asarray([item.pos for item in data["img_database"]]))
                o3d.io.write_point_cloud("output/cameras.ply", pc)
                # Store normalized sample points
                pc.points = o3d.utility.Vector3dVector(data["sample_points"])
                o3d.io.write_point_cloud("output/1.ply", pc)
                # Store points inside frustum
                pc.points = o3d.utility.Vector3dVector(data["sample_points"][visibility_inside_frustum[id_test_img]])
                o3d.io.write_point_cloud("output/2.ply", pc)
                # Store points that are collision-free
                pc.points = o3d.utility.Vector3dVector(
                    data["sample_points"][visibility_intersection_free[id_test_img] == 1])
                o3d.io.write_point_cloud("output/3.ply", pc)
                # Store points that are visible to both
                pc.points = o3d.utility.Vector3dVector(data["sample_points"][data["final_visibility"][id_test_img]])
                o3d.io.write_point_cloud("output/4.ply", pc)
                pc.clear()

            # Sample segments
            # Both start and end point near surface
            p = data["sample_points"]
            v = data["final_visibility"]
            num_segment = int(1e6)
            index1 = np.random.randint(0, num_point_near_surface, num_segment)
            index2 = np.random.randint(0, num_point_near_surface, num_segment)
            segments1 = np.stack([p[index1], p[index2]], axis=1)
            visibility1 = np.logical_and(v[:,index1],v[:,index2])

            # Only start or end point near surface
            index1 = np.random.randint(0, num_point_near_surface, num_segment)
            index2 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
            segments21 = np.stack([p[index1], p[index2]], axis=1)
            segments22 = np.stack([p[index2], p[index1]], axis=1)
            visibility21 = np.logical_and(v[:,index1],v[:,index2])
            visibility22 = np.logical_and(v[:,index2],v[:,index1])

            # None of the points near surface
            index1 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
            index2 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
            segments3 = np.stack([p[index1], p[index2]], axis=1)
            visibility3 = np.logical_and(v[:,index1],v[:,index2])

            segments = np.concatenate((segments1, segments21, segments22, segments3), axis=0)
            segments_visibility = np.concatenate((visibility1, visibility21, visibility22, visibility3), axis=1)
            data["segments"] = segments
            data["segments_visibility"] = segments_visibility
            pass

        return data

    def prepare_model1(self, v_imgs, v_nif_dir):
        print("Start to load image models")
        img_names = [item.img_name for item in v_imgs]
        img_models = {}

        def load_img_model(img_name):
            if os.path.exists(os.path.join(v_nif_dir, img_name)):
                checkpoint_name = [item for item in os.listdir(os.path.join(v_nif_dir, img_name)) if
                                   item[-4:] == "ckpt"]
                assert len(checkpoint_name) == 1
                state_dict = torch.load(os.path.join(v_nif_dir, img_name, checkpoint_name[0]))["state_dict"]
                fake_cfg = {
                    "trainer": {
                        "learning_rate": 0,
                        "batch_size": 0,
                        "num_worker": 0,
                        "output": "output",
                    },
                    "dataset": {
                        "img_size": [600, 400],
                    }
                }
                img_model = Phase1(fake_cfg, v_imgs[0].img_path)
                img_model.load_state_dict(state_dict, strict=True)
                img_model.eval()
                return img_model
                img_models[img_name] = img_model
            else:
                print("cannot find model for img {}".format(img_name))

        # for img_name in tqdm(img_names):
        img_models = thread_map(load_img_model, img_names)
        img_models = dict(zip(img_names, img_models))
        self.model = Explorer(img_models)
        return img_models

    def prepare_model2(self, v_imgs):
        img_size = self.hydra_conf["dataset"]["trained_img_size"]

        def read_img(v_path):
            original_img = cv2.imread(v_path,cv2.IMREAD_UNCHANGED)
            resized_img = cv2.resize(original_img, img_size)
            transform_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
            gx, gy = np.gradient(gray_img)
            return transform_img, np.stack((gx,gy), axis=2)

        imgs = {item.img_name:read_img(item.img_path) for item in v_imgs}
        model = Segment_explorer(imgs)
        return model, imgs

    def forward(self, v_data):
        pass

    def train_dataloader(self):
        self.train_dataset = Blender_Segment_dataset(
            self.data,
            self.imgs,
            "validation",
        )
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=Blender_Segment_dataset.collate_fn,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Blender_Segment_dataset(
            self.data,
            self.imgs,
            "validation",
        )
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=Blender_Segment_dataset.collate_fn,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def test_dataloader(self):
        self.test_dataset = Geometric_dataset_inference(
            self.hydra_conf["model"]["marching_cube_resolution"],
            self.hydra_conf["trainer"]["batch_size"],
        )
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          # collate_fn=Geometric_dataset.collate_fn,
                          )

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': CosineAnnealingLR(optimizer, T_max=500., eta_min=3e-5),
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        predicted_prediction = self.model(batch)
        loss = self.model.loss(predicted_prediction, batch)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch["id"].shape[0])

        return loss if loss != 0 else torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch, batch_idx):
        predicted_prediction = self.model(batch)
        loss = self.model.loss(predicted_prediction, batch)

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch["id"].shape[0])

        return loss

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        predicted_sdf = -torch.cat(result, dim=0).cpu().numpy().astype(np.float32)
        resolution = self.hydra_conf["model"]["marching_cube_resolution"]
        predicted_sdf = predicted_sdf.reshape([resolution, resolution, resolution])
        vertices, triangles = mcubes.marching_cubes(predicted_sdf, 0)
        if vertices.shape[0] != 0:
            mcubes.export_obj(vertices, triangles,
                              os.path.join("output", "model_of_epoch_{}.obj".format(self.trainer.current_epoch)))

    def test_step(self, batch, batch_idx):
        batch = [batch[0][0], batch[1][0]]
        predicted_sdf = self.forward(batch)
        return predicted_sdf

    def test_epoch_end(self, result):
        predicted_sdf = -torch.cat(result, dim=0).cpu().numpy().astype(np.float32)
        resolution = self.hydra_conf["model"]["marching_cube_resolution"]
        predicted_sdf = predicted_sdf.reshape([resolution, resolution, resolution])
        vertices, triangles = mcubes.marching_cubes(predicted_sdf, 0)
        mcubes.export_obj(vertices, triangles,
                          os.path.join("outputs", "model_of_test.obj"))


@hydra.main(config_name="phase2_annoying.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    print(OmegaConf.to_yaml(v_cfg))
    seed_everything(0)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=10000,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=16,
        reload_dataloaders_every_n_epochs=v_cfg["dataset"]["resample_after_n_epoches"]
        # profiler=PyTorchProfiler(),
    )

    model = Phase2(v_cfg)
    if v_cfg["trainer"].resume_from_checkpoint is not None:
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    if False:
        import open3d as o3d

        sys.path.append("thirdparty/sdf_computer/build/")
        import pysdf

        mesh = o3d.io.read_triangle_mesh(
            "/mnt/d/Projects/NeuralRecon/Test_data/OBL_L7/Test_imgs2_colmap_neural/sparse_align/detailed_l7_with_ground.ply")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if True:
            sdf_computer = pysdf.SDF_computer(vertices[faces])
            query_points = sdf_computer.compute_sdf(int(1e6), int(1e6), int(1e6))
            a = o3d.geometry.PointCloud()
            selected_points = query_points[query_points[:, 3] > 0.07][:, :3]
            a.points = o3d.utility.Vector3dVector(selected_points)
            o3d.io.write_point_cloud("/mnt/d/Projects/NeuralRecon/tmp/sdf.ply", a)
    main()
