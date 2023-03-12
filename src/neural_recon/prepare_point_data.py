import shutil
import sys, os

sys.path.append("thirdparty/sdf_computer/build/")
import ctypes
import time
from itertools import chain

import cv2
import psutil

import pysdf

import hydra
from lightning_lite import seed_everything
from omegaconf import DictConfig, OmegaConf

from shared.common_utils import to_homogeneous
from shared.sdf_computer import normalize_intrinsic_and_extrinsic
from src.neural_recon.colmap_io import Image, check_visibility, read_dataset

from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, Array, RawArray
import ray
from typing import List

import mcubes
import tinycudann as tcnn

import PIL.Image
import numpy as np
import open3d
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import math

from src.neural_recon.generate_gt_point import compute_loss, ProgressActor
import open3d as o3d


@hydra.main(config_name="point_based_regression.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(hydra_conf: DictConfig):
    print(OmegaConf.to_yaml(hydra_conf))
    seed_everything(0)

    print("Start to load blender")
    pool = Pool(16)

    # Load data
    if True:
        bounds_min = np.array(hydra_conf["dataset"]["scene_boundary"][:3], dtype=np.float32)
        bounds_max = np.array(hydra_conf["dataset"]["scene_boundary"][3:], dtype=np.float32)
        bounds_center = (bounds_min + bounds_max) / 2
        bounds_size = (bounds_max - bounds_min).max()
        original_img_size = hydra_conf["dataset"]["original_img_size"]
        trained_img_size = hydra_conf["dataset"]["trained_img_size"]

        root_data_dir = hydra_conf["dataset"]["colmap_dir"]
        data = {}
        if False:
            # Read intrinsic
            intrinsic = np.loadtxt(os.path.join(root_data_dir, "cameras.txt")).astype(np.float32)

            # Read poses
            extrinsics = []
            for i in range(0, 4):
                extrinsic = np.loadtxt(os.path.join(root_data_dir, "{}.txt".format(i))).astype(np.float32)
                extrinsics.append(extrinsic)

            normalized_intrinsic, normalized_extrinsic, normalized_positions, normalized_projections = normalize_intrinsic_and_extrinsic(
                bounds_center, bounds_size, original_img_size, intrinsic, np.stack(extrinsics))


            data["img_database"]: List[Image] = [
                Image(
                    id_img=i,
                    img_name="{}.png".format(i),
                    img_path=os.path.join(root_data_dir, "{}.png".format(i)),
                    pos=normalized_positions[i],
                    intrinsic=intrinsic,
                    extrinsic=normalized_extrinsic[i],
                    projection=normalized_projections[i],
                    detected_points=np.zeros((1, 1)),
                    detected_lines=np.zeros((1, 1)),
                    line_field=np.zeros((1, 1)),
                    line_field_path="",
                    img_size=(1920, 1080)
                ) for i in range(len(extrinsics))]

            # Read mesh and normalize it
            mesh = o3d.io.read_triangle_mesh(os.path.join(root_data_dir, "annoying.ply"))
            data["gt_mesh_vertices"] = np.asarray(mesh.vertices)
            data["gt_mesh_faces"] = np.asarray(mesh.triangles)
            data["gt_mesh_vertices"] = (data["gt_mesh_vertices"] - bounds_center) / bounds_size + 0.5
        else:
            data["img_database"], _ = read_dataset(root_data_dir, [bounds_min,bounds_max])
            # Read mesh and normalize it
            mesh = o3d.io.read_triangle_mesh(os.path.join(root_data_dir, "detailed_l7_with_ground.ply"))
            data["gt_mesh_vertices"] = np.asarray(mesh.vertices)
            data["gt_mesh_faces"] = np.asarray(mesh.triangles)
            data["gt_mesh_vertices"] = (data["gt_mesh_vertices"] - bounds_center) / bounds_size + 0.5

    # Compute gt visibility
    if True:
        # Set -1 to skip the test
        print("Start to sample points and compute sdf")
        id_test_img = -1
        sdf_computer = pysdf.PYSDF_computer()
        # Fix the bounds
        sdf_computer.setup_bounds(
            np.append(bounds_center, bounds_size)
        )
        sdf_computer.setup_mesh(data["gt_mesh_vertices"][data["gt_mesh_faces"]],
                                False)  # Do not automatically compute the bounds
        # Sample points and calculate sdf
        num_point_on_surface = int(hydra_conf["dataset"]["num_sample"][0])
        num_point_near_surface = int(hydra_conf["dataset"]["num_sample"][1])
        num_point_uniform = int(hydra_conf["dataset"]["num_sample"][2])
        sdf = sdf_computer.compute_sdf(num_point_on_surface, num_point_near_surface, num_point_uniform, False)
        data["sample_points"] = sdf[:, :3]
        data["sample_distances"] = sdf[:, 3:]

        print("Start to check visibility")
        # Start to check visibility
        print("Check frustum")
        check_visibility(np.zeros((4, 4), dtype=np.float32),
                         np.zeros((2, 3), np.float32))  # Dummy, just for compiling the function
        visibility_inside_frustum = pool.map(
            partial(check_visibility, v_points=data["sample_points"]),
            np.stack([item.projection for item in data["img_database"]]),
            chunksize=min(10, len(data["img_database"]) // 2))
        visibility_inside_frustum = np.stack(visibility_inside_frustum, axis=0)

        print("Check intersection")
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
        pool.close()

    # Prepare GT data
    if True:
        print("Start to calculate gt loss")
        shutil.rmtree("output/gt_loss", ignore_errors=True)
        def read_img(v_path):
            original_img = cv2.imread(v_path, cv2.IMREAD_UNCHANGED)
            resized_img = cv2.resize(original_img, trained_img_size)
            transform_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            return transform_img

        imgs = {item.img_name: read_img(item.img_path) for item in data["img_database"]}
        projections = [item.projection for item in data["img_database"]]
        Ks = [item.intrinsic for item in data["img_database"]]
        img_names = [item.img_name for item in data["img_database"]]

        num_points = data["sample_points"].shape[0]

        num_worker = psutil.cpu_count(logical=True)
        num_sample_per_worker = num_points // num_worker + 1
        print("Start to ray with {} workers and {} samples per worker".format(num_worker, num_sample_per_worker))
        ray.init(include_dashboard=True,
                 dashboard_host="0.0.0.0",
                 dashboard_port=15001,
                 num_cpus=num_worker,
                 # num_cpus=0,
                 # local_mode=True
                 )

        imgs_id = ray.put(imgs)
        projections_id = ray.put(projections)
        img_names_id = ray.put(img_names)

        progress_actor = ProgressActor.remote(num_points)

        print("Start a dummy task to compile numba function")
        dummy_task = compute_loss.remote(
            0,
            10,
            data["sample_points"],
            data["final_visibility"],
            projections,
            img_names,
            imgs,
            progress_actor,
            0
        )
        ray.get(dummy_task)
        print("Start a calculate")
        a = time.time()
        result_ids = []
        for i in range(num_worker):
            result_ids.append(compute_loss.remote(
                i,
                num_sample_per_worker,
                data["sample_points"],
                data["final_visibility"],
                projections_id,
                img_names_id,
                imgs_id,
                progress_actor,
                -1
            ))
        # Query progress periodically.
        while True:
            progress = ray.get(progress_actor.get_progress.remote())
            print(f"Progress: {int(progress * 100)}%")
            if progress > 1:
                break
            time.sleep(9)
        # Get the results.
        results = ray.get(result_ids)
        print("End of calculation, using {} minutes".format((time.time() - a) / 60))
    ray.shutdown()
    print("Start to save")
    np.save("output/gt_loss/data", data)
    del data
    np.save("output/gt_loss/imgs", imgs)
    del imgs
    gt_loss = np.concatenate([np.load("output/gt_loss/sub/gt_loss_{}.npy".format(i)) for i in range(num_worker)])
    np.save("output/gt_loss/gt_loss", gt_loss)
    del gt_loss
    projected_points = np.concatenate(
        [np.load("output/gt_loss/sub/projected_points_{}.npy".format(i), allow_pickle=True) for i in range(num_worker)])
    np.save("output/gt_loss/projected_points", projected_points)
    del projected_points
    print("Done")
    pass


if __name__ == '__main__':
    main()
