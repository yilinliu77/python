import ctypes
import sys, os
import time

import cv2

sys.path.append("thirdparty/sdf_computer/build/")
import pysdf

import hydra
from lightning_lite import seed_everything
from omegaconf import DictConfig, OmegaConf

from shared.common_utils import to_homogeneous
from shared.sdf_computer import normalize_intrinsic_and_extrinsic
from src.neural_recon.colmap_io import Image, check_visibility

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

from src.neural_recon.generate_gt import compute_loss
import open3d as o3d


@hydra.main(config_name="phase2_annoying.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
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
        # Read intrinsic
        intrinsic = np.loadtxt(os.path.join(root_data_dir, "cameras.txt")).astype(np.float32)

        # Read poses
        extrinsics = []
        for i in range(0, 4):
            extrinsic = np.loadtxt(os.path.join(root_data_dir, "{}.txt".format(i))).astype(np.float32)
            extrinsics.append(extrinsic)

        normalized_intrinsic, normalized_extrinsic, normalized_positions, normalized_projections = normalize_intrinsic_and_extrinsic(
            bounds_center, bounds_size, original_img_size, intrinsic, np.stack(extrinsics))

        data = {}
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

        # Sample segments
        # Both start and end point near surface
        print("Sample segments")
        p = data["sample_points"]
        v = data["final_visibility"]
        num_segment = int(hydra_conf["dataset"]["num_sample"][3])
        index1 = np.random.randint(0, num_point_near_surface, num_segment)
        index2 = np.random.randint(0, num_point_near_surface, num_segment)
        segments1 = np.stack([p[index1], p[index2]], axis=1)
        visibility1 = np.logical_and(v[:, index1], v[:, index2])

        # Only start or end point near surface
        index1 = np.random.randint(0, num_point_near_surface, num_segment)
        index2 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
        segments21 = np.stack([p[index1], p[index2]], axis=1)
        segments22 = np.stack([p[index2], p[index1]], axis=1)
        visibility21 = np.logical_and(v[:, index1], v[:, index2])
        visibility22 = np.logical_and(v[:, index2], v[:, index1])

        # None of the points near surface
        index1 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
        index2 = np.random.randint(num_point_near_surface, p.shape[0], num_segment)
        segments3 = np.stack([p[index1], p[index2]], axis=1)
        visibility3 = np.logical_and(v[:, index1], v[:, index2])

        segments = np.concatenate((segments1, segments21, segments22, segments3), axis=0)
        segments_visibility = np.concatenate((visibility1, visibility21, visibility22, visibility3), axis=1)
        data["segments"] = segments
        data["segments_visibility"] = segments_visibility

    # Prepare GT data
    if True:
        def read_img(v_path):
            original_img = cv2.imread(v_path, cv2.IMREAD_UNCHANGED)
            resized_img = cv2.resize(original_img, trained_img_size)
            transform_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            # 1/128 makes the sum of the kernel to be 1
            gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5, scale=1 / 128)
            gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=5, scale=1 / 128)
            # 255 makes the range of gradient lies in [-1,1]
            # length of the normal vector will be [0, 1]
            gradient = np.stack((gx, gy), axis=2) / 255.
            return transform_img, gradient

        imgs = {item.img_name: read_img(item.img_path) for item in data["img_database"]}
        projections = [item.projection for item in data["img_database"]]
        img_names = [item.img_name for item in data["img_database"]]

        pool.close()
        num_segments = data["segments"].shape[0]

        ray.init(include_dashboard=True,
                 dashboard_host="0.0.0.0",
                 dashboard_port=19999,
                 # local_mode=True
                 )
        dummy_task = compute_loss.remote(
            0,
            data["segments"][0],
            data["segments_visibility"][:, 0],
            projections,
            img_names,
            imgs
        )
        ray.get(dummy_task)
        imgs_id = ray.put(imgs)
        projections_id = ray.put(projections)
        img_names_id = ray.put(img_names)

        result_ids = []
        a = time.time()
        for i in range(num_segments):
            result_ids.append(compute_loss.remote(
                i,
                data["segments"][i],
                data["segments_visibility"][:, i],
                projections_id,
                img_names_id,
                imgs_id
            ))

        # Get the results.
        results = ray.get(result_ids)
        projected_segments = [item[0] for item in results]
        gt_loss = np.asarray([item[1:] for item in results]).astype(np.float16)
        print(time.time() - a)

    os.makedirs("output/gt_loss", exist_ok=True)
    np.save("output/gt_loss/gt_loss", gt_loss)
    np.save("output/gt_loss/data", data)
    np.save("output/gt_loss/imgs", imgs)
    np.save("output/gt_loss/projected_segments", projected_segments)
    print("Done")
    pass


if __name__ == '__main__':
    main()
