import itertools
import sys, os
import time
from copy import copy
from typing import List

import scipy.spatial
from torch.distributions import Binomial
from torch.nn.utils.rnn import pad_sequence

from src.neural_recon.init_segments import compute_init_based_on_similarity
from src.neural_recon.losses import loss1, loss2, loss3, loss4, loss5

# sys.path.append("thirdparty/sdf_computer/build/")
# import pysdf
# from src.neural_recon.phase12 import Siren2

import math

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
import networkx as nx
from torch_scatter import scatter_add, scatter_min, scatter_mean
import faiss
# import torchsort

# import mcubes
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from tqdm import tqdm, trange
import ray
import platform
import shutil

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_recon.optimize_segment import compute_initial_normal, compute_roi, sample_img_prediction, \
    compute_initial_normal_based_on_pos, compute_initial_normal_based_on_camera, sample_img, sample_img_prediction2
from shared.common_utils import *

from src.neural_recon.colmap_io import read_dataset, Image, Point_3d, check_visibility
# from src.neural_recon.phase1 import NGPModel
from scipy.spatial import Delaunay
from math import ceil


class Collision_checker:
    def __init__(self):
        self.triangles = None
        self.tri_to_patch = None
        self.n_ = None
        self.d = None

    def check_ray_batch(self, patch_id_c, v_origin, v_direction):
        check_triangle = self.triangles[self.tri_to_patch != patch_id_c]
        final_flag = torch.ones((v_origin.shape[0], check_triangle.shape[0]), dtype=torch.bool,
                                device=v_origin.device)

        v0v1 = (check_triangle[:, 1] - check_triangle[:, 0])[None, :]  # 1,M,3
        v0v2 = (check_triangle[:, 2] - check_triangle[:, 0])[None, :]  # 1,M,3

        original_distance = torch.linalg.norm(v_direction, dim=-1, keepdim=True)
        ray_direction_ = (v_direction / original_distance)[:, None]  # N,1,3
        ray_origin = v_origin[:, None]  # N,1,3

        pvec = torch.cross(ray_direction_, v0v2, dim=-1)
        det = torch.sum(v0v1 * pvec, dim=-1)

        # det = v0v1.sum(-1) * torch.cross(ray_direction_, v0v2).sum(-1)

        # Check parallel
        final_flag[torch.abs(det) < 1e-8] = 0

        invDet = 1.0 / det
        tvec = ray_origin - check_triangle[:, 0][None, :]
        u = torch.sum(tvec * pvec, dim=-1) * invDet

        final_flag[torch.logical_or(u < 0, u > 1)] = 0

        qvec = torch.cross(tvec, v0v1, dim=-1)
        v = torch.sum(ray_direction_ * qvec, dim=-1) * invDet

        final_flag[torch.logical_or(v < 0, u + v > 1)] = 0
        t = torch.sum(v0v2 * qvec, dim=-1) * invDet
        final_flag[t < 1e-8] = 0
        final_flag[t > original_distance - 1e-12] = 0
        # intersected_point = ray_origin + ray_direction_ * t
        return torch.max(final_flag, dim=-1)[0]

    # v_origin: N,3
    # v_direction: N,3
    # self.triangles: M,3
    # self.triangles: M,3,3
    # self.n_: M,3
    # self.d: M,
    def check_ray(self, patch_id_c, v_origin, v_direction, batch_size=20000):
        if self.triangles is None or self.tri_to_patch.sum() == 0:
            return torch.zeros_like(v_origin[:, 0], dtype=torch.bool)
        else:
            # return torch.zeros_like(v_origin[:, 0], dtype=torch.bool)
            num_batches = (v_origin.shape[0] + batch_size - 1) // batch_size
            output = []
            for i in range(num_batches):
                cur_batch_s = i * batch_size
                cur_batch_e = min((i + 1) * batch_size, v_origin.shape[0])
                output.append(self.check_ray_batch(patch_id_c,
                                                   v_origin[cur_batch_s: cur_batch_e],
                                                   v_direction[cur_batch_s: cur_batch_e]))
            return torch.cat(output)

    def add_triangles(self, v_triangles, tri_to_patch):
        if self.triangles is None:
            self.triangles = v_triangles
            self.tri_to_patch = tri_to_patch
        else:
            self.triangles = torch.cat((self.triangles, v_triangles))
            self.tri_to_patch = torch.cat((self.tri_to_patch, tri_to_patch))

    def clear(self):
        self.triangles = None

    def save_ply(self, v_file):
        vertices = []
        polygons = []
        acc_num_vertices = 0
        triangles = self.triangles.cpu().numpy()
        for id_tri in range(triangles.shape[0]):
            vertices.append(triangles[id_tri, 0])
            vertices.append(triangles[id_tri, 1])
            vertices.append(triangles[id_tri, 2])
            polygons.append([acc_num_vertices, acc_num_vertices + 1, acc_num_vertices + 2])
            acc_num_vertices += 3
            pass
        vertices = np.stack(vertices, axis=0)
        with open(v_file, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex {}\nproperty float x\nproperty float y\nproperty float z\n".format(
                    acc_num_vertices))
            f.write("element face {}\nproperty list uchar int vertex_index\n".format(len(polygons)))
            f.write("end_header\n")
            for ver in vertices:
                f.write("{} {} {}\n".format(ver[0], ver[1], ver[2]))
            for polygon in polygons:
                f.write("{}".format(len(polygon)))
                for item in polygon:
                    f.write(" {}".format(item))
                f.write("\n")
            pass
        print("Save done")
        return
