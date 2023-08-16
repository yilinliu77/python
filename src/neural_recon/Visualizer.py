import os
import time

import cv2
import torch

from shared.common_utils import to_homogeneous_tensor
from src.neural_recon.geometric_util import intersection_of_ray_and_all_plane
import numpy as np


class Visualizer:
    def __init__(self,
                 num_patches,
                 v_log_root,
                 v_imgs,
                 intrinsic,
                 transformation
                 ):
        self.num_patches = num_patches
        self.log_root = v_log_root
        self.imgs = [cv2.cvtColor((item * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for item in v_imgs.cpu().numpy()]
        self.intrinsic = intrinsic
        self.transformation = transformation

        for i in range(self.num_patches):
            os.mkdir(os.path.join(self.log_root, "patch_{}".format(i)))
            os.mkdir(os.path.join(self.log_root, "patch_{}/samples".format(i)))

        self.timer = {
            "Sample": 0,
            "Construct": 0,
            "Sample points": 0,
            "Collision": 0,
            "NCC1": 0,
            "NCC2": 0,
            "Edge": 0,
            "Glue": 0,
            "Loss": 0,
            "Update": 0,
        }
        self.timer_ = time.time()
        pass

    def start_iter(self):
        self.timer_ = time.time()

    def update_timer(self, name):
        self.timer[name] += time.time() - self.timer_
        self.timer_ = time.time()

    def update_sample_plane(self, samples_abcd, v_ray, id_vertexes, v_iter):
        num_sample = samples_abcd.shape[1]
        for i_patch in range(self.num_patches):
            self.save_planes("patch_{}/samples/{}.ply".format(i_patch, v_iter),
                             samples_abcd[i_patch],
                             v_ray,
                             [id_vertexes[i_patch]] * num_sample
                             )

        pass

    def save_planes(self, file_path, planes, v_ray, id_vertexes):
        vertices = []
        polygons = []
        acc_num_vertices = 0
        for i in range(planes.shape[0]):
            intersection_points = intersection_of_ray_and_all_plane(planes[i:i + 1],
                                                                    v_ray[id_vertexes[i]])[0]
            vertices.append(intersection_points)
            polygons.append(np.arange(intersection_points.shape[0]) + acc_num_vertices)
            acc_num_vertices += intersection_points.shape[0]

        vertices = torch.cat(vertices, dim=0).cpu().numpy()
        with open(os.path.join(self.log_root, file_path), "w") as f:
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
        return

    def viz_patch_2d(self,
                     v_patch_id,
                     v_iter,

                     samples_abcd,
                     sample_points_on_face_src,

                     remain_flag,
                     local_edge_pos,
                     local_centroid,

                     final_loss,
                     ncc_loss,
                     edge_loss,
                     id_best,

                     ):
        num_sample = samples_abcd.shape[0]
        num_img = len(self.imgs)-1
        for i_img in range(num_img):
            img_ref = self.imgs[0].copy()
            imgs_src = [item.copy() for item in self.imgs[1:]]

            # Draw points
            points_ref = (self.intrinsic @ sample_points_on_face_src.T).T.cpu().numpy()
            points_ref = points_ref[:,:2] / points_ref[:, 2:]
            points_ref = np.around(points_ref * img_ref.shape[:2]).astype(np.int64)

            points_src = (self.transformation[i_img] @ to_homogeneous_tensor(sample_points_on_face_src).T).T.cpu().numpy()
            points_src = points_src[:, :2] / points_src[:, 2:3]
            points_src = np.around(points_src * img_ref.shape[:2]).astype(np.int64)

            for i_point in range(points_ref.shape[0]):
                cv2.circle(img_ref, points_ref[i_point], 1, (0, 255, 0), 1)
                cv2.circle(imgs_src[i_img], points_src[i_point], 1, (0, 255, 0), 1)

            cv2.putText(imgs_src[i_img],
                        "NCC: {:.4f}; Edge: {:.4f}; Final: {:.4f}".format(
                            ncc_loss[i_img, i_sample].item(), edge_loss[i_img, i_sample].item(), final_loss[i_img, i_sample].item()
                        ),
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            total_img = np.stack(
                [img_ref] + imgs_src[:-1], axis=1).reshape(1600, 5, 800, 3).reshape(1600, 4000, 3)
            total_img = np.concatenate([img_ref] + imgs_src[:-1], axis=1)
            cv2.imwrite(os.path.join(self.log_root, "patch_{}/{}.png".format(v_patch_id, v_iter)), total_img)
            pass
        pass
