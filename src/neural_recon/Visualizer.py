import os
import random
import time

import cv2
import torch

from shared.common_utils import to_homogeneous_tensor
from src.neural_recon.geometric_util import intersection_of_ray_and_all_plane
import numpy as np


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


class Visualizer:
    def __init__(self,
                 num_patches,
                 v_log_root,
                 v_imgs,
                 intrinsic,
                 extrinsic_ref_cam,
                 transformation,
                 debug_patch_id_list=None
                 ):
        self.num_patches = num_patches
        self.log_root = v_log_root
        self.imgs = [cv2.cvtColor((item * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for item in v_imgs.cpu().numpy()]
        self.intrinsic = intrinsic
        self.extrinsic_ref_cam = extrinsic_ref_cam
        self.transformation = transformation
        self.tri_colors = [generate_random_color() for _ in range(100)]

        self.viz_interval = 500
        self.viz_debug_iter = []

        os.mkdir(os.path.join(self.log_root, "0total"))
        if debug_patch_id_list is not None:
            for i in debug_patch_id_list:
                os.mkdir(os.path.join(self.log_root, "patch_{}".format(i)))
                os.mkdir(os.path.join(self.log_root, "patch_{}/samples".format(i)))
        else:
            for i in range(self.num_patches):
                os.mkdir(os.path.join(self.log_root, "patch_{}".format(i)))
                os.mkdir(os.path.join(self.log_root, "patch_{}/samples".format(i)))

        self.timer = {
            "Sample": 0,
            "Construct": 0,
            "Construct_p1": 0,
            "Construct_p2": 0,
            "Construct_p3": 0,
            "Sample points": 0,
            "Collision": 0,
            "NCC1": 0,
            "NCC2": 0,
            "Edge": 0,
            "Glue": 0,
            "Loss": 0,
            "Update1": 0,
            "Update2": 0,
        }
        self.timer_ = time.time()
        pass

    def start_iter(self):
        self.timer_ = time.time()

    def update_timer(self, name):
        self.timer[name] += time.time() - self.timer_
        self.timer_ = time.time()

    def update_sample_plane(self, patch_id_list, samples_abcd, v_ray, id_vertexes, v_iter):
        num_sample = samples_abcd.shape[1]
        for idx in range(len(patch_id_list)):
            patch_id = patch_id_list[idx]
            self.save_planes("patch_{}/samples/{}.ply".format(patch_id, v_iter),
                             samples_abcd[idx],
                             v_ray,
                             [id_vertexes[patch_id]] * num_sample
                             )

        pass

    def save_planes(self, file_path, planes, v_ray, id_vertexes, transform_to_world=True):
        vertices = []
        polygons = []
        acc_num_vertices = 0
        for i in range(planes.shape[0]):
            intersection_points = intersection_of_ray_and_all_plane(planes[i:i + 1],
                                                                    v_ray[id_vertexes[i]])[0]
            if transform_to_world:
                intersection_points = (torch.linalg.inv(self.extrinsic_ref_cam)
                                       @ to_homogeneous_tensor(intersection_points).T).T
                intersection_points[:,0:3] /= intersection_points[:,3:4]
                intersection_points = intersection_points[:,0:3]

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

    def project(self, points, matrix, img_shape):
        shape = points.shape
        assert shape[-1] == matrix.shape[-1]
        if len(shape) != 2:
            points = points.reshape(-1, shape[-1])
        points_ref = (matrix @ points.T).T.cpu().numpy()
        points_ref = points_ref[:, :2] / points_ref[:, 2:3]
        points_ref = np.around(points_ref * img_shape).astype(np.int64)
        if len(shape) != 2:
            points_ref = points_ref.reshape(shape[:-1] + (2,))
        return points_ref

    def viz_patch_2d(self,
                     v_patch_id,
                     v_iter,

                     samples_abcd,
                     sample_points_on_face_src,
                     num_sample_points_per_tri,

                     remain_flag,
                     local_edge_pos,
                     local_centroid,

                     final_loss,
                     final_loss_sum,
                     ncc_loss,
                     edge_loss,
                     id_best,
                     ):
        if (v_iter % self.viz_interval != 0 or v_iter // self.viz_interval == 0) and v_iter not in self.viz_debug_iter:
            return
        print("Start to viz iter {} patch {}".format(v_iter, v_patch_id))
        num_sample = samples_abcd.shape[0]
        num_triangles = num_sample_points_per_tri.shape[0] // num_sample
        num_img = len(self.imgs)

        img_shape = self.imgs[0].shape[:2]
        img_shape = img_shape[::-1]

        input_imgs = [self.imgs[0].copy()] + [item.copy() for item in self.imgs[1:]]
        input_points_2d = []
        input_points_2d.append(self.project(sample_points_on_face_src, self.intrinsic, img_shape))
        for i_img in range(num_img - 1):
            input_points_2d.append(self.project(
                to_homogeneous_tensor(sample_points_on_face_src), self.transformation[i_img], img_shape))

        input_wire = torch.cat(
            (local_edge_pos, torch.tile(local_centroid[:, None, None, :], (1, num_triangles, 1, 1)),), dim=2)
        input_wires = [self.project(input_wire, self.intrinsic, img_shape)]
        for i_img in range(num_img - 1):
            input_wires.append(self.project(
                to_homogeneous_tensor(input_wire), self.transformation[i_img], img_shape))

        id_points = num_sample_points_per_tri.cumsum(0)
        id_points = torch.cat((torch.zeros_like(id_points[0:1]), id_points), dim=0)

        sorted_index = torch.argsort(final_loss_sum)

        for i_img in range(num_img - 1):
            # Draw points
            imgs = []
            for i_sample in range(num_sample):
                img = input_imgs[i_img].copy()

                i_start = sorted_index[i_sample] * num_triangles
                i_end = (sorted_index[i_sample] + 1) * num_triangles
                points_2d = input_points_2d[i_img][id_points[i_start]:id_points[i_end]]
                local_remain_flag = remain_flag[i_img-1][id_points[i_start]:id_points[i_end]].cpu().numpy()

                point_color = []
                for i_triangle in range(num_triangles):
                    cv2.line(img,
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 0],
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 1],
                             (0, 255, 0), 1)
                    cv2.line(img,
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 0],
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 2],
                             (0, 255, 0), 1)
                    cv2.line(img,
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 1],
                             input_wires[i_img][sorted_index[i_sample], i_triangle, 2],
                             (0, 255, 0), 1)
                    point_color += [
                        self.tri_colors[i_triangle] for i in range(num_sample_points_per_tri[i_start + i_triangle])]
                assert len(point_color) == points_2d.shape[0]
                point_color = np.stack(point_color)
                point_color[~local_remain_flag] = 0
                points_2d = np.clip(points_2d, 0, img_shape[1] - 1)
                img[points_2d[:, 1], points_2d[:, 0]] = point_color
                if i_img == 0:
                    cv2.putText(img,
                                "Sample: {:02d}; Final: {:.4f}".format(
                                    sorted_index[i_sample],
                                    final_loss_sum[sorted_index[i_sample]].item(),
                                ),
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img,
                                "Sample: {:02d}; NCC: {:.4f}; Edge: {:.4f}; Final: {:.4f}/{:.4f}".format(
                                    sorted_index[i_sample],
                                    ncc_loss[i_img-1, sorted_index[i_sample]].item(),
                                    edge_loss[i_img-1, sorted_index[i_sample]].item(),
                                    final_loss[i_img-1, sorted_index[i_sample]].item(),
                                    final_loss_sum[sorted_index[i_sample]].item(),
                                ),
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                imgs.append(img)
            total_img = np.stack(imgs, axis=0).reshape(10, 10, 968, 1296, 3)
            total_img = np.transpose(total_img, (0, 2, 1, 3, 4)).reshape(968*10, 1296*10, 3)
            cv2.imwrite(os.path.join(
                self.log_root, "patch_{}/image_{}_iter_{}.png".format(v_patch_id, i_img, v_iter)), total_img)
            pass
        print("Done viz iter {} patch {}".format(v_iter, v_patch_id))
        pass

    def viz_results(self,
                    v_iter,
                    planes, v_ray, id_vertexes, debug_id_list=None,
                    ):
        if debug_id_list is not None:
            planes = planes[debug_id_list]
            id_vertexes_ = []
            for debug_id in debug_id_list:
                id_vertexes_.append(id_vertexes[debug_id])
            id_vertexes = id_vertexes_

        num_patch = planes.shape[0]
        num_img = len(self.imgs)
        img_shape = self.imgs[0].shape[:2]
        img_shape = img_shape[::-1]

        all_intersection_end_points = intersection_of_ray_and_all_plane(planes,
                                                                        v_ray)
        edges_idx = [
            [[i_point, (i_point + 1) % len(id_vertexes[i_patch])] for i_point in range(len(id_vertexes[i_patch]))]
            for i_patch in range(num_patch)]
        all_intersection_end_points = torch.cat([
            all_intersection_end_points[idx][item][edges_idx[idx],:] for idx, item in enumerate(id_vertexes)], dim=0)

        input_imgs = [self.imgs[0].copy()] + [item.copy() for item in self.imgs[1:]]
        input_points_2d = []
        input_points_2d.append(self.project(all_intersection_end_points, self.intrinsic, img_shape))
        for i_img in range(num_img - 1):
            input_points_2d.append(self.project(
                to_homogeneous_tensor(all_intersection_end_points), self.transformation[i_img], img_shape))
        input_points_2d = np.stack(input_points_2d, axis=0)
        #input_points_2d = np.clip(input_points_2d, 0, img_shape[0] - 1)
        #input_points_2d = input_points_2d[..., np.flip(np.arange(input_points_2d.shape[-1]))]
        for i_img in range(num_img):
            # Draw points
            for i_line in range(input_points_2d[i_img].shape[0]):
                cv2.line(input_imgs[i_img],
                         input_points_2d[i_img][i_line,0],
                         input_points_2d[i_img][i_line,1],
                         (0, 255, 0), 1)
        #total_img = np.stack(input_imgs[:10], axis=0).reshape(2, 5, 800, 800, 3)
        #total_img = np.transpose(total_img, (0, 2, 1, 3, 4)).reshape(1600, 4000, 3)
        total_img = np.stack(input_imgs[:10], axis=0).reshape(2, 5, 968, 1296, 3)
        total_img = np.transpose(total_img, (0, 2, 1, 3, 4)).reshape(968*2, 1296*5, 3)
        cv2.imwrite(os.path.join(
            self.log_root, "0total/iter_{}.png".format(v_iter,)), total_img)

        self.save_planes("0total/iter_{}.ply".format(v_iter), planes, v_ray, id_vertexes)
        pass




