import time

import networkx as nx
import numpy as np
import torch
from torch_scatter import scatter_mean

from shared.common_utils import normalize_tensor, normalize_vector, to_homogeneous_tensor
from src.neural_recon.geometric_util import compute_area, intersection_of_ray_and_plane
from src.neural_recon.optimize_segment import sample_img

def dilate_edge(v_gradient_img, v_num_iter=0):
    if isinstance(v_gradient_img, torch.Tensor):
        v_gradient_img = v_gradient_img.cpu().numpy()
    edge_field = np.linalg.norm(v_gradient_img, axis=-1) > 0.01
    edge_pixels = np.column_stack(np.where(edge_field))[:, ::-1]
    edge_gradients = v_gradient_img[edge_field]

    gd = normalize_vector(edge_gradients)
    dilate_edge_maps = v_gradient_img.copy()
    for i in range(1, v_num_iter + 1):
        related_pixels = np.round(edge_pixels + gd * i).astype(np.int64)
        old_norm = np.linalg.norm(dilate_edge_maps[related_pixels[:, 1], related_pixels[:, 0]], axis=-1)
        new_norm = np.linalg.norm(edge_gradients, axis=-1)
        mask = new_norm > old_norm
        dilate_edge_maps[related_pixels[mask, 1], related_pixels[mask, 0]] = edge_gradients[mask]
        related_pixels = np.round(edge_pixels - gd * i).astype(np.int64)
        old_norm = np.linalg.norm(dilate_edge_maps[related_pixels[:, 1], related_pixels[:, 0]], axis=-1)
        new_norm = np.linalg.norm(edge_gradients, axis=-1)
        mask = new_norm > old_norm
        dilate_edge_maps[related_pixels[mask, 1], related_pixels[mask, 0]] = edge_gradients[mask]

    return dilate_edge_maps


# v_points: (M,3) M 3D points in the camera coordinate of image 1
def get_projections(v_points, v_intrinsic, v_transformation, ):
    points_2d1 = (v_intrinsic @ v_points.T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-8)
    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_points).T).T
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-8)

    valid_mask = torch.logical_and(points_2d2 < 1, points_2d2 > 0).all(dim=-1)
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    return points_2d1, points_2d2, valid_mask


# v_points: (M,3) M 3D points in the camera coordinate of image 1
def get_projections_batch(v_points, v_intrinsic, v_transformation, ):
    points_2d1 = (v_intrinsic @ v_points.T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-8)
    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_points).T).transpose(1,2)
    points_2d2 = points_2d2[:, :, :2] / (points_2d2[:, :, 2:3] + 1e-8)

    valid_mask = torch.logical_and(points_2d2 < 1, points_2d2 > 0).all(dim=-1)
    points_2d2 = torch.clamp(points_2d2, 0, 0.999999)

    return points_2d1, points_2d2, valid_mask

def sample_imgs(v_img1, v_img2, points_2d1, points_2d2):
    sample_img1 = sample_img(v_img1[None, :].permute(0, 3, 1, 2), points_2d1[None, :, :])
    sample_img2 = sample_img(v_img2[None, :].permute(0, 3, 1, 2), points_2d2[None, :, :])
    return sample_img1, sample_img2

def sample_imgs_batch2(v_img2, points_2d2):
    sample_img2 = sample_img(v_img2[:,:,:,None].permute(0, 3, 1, 2), points_2d2)
    return sample_img2

def sample_imgs_batch(v_img1, v_img2, points_2d1, points_2d2):
    num_source = v_img2.shape[0]
    sample_img1 = sample_img(v_img1[None, :].permute(0, 3, 1, 2), points_2d1[None, :, :])
    sample_img2 = sample_img(v_img2.permute(0, 3, 1, 2), points_2d2)
    return sample_img1, sample_img2

class Bilateral_ncc_computer:
    def __init__(self, v_enable_spatial_weights=True, v_enable_color_weights=True,
                 v_sigma_spatial=5, v_sigma_color=1, v_window_size=7):
        self.enable_spatial_weights = v_enable_spatial_weights
        self.enable_color_weights = v_enable_color_weights

        self.sigma_spatial = v_sigma_spatial
        self.sigma_color = v_sigma_color
        self.window_size = v_window_size

        # Spatial weights
        sigma_spatial = self.sigma_spatial
        spatial_normalization_ = 1. / (2. * sigma_spatial * sigma_spatial)
        spatial_weights = torch.stack(torch.meshgrid(
            torch.arange(self.window_size, dtype=torch.float32),
            torch.arange(self.window_size, dtype=torch.float32),
            indexing="xy"
        ), dim=2) - self.window_size // 2
        spatial_weights = torch.linalg.norm(spatial_weights, dim=-1)
        self.spatial_weights = spatial_weights ** 2 * spatial_normalization_

        pass

    def bilateral_ncc_(self, v_img1, v_img2):
        batch_size = v_img1.shape[0]
        window_size = v_img1.shape[1]
        device = v_img1.device

        # Add weights from 1) position 2) color difference
        # 1) position
        spatial_weights = self.spatial_weights.to(device)

        # 2) color difference
        sigma_color = self.sigma_color
        color_normalization_ = 1. / (2. * sigma_color * sigma_color)
        color_weights1 = (v_img1 - v_img1[:, 3:4, 3:4]) ** 2 * color_normalization_
        color_weights2 = (v_img2 - v_img2[:, :, 3:4, 3:4]) ** 2 * color_normalization_

        if self.enable_spatial_weights and self.enable_color_weights:
            final_weights = torch.exp(-spatial_weights[None, :] - color_weights1 - color_weights2)
        elif self.enable_spatial_weights:
            final_weights = torch.exp(-spatial_weights[None, :])
        elif self.enable_color_weights:
            final_weights = torch.exp(-color_weights1 - color_weights2)
        else:
            final_weights = torch.ones_like(v_img1)
        # final_weights = final_weights / final_weights.sum(dim=[1,2],keepdim=True)

        v_img1 = (v_img1 * final_weights)
        v_img2 = (v_img2 * final_weights)

        norm_img1 = v_img1 - v_img1.mean(dim=[2, 3], keepdim=True)
        norm_img2 = v_img2 - v_img2.mean(dim=[2, 3], keepdim=True)

        ncc1 = torch.sum(norm_img1 * norm_img2, dim=[2, 3])
        ncc2 = torch.sqrt(torch.clamp_min(
            torch.sum(norm_img1 ** 2, dim=[2, 3]) * torch.sum(norm_img2 ** 2, dim=[2, 3]),
            1e-8))
        ncc2 = torch.clamp_min(ncc2, 1e-6)
        ncc = 1 - ncc1 / ncc2
        return ncc

    def compute(self, v_points_c, v_normal_c, v_intrinsic, v_transformation,
                v_img1, v_img2):
        # 1. Prepare variables
        device = v_points_c.device
        num_points = v_points_c.shape[0]
        world_up_vector = torch.zeros_like(v_normal_c)
        world_up_vector[:, 2] = 1
        right_vector = normalize_tensor(torch.cross(world_up_vector, v_normal_c))
        up_vector = normalize_tensor(torch.cross(v_normal_c, right_vector))

        height, width = v_img1.shape[:2]
        resolution = 1 / min(height, width) * torch.linalg.norm(v_points_c, dim=-1)

        # 2. Sample points in the window
        index = torch.arange(self.window_size, device=device) - self.window_size // 2
        index = torch.stack(torch.meshgrid(index, index, indexing="xy"), dim=2)
        window_points = \
            (index[:, :, 0].view(1, -1, 1) * right_vector[:, None, :] * resolution[:, None, None]).view(
                num_points, self.window_size, self.window_size, 3) + \
            (index[:, :, 1].view(1, -1, 1) * up_vector[:, None, :] * resolution[:, None, None]).view(
                num_points, self.window_size, self.window_size, 3)
        window_points = window_points + v_points_c[:, None, None, :]

        # 3. Project points in both images
        points_2d1, points_2d2, valid_mask = get_projections(
            window_points.reshape(-1, 3), v_intrinsic, v_transformation)

        sample_imgs1, sample_imgs2 = sample_imgs(v_img1[:, :, None], v_img2[:, :, None], points_2d1, points_2d2)
        sample_imgs1 = sample_imgs1.reshape(num_points, self.window_size, self.window_size)
        sample_imgs2 = sample_imgs2.reshape(num_points, self.window_size, self.window_size)
        valid_mask = valid_mask.reshape(num_points, self.window_size, self.window_size).all(dim=1).all(dim=1)

        # 4. Compute the ncc
        ncc = self.bilateral_ncc_(sample_imgs1, sample_imgs2[None,:])[0]

        # Visualize
        if False:
            points_2d1 = points_2d1.reshape(num_points, v_window_size * v_window_size, 2)
            points_2d2 = points_2d2.reshape(num_points, v_window_size * v_window_size, 2)

            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]
            p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
            p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
            point_color = (0, 255, 255)
            point_thickness = 3

            for i in range(num_points):
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img11[p_2d1[i, :, 1], p_2d1[i, :, 0]] = point_color
                img21[p_2d2[i, :, 1], p_2d2[i, :, 0]] = point_color

                cv2.imshow("1", np.concatenate([
                    np.concatenate([img11, img21], axis=1),
                ], axis=0))
                cv2.waitKey()

        ncc[~valid_mask] = torch.inf
        return ncc

    def compute_batch(self, v_points_c, v_normal_c, v_intrinsic, v_transformations,
                      v_img_ref, v_img_source):
        time_table = [0]*10
        cur_time= time.time()

        # 1. Prepare variables
        device = v_points_c.device
        num_points = v_points_c.shape[0]
        world_up_vector = torch.zeros_like(v_normal_c)
        world_up_vector[:, 2] = 1
        right_vector = normalize_tensor(torch.cross(world_up_vector, v_normal_c))
        up_vector = normalize_tensor(torch.cross(v_normal_c, right_vector))

        height, width = v_img_ref.shape[:2]
        resolution = 1 / min(height, width) * torch.linalg.norm(v_points_c, dim=-1)
        time_table[0]+=time.time()-cur_time
        cur_time = time.time()

        # 2. Sample points in the window
        index = torch.arange(self.window_size, device=device) - self.window_size // 2
        index = torch.stack(torch.meshgrid(index, index, indexing="xy"), dim=2)
        window_points = \
            (index[:, :, 0].view(1, -1, 1) * right_vector[:, None, :] * resolution[:, None, None]).view(
                num_points, self.window_size, self.window_size, 3) + \
            (index[:, :, 1].view(1, -1, 1) * up_vector[:, None, :] * resolution[:, None, None]).view(
                num_points, self.window_size, self.window_size, 3)
        window_points = window_points + v_points_c[:, None, None, :]
        time_table[1] += time.time() - cur_time
        cur_time = time.time()

        # 3. Project points in both images
        points_2d1, points_2d2, valid_mask = get_projections_batch(
            window_points.reshape(-1, 3), v_intrinsic, v_transformations)

        sample_imgs1, sample_imgs2 = sample_imgs_batch(v_img_ref[:, :, None], v_img_source[:, :,:, None], points_2d1, points_2d2)
        sample_imgs1 = sample_imgs1.reshape(num_points, self.window_size, self.window_size)
        sample_imgs2 = sample_imgs2.reshape(-1, num_points, self.window_size, self.window_size)
        valid_mask = valid_mask.reshape(-1, num_points, self.window_size, self.window_size).all(dim=2).all(dim=2)
        time_table[2] += time.time() - cur_time
        cur_time = time.time()

        # 4. Compute the ncc
        ncc = self.bilateral_ncc_(sample_imgs1, sample_imgs2)

        # Visualize, do not adapt to batch optimization yet
        if False:
            points_2d1 = points_2d1.reshape(num_points, v_window_size * v_window_size, 2)
            points_2d2 = points_2d2.reshape(num_points, v_window_size * v_window_size, 2)

            img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
            shape = img11.shape[:2][::-1]
            p_2d1 = np.around(points_2d1.cpu().numpy() * shape).astype(np.int64)
            p_2d2 = np.around(points_2d2.cpu().numpy() * shape).astype(np.int64)
            point_color = (0, 255, 255)
            point_thickness = 3

            for i in range(num_points):
                img11 = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                img21 = cv2.cvtColor(src_imgs[0], cv2.COLOR_GRAY2BGR)
                img11[p_2d1[i, :, 1], p_2d1[i, :, 0]] = point_color
                img21[p_2d2[i, :, 1], p_2d2[i, :, 0]] = point_color

                cv2.imshow("1", np.concatenate([
                    np.concatenate([img11, img21], axis=1),
                ], axis=0))
                cv2.waitKey()

        ncc[~valid_mask] = torch.inf
        time_table[3] += time.time() - cur_time
        cur_time = time.time()
        return ncc


# v_max_gradient_norm: If the norm of the gradient direction less than this value, we reduce the influence of this pixel
# v_edge_power: Final weight = torch.pow(weight, value). Higher value produces lower edge loss
class Edge_loss_computer:
    def __init__(self, v_sample_density, v_max_gradient_norm=0.1, v_edge_power=2):
        self.sample_density = v_sample_density
        self.max_gradient_norm = v_max_gradient_norm
        self.edge_power = v_edge_power
        pass

    def compute(self, v_edge_points, v_intrinsic, v_transformation,
                v_gradient1, v_gradient2, v_num_hypothesis=100):
        num_edge, _, _ = v_edge_points.shape

        # 1. Get the projected points in both images
        points_2d1, points_2d2, valid_mask = get_projections(v_edge_points.view(-1, 3),
                                                             v_intrinsic, v_transformation)

        points_2d1 = points_2d1.reshape(num_edge, 2, 2)
        points_2d2 = points_2d2.reshape(num_edge, 2, 2)
        valid_mask = valid_mask.reshape(num_edge, 2).all(dim=1)

        # 2. Sample the same number of points along the segments
        num_horizontal = torch.clamp(
            (torch.linalg.norm(v_edge_points[:, 0] - v_edge_points[:, 1], dim=-1) / self.sample_density).to(
                torch.long),
            2, 1000)

        begin_idxes = num_horizontal.cumsum(dim=0)
        begin_idxes = begin_idxes.roll(1)
        begin_idxes[0] = 0
        dx = torch.arange(num_horizontal.sum(), device=v_edge_points.device) - \
             begin_idxes.repeat_interleave(num_horizontal)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal)
        dir1 = points_2d1[:, 1] - points_2d1[:, 0]
        sampled_points1 = points_2d1[:, 0].repeat_interleave(num_horizontal, dim=0) \
                          + dx[:, None] * dir1.repeat_interleave(num_horizontal, dim=0)

        dir2 = points_2d2[:, 1] - points_2d2[:, 0]
        sampled_points2 = points_2d2[:, 0].repeat_interleave(num_horizontal, dim=0) \
                          + dx[:, None] * dir2.repeat_interleave(num_horizontal, dim=0)

        # 3. Sample pixel direction
        # The direction in first img is used to decide the weight of the edge loss
        # If it is not an edge in the first image, then we are unlikely to calculate the edge loss in the second img
        # Also we do not calculate black area
        sample_imgs1, sample_imgs2 = sample_imgs(v_gradient1, v_gradient2,
                                                 sampled_points1, sampled_points2)
        sample_imgs1 = sample_imgs1[0]
        sample_imgs2 = sample_imgs2[0]

        edge_directions1 = normalize_tensor(points_2d1[:, 0] - points_2d1[:, 1])

        weight1 = torch.clamp_max(torch.norm(sample_imgs1, dim=-1) / self.max_gradient_norm, 1)
        weight2 = 1 - ((normalize_tensor(sample_imgs1) * edge_directions1.repeat_interleave(num_horizontal, dim=0))
                       .sum(dim=-1).abs())

        weight = torch.pow(weight1 * weight2, self.edge_power)

        sample_imgs2 = normalize_tensor(sample_imgs2)
        edge_directions2 = normalize_tensor(points_2d2[:, 0] - points_2d2[:, 1])
        edge_directions2 = edge_directions2.repeat_interleave(num_horizontal, dim=0)
        black_mask = torch.linalg.norm(sample_imgs2, dim=-1) < 1e-8

        edge_loss2 = torch.abs(torch.sum(edge_directions2 * sample_imgs2, dim=-1))
        edge_loss2[black_mask] = 1
        weighted_edge_loss = edge_loss2 * weight

        weight_edge = num_horizontal.reshape(v_num_hypothesis, -1)
        weight_edge = (weight_edge / weight_edge.sum(dim=1, keepdim=True)).reshape(-1)
        weighted_edge_loss = weighted_edge_loss * weight_edge.repeat_interleave(num_horizontal)
        mean_edge_loss = scatter_mean(weighted_edge_loss,
                                      torch.arange(
                                          num_horizontal.shape[0], device=edge_directions2.device
                                      ).repeat_interleave(num_horizontal))

        return mean_edge_loss, valid_mask, num_horizontal

    def compute_batch(self, v_edge_points, v_intrinsic, v_transformation,
                v_gradient1, v_gradient2, img2s, v_num_hypothesis=100):
        num_edge, _, _ = v_edge_points.shape

        # 1. Get the projected points in both images
        points_2d1, points_2d2, valid_mask = get_projections_batch(v_edge_points.view(-1, 3),
                                                             v_intrinsic, v_transformation)

        points_2d1 = points_2d1.reshape(num_edge, 2, 2)
        points_2d2 = points_2d2.reshape(-1, num_edge, 2, 2)
        valid_mask = valid_mask.reshape(-1, num_edge, 2).all(dim=2)

        # 2. Sample the same number of points along the segments
        num_horizontal = torch.clamp(
            (torch.linalg.norm(v_edge_points[:, 0] - v_edge_points[:, 1], dim=-1) / self.sample_density).to(
                torch.long),
            2, 1000)

        begin_idxes = num_horizontal.cumsum(dim=0)
        begin_idxes = begin_idxes.roll(1)
        begin_idxes[0] = 0
        dx = torch.arange(num_horizontal.sum(), device=v_edge_points.device) - \
             begin_idxes.repeat_interleave(num_horizontal)
        dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal)
        dir1 = points_2d1[:, 1] - points_2d1[:, 0]
        sampled_points1 = points_2d1[:, 0].repeat_interleave(num_horizontal, dim=0) \
                          + dx[:, None] * dir1.repeat_interleave(num_horizontal, dim=0)

        dir2 = points_2d2[:,:, 1] - points_2d2[:,:, 0]
        sampled_points2 = points_2d2[:,:, 0].repeat_interleave(num_horizontal, dim=1) \
                          + dx[None,:, None] * dir2.repeat_interleave(num_horizontal, dim=1)

        # 3. Sample pixel direction
        # The direction in first img is used to decide the weight of the edge loss
        # If it is not an edge in the first image, then we are unlikely to calculate the edge loss in the second img
        # Also we do not calculate black area
        sample_imgs1, sample_imgs2 = sample_imgs_batch(v_gradient1, v_gradient2,
                                                 sampled_points1, sampled_points2)
        sample_pixel_imgs2 = sample_imgs_batch2(img2s, sampled_points2)

        edge_directions1 = normalize_tensor(points_2d1[:, 0] - points_2d1[:, 1])

        #weight1 = torch.clamp_max(torch.norm(sample_imgs1, dim=-1) / self.max_gradient_norm, 1)
        mean_edge_grad_norm = scatter_mean(torch.norm(sample_imgs1, dim=-1),
                                      torch.arange(
                                          num_horizontal.shape[0], device=sample_imgs1.device
                                      ).repeat_interleave(num_horizontal), dim=1)
        valid_edge_mask = (mean_edge_grad_norm > 0.01)

        weight2 = 1 - ((normalize_tensor(sample_imgs1) * edge_directions1.repeat_interleave(num_horizontal, dim=0))
                       .sum(dim=-1).abs())

        weight = torch.pow(weight2, self.edge_power)

        sample_imgs2 = normalize_tensor(sample_imgs2)
        edge_directions2 = normalize_tensor(points_2d2[:,:, 0] - points_2d2[:,:, 1])
        edge_directions2 = edge_directions2.repeat_interleave(num_horizontal, dim=1)
        edge_loss2 = torch.abs(torch.sum(edge_directions2 * sample_imgs2, dim=-1))

        # mean_edge_pixel_norm = scatter_mean(sample_pixel_imgs2.squeeze(2),
        #                               torch.arange(
        #                                   num_horizontal.shape[0], device=sample_imgs1.device
        #                               ).repeat_interleave(num_horizontal), dim=1)
        black_mask = (sample_pixel_imgs2 < 1e-8).squeeze(2)
        edge_loss2[black_mask] = 1

        weighted_edge_loss = edge_loss2 * weight

        weight_edge = num_horizontal.reshape(v_num_hypothesis, -1)
        weight_edge = (weight_edge / weight_edge.sum(dim=1, keepdim=True)).reshape(-1)
        weighted_edge_loss = weighted_edge_loss * weight_edge.repeat_interleave(num_horizontal)
        #weighted_edge_loss[black_mask] = 1
        mean_edge_loss = scatter_mean(weighted_edge_loss,
                                      torch.arange(
                                          num_horizontal.shape[0], device=edge_directions2.device
                                      ).repeat_interleave(num_horizontal), dim=1)

        mean_edge_loss[~valid_edge_mask.tile(mean_edge_loss.shape[0], 1)] = torch.nan

        return mean_edge_loss, valid_mask, num_horizontal


def compute_regularization(v_triangles, v_intrinsic, v_transformation):
    points_2d1 = (v_intrinsic @ v_triangles.reshape(-1, 3).T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-8)
    points_2d1 = points_2d1.reshape(-1, 3, 2)

    area1 = compute_area(points_2d1).reshape(v_triangles.shape[0], -1).sum(dim=1)

    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_triangles.reshape(-1, 3)).T).T
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-8)
    points_2d2 = points_2d2.reshape(-1, 3, 2)

    area2 = compute_area(points_2d2).reshape(v_triangles.shape[0], -1).sum(dim=1)
    weight = (area1 - area2).abs() / area1
    weight[weight < 2 / 3] = 0
    weight[weight > 1] = 1
    return weight

def compute_regularization_batch(v_triangles, v_intrinsic, v_transformation):
    points_2d1 = (v_intrinsic @ v_triangles.reshape(-1, 3).T).T
    points_2d1 = points_2d1[:, :2] / (points_2d1[:, 2:3] + 1e-8)
    points_2d1 = points_2d1.reshape(-1, 3, 2)

    area1 = compute_area(points_2d1).reshape(v_triangles.shape[0], -1).sum(dim=1)

    points_2d2 = (v_transformation @ to_homogeneous_tensor(v_triangles.reshape(-1, 3)).T).T
    points_2d2 = points_2d2[:, :2] / (points_2d2[:, 2:3] + 1e-8)
    points_2d2 = points_2d2.reshape(-1, 3, 2)

    area2 = compute_area(points_2d2).reshape(v_triangles.shape[0], -1).sum(dim=1)
    weight = (area1 - area2).abs() / area1
    weight[weight < 2 / 3] = 0
    weight[weight > 1] = 1
    return weight


class Glue_loss_computer:
    def __init__(self, v_dual_graph: nx.Graph, v_rays_c):
        self.dual_graph = v_dual_graph
        num_patch = v_dual_graph.number_of_nodes()
        # Prepare the index of planes and vertex in advance to avoid the for-loop when computing the glue loss
        self.plane_index = [[] for _ in range(num_patch)]
        self.rays = [[] for _ in range(num_patch)]
        self.vertex_index = [[] for _ in range(num_patch)]
        for id_patch in range(num_patch):
            for id_nearby_patch in v_dual_graph[id_patch]:
                for id_vertex in v_dual_graph[id_patch][id_nearby_patch]["adjacent_vertices"]:
                    self.vertex_index[id_patch].append(
                        np.where(np.array(v_dual_graph.nodes[id_patch]["id_vertex"]) == id_vertex)[0].item())
                    self.plane_index[id_patch].append(id_nearby_patch)
                    self.rays[id_patch].append(v_rays_c[id_vertex])
            self.rays[id_patch] = torch.stack(self.rays[id_patch], dim=0)

    def compute(self, v_patch_id, v_optimized_abcd_list, v_vertex_pos):
        nearby_points = intersection_of_ray_and_plane(
            v_optimized_abcd_list[self.plane_index[v_patch_id]],
            self.rays[v_patch_id]
        )[1]

        delta_distances_ = v_vertex_pos[:, self.vertex_index[v_patch_id]] - nearby_points[None, :]
        delta_distances = (delta_distances_ ** 2).sum(dim=2).mean(dim=1)

        return delta_distances


class Regularization_loss_computer:
    def __init__(self, v_dual_graph: nx.Graph):
        self.dual_graph = v_dual_graph
        num_patch = v_dual_graph.number_of_nodes()
        self.vertical_weight = 0.5
        self.parallel_weight = 0.5

        # Prepare the index of planes and vertex in advance to avoid the for-loop when computing the glue loss
        self.nearby_plane_idx = [[] for _ in range(num_patch)]
        for id_patch in range(num_patch):
            self.nearby_plane_idx[id_patch] = list(v_dual_graph[id_patch].keys())

    def compute(self, v_patch_id, v_optimized_abcd_list):
        cur_plane_abcd = v_optimized_abcd_list[v_patch_id]
        n1, d1 = cur_plane_abcd[0:3], cur_plane_abcd[3]
        nearby_plane_abcd = v_optimized_abcd_list[self.nearby_plane_idx[v_patch_id]]
        n2, d2 = nearby_plane_abcd[:, 0:3], nearby_plane_abcd[:, 3]
        # 计算点积，形状为 (num_nearby_plane,)
        dot_product = torch.matmul(n1, n2.T)

        # 正则化项1: 垂直约束 (n1与n2的点积接近0)
        vertical_constraint = torch.abs(dot_product)

        # 正则化项2: 平行约束 (n1与n2的点积接近1或-1)
        parallel_constraint = torch.abs(torch.abs(dot_product) - 1)

        regularization_loss = torch.min(vertical_constraint,parallel_constraint)
        return regularization_loss.sum().squeeze()

class Mutex_loss_computer:
    def __init__(self):
        pass

    def compute(self, v_patch_id, vertex_pos):
        # Inner regularization distance
        # diff = vertex_pos.unsqueeze(1) - vertex_pos.unsqueeze(0)
        # distances = torch.sqrt((diff ** 2).sum(-1))
        # std_mean_distances = distances.mean(dim=1) / distances.std(dim=1)
        # regularization_loss = 1 - std_mean_distances.mean(dim=0) / std_mean_distances.max(dim=0)

        # Mutex distance
        centroid = vertex_pos.mean(dim=1)
        dis2centroid = torch.sqrt(((vertex_pos - centroid.unsqueeze(1))**2).sum(-1))
        return 1 - dis2centroid.min(dim=1)[0] / dis2centroid.max(dim=1)[0]
