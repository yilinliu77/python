import open3d

from shared.perspective_geometry import extract_fundamental_from_projection

import cv2
import numpy as np
import torch
from shared.common_utils import to_homogeneous, to_homogeneous_tensor, pad_imgs, normalize_tensor
from src.neural_recon.losses import loss2, loss3, loss4
from src.neural_recon.optimize_segment import optimize_single_segment_tensor, sample_img
from src.neural_recon.bak.phase1 import Phase1


def sample_points_2d(v_edge_points, v_num_horizontal):
    device = v_edge_points.device
    cur_dir = v_edge_points[:, 1] - v_edge_points[:, 0]
    cur_length = torch.linalg.norm(cur_dir, dim=-1) + 1e-6

    cur_dir_h = torch.cat((cur_dir, torch.zeros_like(cur_dir[:, 0:1])), dim=1)
    z_axis = torch.zeros_like(cur_dir_h)
    z_axis[:, 2] = 1
    edge_up = normalize_tensor(torch.cross(cur_dir_h, z_axis, dim=1)[:, :2]) * 0.0125
    # The vertical length is 10 -> 10/800 = 0.0125

    # Compute interpolated point
    num_horizontal = v_num_horizontal
    num_half_vertical = 10
    num_coordinates_per_edge = num_horizontal * num_half_vertical * 2

    begin_idxes = num_horizontal.cumsum(dim=0)
    total_num_x_coords = begin_idxes[-1]
    begin_idxes = begin_idxes.roll(1)  # Used to calculate the value
    begin_idxes[0] = 0  # (M,)
    dx = torch.arange(num_horizontal.sum(), device=device) - \
         begin_idxes.repeat_interleave(num_horizontal)  # (total_num_x_coords,)
    dx = dx / (num_horizontal - 1).repeat_interleave(num_horizontal)
    dy = torch.arange(num_half_vertical, device=device) / (num_half_vertical - 1)
    dy = torch.cat((torch.flip(-dy, dims=[0]), dy))

    # Meshgrid
    total_num_coords = total_num_x_coords * dy.shape[0]
    coords_x = dx.repeat_interleave(torch.ones_like(dx, dtype=torch.long) * dy.shape[0])  # (total_num_coords,)
    coords_y = torch.tile(dy, (total_num_x_coords,))  # (total_num_coords,)
    coords = torch.stack((coords_x, coords_y), dim=1)

    interpolated_coordinates = \
        cur_dir.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_x[:, None] + \
        edge_up.repeat_interleave(num_coordinates_per_edge, dim=0) * coords_y[:, None] + \
        v_edge_points[:, 0].repeat_interleave(num_coordinates_per_edge, dim=0)

    return num_coordinates_per_edge, interpolated_coordinates


if __name__ == '__main__':
    img1_ = cv2.imread(r"D:\DATASET\SIGA2023\Mechanism\ABC-NEF-COLMAP\00000077\imgs\0_colors.png",cv2.IMREAD_GRAYSCALE)
    img2_ = cv2.imread(r"D:\DATASET\SIGA2023\Mechanism\ABC-NEF-COLMAP\00000077\imgs\5_colors.png",cv2.IMREAD_GRAYSCALE)
    img1 = torch.from_numpy(img1_).unsqueeze(0).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2_).unsqueeze(0).unsqueeze(0) / 255.

    sp1 = torch.tensor((330,526), dtype=torch.float32) / 800.
    ep1 = torch.tensor((436,529), dtype=torch.float32) / 800.

    sp2 = torch.tensor((310,516), dtype=torch.float32) / 800.
    ep2 = torch.tensor((392,526), dtype=torch.float32) / 800.

    num_horizontal = torch.tensor((100,),dtype=torch.long)
    num_edges, sample_points1 = sample_points_2d(torch.stack((sp1,ep1)).unsqueeze(0), num_horizontal)
    _, sample_points2 = sample_points_2d(torch.stack((sp2,ep2)).unsqueeze(0), num_horizontal)

    sample_imgs1 = sample_img(img1, sample_points1[None, :, :])[0]
    sample_imgs2 = sample_img(img2, sample_points2[None, :, :])[0]

    # similarity_loss, black_area_in_img1 = loss2(sample_imgs1, sample_imgs2.unsqueeze(0), num_edges)
    similarity_loss, black_area_in_img1 = loss4(sample_imgs1, sample_imgs2.unsqueeze(0), num_edges)
    print("{:.3f}".format(similarity_loss.item()))

    cv2.namedWindow("1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1", 1600, 900)
    cv2.moveWindow("1", 0, 0)

    viz_img1 = cv2.cvtColor(img1_, cv2.COLOR_GRAY2BGR)
    cv2.line(viz_img1,
             (sp1*800).numpy().astype(np.int64),
             (ep1*800).numpy().astype(np.int64),
             (0,0,255), 1)
    viz_img2 = cv2.cvtColor(img2_, cv2.COLOR_GRAY2BGR)
    cv2.line(viz_img2,
             (sp2*800).numpy().astype(np.int64),
             (ep2*800).numpy().astype(np.int64),
             (0,0,255), 1)
    cv2.imshow("1",np.concatenate((viz_img1,viz_img2),axis=1))
    cv2.waitKey()
