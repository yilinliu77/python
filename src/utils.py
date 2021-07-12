import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def convert_standard_normalization_back(v_img: np.ndarray) -> np.ndarray:
    img = np.clip((v_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).astype(
        np.uint8)
    return img

def project_box3d_to_img(v_data,v_p2, v_pitch_45, v_width=None, v_height=None):
    camera_corners = []
    for id_batch in range(len(v_data)):
        corners = torch.tensor(
            [
                [-v_data[id_batch].l / 2,
                 0,
                 -v_data[id_batch].w / 2],
                [-v_data[id_batch].l / 2,
                 0,
                 v_data[id_batch].w / 2],
                [v_data[id_batch].l / 2,
                 0,
                 -v_data[id_batch].w / 2],
                [v_data[id_batch].l / 2,
                 0,
                 v_data[id_batch].w / 2],
                [-v_data[id_batch].l / 2,
                 -v_data[id_batch].h,
                 -v_data[id_batch].w / 2],
                [-v_data[id_batch].l / 2,
                 -v_data[id_batch].h,
                 v_data[id_batch].w / 2],
                [v_data[id_batch].l / 2,
                 -v_data[id_batch].h,
                 -v_data[id_batch].w / 2],
                [v_data[id_batch].l / 2,
                 -v_data[id_batch].h,
                 v_data[id_batch].w / 2],
                [0,
                 -v_data[id_batch].h / 2,
                 0],

            ]
        ).float()

        # Rotate through Y axis
        # Both upper of lower case is accept. The box is currently at the origin
        yaw_rotation_matrix = torch.tensor(
            R.from_euler("xyz", [0, v_data[id_batch].ry, 0]).as_matrix()).float()
        corners = torch.matmul(yaw_rotation_matrix, corners.T).T

        corners = corners + torch.tensor([
            v_data[id_batch].x,
            v_data[id_batch].y,
            v_data[id_batch].z
        ]).float()  # [N, 8, 3]

        # Rotate through Y axis
        # Should be global coordinates, upper case in scipy's Rotation
        if v_pitch_45:
            rotation_matrix = torch.tensor(
                R.from_euler("XYZ", [math.pi / 4, 0, 0]).as_matrix()).float()
            corners = torch.matmul(rotation_matrix, corners.T).T

        camera_corners.append(torch.cat([corners,
                                         torch.ones_like(corners[:, 0:1])],
                                        dim=-1))  # [N, 8, 4]
        # with open("{}.xyz".format(id_batch), "w") as f:
        #     for point in camera_corners[-1]:
        #         f.write("{} {} {}\n".format(point[0].item(), point[1].item(), point[2].item()))
        pass
    camera_corners = torch.stack(camera_corners, dim=0)
    camera_corners = torch.matmul(torch.tensor(v_p2).float(),
                                  camera_corners.transpose(1, 2)).transpose(1, 2)  # [N, 8, 3]

    homo_coord = (camera_corners / (camera_corners[:, :, 2:] + 1e-6))[:, :, :2]  # [N, 8, 3]
    if v_width is not None:
        homo_coord = torch.stack([
            torch.clamp(homo_coord[:, :, 0], 0, v_width),
            torch.clamp(homo_coord[:, :, 1], 0, v_height),
        ], dim=-1)
    return homo_coord