import time
from typing import List

import cv2
import numpy as np
import torch


def debug_imgs(v_imgs: List[np.ndarray]) -> None:
    if not isinstance(v_imgs, List):
        print("Need to input a list of np.ndarray")
        raise

    if cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) <= 0:
        cv2.namedWindow("test", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("test", 1600, 900)
    imgs = np.concatenate(v_imgs, axis=1)
    # if imgs.shape[2] == 3:
    #     imgs = imgs[:, :, [2, 1, 0]]
    cv2.imshow("test", imgs)
    cv2.waitKey()
    # cv2.destroyWindow("test")

def refresh_timer(a):
    delta = time.time() - a
    return delta, time.time()

###
# Numpy
###

def to_homogeneous_vector(v_array: np.ndarray):
    return np.insert(v_array, v_array.shape[v_array.ndim - 1], 1, axis=v_array.ndim - 1)


def to_homogeneous(v_array: np.ndarray):
    if v_array.ndim == 1:
        result = np.zeros(v_array.shape[0] + 1, dtype=v_array.dtype)
        result[-1] = 1
        result[:-1] = v_array
        return result
    else:
        result = np.zeros((4, 4), dtype=v_array.dtype)
        result[3, 3] = 1
        if v_array.shape[0] == 3 and v_array.shape[1] == 4:
            result[:3, :4] = v_array
            return result
        elif v_array.shape[0] == 3 and v_array.shape[1] == 3:
            result[:3, :3] = v_array
            return result
        elif v_array.shape[0] == 4 and v_array.shape[1] == 4:
            return v_array
        else:
            raise


# v_segments: (N, 6)
def save_line_cloud(v_file_path: str, v_segments: np.ndarray):
    with open(v_file_path, "w") as f:
        for segment in v_segments:
            f.write("v {} {} {}\n".format(segment[0], segment[1], segment[2]))
            f.write("v {} {} {}\n".format(segment[3], segment[4], segment[5]))
        for i in range(v_segments.shape[0]):
            f.write("l {} {}\n".format(i * 2 + 1, i * 2 + 2))


def normalize_vector(v_vector):
    length = np.linalg.norm(v_vector, axis=-1, keepdims=True) + 1e-8
    # assert length != 0
    return v_vector / length


def padding(array, desired_height, desired_width):
    """
    :param array: numpy array
    :param desired_height: desired height
    :param desired_width: desired width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = (desired_height - h) // 2
    aa = desired_height - a - h

    b = (desired_width - w) // 2
    bb = desired_width - b - w

    if len(array.shape) == 3:
        return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')
    elif len(array.shape) == 2:
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        raise


def pad_and_enlarge_along_y(v_small_img, v_big_img):
    source_w = v_small_img.shape[1]
    source_h = v_small_img.shape[0]
    target_w = v_big_img.shape[1]
    target_h = v_big_img.shape[0]

    if target_w / source_w < target_h / source_h:
        resized_shape = (target_w, int(target_w / source_w * source_h))
    else:
        resized_shape = (int(target_h / source_h * source_w), target_h)

    padding_img = padding(cv2.resize(v_small_img, resized_shape, interpolation=cv2.INTER_AREA), target_h, target_w)
    return np.concatenate((v_big_img,padding_img),axis=1)


###
# Tensor
###
def to_homogeneous_tensor(v_array: torch.Tensor):
    ndims = v_array.dim()
    one = torch.ones_like(v_array[..., 0:1])
    return torch.cat((v_array, one), dim=ndims - 1)


def to_homogeneous_mat_tensor(v_array: torch.Tensor):
    result = torch.zeros((4, 4), dtype=v_array.dtype, device=v_array.device)
    result[3, 3] = 1
    if v_array.shape[0] == 3 and v_array.shape[1] == 4:
        result[:3, :4] = v_array
        return result
    elif v_array.shape[0] == 3 and v_array.shape[1] == 3:
        result[:3, :3] = v_array
        return result
    elif v_array.shape[0] == 4 and v_array.shape[1] == 4:
        return v_array
    else:
        raise


def normalize_tensor(v_vector):
    length = torch.linalg.norm(v_vector+1e-6, dim=-1, keepdim=True)
    # assert length != 0
    return v_vector / length


def normalized_torch_img_to_numpy(v_tensor: torch.Tensor):
    assert len(v_tensor.shape) == 4 or len(v_tensor.shape) == 3
    if len(v_tensor.shape) == 4:
        t = v_tensor.permute(0, 2, 3, 1).detach().cpu()
    elif len(v_tensor.shape) == 3:
        t = v_tensor.permute(1, 2, 0).detach().cpu()
    else:
        raise
    return np.clip((t.numpy() * 255.).astype(np.uint8), 0, 255)

# Physics notions(ISO) are used here
# https://en.wikipedia.org/wiki/Spherical_coordinate_system
# Phi: xy plane; [-pi,pi]
# Theta: z; [0,pi]
def vector_to_sphere_coordinate(v_tensor):
    normalized_tensor = normalize_tensor(v_tensor)
    phi = torch.atan2(normalized_tensor[...,1],normalized_tensor[...,0])
    theta = torch.atan2(torch.sqrt(normalized_tensor[...,1]**2+normalized_tensor[...,0]**2),normalized_tensor[...,2])
    return torch.stack((phi,theta),dim=-1)

def sphere_coordinate_to_vector(v_phi, v_theta):
    x = torch.cos(v_phi) * torch.sin(v_theta)
    y = torch.sin(v_phi) * torch.sin(v_theta)
    z = torch.cos(v_theta)
    return torch.stack((x,y,z),dim=-1)

# These two functions are calculating the rotation matrix for an arrow that created by open3d
def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def get_line_mesh(v_path, v_points, v_lines):
    with open(v_path,"w") as f:
        for item in v_points:
            f.write("v {} {} {} 1 0 0\n".format(item[0],item[1],item[2]))
        for item in v_lines:
            f.write("l {} {}\n".format(item[0] + 1, item[1] + 1))
