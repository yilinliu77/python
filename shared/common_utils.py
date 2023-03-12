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
            f.write("v {} {} {}\n".format(segment[0],segment[1],segment[2]))
            f.write("v {} {} {}\n".format(segment[3],segment[4],segment[5]))
        for i in range(v_segments.shape[0]):
            f.write("l {} {}\n".format(i*2+1, i*2+2))

def normalize_vector(v_vector):
    length = np.linalg.norm(v_vector)
    assert length!=0
    return v_vector/length

def normalized_torch_img_to_numpy(v_tensor: torch.Tensor):
    assert len(v_tensor.shape) == 4 or len(v_tensor.shape) == 3
    if len(v_tensor.shape) == 4:
        t = v_tensor.permute(0,2,3,1).detach().cpu()
    elif len(v_tensor.shape) == 3:
        t = v_tensor.permute(1,2,0).detach().cpu()
    else:
        raise
    return (t.numpy() * 255.).astype(np.uint8)