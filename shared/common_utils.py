from typing import List

import cv2
import numpy as np


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
