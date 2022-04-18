from typing import List

import cv2
import numpy as np


def debug_imgs(v_imgs: List[np.ndarray]) -> None:
    if not isinstance(v_imgs, List):
        print("Need to input a list of np.ndarray")
        raise

    if cv2.getWindowProperty("test",cv2.WND_PROP_VISIBLE) <= 0:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 1600, 900)
    imgs = np.concatenate(v_imgs, axis=1)
    if imgs.shape[2] == 3:
        imgs = imgs[:, :, [2, 1, 0]]
    cv2.imshow("test", imgs)
    cv2.waitKey()
    # cv2.destroyWindow("test")
