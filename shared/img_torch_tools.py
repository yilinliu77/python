import torch
import numpy as np


def get_img_from_tensor(v_tensor: torch.Tensor):
    img = v_tensor.cpu().numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    img = img * 255
    img = np.asarray(img, dtype=np.uint8)
    return img
