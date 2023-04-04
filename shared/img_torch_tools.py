import torch
import numpy as np


def get_img_from_tensor(v_tensor: torch.Tensor):
    img = v_tensor.cpu().numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    img = img * 255
    img = np.asarray(img, dtype=np.uint8)
    return img

def print_model_size(v_model):
    param_size = 0
    for param in v_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in v_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    buffer_mb = buffer_size / 1024 ** 2
    param_mb = param_size / 1024 ** 2
    print('Total size: {:.3f}MB; Buffer: {:.3f}MB; Parameters: {:.3f}MB'.format(size_all_mb, buffer_mb, param_mb))