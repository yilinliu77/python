import os
import time

import numba as nb
import torch
import numpy as np

from src.regress_reconstructability_hyper_parameters.dataset import Regress_hyper_parameters_img_dataset


@nb.njit
def str_to_int(s):
    final_index, result = len(s) - 1, 0
    for i,v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result

@nb.jit(nopython=True)
def str_to_float(v_str: str) -> float:
    dot_pos: int = 0
    candidate_str = v_str
    if v_str[0] == "-":
        candidate_str = candidate_str[1:]

    for index in range(len(candidate_str)):
        if candidate_str[index] == ".":
            dot_pos=index
    # print(dot_pos)
    if dot_pos == 0:
        raise

    result: float = 0
    for index in range(len(candidate_str)):
        if index < dot_pos:
            result += 10. ** (dot_pos - index - 1) * str_to_int(candidate_str[index])
        elif index > dot_pos:
            result += 10. ** (dot_pos - index)  * str_to_int(candidate_str[index])

    if v_str[0] == "-":
        result = result * -1
    return result

if __name__ == '__main__':
    a = torch.tensor()


    data_file = r"D:\Projects\Reconstructability\training_data\v7\chengbao_coarse_ds_0090\views.npy"
    cur = time.time()
    data = np.load(data_file).copy()
    print("Read file: ",time.time() - cur)

    mean_time = 0
    for i in range(10000):
        cur = time.time()
        np.log(data[np.random.randint(0,data.shape[0])] ** 2)
        mean_time += time.time()-cur
    print("In memory test: {:.2f}".format(mean_time / 10))
    mean_time = 0
    for i in range(10000):
        cur = time.time()
        item = np.load(data_file,mmap_mode="r")[np.random.randint(0,data.shape[0])]
        np.log(item ** 2)
        mean_time += time.time()-cur
    print("Out memory test: {:.2f}".format(mean_time / 10))

    aa = np.load(data_file,mmap_mode="r")
    mean_time = 0
    for i in range(10000):
        cur = time.time()
        item = aa[np.random.randint(0,data.shape[0])]
        np.log(item ** 2)
        mean_time += time.time()-cur
    print("Out memory test: {:.2f}".format(mean_time / 10))

    pass