import sys
from functools import partial
from multiprocessing import Pool
import os
import numpy as np

if __name__ == '__main__':
    list1 = [item.strip() for item in open(r"src/brepnet/data/list/deduplicated_deepcad_testing.txt").readlines()]
    list2 = os.listdir(r"/mnt/d/deepcad_v6")
    final_list = list(set(list1) & set(list2))
    final_list.sort()
    np.savetxt(r"src/brepnet/data/list/deduplicated_deepcad_testing_brepnet.txt", final_list, fmt="%s")
