import sys
from functools import partial
from multiprocessing import Pool
import os
import numpy as np

if __name__ == '__main__':
    list1 = [item.strip() for item in open(r"src/brepnet/data/list/deduplicated_deepcad_7_30_final.txt").readlines()]
    list2 = [item.strip() for item in open(r"src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt").readlines()]
    final_list = list(set(list1) & set(list2))
    final_list.sort()
    np.savetxt(r"src/brepnet/data/list/new/deduplicated_deepcad_testing_7_30.txt", final_list, fmt="%s")
