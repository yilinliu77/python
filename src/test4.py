import sys
from functools import partial
from multiprocessing import Pool
import os
import numpy as np

if __name__ == '__main__':
    list1 = [item.strip()[:8] for item in open(r"src/process_abc/parsenet_test_ids.txt").readlines()]
    list2 = [item.strip() for item in open(r"src/brepnet/data/list/deduplicated_deepcad_7_30.txt").readlines()]
    list3 = [item.strip() for item in open(r"src/process_abc/abc_single_solid.txt").readlines()]
    list4 = [item.strip() for item in open(r"src/process_abc/abc_730.txt").readlines()]
    final_list = list(set(list1) & set(list2))
    final_list.sort()
    pass