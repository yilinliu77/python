from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, pickle
import numpy as np
import trimesh

brepgen_data = r"D:/brepnet/paper/valid/brepgen_filtered_valid_relation.npy"
ours_data = r"D:/brepnet/paper/valid/1127_730_li_270k_1gpu_valid_relation.npy"

if __name__ == "__main__":
    brepgen = np.load(brepgen_data)
    ours = np.load(ours_data)

    valid_range = np.arange(6,30,4)

    brepgen_idx = np.digitize(brepgen[:,0], bins=valid_range)
    brepgen_num = [0 for _ in range(len(valid_range))]
    for i in range(1, len(valid_range) + 1):
        brepgen_num[i-1] = brepgen[brepgen_idx == i,1].sum() / np.sum(brepgen_idx == i)

    ours_idx = np.digitize(ours[:,0], bins=valid_range)
    ours_num = [0 for _ in range(len(valid_range))]
    for i in range(1, len(valid_range) + 1):
        ours_num[i-1] = ours[ours_idx == i,1].sum() / np.sum(ours_idx == i)

    plt.plot(valid_range, brepgen_num, label='BRepGen', color=np.array((255,191,1,255))/255.)
    plt.plot(valid_range, ours_num, label='Ours', color=np.array((128,195,28,255))/255.)
    plt.ylabel("Valid Shape Number")
    plt.xlabel("Face Number")
    plt.title("Relation between shape complexity and shape validity")
    plt.legend()
    plt.show()
    # print(data.mean())
