from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, pickle
import numpy as np
import trimesh

deepcad_pkl = r"D:/brepnet/paper/uncond_lfd/deepcad_filtered_lfd.pkl"
brepgen_pkl = r"D:/brepnet/paper/uncond_lfd/brepgen_filtered_lfd.pkl"
ours_pkl = r"D:/brepnet/paper/uncond_lfd/1203_deepcad_730_li_0.0001_0.02_mean_11k_11m_lfd.pkl"

if __name__ == "__main__":
    _, _, deepcad = pickle.load(open(deepcad_pkl, "rb"))
    deepcad = deepcad.min(axis=1)
    _, _, brepgen = pickle.load(open(brepgen_pkl, "rb"))
    brepgen = brepgen.min(axis=1)
    _, _, ours = pickle.load(open(ours_pkl, "rb"))
    ours = ours.min(axis=1)

    deepcad_his, deepcad_edges = np.histogram(deepcad, bins=20, range=(0, 4500))
    brepgen_his, brepgen_edges = np.histogram(brepgen, bins=20, range=(0, 4500))
    ours_his, ours_edges = np.histogram(ours, bins=20, range=(0, 4500))
    plt.barh(deepcad_edges[:-1] - 50, deepcad_his, height=50, color=np.array((0,148,126,255))/255., label='DeepCAD')
    plt.barh(brepgen_edges[:-1], brepgen_his, height=50, color=np.array((255,191,1,255))/255., label='BRepGen')
    plt.barh(ours_edges[:-1] + 50, ours_his, height=50, color=np.array((128,195,28,255))/255., label='Ours')

    plt.xlim(0, 100)
    plt.title("Light Field Distance (LFD) Distribution of the Generated Shapes")
    plt.xlabel("Frequency")
    plt.ylabel("Light Field Distance (LFD)")
    plt.legend()
    plt.savefig(r"D:/brepnet/paper/uncond_lfd/lfd.pdf", dpi=600)
    plt.show()
    # print(data.mean())
