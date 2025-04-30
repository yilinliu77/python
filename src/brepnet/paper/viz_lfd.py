from pathlib import Path

from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, pickle
import numpy as np
import trimesh

deepcad_pkl = r"D:/brepnet/paper/uncond_lfd/deepcad_filtered_lfd.pkl"
brepgen_pkl = r"D:/brepnet/paper/uncond_lfd/brepgen_filtered_lfd.pkl"
ours_pkl = r"D:/brepnet/paper_imgs/uncond_results/1203_deepcad_730_li_0.0001_0.02_mean_11k_1000k_lfd.pkl"

if __name__ == "__main__":
    _, _, deepcad = pickle.load(open(deepcad_pkl, "rb"))
    deepcad = deepcad.min(axis=1)
    _, _, brepgen = pickle.load(open(brepgen_pkl, "rb"))
    brepgen = brepgen.min(axis=1)
    _, _, ours = pickle.load(open(ours_pkl, "rb"))
    ours = ours.min(axis=1)

    index = np.arange(ours.shape[0])
    np.random.shuffle(index)
    ours = ours[index[:800]]

    deepcad_his, deepcad_edges = np.histogram(deepcad, bins=20, range=(0, 4500))
    brepgen_his, brepgen_edges = np.histogram(brepgen, bins=20, range=(0, 4500))
    ours_his, ours_edges = np.histogram(ours, bins=20, range=(0, 4500))
    # plt.barh(ours_edges[:-1] + 50, ours_his, height=50, color=np.array((128,195,28,255))/255., label='Ours')
    # plt.barh(brepgen_edges[:-1], brepgen_his, height=50, color=np.array((255,191,1,255))/255., label='BRepGen')
    # plt.barh(deepcad_edges[:-1] - 50, deepcad_his, height=50, color=np.array((0,148,126,255))/255., label='DeepCAD')

    ours_pline = make_interp_spline(ours_edges[:-1], ours_his, k=3)
    x = np.linspace(0, 4500, 100)
    ours_p = ours_pline(x)
    brepgen_pline = make_interp_spline(brepgen_edges[:-1], brepgen_his, k=3)
    brepgen_p = brepgen_pline(x)
    deepcad_pline = make_interp_spline(deepcad_edges[:-1], deepcad_his, k=3)
    deepcad_p = deepcad_pline(x)

    plt.plot(x, ours_p, 'k', linewidth=2, label='Ours')
    plt.plot(x, brepgen_p, 'r', linewidth=2, label='BRepGen')
    plt.plot(x, deepcad_p, 'b', linewidth=2, label='DeepCAD')

    plt.xlim(0, 4500)
    plt.title("Light Field Distance (LFD) Distribution of the Generated Shapes")
    plt.ylabel("Frequency")
    plt.xlabel("Light Field Distance (LFD)")
    plt.legend()
    plt.savefig(r"D:/brepnet/paper/uncond_lfd/lfd.pdf", dpi=600)
    plt.show()
    # print(data.mean())
