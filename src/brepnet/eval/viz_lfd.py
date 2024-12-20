from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, pickle
import numpy as np
import trimesh

pkl = r"/mnt/d/uncond_results/1127/1127_730_li_120k_lfd.pkl"
# pkl = r"/mnt/d/uncond_results/1106/1108_730_li_600k_30_lfd.pkl"
output_path = r"/mnt/d/uncond_results/1127/1127_730_li_120k_lfd.png"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl = sys.argv[1]
        output_path = sys.argv[2]

    src_folder_list, nearest_name, data_ori = pickle.load(open(pkl, "rb"))
    data = data_ori.min(axis=1)

    his, bin_edges = np.histogram(data, bins=45, range=(0, 4500))
    plt.xlim(0, 100)
    plt.barh(bin_edges[:-1], his, height=50)
    plt.title("Light Field Distance (LFD) Distribution of the Generated Shapes")
    plt.xlabel("Frequency")
    plt.ylabel("Light Field Distance (LFD)")
    plt.savefig(output_path, dpi=600)

    print("Mean: ", data.mean())
    print("Max: ", data.max())
    print("Min: ", data.min())
    print("Std: ", data.std())
    print("Median: ", np.median(data))
    print("1%: ", np.percentile(data, 1))
    print("10%: ", np.percentile(data, 10))
    print("25%: ", np.percentile(data, 25))
    print("50%: ", np.percentile(data, 50))
    print("75%: ", np.percentile(data, 75))
    print("90%: ", np.percentile(data, 90))
    print("99%: ", np.percentile(data, 99))

    if True:
        small_ids = np.where(data < 100)
        # data_root = Path("/mnt/d/brepgen_train")
        # data_root = Path("/mnt/d/deepcad_train_v6")
        # pkl = Path(pkl)
        # src_dir = pkl.parent / Path(pkl).name.replace("_lfd.pkl", "_post")
        # dst_file = pkl.parent / Path(pkl).name.replace("_lfd.pkl", "_lfd.ply")
        # mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        # offset = 0
        # for id in tqdm(small_ids[0]):
        #     item = trimesh.load(str(src_dir / src_folder_list[id] / "recon_brep.stl"))
        #     delta_x = offset % 20 * 9
        #     delta_y = offset // 20 * 3
        #     item.vertices += [delta_x, delta_y, 0]
        #     mesh = mesh + item
        #     item = trimesh.load(str(data_root / nearest_name[id] / "mesh.ply"))
        #     delta_x = offset % 20 * 9 + 3
        #     delta_y = offset // 20 * 3
        #     item.vertices += [delta_x, delta_y, 0]
        #     mesh = mesh + item
        #     offset += 1
        # mesh.export(str(dst_file))
        print(f"{small_ids[0].shape[0]} shapes' LFD is less than 100")
    pass
