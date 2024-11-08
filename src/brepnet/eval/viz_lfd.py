from matplotlib import pyplot as plt
import sys, pickle
import numpy as np

pkl = r"/mnt/d/uncond_results/1106/1103_wcube_uncond_gaussian_epsilon_730_li_30_22m_30_lfd.pkl"
output_path = r"lfd.png"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl = sys.argv[1]
        output_path = sys.argv[2]
    
    src_folder_list, nearest_name, data = pickle.load(open(pkl, "rb"))
    data = data.min(axis=1)

    his, bin_edges = np.histogram(data, bins=45, range=(0, 4500))
    plt.xlim(0, 100)
    plt.barh(bin_edges[:-1], his, height=50)
    plt.title("Light Field Distance (LFD) Distribution of the Generated Shapes")
    plt.xlabel("Frequency")
    plt.ylabel("Light Field Distance (LFD)")
    plt.savefig(output_path, dpi=600)
    print(data.mean())
    pass

