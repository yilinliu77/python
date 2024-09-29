import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm


def get_bins(v_root):
    v_root = Path(v_root)
    files = [item for item in os.listdir(v_root) if item.endswith("feature.npz")]
    files.sort()

    features = []
    for file in tqdm(files[:100]):
        feature = np.load(v_root / file)["face_features"]
        features.append(feature)
    features = np.concatenate(features, axis=0)

    bins = np.histogram(features, bins=32, range=(-10, 10))
    return (bins[0], (bins[1][:-1] + bins[1][1:]) / 2)

if __name__ == '__main__':
    bins1 = get_bins("D:/brepnet/Test_AutoEncoder_context_KL")
    bins2 = get_bins("D:/brepnet/Test_AutoEncoder_0925")
    bins3 = get_bins("D:/brepnet/Test_AutoEncoder_0925_gaussian")

    # Draw
    plt.figure()
    plt.plot(bins2[1], bins2[0], color="black", label="No KL")
    plt.plot(bins1[1], bins1[0], color="purple", label="Small KL")
    plt.plot(bins3[1], bins3[0], color="orange", label="Large KL")
    # plt.hist(bins2[0], bins=32, range=(-10, 10), color='blue', label='0925')
    # plt.hist(bins3[0], bins=32, range=(-10, 10), color='green', label='0925_gaussian')
    plt.legend()
    plt.show()

