import os
from pathlib import Path
import shutil

import ray
from tqdm import tqdm

from src.brepnet.test.viz_data_test import prefix

v_root = Path("D:/brepnet/Test_AutoEncoder_context")


@ray.remote
def process(prefiex):
    for prefix in prefiex:
        shutil.move(v_root / (prefix + "_feature.npy"), v_root / prefix / "feature.npy")


if __name__ == "__main__":
    prefixes = [item[:8] for item in os.listdir(v_root) if item.endswith("feature.npy")]

    ray.init()

    tasks = []
    for i in range(0, len(prefixes), 1000):
        tasks.append(process.remote(prefixes[i:min(len(prefixes), i + 1000)]))
    ray.get(tasks)

