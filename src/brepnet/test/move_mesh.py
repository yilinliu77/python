import os
from pathlib import Path
import shutil

import ray
from tqdm import tqdm

from src.brepnet.test.viz_data_test import prefix

v_root = Path("C:/Users/yilin/Desktop/test_output/")
v_output = Path("C:/Users/yilin/Desktop/test_output/mesh")


def process(prefiex):
    num_valid = 0
    for prefix in tqdm(prefiex):
        if not (v_root / prefix / "recon_brep.ply").exists():
            print("No mesh found for {}".format(prefix))
            continue
        shutil.copyfile(v_root / prefix / "recon_brep.ply", v_output / "{}.ply".format(prefix))
        num_valid+=1

    print("{}/{}".format(num_valid, len(prefiex)))


if __name__ == "__main__":
    v_output.mkdir(exist_ok=True, parents=True)
    prefixes = [item[:8] for item in os.listdir(v_root) if item.startswith("00")]
    process(prefixes)

