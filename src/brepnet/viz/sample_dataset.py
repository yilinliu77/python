from pathlib import Path
import random

from tqdm import tqdm
import trimesh

data_root = Path("/mnt/d/brepgen_train")
names = open(r"src/brepnet/data/list/deduplicated_deepcad_training_30.txt").readlines()

random.shuffle(names)
names = names[:1600]

mesh = trimesh.Trimesh()

for idx, name in enumerate(tqdm(names)):
    delta = [idx//40*3, idx%40*3, 0]
    name = name.strip()
    item = trimesh.load(data_root/name/"mesh.ply")
    item.vertices += delta
    mesh += item

mesh.export("deepcad_training_30.ply")