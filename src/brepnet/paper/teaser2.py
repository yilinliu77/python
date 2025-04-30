import random
import os
import shutil
from pathlib import Path
from random import shuffle

from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Extend.DataExchange import read_step_file

from shared.occ_utils import get_primitives
from src.brepnet.eval.check_valid import check_step_valid_soild

random.seed(77)

root = Path(r"D:/brepnet/paper_imgs/uncond_results/1203_deepcad_730_li_0.0001_0.02_mean_11k_1000k_post")
output_root = Path(r"D:/brepnet/paper/teaser4/uncond4/step")

b=os.listdir(root)
b.sort()
shuffle(b)

id_cur = 0
for prefix in b:
    _, shape = check_step_valid_soild(root/f"{prefix}/recon_brep.step",return_shape=True)
    if len(get_primitives(shape, TopAbs_FACE)) < 10:
        continue
    id_cur += 1
    print(prefix)
    if (root/f"{prefix}/success.txt").exists():
        print("success")
    else:
        print("fail")

    shutil.copyfile(root/f"{prefix}/recon_brep.step", output_root/f"{prefix}.step")
    if (root/f"{prefix}/recon_brep.stl").exists():
        shutil.copyfile(root/f"{prefix}/recon_brep.stl", output_root/f"{prefix}.stl")

    if id_cur >= 10:
        break