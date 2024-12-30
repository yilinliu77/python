import shutil
from pathlib import Path
from tqdm import tqdm

from src.brepnet.paper.step_to_obj import import_step_file_as_obj

succ_root = Path(r"C:/Users/yilin/Desktop/1224_txt/succ")
src_root = Path(r"D:/brepnet/deepcad_v6")
dst_root = Path(r"C:/Users/yilin/Desktop/1224_txt/output")

for succ_folder in tqdm(succ_root.iterdir()):
    succ_name = succ_folder.stem
    prefix = succ_name.split("_")[1]
    out_folder = dst_root / prefix
    out_folder.mkdir(exist_ok=True, parents=True)
    import_step_file_as_obj(succ_folder / "recon_brep.step", out_folder, 0.01, 0.005, 100)
    import_step_file_as_obj(src_root / prefix / "normalized_shape.step", out_folder, 0.01, 0.005, 100)
    shutil.copyfile(succ_folder/f"{prefix}_txt.txt", out_folder/f"{prefix}_txt.txt")
