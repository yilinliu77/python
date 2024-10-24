import os
import shutil

from tqdm import tqdm

from src.brepnet.eval.check_valid import check_step_valid_soild, load_data_with_prefix
from src.brepnet.post.utils import *
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_SOLID

from src.brepnet.eval.eval_brepgen import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--ref_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True, default="")
    args = parser.parse_args()
    src_root = args.src_root
    ref_root = args.ref_root

    # Load reference pcd
    num_cpus = multiprocessing.cpu_count()
    ref_pcs = []
    shape_paths = load_data_with_prefix(ref_root, '.ply')
    load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, shape_paths)
    for pc in tqdm(load_iter, total=len(shape_paths)):
        if len(pc) > 0:
            ref_pcs.append(pc)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("real point clouds: {}".format(ref_pcs.shape))

    # Load fake pcd
    src_pcs = []
    if args.prefix:
        shape_paths = load_data_with_prefix(os.path.join(src_root, args.prefix), '.ply')
    else:
        shape_paths = load_data_with_prefix(src_root, '.ply')
    load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, shape_paths)
    for pc in tqdm(load_iter, total=len(shape_paths)):
        if len(pc) > 0:
            src_pcs.append(pc)
    src_pcs = np.stack(src_pcs, axis=0)

    pass
