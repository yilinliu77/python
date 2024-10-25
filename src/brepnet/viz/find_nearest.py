import glob
import os
import shutil

from tqdm import tqdm
from src.brepnet.eval.check_valid import check_step_valid_soild, load_data_with_prefix
from src.brepnet.post.utils import *
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_SOLID

from src.brepnet.eval.eval_brepgen import *

from chamferdist import ChamferDistance

from src.brepnet.viz.sort_and_merge import arrange_meshes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--src_pc_root", type=str, required=True)
    parser.add_argument("--ref_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--txt", type=str, required=False, default="")
    args = parser.parse_args()
    src_root = args.src_root
    src_pc_root = args.src_pc_root
    ref_root = args.ref_root

    if not os.path.exists(src_root) or not os.path.exists(ref_root) or not os.path.exists(src_pc_root):
        raise ValueError("Invalid path")

    print("\nLoading reference point clouds...")
    # Load reference pcd
    num_cpus = multiprocessing.cpu_count()
    ref_pcs = []
    if args.txt:
        with open(args.txt, "r") as f:
            ref_folders = [line.strip() for line in f.readlines()]
        ref_shape_paths = [os.path.join(ref_root, folder, "pc.ply") for folder in ref_folders]
    else:
        ref_shape_paths = load_data_with_prefix(ref_root, 'pc.ply')

    assert len(ref_shape_paths) > 0
    ref_shape_paths.sort()
    load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, ref_shape_paths)
    for pc in tqdm(load_iter, total=len(ref_shape_paths)):
        if len(pc) > 0:
            ref_pcs.append(pc)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("real point clouds: {}".format(ref_pcs.shape))

    # Load fake pcd
    print("\nLoading fake point clouds...")
    src_pcs = []
    if args.prefix:
        src_shape_paths = [os.path.join(src_pc_root, args.prefix + ".ply")]
    else:
        src_shape_paths = load_data_with_prefix(src_pc_root, '.ply')

    assert len(src_shape_paths) > 0
    src_shape_paths.sort()
    load_iter = multiprocessing.Pool(num_cpus).imap(collect_pc2, src_shape_paths)
    for pc in tqdm(load_iter, total=len(src_shape_paths)):
        if len(pc) > 0:
            src_pcs.append(pc)
    src_pcs = np.stack(src_pcs, axis=0)
    print("fake point clouds: {}".format(src_pcs.shape))

    device = torch.device("cuda")
    src_pcs = torch.from_numpy(src_pcs).to(device).to(torch.float32)
    ref_pcs = torch.from_numpy(ref_pcs).to(device).to(torch.float32)

    print("\nFinding nearest...")
    chamferdist = ChamferDistance()
    batch_size = 10
    for i in tqdm(range(src_pcs.shape[0])):
        local_out_root = os.path.join(src_root, os.path.basename(src_shape_paths[i])[:-4])
        cf2ref = chamferdist(src_pcs[i].unsqueeze(0).repeat(ref_pcs.shape[0], 1, 1), ref_pcs, batch_reduction=None, bidirectional=False)
        topk_near_idx = torch.topk(-cf2ref, 10).indices.tolist()
        with open(os.path.join(local_out_root, "nearest.txt"), "w") as f:
            f.write("Source: {}\n".format(os.path.basename(local_out_root)))
            f.write("Top10 Nearest:\n")
            for ref_idx in topk_near_idx:
                f.write(os.path.basename(os.path.dirname(ref_shape_paths[ref_idx])) + "\n")
        near_mesh_paths = [os.path.join(os.path.dirname(ref_shape_paths[ref_idx]), "mesh.ply") for ref_idx in topk_near_idx]
        src_mesh_paths = glob.glob(os.path.join(local_out_root, "*.stl"))
        mesh_paths = src_mesh_paths + near_mesh_paths
        arrange_meshes(mesh_paths, os.path.join(local_out_root, "nearest.ply"))
