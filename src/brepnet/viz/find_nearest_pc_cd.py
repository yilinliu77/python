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


def find_nearest(local_out_root, src_pcs_c, ref_pcs, ref_shape_paths):
    chamferdist = ChamferDistance()
    cf2ref = chamferdist(src_pcs_c.unsqueeze(0).repeat(ref_pcs.shape[0], 1, 1), ref_pcs,
                         batch_reduction=None, point_reduction='mean', bidirectional=True)
    topk_near_idx = torch.topk(-cf2ref, 10).indices.tolist()
    with open(os.path.join(local_out_root, "nearest.txt"), "w") as f:
        f.write("Source: {}\n".format(os.path.basename(local_out_root)))
        f.write("Top10 Nearest:\n")
        for ref_idx in topk_near_idx:
            folder_name = os.path.basename(os.path.dirname(ref_shape_paths[ref_idx]))
            f.write(f"{folder_name} {cf2ref[ref_idx]}\n")
    near_mesh_paths = [os.path.join(os.path.dirname(ref_shape_paths[ref_idx]), "mesh.ply") for ref_idx in topk_near_idx]
    src_mesh_paths = glob.glob(os.path.join(local_out_root, "*.stl"))
    mesh_paths = src_mesh_paths + near_mesh_paths
    arrange_meshes(mesh_paths, os.path.join(local_out_root, "nearest.ply"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_post", type=str, required=True)
    parser.add_argument("--fake_pcd_root", type=str, required=False)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--txt", type=str, required=False, default="")
    parser.add_argument("--use_ray", action='store_true')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_gpus_task", type=float, default=0.5)

    args = parser.parse_args()
    fake_post = args.fake_post
    fake_pcd_root = args.fake_pcd_root
    train_root = args.train_root

    if not os.path.exists(fake_post) or not os.path.exists(train_root) or not os.path.exists(fake_pcd_root):
        print("fake_post: ", fake_post)
        print("train_root: ", train_root)
        print("fake_pcd_root: ", fake_pcd_root)
        raise ValueError("Invalid path")

    print("\nLoading reference point clouds...")
    # Load reference pcd
    num_cpus = multiprocessing.cpu_count()
    ref_pcs = []
    if args.txt:
        with open(args.txt, "r") as f:
            ref_folders = [line.strip() for line in f.readlines()]
        ref_shape_paths = [os.path.join(train_root, folder, "pc.ply") for folder in ref_folders]
    else:
        ref_shape_paths = load_data_with_prefix(train_root, 'pc.ply')

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
        src_shape_paths = [os.path.join(fake_pcd_root, args.prefix + ".ply")]
    else:
        src_shape_paths = load_data_with_prefix(fake_pcd_root, '.ply')

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
    if not args.use_ray:
        for i in tqdm(range(src_pcs.shape[0])):
            local_out_root = os.path.join(fake_post, os.path.basename(src_shape_paths[i])[:-4])
            find_nearest(local_out_root, src_pcs[i], ref_pcs, ref_shape_paths)
    else:
        find_nearest_remote = ray.remote(num_gpus=args.num_gpus_task)(find_nearest)
        ray.init(
                # local_mode=True,
                num_gpus=args.num_gpus,
        )
        futures = []
        for i in tqdm(range(src_pcs.shape[0])):
            local_out_root = os.path.join(fake_post, os.path.basename(src_shape_paths[i])[:-4])
            futures.append(find_nearest_remote.remote(local_out_root, src_pcs[i], ref_pcs, ref_shape_paths))
        ray.get(futures)
        ray.shutdown()
