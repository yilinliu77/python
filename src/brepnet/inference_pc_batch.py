import argparse
import shutil
import sys
import os.path
import time
from pathlib import Path
import random

import numpy as np
# import open3d as o3d
import ray
from PIL import Image
from tqdm import tqdm

sys.path.append('../../../')
from src.brepnet.post.utils import export_edges
from src.brepnet.diffusion_model import Diffusion_condition
from src.brepnet.post.construct_brep import construct_brep_from_datanpz

import torch
from lightning_fabric import seed_everything

os.environ["HTTP_PROXY"] = "http://172.31.178.126:7890"
os.environ["HTTPS_PROXY"] = "http://172.31.178.126:7890"


def choose_best(cond_root, folder, output_dir, num_proposals=32):
    valid_results = []
    all_results = []
    for i in range(num_proposals):
        if (output_dir / "after_post" / f"{folder}_{i:02d}" / "recon_brep.stl").exists():
            valid_results.append(output_dir / "after_post" / f"{folder}_{i:02d}" / "recon_brep.stl")
        if (output_dir / "after_post" / f"{folder}_{i:02d}" / "separate_faces.ply").exists():
            all_results.append(output_dir / "after_post" / f"{folder}_{i:02d}" / "separate_faces.ply")

    results = valid_results if len(valid_results) > 0 else all_results

    assert len(results) > 0, f"No valid results for {folder}"

    import open3d as o3d
    gt_pc = o3d.io.read_point_cloud(str(cond_root / folder / "pc.ply"))

    # eval 32 proposals
    num_sample_ratio = 1
    cd_dis_to_pc = []
    for i in range(len(results)):
        recon_mesh = o3d.io.read_triangle_mesh(str(results[i]))
        if len(recon_mesh.vertices) == 0:
            return np.Inf
        recon_pc = recon_mesh.sample_points_poisson_disk(len(gt_pc.points) * num_sample_ratio)
        dis = np.mean(gt_pc.compute_point_cloud_distance(recon_pc))
        cd_dis_to_pc.append(dis)

    best_results = results[np.argmin(cd_dis_to_pc)]
    shutil.copytree(best_results.parent, output_dir / "best_post" / folder, dirs_exist_ok=True)


def downsample_pc(points, n):
    sample_idx = random.sample(list(range(points.shape[0])), n)
    return points[sample_idx]


if __name__ == '__main__':
    conf = {
        "name"               : "Diffusion_condition",
        "train_decoder"      : False,
        "stored_z"           : False,
        "use_mean"           : True,
        "diffusion_latent"   : 768,
        "diffusion_type"     : "epsilon",
        "loss"               : "l2",
        "pad_method"         : "random",
        "num_max_faces"      : 30,
        "beta_schedule"      : "squaredcos_cap_v2",
        "beta_start"         : 0.0001,
        "beta_end"           : 0.02,
        "variance_type"      : "fixed_small",
        "addition_tag"       : False,
        "autoencoder"        : "AutoEncoder_1119_light",
        "with_intersection"  : True,
        "dim_latent"         : 8,
        "dim_shape"          : 768,
        "sigmoid"            : False,
        "in_channels"        : 6,
        "gaussian_weights"   : 1e-6,
        "norm"               : "layer",
        "autoencoder_weights": "",
        "is_aug"             : False,
        "condition"          : [],
        "cond_prob"          : []
    }

    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--autoencoder_weights', type=str, required=True)
    parser.add_argument('--diffusion_weights', type=str, required=True)
    parser.add_argument('--condition', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default="./inference_output")
    parser.add_argument("--list", required=True)
    parser.add_argument("--cond_root", required=True)
    parser.add_argument("--cond_pc_num", type=int, default=2048)
    parser.add_argument("--num_proposals", type=int, default=32)
    parser.add_argument("--only_best", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=100)
    parser.add_argument("--random_choose_num", type=int, required=False)
    parser.add_argument("--from_scratch", action="store_true")

    args = parser.parse_args()
    conf["autoencoder_weights"] = args.autoencoder_weights
    conf["diffusion_weights"] = args.diffusion_weights
    conf["condition"] = args.condition

    seed_everything(0)

    from_scratch = args.from_scratch
    if not from_scratch:
        print("Not from scratch, will use the existing data in output_dir.")
    else:
        print("From scratch, removing the existing data in output_dir.")
        shutil.rmtree(args.output_dir, ignore_errors=True) if os.path.exists(args.output_dir) else None

    cond_pc_num = int(args.cond_pc_num)
    num_proposals = int(args.num_proposals)

    assert args.list is not None, "list is required."
    assert args.cond_root is not None, "cond_root is required when list is provided."

    with open(args.list, "r") as f:
        args.input = [os.path.join(args.cond_root, line.strip(), 'pc.ply') for line in f.readlines()]
        for each in args.input:
            assert os.path.exists(each), f"File {each} not found."
        if args.random_choose_num is not None:
            print(f"Random choose {args.random_choose_num} from {len(args.input)}")
            args.input = np.random.choice(args.input, args.random_choose_num, replace=False)

    if args.only_best is not None:
        assert "pc" in args.condition, "Only pc is supported when list is provided."

    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    model = Diffusion_condition(conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_weights = torch.load(conf["diffusion_weights"], map_location=device, weights_only=False)["state_dict"]
    diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
    diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
    autoencoder_weights = torch.load(conf["autoencoder_weights"], map_location=device, weights_only=False)["state_dict"]
    autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
    autoencoder_weights = {"ae_model." + k: v for k, v in autoencoder_weights.items()}
    diffusion_weights.update(autoencoder_weights)
    diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
    model.load_state_dict(diffusion_weights, strict=False)
    model.to(device)
    model.eval()

    # 1. Generate data
    print("We have {} data".format(len(args.input)))
    output_dir = Path(args.output_dir)
    (output_dir / "network_pred").mkdir(parents=True, exist_ok=True)

    ts = time.time()
    batch_size = 1024 // num_proposals  # 1024 in RTX 4090
    # for id_item, fileitem in enumerate(tqdm(args.input)):
    for batch_idx in range(0, len(args.input), batch_size):
        hitted_fileitems = args.input[batch_idx:batch_idx + batch_size]
        data = {
            "conditions": {
            }
        }

        valid_hitted_fileitems = []
        for id_item, fileitem in enumerate(tqdm(hitted_fileitems)):
            name = os.path.basename(os.path.dirname(fileitem))
            if os.path.exists(output_dir / "network_pred" / f"{name}_00"):
                continue
            valid_hitted_fileitems.append(fileitem)
            if "pc" in conf["condition"]:
                input_file = Path(fileitem)
                name = input_file.stem
                if name == "pc":
                    name = f"{input_file.parent.stem}_pc"
                if not input_file.exists():
                    print(f"File {input_file} not found.")
                    exit(1)

                import open3d as o3d

                pcd = o3d.io.read_point_cloud(str(input_file))
                points = np.array(pcd.points)
                if pcd.has_normals():
                    normals = np.array(pcd.normals)
                else:
                    normals = np.zeros_like(points)

                # Normalize
                bbox_min = np.min(points, axis=0)
                bbox_max = np.max(points, axis=0)
                center = (bbox_min + bbox_max) / 2
                points -= center
                scale = np.max(bbox_max - bbox_min)
                points /= scale
                points *= 0.9 * 2

                points = np.concatenate([points, normals], axis=1)

                if points.shape[0] > cond_pc_num:
                    points = downsample_pc(points, cond_pc_num)
                elif points.shape[0] < cond_pc_num:
                    # upsample
                    raise NotImplementedError
                assert points.shape[0] == cond_pc_num
                points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
                if "points" not in data["conditions"]:
                    data["conditions"]["points"] = torch.empty((0, 1, points_tensor.shape[0], points_tensor.shape[1]),
                                                               device=points_tensor.device)
                data["conditions"]["points"] = torch.cat((data["conditions"]["points"],
                                                          points_tensor[None, None, :, :].repeat(num_proposals, 1, 1, 1)), dim=0)
            elif "txt" in conf["condition"]:
                raise NotImplementedError
                data["conditions"]["txt"] = [fileitem for item in range(num_proposals)]
                name = f"{id_item:02d}"
            elif "sketch" in conf["condition"]:
                raise NotImplementedError
                input_file = Path(fileitem)
                name = input_file.stem
                if not input_file.exists():
                    print(f"File {input_file} not found.")
                    exit(1)

                import torchvision.transforms as T

                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                img = np.array(Image.open(input_file).convert("RGB"))
                # img[img>100]=255
                img = transform(img).to(device)
                img = img[None, None, :].repeat(num_proposals, 1, 1, 1, 1)
                data["conditions"]["imgs"] = img
                data["conditions"]["img_id"] = torch.tensor([[0]], device=device).repeat(num_proposals, 1)
            else:
                print("Unknown condition")
                exit(1)

        if len(valid_hitted_fileitems) == 0:
            continue

        # run in batch
        with torch.no_grad():
            network_preds = model.inference(num_proposals * len(valid_hitted_fileitems), device, v_data=data, v_log=True)

        # unpack num_proposals network_preds for each fileitem
        for id_item, fileitem in enumerate(tqdm(valid_hitted_fileitems)):
            local_network_preds = network_preds[id_item * num_proposals:(id_item + 1) * num_proposals]
            for idx in range((len(local_network_preds))):
                name = os.path.basename(os.path.dirname(fileitem))
                prefix = f"{name}_{idx:02d}"
                (output_dir / "network_pred" / prefix).mkdir(parents=True, exist_ok=True)
                recon_data = local_network_preds[idx]
                export_edges(recon_data["pred_edge"], str(output_dir / "network_pred" / prefix / f"edge.obj"))
                np.savez_compressed(str(output_dir / "network_pred" / prefix / f"data.npz"),
                                    pred_face_adj_prob=recon_data["pred_face_adj_prob"],
                                    pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
                                    pred_face=recon_data["pred_face"],
                                    pred_edge=recon_data["pred_edge"],
                                    pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
                                    )
    te = time.time()
    with open(output_dir / "time1_gen.txt", "a") as f:
        f.write(f"time1_gen: {te - ts} s\n")

    # Start post-processing
    ts = time.time()
    ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8080,
            num_cpus=int(args.num_cpus),
    )
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)

    all_folders = os.listdir(output_dir / "network_pred")
    all_folders.sort()

    tasks = []
    # all_folders = [folder for folder in all_folders if not (output_dir / "after_post" / folder).exists()]
    print(f"Start post processing, folder num: {len(all_folders)}")
    for i in range(len(all_folders)):
        if (output_dir / "after_post" / all_folders[i]).exists():
            continue
        tasks.append(construct_brep_from_datanpz_ray.remote(
                output_dir / "network_pred", output_dir / "after_post",
                all_folders[i],
                v_drop_num=2,
                use_cuda=False, from_scratch=True,
                is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
        ))
    results = []
    for i in tqdm(range(len(all_folders))):
        try:
            results.append(ray.get(tasks[i]))
        except:
            results.append(None)

    te = time.time()
    with open(output_dir / "time2_post.txt", "a") as f:
        f.write(f"time2_post: {te - ts} s\n")

    # Start choose best post
    ts = time.time()
    if args.only_best:
        assert args.cond_root is not None
        cond_root = Path(args.cond_root)
        set_all_folders = set()
        for folder in all_folders:
            set_all_folders.add(folder.split("_")[0])
        all_folders = list(set_all_folders)

        print(f"\nStart choose best post, folder num: {len(all_folders)}\n")
        os.makedirs(output_dir / "best_post", exist_ok=True)

        choose_best_remote = ray.remote(choose_best)
        futures = []
        for folder in tqdm(all_folders):
            if (output_dir / "best_post" / folder).exists():
                continue
            futures.append(choose_best_remote.remote(cond_root, folder, output_dir, num_proposals))
        for future in tqdm(futures):
            ray.get(future)

    te = time.time()
    with open(output_dir / "time3_choose_best.txt", "a") as f:
        f.write(f"time3_choose_best: {te - ts} s\n")
    print("Done.")
