import argparse
import sys
import os.path
from pathlib import Path
import numpy as np
import open3d as o3d
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

if __name__ == '__main__':
    conf = {
        "name": "Diffusion_condition",
        "train_decoder": False,
        "stored_z": False,
        "use_mean": True,
        "diffusion_latent": 768,
        "diffusion_type": "epsilon",
        "loss": "l2",
        "pad_method": "random",
        "num_max_faces": 30,
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "variance_type": "fixed_small",
        "addition_tag": False,
        "autoencoder": "AutoEncoder_1119_light",
        "with_intersection": True,
        "dim_latent": 8,
        "dim_shape": 768,
        "sigmoid": False,
        "in_channels": 6,
        "gaussian_weights": 1e-6,
        "norm": "layer",
        "autoencoder_weights": "",
        "is_aug": False,
        "condition": [],
        "cond_prob": []
    }

    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--autoencoder_weights', type=str, required=True)
    parser.add_argument('--diffusion_weights', type=str, required=True)
    parser.add_argument('--condition', nargs='+', required=True)
    parser.add_argument('--input', nargs='+', type=str)
    parser.add_argument('--output_dir', type=str, default="./inference_output")

    args = parser.parse_args()
    conf["autoencoder_weights"] = args.autoencoder_weights
    conf["diffusion_weights"] = args.diffusion_weights
    conf["condition"] = args.condition

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    model = Diffusion_condition(conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_weights = torch.load(conf["diffusion_weights"], map_location=device, weights_only=False)["state_dict"]
    diffusion_weights = {k: v for k, v in diffusion_weights.items() if "ae_model" not in k}
    diffusion_weights = {k[6:]: v for k, v in diffusion_weights.items() if "model" in k}
    autoencoder_weights = torch.load(conf["autoencoder_weights"], map_location=device, weights_only=False)["state_dict"]
    autoencoder_weights = {k[6:]: v for k, v in autoencoder_weights.items() if "model" in k}
    autoencoder_weights = {"ae_model."+k: v for k, v in autoencoder_weights.items()}
    diffusion_weights.update(autoencoder_weights)
    diffusion_weights = {k: v for k, v in diffusion_weights.items() if "camera_embedding" not in k}
    model.load_state_dict(diffusion_weights, strict=False)
    model.to(device)
    model.eval()

    print("We have {} data".format(len(args.input)))
    output_dir = Path(args.output_dir)
    (output_dir / "network_pred").mkdir(parents=True, exist_ok=True)

    for id_item, fileitem in enumerate(tqdm(args.input)):
        data = {
            "conditions": {
            }
        }
        num_proposals = 32
        if "pc" in conf["condition"]:
            input_file = Path(fileitem)
            name = input_file.stem
            if not input_file.exists():
                print(f"File {input_file} not found.")
                exit(1)

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
            num_sample = 8192
            index = np.random.choice(points.shape[0], num_sample, replace=False)
            points = points[index]
            points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
            data["conditions"]["points"] = points_tensor[None, None, :, :].repeat(num_proposals, 1, 1, 1)
        elif "txt" in conf["condition"]:
            data["conditions"]["txt"] = [fileitem for item in range(num_proposals)]
            name = f"{id_item:02d}"
        elif "sketch" in conf["condition"]:
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
            img[img>100]=255
            img = transform(img).to(device)
            img = img[None, None, :].repeat(num_proposals, 1, 1, 1, 1)
            data["conditions"]["imgs"] = img
            data["conditions"]["img_id"] = torch.tensor([[0]], device=device).repeat(num_proposals, 1)
        else:
            print("Unknown condition")
            exit(1)

        with torch.no_grad():
            network_preds = model.inference(num_proposals, device, v_data=data, v_log=True)

        for idx in range((len(network_preds))):
            prefix = f"{name}_{idx:02d}"
            (output_dir/"network_pred"/prefix).mkdir(parents=True, exist_ok=True)
            recon_data = network_preds[idx]
            export_edges(recon_data["pred_edge"], str(output_dir / "network_pred" / prefix / f"edge.obj"))
            np.savez_compressed(str(output_dir / "network_pred" / prefix / f"data.npz"),
                                pred_face_adj_prob=recon_data["pred_face_adj_prob"],
                                pred_face_adj=recon_data["pred_face_adj"].cpu().numpy(),
                                pred_face=recon_data["pred_face"],
                                pred_edge=recon_data["pred_edge"],
                                pred_edge_face_connectivity=recon_data["pred_edge_face_connectivity"],
                                )

    print("Start post processing")
    num_cpus = 8
    ray.init(
        dashboard_host="0.0.0.0",
        dashboard_port=8080,
        num_cpus=num_cpus,
    )
    construct_brep_from_datanpz_ray = ray.remote(num_cpus=1, max_retries=0)(construct_brep_from_datanpz)

    all_folders = os.listdir(output_dir / "network_pred")
    all_folders.sort()

    tasks = []
    for i in range(len(all_folders)):
        tasks.append(construct_brep_from_datanpz_ray.remote(
            output_dir / "network_pred", output_dir/"after_post",
            all_folders[i],
            v_drop_num=3,
            use_cuda=False, from_scratch=True,
            is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False,
        ))
    results = []
    for i in tqdm(range(len(all_folders))):
        try:
            results.append(ray.get(tasks[i]))
        except:
            results.append(None)
    print("Done.")
