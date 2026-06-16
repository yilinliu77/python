"""Point-cloud -> diffusion inference under the org Ray+Lightning architecture.

Vanilla adsk_raylab.RayLightningExperiment, no hydra/config files -- plain
argparse. A LightningDataModule streams point clouds; a LightningModule wraps
the brepnet Diffusion_condition model, loads its own checkpoint in __init__,
and writes predictions to S3 in test_step.

Only GPU inference runs here. B-rep post-processing stays separate:
    python -m src.brepnet.pc_to_brep --only_post --output_dir <parent-of-network_pred>

Example (ray cluster):
    python -m src.brepnet.infer_ray_lightning \
        --ckpt s3://bucket/ckpts/0218_abc_pc_li_1450k.ckpt \
        --pc_dir s3://bucket/pc_test \
        --output_dir s3://bucket/out1/network_pred \
        --s3_results_uri s3://bucket/ray_results/pc_infer \
        --gpu 8 --worker_node_type p5.48xlarge --worker_node_priority background

Local (no ray cluster / no adsk_raylab):
    python -m src.brepnet.infer_ray_lightning --no_ray \
        --ckpt 0218_abc_pc_li_1450k.ckpt --pc_dir inference_data/pc_test \
        --output_dir ./out1/network_pred --limit 5

The diffusion checkpoint embeds the matching ae_model, so loading the whole
state_dict gives the correct autoencoder -- no separate AE file is needed.
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from lightning.pytorch import Trainer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.brepnet.diffusion_model import Diffusion_condition

try:
    from adsk_raylab.ray_lightning import RayLightningExperiment
except Exception:  # noqa: BLE001
    RayLightningExperiment = None
try:
    from cloudpathlib import S3Path
except Exception:  # noqa: BLE001
    S3Path = None


def is_s3(p):
    return str(p).startswith("s3://")


def seed_worker(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


def _download(uri, dst_dir):
    """Download an s3 file to dst_dir, return local path."""
    if S3Path is not None:
        p = S3Path(uri)
        local = Path(dst_dir) / p.name
        p.download_to(str(local))
        return str(local)
    import subprocess
    local = Path(dst_dir) / Path(uri).name
    subprocess.run(["aws", "s3", "cp", uri, str(local), "--quiet"], check=True)
    return str(local)


def build_conf(num_max_faces):
    return {
        "name": "Diffusion_condition", "train_decoder": False, "stored_z": False,
        "use_mean": True, "diffusion_latent": 768, "diffusion_type": "epsilon",
        "loss": "l2", "pad_method": "random", "num_max_faces": num_max_faces,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "variance_type": "fixed_small", "addition_tag": False,
        "autoencoder": "AutoEncoder_1119_light", "with_intersection": True,
        "dim_latent": 8, "dim_shape": 768, "sigmoid": False, "in_channels": 6,
        "gaussian_weights": 1e-6, "norm": "layer", "autoencoder_weights": None,
        "is_aug": False, "condition": ["pc"], "cond_prob": [], "pc_encoder": "pointnet2",
    }


def export_edges(l_v, v_file):
    with open(v_file, "w") as f:
        line_str, n = "", 0
        for edge in l_v:
            for v in edge:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(edge.shape[0] - 1):
                line_str += f"l {i + n + 1} {i + n + 2}\n"
            n += edge.shape[0]
        f.write(line_str)


# --------------------------------------------------------------------------- #
class PCInferDataset(Dataset):
    def __init__(self, pc_dir, num_points, test_list=None, limit=0):
        self.num_points = num_points
        self.is_s3 = is_s3(pc_dir)
        PPath = S3Path if self.is_s3 else Path
        root = PPath(pc_dir)
        if test_list:
            with tempfile.TemporaryDirectory() as td:
                lf = _download(test_list, td) if is_s3(test_list) else test_list
                names = [x.strip() for x in open(lf).readlines() if x.strip()]
            names = [n if n.endswith(".ply") else n + ".ply" for n in names]
            self.files = [root / n for n in names]
        else:
            self.files = sorted(p for p in root.iterdir() if str(p).endswith(".ply"))
        if limit > 0:
            self.files = self.files[:limit]
        print(f"PCInferDataset: {len(self.files)} point clouds")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        name = Path(str(f)).stem
        if self.is_s3:
            with tempfile.TemporaryDirectory() as td:
                local = Path(td) / Path(str(f)).name
                f.download_to(str(local))
                pcd = o3d.io.read_point_cloud(str(local))
        else:
            pcd = o3d.io.read_point_cloud(str(f))
        if not pcd.has_normals():
            raise ValueError(f"{f} has no normals (model needs 6-channel input).")
        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals)
        lo, hi = pts.min(0), pts.max(0)
        pts = (pts - (lo + hi) / 2) / np.max(hi - lo) * 0.9 * 2
        pc = np.concatenate([pts, nrm], axis=1)
        replace = self.num_points > pc.shape[0]
        sel = np.random.choice(pc.shape[0], self.num_points, replace=replace)
        return {"points": torch.from_numpy(pc[sel].astype(np.float32)), "prefix": name}


class PCInferDataModule(pl.LightningDataModule):
    def __init__(self, pc_dir, num_points=8192, test_list=None, limit=0,
                 batch_size=1, num_worker=4):
        super().__init__()
        self.pc_dir, self.num_points, self.test_list, self.limit = pc_dir, num_points, test_list, limit
        self.batch_size, self.num_worker = batch_size, num_worker

    def test_dataloader(self):
        ds = PCInferDataset(self.pc_dir, self.num_points, self.test_list, self.limit)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_worker,
                          shuffle=False, pin_memory=False, worker_init_fn=seed_worker)


# --------------------------------------------------------------------------- #
class PCInferModule(pl.LightningModule):
    def __init__(self, ckpt=None, num_samples=16, num_max_faces=100,
                 ae_dropout=False, output_dir=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_samples = num_samples
        self.ae_dropout = ae_dropout
        self.model = Diffusion_condition(build_conf(num_max_faces))
        if ckpt:
            path = ckpt
            if is_s3(ckpt):
                path = _download(ckpt, tempfile.mkdtemp())
            sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            print(self.load_state_dict(sd, strict=False))  # keys are model.* -> aligns
        if output_dir is None:
            self.test_root = None
        elif is_s3(output_dir):
            self.test_root = S3Path(output_dir)
        else:
            self.test_root = Path(output_dir)
            self.test_root.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def setup(self, stage=None):
        self.model.eval()
        if self.ae_dropout:
            self.model.ae_model.train()

    def _save(self, root, recon):
        root.mkdir(parents=True, exist_ok=True)
        pf = recon["pred_face"].astype(np.float32)
        export_edges(recon["pred_edge"], str(root / "edge.obj"))
        np.savez_compressed(str(root / "data.npz"),
                            pred_face_adj_prob=recon["pred_face_adj_prob"],
                            pred_face_adj=recon["pred_face_adj"].cpu().numpy(),
                            pred_face=pf, pred_edge=recon["pred_edge"],
                            pred_edge_face_connectivity=recon["pred_edge_face_connectivity"])
        o3d.io.write_point_cloud(str(root / "pred_face_points.ply"),
                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pf.reshape(-1, 3))))

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        points, prefixes = batch["points"], batch["prefix"]
        ac = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device.type == "cuda"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for b in range(points.shape[0]):
                pc = points[b].to(self.device)[None, None].repeat(self.num_samples, 1, 1, 1)
                with ac:
                    preds = self.model.inference(self.num_samples, self.device,
                                                 v_data={"conditions": {"points": pc}}, v_log=False)
                for idx, recon in enumerate(preds):
                    self._save(tmp / prefixes[b] / f"{idx:02d}", recon)
            if self.test_root is not None:
                if S3Path is not None and isinstance(self.test_root, S3Path):
                    self.test_root.upload_from(str(tmp), force_overwrite_to_cloud=True)
                else:
                    shutil.copytree(str(tmp), str(self.test_root), dirs_exist_ok=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("pc -> diffusion inference (Ray+Lightning)")
    ap.add_argument("--ckpt", required=True, help="diffusion checkpoint (local or s3://)")
    ap.add_argument("--pc_dir", required=True, help="dir of .ply clouds (local or s3://)")
    ap.add_argument("--output_dir", required=True, help="network_pred output dir (local or s3://)")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--num_points", type=int, default=8192)
    ap.add_argument("--num_max_faces", type=int, default=100)
    ap.add_argument("--ae_dropout", action="store_true")
    ap.add_argument("--test_list", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    # ray
    ap.add_argument("--gpu", type=int, default=1, help="num ray workers (GPUs)")
    ap.add_argument("--num_cpus", type=int, default=4, help="cpus per worker")
    ap.add_argument("--worker_node_type", default="p5.48xlarge")
    ap.add_argument("--worker_node_priority", default="background")
    ap.add_argument("--exp_name", default="pc_infer")
    ap.add_argument("--s3_results_uri", default="s3://autodesk-adpcdl-965535024567-p-ue1-internal-brepfaceted/private/yl-voronoi/training_logs")
    ap.add_argument("--max_failures", type=int, default=50)
    ap.add_argument("--no_ray", action="store_true", help="run a local Trainer instead of ray")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    model_kwargs = dict(ckpt=args.ckpt, num_samples=args.num_samples,
                        num_max_faces=args.num_max_faces, ae_dropout=args.ae_dropout,
                        output_dir=args.output_dir)
    data_kwargs = dict(pc_dir=args.pc_dir, num_points=args.num_points, test_list=args.test_list,
                       limit=args.limit, batch_size=args.batch_size, num_worker=args.num_cpus)
    trainer_kwargs = {"accelerator": "auto", "precision": args.precision, "limit_test_batches": 1.0}

    if args.no_ray or RayLightningExperiment is None:
        if RayLightningExperiment is None and not args.no_ray:
            print("[warn] adsk_raylab unavailable; running local Trainer().test()")
        model = PCInferModule(**model_kwargs)
        dm = PCInferDataModule(**data_kwargs)
        Trainer(**trainer_kwargs).test(model, dm)
    else:
        RayLightningExperiment(
            exp_name=args.exp_name,
            model_class=PCInferModule, model_class_kwargs=model_kwargs,
            data_class=PCInferDataModule, data_class_kwargs=data_kwargs,
            trainer_kwargs=trainer_kwargs,
            num_workers=args.gpu, num_cpus_per_worker=args.num_cpus, use_gpu=True,
            worker_node_type=args.worker_node_type,
            worker_node_priority=args.worker_node_priority,
            s3_results_uri=args.s3_results_uri,
            max_failures=args.max_failures,
        ).test()


if __name__ == "__main__":
    main()
