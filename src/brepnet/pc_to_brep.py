"""End-to-end: point clouds -> diffusion inference -> rebuilt B-reps.

Multi-node + S3 aware. Designed for a ray cluster spanning several nodes that do
NOT share a filesystem: every worker pulls its own inputs and pushes its own
outputs (via S3), and cross-node coordination goes through ray. Single-node /
local paths still work unchanged.

Usage (local, single node):
    python -m src.brepnet.pc_to_brep \
        --diffusion_weights 0218_abc_pc_li_1450k.ckpt \
        --autoencoder_weights 1119_abc_aug1_11k.ckpt \
        --pc_dir inference_data/pc_test \
        --output_dir ./inference_output

Usage (multi-node, S3 exchange):
    # on each node: ray start --address=<head>   (head: ray start --head)
    python -m src.brepnet.pc_to_brep \
        --diffusion_weights  s3://bucket/ckpts/0218_abc_pc_li_1450k.ckpt \
        --autoencoder_weights s3://bucket/ckpts/1119_abc_aug1_11k.ckpt \
        --pc_dir   s3://bucket/pc_test \
        --output_dir s3://bucket/runs/out1 \
        --ray_address auto

For multi-node runs every path SHOULD be s3:// (local paths are only visible to
the node that holds them). The driver never assumes a shared filesystem.

Pipeline:
  1. Distributed inference: shard clouds across ray GPU workers (one per cluster
     GPU). Each worker downloads ckpts to node-local cache once, then for each
     assigned cloud: fetch -> diffuse (fp16) -> decode (fp32) -> write proposals
     -> publish to <output>/network_pred/<name>/<idx>/.
  2. Parallel post: one ray CPU task per proposal -> fetch data.npz -> rebuild
     B-rep (construct_brep, unchanged) -> publish to <output>/after_post/...
     Per-shape timeout counted from real worker start (ray actor registry).
  3. Valid solids copied to <output>/success_brep/<name>_00.step, _01.step, ...

Config defaults match the 0218_abc_pc_li checkpoint family:
  AutoEncoder_1119_light, PointNet2 pc encoder, linear beta schedule, epsilon
  prediction, random padding, num_max_faces=100, drop_num=1, fp32 decode.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import OCC (via the post module) BEFORE torch/open3d, or OCC's libtiff fails
# with an undefined-symbol error. Guarded so inference-only envs still work.
try:
    from src.brepnet.post.construct_brep import construct_brep_from_datanpz
    _POST_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001
    construct_brep_from_datanpz = None
    _POST_IMPORT_ERROR = _e

import numpy as np
import open3d as o3d
import torch
import ray
from tqdm import tqdm
from lightning_fabric import seed_everything

from src.brepnet.diffusion_model import Diffusion_condition

# per-node checkpoint cache (shared by all actors on a node; atomic download)
NODE_CKPT_CACHE = Path("/tmp/pc2brep_ckpt_cache")


# --------------------------------------------------------------------------- #
# S3 / local path helpers (uses the aws CLI; works without boto3)
# --------------------------------------------------------------------------- #
def is_s3(p):
    return str(p).startswith("s3://")


def _aws(*args):
    subprocess.run(["aws", "s3", *args], check=True)


def s3_cp(src, dst):
    _aws("cp", str(src), str(dst), "--quiet")


def s3_sync(src, dst):
    _aws("sync", str(src), str(dst), "--quiet")


def uri_join(base, *parts):
    if is_s3(base):
        return str(base).rstrip("/") + "/" + "/".join(parts)
    return str(Path(base, *parts))


def s3_list_ply(uri):
    uri = str(uri).rstrip("/") + "/"
    out = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True, check=True).stdout
    names = []
    for line in out.splitlines():
        toks = line.split()
        if toks and toks[-1].endswith(".ply"):
            names.append(toks[-1])
    return sorted(names)


def materialize_file(path, cache_dir):
    """Return a node-local path for a file that may live on S3 (atomic download,
    shared across actors on the same node)."""
    if not is_s3(path):
        return str(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(str(path)).name
    if local.exists():
        return str(local)
    tmp = cache_dir / f"{Path(str(path)).name}.tmp.{os.getpid()}"
    s3_cp(path, tmp)
    os.replace(tmp, local)
    return str(local)


def fetch_input(uri, work_dir):
    """Return a local path for an input that may be on S3."""
    if not is_s3(uri):
        return str(uri)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    lp = work_dir / Path(str(uri)).name
    s3_cp(uri, lp)
    return str(lp)


def publish_dir(local_dir, dest):
    """Push a local directory to dest (S3 sync or local copy-merge)."""
    if is_s3(dest):
        s3_sync(local_dir, dest)
    else:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, dest, dirs_exist_ok=True)


def copy_uri(src, dst):
    """Copy a single file; handles any s3/local combination."""
    if is_s3(src) or is_s3(dst):
        _aws("cp", str(src), str(dst), "--quiet")
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def list_input_uris(pc_dir, limit):
    if is_s3(pc_dir):
        names = s3_list_ply(pc_dir)
        if limit > 0:
            names = names[:limit]
        return [uri_join(pc_dir, n) for n in names]
    files = sorted(glob.glob(os.path.join(str(pc_dir), "*.ply")))
    if limit > 0:
        files = files[:limit]
    return files


def list_proposals(net_pred_base):
    """Return ['<name>/<idx>', ...] for proposals under <base>/network_pred."""
    if is_s3(net_pred_base):
        base = str(net_pred_base).rstrip("/") + "/"
        out = subprocess.run(["aws", "s3", "ls", base, "--recursive"],
                             capture_output=True, text=True, check=True).stdout
        props = set()
        for line in out.splitlines():
            key = line.split()[-1] if line.split() else ""
            if key.endswith("/data.npz"):
                parts = key.split("/")
                props.add(parts[-3] + "/" + parts[-2])
        return sorted(props)
    base = Path(net_pred_base)
    res = []
    if base.exists():
        for name in sorted(os.listdir(base)):
            nd = base / name
            if nd.is_dir():
                for idx in sorted(os.listdir(nd)):
                    if (nd / idx / "data.npz").exists():
                        res.append(f"{name}/{idx}")
    return res


# --------------------------------------------------------------------------- #
# model / inference helpers
# --------------------------------------------------------------------------- #
def export_edges(l_v, v_file):
    with open(v_file, "w") as f:
        line_str, num_points = "", 0
        for edge in l_v:
            for v in edge:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(0, edge.shape[0] - 1):
                line_str += f"l {i + num_points + 1} {i + num_points + 2}\n"
            num_points += edge.shape[0]
        f.write(line_str)


def build_conf(num_max_faces):
    return {
        "name": "Diffusion_condition", "train_decoder": False, "stored_z": False,
        "use_mean": True, "diffusion_latent": 768, "diffusion_type": "epsilon",
        "loss": "l2", "pad_method": "random", "num_max_faces": num_max_faces,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "variance_type": "fixed_small", "addition_tag": False,
        "autoencoder": "AutoEncoder_1119_light", "with_intersection": True,
        "dim_latent": 8, "dim_shape": 768, "sigmoid": False, "in_channels": 6,
        "gaussian_weights": 1e-6, "norm": "layer", "autoencoder_weights": "",
        "is_aug": False, "condition": ["pc"], "cond_prob": [], "pc_encoder": "pointnet2",
    }


def load_model(conf, diffusion_ckpt, autoencoder_ckpt, device):
    conf = dict(conf)
    conf["autoencoder_weights"] = autoencoder_ckpt
    model = Diffusion_condition(conf)
    w = torch.load(diffusion_ckpt, map_location=device, weights_only=False)["state_dict"]
    w = {k: v for k, v in w.items() if "ae_model" not in k}
    w = {k[6:]: v for k, v in w.items() if "model" in k}
    w = {k: v for k, v in w.items() if "camera_embedding" not in k}
    missing, unexpected = model.load_state_dict(w, strict=False)
    miss = [k for k in missing if not k.startswith("ae_model")]
    unexp = [k for k in unexpected if not k.startswith("ae_model")]
    print(f"[load] diffusion weights: missing(non-ae)={len(miss)} unexpected(non-ae)={len(unexp)}")
    model.to(device)
    model.eval()
    return model


def load_pc(input_file, num_sample, device, num_samples):
    pcd = o3d.io.read_point_cloud(str(input_file))
    if not pcd.has_normals():
        raise ValueError(f"Point cloud {input_file} has no normal vectors; "
                         f"the model requires per-point normals (6-channel input).")
    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    bbox_min, bbox_max = points.min(0), points.max(0)
    center = (bbox_min + bbox_max) / 2
    points = (points - center) / np.max(bbox_max - bbox_min) * 0.9 * 2
    points = np.concatenate([points, normals], axis=1)
    replace = num_sample > points.shape[0]
    idx = np.random.choice(points.shape[0], num_sample, replace=replace)
    pts = torch.tensor(points[idx], dtype=torch.float32, device=device)
    return pts[None, None].repeat(num_samples, 1, 1, 1)


def infer_one(model, f, out_root, num_samples, num_points, device, use_fp16=True):
    """Run inference for ONE point cloud, writing out_root/<name>/<idx>/."""
    out_root = Path(out_root)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=(use_fp16 and device.type == "cuda"))
    name = Path(f).stem
    data = {"conditions": {"points": load_pc(f, num_points, device, num_samples)}}
    with torch.no_grad(), autocast_ctx:
        preds = model.inference(num_samples, device, v_data=data, v_log=False)
    n_faces = 0
    for idx, recon in enumerate(preds):
        d = out_root / name / f"{idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        pf = recon["pred_face"].astype(np.float32)
        export_edges(recon["pred_edge"], str(d / "edge.obj"))
        np.savez_compressed(str(d / "data.npz"),
                            pred_face_adj_prob=recon["pred_face_adj_prob"],
                            pred_face_adj=recon["pred_face_adj"].cpu().numpy(),
                            pred_face=pf,
                            pred_edge=recon["pred_edge"],
                            pred_edge_face_connectivity=recon["pred_edge_face_connectivity"])
        o3d.io.write_point_cloud(str(d / "pred_face_points.ply"),
                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pf.reshape(-1, 3))))
        if idx == 0:
            n_faces = int(pf.shape[0])
    return name, n_faces


# --------------------------------------------------------------------------- #
# stage 1: distributed inference (ray GPU workers, self-contained I/O)
# --------------------------------------------------------------------------- #
@ray.remote
class InferenceActor:
    def __init__(self, conf, diffusion_uri, ae_uri, ae_dropout, use_fp16):
        import torch as _torch
        self.use_fp16 = use_fp16
        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        dif = materialize_file(diffusion_uri, NODE_CKPT_CACHE)
        ae = materialize_file(ae_uri, NODE_CKPT_CACHE)
        self.model = load_model(conf, dif, ae, self.device)
        if ae_dropout:
            self.model.ae_model.train()  # force AE dropout on during decode

    def infer(self, input_uris, out_base, num_samples, num_points):
        work = Path(tempfile.mkdtemp(prefix="pc2brep_infer_"))
        net_local = work / "network_pred"
        stats = []
        try:
            for uri in input_uris:
                local = fetch_input(uri, work / "in")
                name, nf = infer_one(self.model, local, net_local, num_samples,
                                     num_points, self.device, self.use_fp16)
                # publish this cloud's proposals, then free node-local disk
                publish_dir(str(net_local / name), uri_join(out_base, "network_pred", name))
                shutil.rmtree(net_local / name, ignore_errors=True)
                if is_s3(uri):
                    try:
                        os.remove(local)
                    except OSError:
                        pass
                stats.append((name, nf))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return stats


def run_inference_distributed(args, input_uris, out_base, conf):
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cluster_gpus)
    num_workers = min(num_workers, len(input_uris)) or 1
    gpus_per = 1 if cluster_gpus > 0 else 0
    print(f"[infer] {len(input_uris)} clouds across {num_workers} worker(s) "
          f"(cluster GPUs={cluster_gpus}), num_samples={args.num_samples}, "
          f"num_points={args.num_points}, ae_dropout={args.ae_dropout}, "
          f"precision=fp16-diffusion/fp32-decode")

    shards = [input_uris[i::num_workers] for i in range(num_workers)]
    shards = [s for s in shards if s]
    actors = [InferenceActor.options(num_gpus=gpus_per).remote(
                  conf, args.diffusion_weights, args.autoencoder_weights, args.ae_dropout, True)
              for _ in shards]
    futs = [a.infer.remote(s, out_base, args.num_samples, args.num_points)
            for a, s in zip(actors, shards)]
    results = ray.get(futs)
    for a in actors:
        ray.kill(a)

    nfs = [n for r in results for _, n in r]
    if nfs:
        print(f"[infer] done: {len(nfs)} clouds, faces/shape min={min(nfs)} "
              f"max={max(nfs)} mean={sum(nfs)/len(nfs):.1f}")


# --------------------------------------------------------------------------- #
# stage 2: parallel post (ray CPU tasks, self-contained I/O)
# --------------------------------------------------------------------------- #
@ray.remote
class StartRegistry:
    """Cross-node record of when each task actually began running."""
    def __init__(self):
        self.t = {}

    def mark(self, key, ts):
        self.t[key] = ts

    def snapshot(self):
        return dict(self.t)


def _post_task(proposal, net_pred_base, after_post_base, drop_num, registry):
    # First action: report real start time to the (cross-node) registry.
    registry.mark.remote(proposal, time.time())
    name, idx = proposal.split("/")
    work = Path(tempfile.mkdtemp(prefix="pc2brep_post_"))
    try:
        in_dir = work / "in" / name / idx
        in_dir.mkdir(parents=True, exist_ok=True)
        copy_uri(uri_join(net_pred_base, name, idx, "data.npz"), str(in_dir / "data.npz"))
        out_local = work / "out"
        construct_brep_from_datanpz(str(work / "in"), str(out_local), f"{name}/{idx}",
                                    v_drop_num=drop_num, use_cuda=False, from_scratch=True,
                                    is_log=False, is_ray=True, is_optimize_geom=True, isdebug=False)
        res_dir = out_local / name / idx
        success = (res_dir / "success.txt").exists()
        if res_dir.exists():
            publish_dir(str(res_dir), uri_join(after_post_base, name, idx))
        return (name, idx, bool(success))
    except Exception:
        return (name, idx, False)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_post(args, output_dir):
    if construct_brep_from_datanpz is None:
        raise RuntimeError(f"post deps unavailable: {_POST_IMPORT_ERROR}")
    from collections import defaultdict

    net_pred_base = uri_join(output_dir, "network_pred")
    after_post_base = uri_join(output_dir, "after_post")
    success_base = uri_join(output_dir, "success_brep")

    folders = list_proposals(net_pred_base)
    n_cpus = int(ray.cluster_resources().get("CPU", os.cpu_count() or 1))
    if args.num_cpus and args.num_cpus > 0:
        n_cpus = args.num_cpus
    print(f"[post] {len(folders)} proposals, ray cpus={n_cpus}, "
          f"timeout={args.timeout}s (from real worker start)")

    registry = StartRegistry.remote()
    remote_fn = ray.remote(num_gpus=0, max_retries=0)(_post_task)

    queue = list(folders)
    inflight = {}   # ref -> {"proposal", "deadline"(None until running)}
    succ = defaultdict(list)
    timed_out = 0
    pbar = tqdm(total=len(folders), desc="post")
    while queue or inflight:
        while queue and len(inflight) < n_cpus:
            p = queue.pop(0)
            t = remote_fn.remote(p, net_pred_base, after_post_base, args.drop_num, registry)
            inflight[t] = {"proposal": p, "deadline": None}
        ready, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=2.0)
        now = time.time()
        for t in ready:
            try:
                name, idx, ok = ray.get(t)
                if ok:
                    succ[name].append(idx)
            except Exception:
                pass
            inflight.pop(t, None)
            pbar.update(1)
        started = ray.get(registry.snapshot.remote()) if inflight else {}
        for t in list(inflight):
            info = inflight[t]
            if info["deadline"] is None:
                if info["proposal"] in started:
                    info["deadline"] = started[info["proposal"]] + args.timeout
                continue
            if now > info["deadline"]:
                ray.cancel(t, force=True)
                inflight.pop(t, None)
                timed_out += 1
                pbar.update(1)
    pbar.close()

    # Transfer valid solids -> success_brep/<name>_NN.step (sequential per shape)
    transferred = 0
    for name in sorted(succ):
        for seq, idx in enumerate(sorted(succ[name])):
            copy_uri(uri_join(after_post_base, name, idx, "recon_brep.step"),
                     uri_join(success_base, f"{name}_{seq:02d}.step"))
            transferred += 1

    inputs = sorted(set(f.split("/")[0] for f in folders))
    print("\n=== post-processing summary ===")
    print(f"proposals processed         : {len(folders)}")
    print(f"valid solids (success.txt)  : {sum(len(v) for v in succ.values())}")
    print(f"timed out (>{args.timeout}s)        : {timed_out}")
    print(f"inputs with >=1 valid solid : {len(succ)}/{len(inputs)}")
    print(f"[post] {transferred} STEP files -> {success_base}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Point clouds -> diffusion -> B-rep (ray multi-node, S3-aware)")
    ap.add_argument('--diffusion_weights', help="local path or s3:// (required unless --only_post)")
    ap.add_argument('--autoencoder_weights', help="local path or s3:// (required unless --only_post)")
    ap.add_argument('--pc_dir', help="dir of .ply clouds; local or s3:// (required unless --only_post)")
    ap.add_argument('--output_dir', default="./inference_output", help="local path or s3:// URI")
    ap.add_argument('--num_samples', type=int, default=16, help="diffusion samples per cloud")
    ap.add_argument('--num_max_faces', type=int, default=100)
    ap.add_argument('--num_points', type=int, default=8192)
    ap.add_argument('--limit', type=int, default=0, help="limit number of inputs (0=all)")
    ap.add_argument('--ae_dropout', action='store_true', help="force AE dropout ON during decode")
    ap.add_argument('--num_workers', type=int, default=0, help="inference workers (0 = one per cluster GPU)")
    ap.add_argument('--ray_address', default=None, help="ray cluster address (e.g. 'auto'); default: local/RAY_ADDRESS")
    ap.add_argument('--seed', type=int, default=0)
    # post
    ap.add_argument('--skip_post', action='store_true', help="inference only")
    ap.add_argument('--only_post', action='store_true', help="post only (predictions must exist under output_dir)")
    ap.add_argument('--num_cpus', type=int, default=-1, help="post concurrency (-1 = all cluster cpus)")
    ap.add_argument('--drop_num', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=600, help="per-shape post timeout (s), running-time only")
    args = ap.parse_args()

    seed_everything(args.seed)

    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=False)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    if args.only_post:
        run_post(args, args.output_dir)
        return

    if not (args.diffusion_weights and args.autoencoder_weights and args.pc_dir):
        ap.error("--diffusion_weights, --autoencoder_weights and --pc_dir are required unless --only_post")

    input_uris = list_input_uris(args.pc_dir, args.limit)
    print(f"[data] {len(input_uris)} input point clouds; output -> {args.output_dir}")
    conf = build_conf(args.num_max_faces)
    run_inference_distributed(args, input_uris, args.output_dir, conf)
    if args.skip_post:
        print("[skip_post] stopping after network predictions (UV grids).")
        return
    run_post(args, args.output_dir)


if __name__ == '__main__':
    main()
