import glob
import os
import shutil
import torch

from tqdm import tqdm
from src.brepnet.eval.check_valid import check_step_valid_soild, load_data_with_prefix
from src.brepnet.post.utils import *
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_SOLID

from src.brepnet.eval.eval_brepgen import *

from chamferdist import ChamferDistance

from src.brepnet.viz.sort_and_merge import arrange_meshes

from torch.nn.utils.rnn import pad_sequence


def load_data_from_npz(data_npz_file):
    data_npz = np.load(data_npz_file, allow_pickle=True)
    # Brepgen
    if 'face_edge_adj' in data_npz:
        faces = data_npz['pred_face']
        face_edge_adj = data_npz['face_edge_adj']
        faces_adj_pair = []
        N = face_edge_adj.shape[0]
        for face_idx1 in range(N):
            for face_idx2 in range(face_idx1 + 1, N):
                face_edges1 = face_edge_adj[face_idx1]
                face_edges2 = face_edge_adj[face_idx2]
                if sorted((face_idx1, face_idx2)) in faces_adj_pair:
                    continue
                if len(set(face_edges1).intersection(set(face_edges2))) > 0:
                    faces_adj_pair.append(sorted((face_idx1, face_idx2)))
        return faces, faces_adj_pair
    # Ours
    if 'sample_points_faces' in data_npz:
        face_points = data_npz['sample_points_faces']  # Face sample points (num_faces*20*20*3)
    elif 'pred_face' in data_npz and 'pred_edge_face_connectivity' in data_npz:
        face_points = data_npz['pred_face']
    else:
        raise ValueError("Invalid data format")

    return face_points[..., :3]


def batch_compute_cf_dis(src_pcs, ref_pcs_list, batch_size=50000, pad_value=10):
    ref_pcs_shape_list = torch.tensor([ref_pcs.shape[0] for ref_pcs in ref_pcs_list]).to(torch.int).to(src_pcs.device)
    unique_ref_pcs_shape_list = torch.unique(ref_pcs_shape_list).tolist()
    cf_src_and_ref = []
    for unique_ref_pcs_shape in unique_ref_pcs_shape_list:
        hits_idx = torch.where(ref_pcs_shape_list == unique_ref_pcs_shape)[0]
        local_ref_pcs = torch.stack(ref_pcs_list[hits_idx[0]:hits_idx[-1] + 1]).reshape(hits_idx.shape[0], -1, 3)
        local_src_pcs = src_pcs.reshape(1, -1, 3).repeat(local_ref_pcs.shape[0], 1, 1)
        local_cf_src_and_ref = chamferdist(local_src_pcs, local_ref_pcs, batch_reduction=None, point_reduction='mean', bidirectional=True)
        cf_src_and_ref.append(local_cf_src_and_ref)
    # for i in range(0, len(ref_pcs_list), batch_size):
    #     local_ref_pcs = pad_sequence(ref_pcs_list[i:i + batch_size], batch_first=True, padding_value=pad_value)
    #     local_ref_pcs = local_ref_pcs.reshape(local_ref_pcs.shape[0], -1, 3)
    #     local_src_pcs = src_pcs.reshape(1, -1, 3).repeat(local_ref_pcs.shape[0], 1, 1)
    #     local_cf_src_and_ref = chamferdist(local_src_pcs, local_ref_pcs, batch_reduction=None, point_reduction='mean', bidirectional=True)
    #     cf_src_and_ref.append(local_cf_src_and_ref)
    return torch.cat(cf_src_and_ref, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    parser.add_argument("--txt", type=str, required=False, default="")
    args = parser.parse_args()
    fake_root = args.fake_root
    train_root = args.train_root
    device = torch.device("cuda")

    if not os.path.exists(fake_root) or not os.path.exists(train_root):
        raise ValueError("Invalid path")

    print("\nLoading reference point clouds...")
    # Load reference pcd
    num_cpus = multiprocessing.cpu_count()
    if args.txt:
        with open(args.txt, "r") as f:
            ref_folders = [line.strip() for line in f.readlines()]
        ref_npz_paths = [os.path.join(train_root, folder, "data.npz") for folder in ref_folders]
    else:
        ref_npz_paths = load_data_with_prefix(train_root, 'data.npz')

    assert len(ref_npz_paths) > 0
    ref_npz_paths.sort()
    load_iter = multiprocessing.Pool(num_cpus).imap(load_data_from_npz, ref_npz_paths)
    ref_pcs = []
    for pc in tqdm(load_iter, total=len(ref_npz_paths)):
        if len(pc) > 0:
            pc = torch.from_numpy(pc).to(torch.float32).to(device)
            ref_pcs.append(pc)
    print("real point clouds: {}".format(len(ref_pcs)))
    sorted_indices = torch.from_numpy(np.argsort([x.shape[0] for x in ref_pcs])).to(torch.int).to(device)
    ref_pcs = [ref_pcs[i] for i in sorted_indices]
    ref_npz_paths = [ref_npz_paths[i] for i in sorted_indices]

    # Load fake pcd
    print("\nLoading fake point clouds...")
    src_pcs = []
    if args.prefix:
        src_npz_paths = [os.path.join(fake_root, args.prefix, "data.npz")]
    else:
        src_npz_paths = load_data_with_prefix(fake_root, 'data.npz')

    assert len(src_npz_paths) > 0
    src_npz_paths.sort()
    load_iter = multiprocessing.Pool(num_cpus).imap(load_data_from_npz, src_npz_paths)
    for pc in tqdm(load_iter, total=len(src_npz_paths)):
        if len(pc) > 0:
            pc = torch.from_numpy(pc).to(torch.float32).to(device)
            src_pcs.append(pc)
    print("fake point clouds: {}".format(len(src_pcs)))

    print("\nFinding nearest...")
    chamferdist = ChamferDistance()
    batch_size = 10
    nearest_list = []
    for i in tqdm(range(len(src_pcs))):
        local_out_root = os.path.join(fake_root, os.path.basename(os.path.dirname(src_npz_paths[i])))
        cf_src_and_ref = batch_compute_cf_dis(src_pcs[i], ref_pcs)
        topk_near_idx = torch.topk(-cf_src_and_ref, 10).indices.tolist()
        with open(os.path.join(local_out_root, "nearest.txt"), "w") as f:
            f.write("Source: {}\n".format(os.path.basename(local_out_root)))
            f.write("Top10 Nearest:\n")
            for ref_idx in topk_near_idx:
                folder_name = os.path.basename(os.path.dirname(ref_npz_paths[ref_idx]))
                f.write(f"{folder_name} {cf_src_and_ref[ref_idx]}\n")
        near_mesh_paths = [os.path.join(os.path.dirname(ref_npz_paths[ref_idx]), "mesh.ply") for ref_idx in topk_near_idx]
        src_mesh_paths = glob.glob(os.path.join(local_out_root, "*.stl"))
        mesh_paths = src_mesh_paths + near_mesh_paths
        arrange_meshes(mesh_paths, os.path.join(local_out_root, "nearest.ply"))
