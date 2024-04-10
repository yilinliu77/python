import json, os
from itertools import combinations
from pathlib import Path
import open3d as o3d
import trimesh
import numpy as np
from tqdm import tqdm

data_root = Path("G:/Dataset/ABC/raw_data/abc_0000_obj_v00")

def sample_points(v_out_root):
    data_split = json.loads(open(r"train_val_test_split_deepcad.json").read())

    for item in tqdm(data_split["train"]):
        idx = item.split("/")[-1]

        if int(idx) > 10000:
            continue

        valid = False
        for ff in os.listdir(data_root / idx):
            if not ff.endswith(".obj"):
                continue

            mesh = trimesh.load_mesh(data_root / idx / ff, process=False, maintain_order=True)
            if mesh.faces.shape[0] < 10 or mesh.vertices.shape[0] < 10:
                continue

            # Get bounding box
            extent = mesh.bounding_box.extents
            diag = np.linalg.norm(extent)
            centroid = mesh.bounding_box.centroid

            mesh.vertices -= centroid
            mesh.vertices /= diag

            points = trimesh.sample.sample_surface(mesh, 4096)[0]
            np.savez_compressed(os.path.join(v_out_root, idx + ".npz"),
                                points=points)


def compute_pared_cd(v_sample_points_dir):
    data = os.listdir(v_sample_points_dir)
    num = len(data)
    id_pairs = list(combinations(range(num), 2))
    dist_matrix = np.zeros(len(id_pairs))
    for idx, (id1,id2) in enumerate(tqdm(id_pairs)):
        data1 = np.load(os.path.join(v_sample_points_dir, data[id1]))["points"]
        data2 = np.load(os.path.join(v_sample_points_dir, data[id2]))["points"]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(data2)
        dists = pcd1.compute_point_cloud_distance(pcd2)
        dist_matrix[idx] = np.asarray(dists).mean()

    np.save(os.path.join(v_sample_points_dir, "../dist_matrix"), dist_matrix)
    threshold = 5e-3
    total_ids = [item[:8] for item in data]
    invalid_ids = [item[:8] for item in np.asarray(data)[np.asarray(id_pairs)[(dist_matrix < threshold)]][:,1]]
    invalid_ids = list(set(invalid_ids))
    valid_ids = list(set(total_ids) - set(invalid_ids))
    valid_ids.sort()
    np.savetxt(os.path.join(v_sample_points_dir, "../valid_ids.txt"), valid_ids, fmt='%s')

if __name__ == '__main__':
    # sample_points(r"G:/Dataset/img2brep/deepcad_10000/sample_points")
    compute_pared_cd(r"G:/Dataset/img2brep/deepcad_10000/sample_points")
    pass
