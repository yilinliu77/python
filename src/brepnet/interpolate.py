import numpy as np
import torch
import trimesh
from lightning_fabric import seed_everything
from tqdm import tqdm

from src.brepnet.dataset import normalize_coord1112
from src.brepnet.diffusion_model import Diffusion_condition

seed_everything(0)

ckpt_path = r"D:/brepnet/uncond_700k.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
conf = {
    "name": "Diffusion_condition",
    "train_decoder": False,
    "stored_z": True,
    "condition": None,
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
    "autoencoder_weights": None,
}

model = Diffusion_condition(conf)
weight = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
weight = {k[6:]: v for k, v in weight.items()}
model.load_state_dict(weight)
model.eval()
model.to(device)

data_path1 = r"D:/brepnet/deepcad_v6/00000272/data.npz"
data_path2 = r"D:/brepnet/deepcad_v6/00000392/data.npz"


def read_data(data_path):
    data_npz = np.load(data_path)
    face_adj = torch.from_numpy(data_npz['face_adj']).to(device)
    edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity']).to(device)
    edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]

    face_points = torch.from_numpy(data_npz['sample_points_faces']).to(dtype).to(device)
    edge_points = torch.from_numpy(data_npz['sample_points_lines']).to(dtype).to(device)

    face_points_norm, face_normal_norm, face_center, face_scale = normalize_coord1112(face_points)
    edge_points_norm, edge_normal_norm, edge_center, edge_scale = normalize_coord1112(edge_points)

    face_norm = torch.cat((face_points_norm, face_normal_norm), dim=-1)
    edge_norm = torch.cat((edge_points_norm, edge_normal_norm), dim=-1)

    face_bbox = torch.cat((face_center, face_scale), dim=-1)
    edge_bbox = torch.cat((edge_center, edge_scale), dim=-1)

    return {
        "face_adj": face_adj,
        "edge_face_connectivity": edge_face_connectivity,
        "face_points": face_points,
        "edge_points": edge_points,
        "face_norm": face_norm,
        "edge_norm": edge_norm,
        "face_bbox": face_bbox,
        "edge_bbox": edge_bbox,
        "attn_mask": torch.zeros((face_points.shape[0], face_points.shape[0]), dtype=torch.bool, device=device),
        "num_face_record": torch.tensor([face_points.shape[0]], dtype=torch.int32, device=device),
    }


with torch.no_grad():
    feature1 = model.ae_model.encode(read_data(data_path1), v_test=True)["face_features"]
    latent1, kl_loss1 = model.ae_model.sample(feature1, False)
    feature2 = model.ae_model.encode(read_data(data_path2), v_test=True)["face_features"]
    latent2, kl_loss2 = model.ae_model.sample(feature2, False)

    mix_latent = torch.cat((
        latent1,
        latent2, latent2, latent2, latent2, latent2, latent2, latent2, latent2,
        latent2, latent2, latent2, latent2, latent2, latent2, latent2, latent2,
        latent2, latent2,
    ), dim=0)

    index = torch.randperm(mix_latent.shape[0])
    mix_latent = mix_latent[index][None, :].repeat(100, 1, 1)

    num_step = 300
    noise = torch.randn_like(mix_latent)
    noisy_mix_latent = model.noise_scheduler.add_noise(mix_latent, noise,
                                                       torch.tensor(num_step, device=device, dtype=torch.int32))

    face_features = noisy_mix_latent
    for t in tqdm(torch.arange(num_step - 1, -1, -1)):
        timesteps = t.reshape(-1).to(device)
        pred_x0 = model.diffuse(face_features, timesteps, v_condition=None)
        face_features = model.noise_scheduler.step(pred_x0, t, face_features).prev_sample

    pred_faces = []
    for i in tqdm(range(face_features.shape[0])):
        face_z_item = face_features[i]
        threshold = 1e-1
        max_faces = face_z_item.shape[0]
        index = torch.stack(torch.meshgrid(torch.arange(max_faces), torch.arange(max_faces), indexing="ij"), dim=2)
        features = face_z_item[index]
        distance = (features[:, :, 0] - features[:, :, 1]).abs().mean(dim=-1)
        final_face_z = []
        for j in range(max_faces):
            valid = True
            for k in final_face_z:
                if distance[j, k] < threshold:
                    valid = False
                    break
            if valid:
                final_face_z.append(j)
        face_z_item = face_z_item[final_face_z]

        result = model.ae_model.inference(face_z_item)
        pred_faces.append(result["pred_face"].reshape(-1, 3))
        trimesh.PointCloud(result["pred_face"].reshape(-1, 3)).export(f"mix_{i}.ply")

    result1 = model.ae_model.inference(latent1)
    result2 = model.ae_model.inference(latent2)

    trimesh.PointCloud(result1["pred_face"].reshape(-1, 3)).export("ori1.ply")
    trimesh.PointCloud(result2["pred_face"].reshape(-1, 3)).export("ori2.ply")
    pass
