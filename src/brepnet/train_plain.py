import time
from pathlib import Path
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
from tqdm import tqdm

from src.brepnet.dataset import normalize_coord, discrete_coord, denormalize_coord
from src.brepnet.model import res_block_1D, Attn_fuser, res_block_2D

import os.path
import torch

from torch.optim import Adam

def read_data():
    dataset_path = r"D:/img2brep/deepcad_whole_test_v5"
    data_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if
                    os.path.isdir(os.path.join(dataset_path, folder))]
    data_folders.sort()
    filepath = os.path.join(dataset_path, r"id_larger_than_64_faces.txt")
    ignore_ids = []
    if os.path.exists(filepath):
        ignore_ids = [item.strip() for item in open(filepath).readlines()]
    else:
        for folder_path in data_folders:
            if not os.path.exists(os.path.join(folder_path, "data.npz")):
                ignore_ids.append(folder_path)
                continue
            data_npz = np.load(os.path.join(folder_path, "data.npz"))
            if data_npz['sample_points_faces'].shape[0] > 64:
                ignore_ids.append(folder_path)
        with open(filepath, "w") as f:
            for item in ignore_ids:
                f.write(item + "\n")

    for folder_path in ignore_ids:
        data_folders.remove(folder_path)
    print("Number of data folders: ", len(data_folders))
    return data_folders


class AutoEncoder_0921(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.face_conv1 = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
        )
        face_pos_embedding = 1e-2 * torch.randn(ds // 8, 16, 16)
        self.face_pos_embedding = nn.Parameter(face_pos_embedding)
        self.face_coords = nn.Sequential(
            res_block_2D(ds // 8, ds // 8, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 8, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8
            res_block_2D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 4, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4
            res_block_2D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 2, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )

        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
        )
        edge_pos_embedding = torch.randn(ds // 8, 16) * 1e-2
        self.edge_pos_embedding = nn.Parameter(edge_pos_embedding)
        self.edge_coords = nn.Sequential(
            res_block_1D(ds // 8, ds // 8, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 8, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8
            res_block_1D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 4, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 2, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )  # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768  # bottlenek_dim
        self.face_attn_proj_in = nn.Linear(df, bd)
        self.face_attn_proj_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 24)

        self.global_feature1 = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )
        self.global_feature2 = nn.Sequential(
            nn.Linear(df * 2, df),
            nn.LeakyReLU(),
            nn.Linear(df, df),
        )

        face_pos_embedding2 = torch.randn(2, df) * 1e-2
        self.face_pos_embedding2 = nn.Parameter(face_pos_embedding2)
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, df * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(df * 2, 1)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=2, mode="bilinear"),  # 4
            nn.ConvTranspose2d(ds // 2, ds // 2, kernel_size=2, stride=2),
            nn.Conv2d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=2, mode="bilinear"),  # 8
            nn.ConvTranspose2d(ds // 4, ds // 4, kernel_size=2, stride=2),
            nn.Conv2d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=2, mode="bilinear"),  # 16
            nn.ConvTranspose2d(ds // 8, ds // 8, kernel_size=2, stride=2),
            nn.Conv2d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c', c=3),
        )
        self.face_center_scale_decoder = nn.Sequential(
            Rearrange("b n -> b n 1 1"),
            nn.Conv2d(dl * 2 * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )

        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            nn.Conv1d(df, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds // 2, ds // 2, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode="linear"),  # 4
            nn.Conv1d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds // 4, ds // 4, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode="linear"),  # 8
            nn.Conv1d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds // 8, ds // 8, kernel_size=2, stride=2),
            # nn.Upsample(scale_factor=2, mode="linear"),  # 16
            nn.Conv1d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c', c=3),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            Rearrange("b n-> b n 1"),
            nn.Conv1d(df * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w -> ... (w c) n', c=3, n=2),
        )

        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df * 2),
                nn.LeakyReLU(),
                nn.Linear(self.df * 2, self.df * 2),
            )

        self.times = {
            "encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]
        intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
        id_false_start = true_intersection_embedding.shape[0]

        features = intersection_embedding + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)

        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

        return loss_edge, features[:id_false_start]

    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            return v_fused_face_features, torch.zeros_like(v_fused_face_features[0, 0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0, 0])

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        return fused_face_features, kl_loss

    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)
        self.times["encoder"] += time.time() - timer
        timer = time.time()

        # Fuser
        edge_face_connectivity = v_data["edge_face_connectivity"]
        # Face graph
        x = face_features
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        x = self.face_attn_proj_in(x)
        x = self.face_attn(x, v_data["attn_mask"])
        x = self.face_attn_proj_out(x)
        fused_face_features = x
        self.times["Fuser"] += time.time() - timer
        timer = time.time()

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=x.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z)
        self.times["global"] += time.time() - timer
        timer = time.time()

        face_z, kl_loss = self.sample(face_z, v_is_test=v_test)
        self.times["Sample"] += time.time() - timer
        timer = time.time()

        # Intersection
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            face_z,
        )
        self.times["Intersection"] += time.time() - timer
        timer = time.time()

        face_points_local = self.face_points_decoder(face_z)
        face_center_scale = self.face_center_scale_decoder(face_z)
        face_center = face_center_scale[..., 0]
        face_scale = torch.sigmoid(face_center_scale[..., 1]) * 2

        edge_points_local = self.edge_points_decoder(intersected_edge_feature)
        edge_center_scale = self.edge_center_scale_decoder(intersected_edge_feature)
        edge_center = edge_center_scale[..., 0]
        edge_scale = torch.sigmoid(edge_center_scale[..., 1]) * 2

        edge_points_local1 = self.edge_points_decoder(edge_features)
        edge_center_scale1 = self.edge_center_scale_decoder(edge_features)
        edge_center1 = edge_center_scale1[..., 0]
        edge_scale1 = torch.sigmoid(edge_center_scale1[..., 1]) * 2
        self.times["Decoder"] += time.time() - timer
        timer = time.time()

        # Loss
        loss = {}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords_norm"] = nn.functional.l1_loss(
            face_points_local,
            v_data["face_points_norm"]
        )
        loss["face_center"] = nn.functional.l1_loss(
            face_center,
            v_data["face_center"]
        )
        loss["face_scale"] = nn.functional.l1_loss(
            face_scale,
            v_data["face_scale"]
        )

        loss["edge_coords_norm1"] = nn.functional.l1_loss(
            edge_points_local1,
            v_data["edge_points_norm"]
        )
        loss["edge_center1"] = nn.functional.l1_loss(
            edge_center1,
            v_data["edge_center"]
        )
        loss["edge_scale1"] = nn.functional.l1_loss(
            edge_scale1,
            v_data["edge_scale"]
        )

        loss["edge_coords_norm"] = nn.functional.l1_loss(
            edge_points_local,
            v_data["edge_points_norm"][edge_face_connectivity[:, 0]]
        )
        loss["edge_center"] = nn.functional.l1_loss(
            edge_center,
            v_data["edge_center"][edge_face_connectivity[:, 0]]
        )
        loss["edge_scale"] = nn.functional.l1_loss(
            edge_scale,
            v_data["edge_scale"][edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = kl_loss
        loss["total_loss"] = sum(loss.values())
        self.times["Loss"] += time.time() - timer
        timer = time.time()

        data = {}
        if v_test:
            pred_data = self.inference(face_z)
            data.update(pred_data)

            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True

            data["gt_face_adj"] = face_adj.reshape(-1)
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()
            data["gt_edge"] = v_data["edge_points"].detach().cpu().numpy()
            data["gt_edge_face_connectivity"] = v_data["edge_face_connectivity"].detach().cpu().numpy()

            loss["face_coords"] = nn.functional.l1_loss(
                data["pred_face"],
                v_data["face_points"]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local, edge_center, edge_scale),
                v_data["edge_points"][v_data["edge_face_connectivity"][:, 0]]
            )
            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local1, edge_center1, edge_scale1),
                v_data["edge_points"]
            )

        return loss["total_loss"], data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1, 2).to(device)
        feature_pair = v_face_features[indexes]

        feature_pair = feature_pair + self.face_pos_embedding2[None, :]
        feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
        feature_pair = self.edge_feature_proj(feature_pair)
        pred = self.classifier(feature_pair)[..., 0]
        pred_labels = torch.sigmoid(pred) > 0.5

        edge_points_local = self.edge_points_decoder(feature_pair[pred_labels])
        edge_center_scale = self.edge_center_scale_decoder(feature_pair[pred_labels])
        edge_center = edge_center_scale[..., 0]
        edge_scale = torch.sigmoid(edge_center_scale[..., 1]) * 2
        pred_edge_points = denormalize_coord(edge_points_local, edge_center, edge_scale)

        face_points_local = self.face_points_decoder(v_face_features)
        face_center_scale = self.face_center_scale_decoder(v_face_features)
        face_center = face_center_scale[..., 0]
        face_scale = torch.sigmoid(face_center_scale[..., 1]) * 2
        pred_face_points = denormalize_coord(face_points_local, face_center, face_scale)

        pred_edge_face_connectivity = torch.cat(
            (torch.arange(pred_edge_points.shape[0], device=device)[:, None], indexes[pred_labels]), dim=1)
        return {
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    data = []
    batch_size = 8
    cache_data = []
    data_folders = read_data()
    num_max_faces = 62
    num_max_edges = 334
    len_edges = []
    len_faces = []
    for folder_path in data_folders:
        data_npz = np.load(os.path.join(folder_path, "data.npz"))

        # Face sample points (num_faces*32*32*3)
        face_points = torch.from_numpy(data_npz['sample_points_faces'])
        line_points = torch.from_numpy(data_npz['sample_points_lines'])

        face_points_norm, face_center, face_scale = normalize_coord(face_points)
        edge_points_norm, edge_center, edge_scale = normalize_coord(line_points)
        face_points_discrete, face_center_discrete, face_scale_discrete = discrete_coord(face_points_norm, face_center,
                                                                                         face_scale, 256)
        edge_points_discrete, edge_center_discrete, edge_scale_discrete = discrete_coord(edge_points_norm, edge_center,
                                                                                         edge_scale, 256)
        # face_points = continuous_coord(face_points_discrete, face_center, face_scale, 256)

        #  Which of two faces intersect and produce an edge (num_intersection, (id_edge, id_face1, id_face2))
        edge_face_connectivity = torch.from_numpy(data_npz['edge_face_connectivity'])
        # Ignore self intersection
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity[:, 1] != edge_face_connectivity[:, 2]]
        #  Which of two edges intersect and produce a vertex (num_intersection, (id_vertex, id_edge1, id_edge2))
        # vertex_edge_connectivity = torch.from_numpy(data_npz['vertex_edge_connectivity'])

        face_adj = torch.from_numpy(data_npz['face_adj'])
        zero_positions = torch.from_numpy(data_npz['zero_positions'])
        if zero_positions.shape[0] > face_adj.shape[0] * 2:
            index = np.random.choice(zero_positions.shape[0], face_adj.shape[0] * 2, replace=False)
            zero_positions = zero_positions[index]
        # Assume the number of true intersection is less than self.max_intersection

        cache_data.append((
            Path(folder_path).stem,
            face_points, line_points,
            face_adj, zero_positions,
            face_points_norm, face_center, face_scale,
            edge_points_norm, edge_center, edge_scale,
            face_points_discrete, face_center_discrete, face_scale_discrete,
            edge_points_discrete, edge_center_discrete, edge_scale_discrete,
            edge_face_connectivity
        ))

        if len(cache_data) == batch_size:
            (
                v_prefix, face_points, edge_points,
                face_adj, zero_positions,
                face_points_norm, face_center, face_scale,
                edge_points_norm, edge_center, edge_scale,
                face_points_discrete, face_center_discrete, face_scale_discrete,
                edge_points_discrete, edge_center_discrete, edge_scale_discrete,
                edge_face_connectivity
            ) = zip(*cache_data)
            bs = len(v_prefix)
            flat_zero_positions = []
            num_face_record = []

            num_faces = 0
            num_edges = 0
            edge_conn_num = []
            for i in range(bs):
                edge_face_connectivity[i][:, 0] += num_edges
                edge_face_connectivity[i][:, 1:] += num_faces
                edge_conn_num.append(edge_face_connectivity[i].shape[0])
                flat_zero_positions.append(zero_positions[i] + num_faces)
                num_faces += face_points[i].shape[0]
                num_edges += edge_points[i].shape[0]
                num_face_record.append(face_points[i].shape[0])
            num_face_record = torch.tensor(num_face_record, dtype=torch.long)
            num_sum_edges = sum(edge_conn_num)
            edge_attn_mask = torch.ones((num_sum_edges, num_sum_edges), dtype=bool)
            id_cur = 0
            for i in range(bs):
                edge_attn_mask[id_cur:id_cur + edge_conn_num[i], id_cur:id_cur + edge_conn_num[i]] = False
                id_cur += edge_conn_num[i]

            num_max_faces = num_face_record.max()
            valid_mask = torch.zeros((bs, num_max_faces), dtype=bool)
            for i in range(bs):
                valid_mask[i, :num_face_record[i]] = True
            attn_mask = torch.ones((num_faces, num_faces), dtype=bool)
            id_cur = 0
            for i in range(bs):
                attn_mask[id_cur:id_cur + face_points[i].shape[0], id_cur: id_cur + face_points[i].shape[0]] = False
                id_cur += face_points[i].shape[0]

            flat_zero_positions = torch.cat(flat_zero_positions, dim=0)
            dtype = torch.float16
            data.append({
                "v_prefix": v_prefix,
                "edge_points": torch.cat(edge_points, dim=0).to(device).to(dtype),
                "face_points": torch.cat(face_points, dim=0).to(device).to(dtype),

                "edge_face_connectivity": torch.cat(edge_face_connectivity, dim=0).to(device),
                "zero_positions": flat_zero_positions.to(device),
                "attn_mask": attn_mask.to(device),
                "edge_attn_mask": edge_attn_mask.to(device),

                "num_face_record": num_face_record.to(device),
                "valid_mask": valid_mask.to(device),

                "face_points_norm": torch.cat(face_points_norm, dim=0).to(device).to(dtype),
                "face_center": torch.cat(face_center, dim=0).to(device).to(dtype),
                "face_scale": torch.cat(face_scale, dim=0).to(device).to(dtype),
                "edge_points_norm": torch.cat(edge_points_norm, dim=0).to(device).to(dtype),
                "edge_center": torch.cat(edge_center, dim=0).to(device).to(dtype),
                "edge_scale": torch.cat(edge_scale, dim=0).to(device).to(dtype),
            })
            cache_data = []

    conf = {
        "dim_latent": 8,
        "dim_shape": 768,
        "norm": "layer",
        "gaussian_weights": 0,
    }
    model = AutoEncoder_0921(conf)
    model.half()
    model.to(device)
    # model = torch.compile(model, dynamic=True)

    model(data[0])
    torch.cuda.synchronize()
    for key in model.times:
        model.times[key] = 0
    optimizer = Adam(model.parameters(), lr=1e-4)
    tt = [0, 0, 0]
    timer = time.time()
    for batch in tqdm(data[:100]):
        torch.cuda.synchronize()
        optimizer.zero_grad()
        dtimer = time.time()
        loss, recon_data = model(batch)
        torch.cuda.synchronize()
        tt[0] += time.time() - dtimer
        dtimer = time.time()
        loss.backward()
        torch.cuda.synchronize()
        tt[1] += time.time() - dtimer
        dtimer = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        tt[2] += time.time() - dtimer
        torch.cuda.synchronize()
    print(tt)
    print(time.time() - timer)
    print(model.times)

    model.float()
    for idx in range(len(data)):
        for key in ["edge_points", "edge_points_norm", "edge_center", "edge_scale",
                    "face_points", "face_points_norm", "face_center", "face_scale"]:
            data[idx][key] = data[idx][key].float()
    model(data[0])
    for key in model.times:
        model.times[key] = 0
    optimizer = Adam(model.parameters(), lr=1e-4)
    tt = [0, 0, 0]
    timer = time.time()
    for batch in tqdm(data[:100]):
        torch.cuda.synchronize()
        optimizer.zero_grad()
        dtimer = time.time()
        loss, recon_data = model(batch)
        torch.cuda.synchronize()
        tt[0] += time.time() - dtimer
        dtimer = time.time()
        loss.backward()
        torch.cuda.synchronize()
        tt[1] += time.time() - dtimer
        dtimer = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        tt[2] += time.time() - dtimer
    print(tt)
    print(time.time() - timer)
    print(model.times)


