import time
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
from vector_quantize_pytorch import FSQ, ResidualFSQ
from vector_quantize_pytorch import FSQ, ResidualFSQ, ResidualLFQ, ResidualVQ

from src.brepnet.dataset import continuous_coord, denormalize_coord, denormalize_coord2

from chamferdist import ChamferDistance

def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()


class res_linear(nn.Module):
    def __init__(self, dim_in, dim_out, norm="none"):
        super(res_linear, self).__init__()
        self.conv = nn.Linear(dim_in, dim_out)
        self.act = nn.LeakyReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.LayerNorm(dim_out)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_1D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h -> ... h c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h c -> ... c h'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class res_block_2D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm="none"):
        super(res_block_2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm == "none":
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = nn.Sequential(
                Rearrange('... c h w -> ... h w c'),
                nn.LayerNorm(dim_out),
                Rearrange('... h w c -> ... c h w'),
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(dim_out)
        else:
            raise ValueError("Norm type not supported")

    def forward(self, x):
        x = x + self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Separate_encoder(nn.Module):
    def __init__(self, dim_shape=256, dim_latent=8, v_conf=None, **kwargs):
        super().__init__()
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, dim_shape, kernel_size=5, stride=1, padding=2),
            res_block_2D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 1 -> b c'),
        )

        self.face_fuser = Attn_fuser_single(dim_shape)
        self.face_proj = nn.Linear(dim_shape, dim_latent * 2)

        self.edge_coords = nn.Sequential(
            Rearrange('b h n -> b n h'),
            nn.Conv1d(3, dim_shape, kernel_size=5, stride=1, padding=2),
            res_block_1D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 -> b c'),
        )

        self.edge_fuser = Attn_fuser_single(dim_shape)

        self.vertex_coords = nn.Sequential(
            Rearrange('b c-> b c 1'),
            nn.Conv1d(3, dim_shape, kernel_size=1, stride=1, padding=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            Rearrange('b c 1 -> b c'),
        )

        self.vertex_fuser = Attn_fuser_single(dim_shape)

    def forward(self, v_data):
        face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        face_features = self.face_coords(v_data["face_points"][face_mask])

        face_attn_mask = get_attn_mask(face_mask)
        face_features = self.face_fuser(face_features, face_attn_mask)
        face_features = self.face_proj(face_features)

        # Edge
        edge_mask = (v_data["edge_points"] != -1).all(dim=-1).all(dim=-1)
        edge_features = self.edge_coords(v_data["edge_points"][edge_mask])

        edge_attn_mask = get_attn_mask(edge_mask)
        edge_features = self.edge_fuser(edge_features, edge_attn_mask)

        # Vertex
        vertex_mask = (v_data["vertex_points"] != -1).all(dim=-1)
        vertex_features = self.vertex_coords(v_data["vertex_points"][vertex_mask])

        vertex_attn_mask = get_attn_mask(vertex_mask)
        vertex_features = self.vertex_fuser(vertex_features, vertex_attn_mask)

        return {
            "face_features": face_features,
            "edge_features": edge_features,
            "vertex_features": vertex_features,
            "face_mask": face_mask,
            "edge_mask": edge_mask,
            "vertex_mask": vertex_mask,
        }



def prepare_connectivity(v_data, ):
    face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
    edge_mask = (v_data["edge_points"] != -1).all(dim=-1).all(dim=-1)
    vertex_mask = (v_data["vertex_points"] != -1).all(dim=-1)
    # ================== Prepare data for flattened features ==================
    edge_index_offsets = reduce(edge_mask.long(), 'b ne -> b', 'sum')
    edge_index_offsets = F.pad(edge_index_offsets.cumsum(dim=0), (1, -1), value=0)
    face_index_offsets = reduce(face_mask.long(), 'b ne -> b', 'sum')
    face_index_offsets = F.pad(face_index_offsets.cumsum(dim=0), (1, -1), value=0)
    vertex_index_offsets = reduce(vertex_mask.long(), 'b ne -> b', 'sum')
    vertex_index_offsets = F.pad(vertex_index_offsets.cumsum(dim=0), (1, -1), value=0)

    vertex_edge_connectivity = v_data["vertex_edge_connectivity"].clone()
    vertex_edge_connectivity_valid = (vertex_edge_connectivity != -1).all(dim=-1)
    # Solve the vertex_edge_connectivity: last two dimension (id_edge)
    vertex_edge_connectivity[..., 1:] += edge_index_offsets[:, None, None]
    # Solve the edge_face_connectivity: first (id_vertex)
    vertex_edge_connectivity[..., 0:1] += vertex_index_offsets[:, None, None]
    vertex_edge_connectivity = vertex_edge_connectivity[vertex_edge_connectivity_valid]
    v_data["vertex_edge_connectivity"] = vertex_edge_connectivity

    edge_face_connectivity = v_data["edge_face_connectivity"].clone()
    edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
    # Solve the edge_face_connectivity: last two dimension (id_face)
    edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
    # Solve the edge_face_connectivity: first dimension (id_edge)
    edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
    edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]
    v_data["edge_face_connectivity"] = edge_face_connectivity
    return


def flatten_to_batch(v_data, v_mask):
    
    pass

def profile_time(time_dict, key, v_timer):
    torch.cuda.synchronize()
    cur = time.time()
    time_dict[key] += cur - v_timer
    return cur

# Full continuous VAE
class AutoEncoder_base(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_base, self).__init__()
        self.dim_shape = 256
        self.dim_latent = 32
        self.time_statics = [0 for _ in range(10)]

        ds = self.dim_shape
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv2d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4

        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w h -> ... w h n',),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(2*self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_2D(ds, ds, 3, 1, 1),
            res_block_2D(ds, ds, 3, 1, 1),
            res_block_2D(ds, ds, 3, 1, 1),
            res_block_2D(ds, ds, 3, 1, 1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Conv2d(ds, 1, kernel_size=1, stride=1, padding=0),
            Rearrange("b 1 1 1 -> b 1")
        )

        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b n w h -> b n (w h)"),
            nn.Conv1d(2*self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )
        self.num_max_items = 500

        # self.face_coords = torch.compile(self.face_coords)
        # self.face_coords_decoder = torch.compile(self.face_coords_decoder)
        # self.classifier = torch.compile(self.classifier)
        # self.edge_coords_decoder = torch.compile(self.edge_coords_decoder)

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        # Classification
        true_features = torch.cat((true_intersection_embedding[:,0], true_intersection_embedding[:,1]), dim=1)
        pred_true = self.classifier(true_features)
        pred_false = self.classifier(torch.cat((false_intersection_embedding[:,0], false_intersection_embedding[:,1]), dim=1))
        
        gt_true = torch.ones_like(pred_true)
        gt_false = torch.zeros_like(pred_false)
        loss_edge = F.binary_cross_entropy_with_logits(pred_true, gt_true) + \
            F.binary_cross_entropy_with_logits(pred_false, gt_false)
        
        return loss_edge, true_features

    def forward(self, v_data, v_test=False, **kwargs):
        # prepare_connectivity(v_data)
        face_features = self.face_coords(v_data["face_points"])
        pre_face_coords = self.face_coords_decoder(face_features,)
        edge_points = v_data["edge_points"]
        edge_face_connectivity = v_data["edge_face_connectivity"]
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]

        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            face_features, 
        )
        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        
        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            true_indexes = indexes[face_adj]
            false_indexes = indexes[~face_adj]
            data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            true_features = torch.cat((face_features[true_indexes[:, 0]], face_features[true_indexes[:, 1]]), dim=1)
            false_features = torch.cat((face_features[false_indexes[:, 0]], face_features[false_indexes[:, 1]]), dim=1)
            features = torch.cat((true_features, false_features), dim=0)
            pred = self.classifier(features)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred

        return loss, data


# Full continuous VAE
class AutoEncoder_featured(AutoEncoder_base):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_featured, self).__init__(v_conf)
        self.dim_shape = 128
        self.dim_latent = 32
        self.time_statics = [0 for _ in range(10)]

        ds = self.dim_shape
        self.edge_coords = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=5, stride=1, padding=2),
            res_block_1D(ds, ds, ks=5, st=1, pa=2),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4

        self.edge_feature_proj = nn.Sequential(
            Rearrange("b n w h -> b n (w h)"),
            nn.Conv1d(2*self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=5, st=1, pa=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=5, st=1, pa=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        )
        self.shared_edge_decoder = nn.Sequential(
            nn.Conv1d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=5, st=1, pa=2),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w-> ... w n',),
        )
        # self.graph_face_edge

    def forward(self, v_data, v_test=False, **kwargs):
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        pre_face_coords = self.face_coords_decoder(face_features,)
        pre_edge_coords1 = self.shared_edge_decoder(edge_features)

        # Intersection
        edge_face_connectivity = v_data["edge_face_connectivity"]
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            face_features, 
        )
        intersected_edge_feature = self.edge_feature_proj(intersected_edge_feature)
        pre_edge_coords = self.shared_edge_decoder(intersected_edge_feature)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            true_indexes = indexes[face_adj]
            false_indexes = indexes[~face_adj]
            data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            true_features = torch.cat((face_features[true_indexes[:, 0]], face_features[true_indexes[:, 1]]), dim=1)
            false_features = torch.cat((face_features[false_indexes[:, 0]], face_features[false_indexes[:, 1]]), dim=1)
            features = torch.cat((true_features, false_features), dim=0)
            pred = self.classifier(features)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred

        return loss, data


# Full continuous VAE
class AutoEncoder_featuredv2(AutoEncoder_base):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_featuredv2, self).__init__(v_conf)
        self.dim_shape = 128
        self.dim_latent = 32
        self.time_statics = [0 for _ in range(10)]

        ds = self.dim_shape
        self.edge_coords = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4
        self.edge_feature_proj = nn.Sequential( 
            nn.Conv2d(2*self.dim_latent, ds, kernel_size=3, stride=1, padding=1), # "b n 4 4"
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2*2
            Rearrange("b n w h -> b n (w h)"), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        )
        self.shared_edge_decoder = nn.Sequential(
            nn.Conv1d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w-> ... w n',),
        )
        
        # self.edge_coords = torch.compile(self.edge_coords)
        # self.edge_feature_proj = torch.compile(self.edge_feature_proj)
        # self.shared_edge_decoder = torch.compile(self.shared_edge_decoder)
        # self.intersection = torch.compile(self.intersection)

    def forward(self, v_data, v_test=False, **kwargs):
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        pre_face_coords = self.face_coords_decoder(face_features,)
        pre_edge_coords1 = self.shared_edge_decoder(edge_features)

        # Intersection
        edge_face_connectivity = v_data["edge_face_connectivity"]
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            face_features, 
        )
        intersected_edge_feature = self.edge_feature_proj(intersected_edge_feature)
        pre_edge_coords = self.shared_edge_decoder(intersected_edge_feature)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            true_indexes = indexes[face_adj]
            false_indexes = indexes[~face_adj]
            data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            true_features = torch.cat((face_features[true_indexes[:, 0]], face_features[true_indexes[:, 1]]), dim=1)
            false_features = torch.cat((face_features[false_indexes[:, 0]], face_features[false_indexes[:, 1]]), dim=1)
            features = torch.cat((true_features, false_features), dim=0)
            pred = self.classifier(features)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred

        return loss, data


# Full continuous VAE
class AutoEncoder_graph(AutoEncoder_base):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph, self).__init__(v_conf)
        self.dim_shape = 128
        self.dim_latent = 32
        self.time_statics = [0 for _ in range(10)]

        ds = self.dim_shape
        self.edge_coords = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4
        self.edge_feature_proj = nn.Sequential(
            Rearrange("b n w h -> b n (w h)"),
            nn.Conv1d(2*self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        )
        self.shared_edge_decoder = nn.Sequential(
            nn.Conv1d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=1, st=1, pa=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w-> ... w n',),
        )
        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                512, 512, 
                heads=1, edge_dim=128,
            ))
            self.graph_face_edge.append(nn.ReLU())

        self.time_statics = {
        }

        # self.edge_coords = torch.compile(self.edge_coords)
        # self.edge_feature_proj = torch.compile(self.edge_feature_proj)
        # self.shared_edge_decoder = torch.compile(self.shared_edge_decoder)
        # self.intersection = torch.compile(self.intersection)

    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        x = face_features.reshape(-1, 512)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]].reshape(-1,128)
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=4, w=4)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features,)
        pre_edge_coords1 = self.shared_edge_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        intersected_edge_feature = self.edge_feature_proj(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.shared_edge_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            true_indexes = indexes[face_adj]
            false_indexes = indexes[~face_adj]
            data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            true_features = torch.cat((face_features[true_indexes[:, 0]], face_features[true_indexes[:, 1]]), dim=1)
            false_features = torch.cat((face_features[false_indexes[:, 0]], face_features[false_indexes[:, 1]]), dim=1)
            features = torch.cat((true_features, false_features), dim=0)
            pred = self.classifier(features)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred

        return loss, data


# Full continuous VAE
class AutoEncoder_graph_bigger(AutoEncoder_graph):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_bigger, self).__init__(v_conf)
        db = 512 # dim_bottleneck
        self.edge_feature_proj = nn.Sequential(
            nn.Conv2d(2*self.dim_latent, db, kernel_size=1, stride=1, padding=0),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            res_block_2D(db, db, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2 * 2
            res_block_2D(db, db, ks=1, st=1, pa=0),
            res_block_2D(db, db, ks=1, st=1, pa=0),
            res_block_2D(db, db, ks=1, st=1, pa=0),
            nn.Conv2d(db, self.dim_latent, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w h -> b n (w h)"), # 4
        )


# Self attention layer
class Attn_fuser_single(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward):
        super().__init__()
        self.atten = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True),
        ])

    def forward(self, v_embedding, v_attn_mask):
        x = v_embedding
        for layer in self.atten:
            x = layer(x, src_mask=v_attn_mask)
        return x

class Attn_fuser(nn.Module):
    def __init__(self, v_layer, num_layers):
        super().__init__()
        self.atten = nn.TransformerEncoder(v_layer, num_layers)

    def forward(self, v_embedding, v_attn_mask):
        return self.atten(v_embedding, mask=v_attn_mask)

# Full continuous VAE
class AutoEncoder_graph_atten(AutoEncoder_graph_bigger):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_atten, self).__init__(v_conf)
        self.face_attn = Attn_fuser_single(512, 2048)
    
    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        x = face_features.reshape(-1, 512)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]].reshape(-1,128)
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Attn
        x = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=4, w=4)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features,)
        pre_edge_coords1 = self.shared_edge_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        intersected_edge_feature = self.edge_feature_proj(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.shared_edge_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            true_indexes = indexes[face_adj]
            false_indexes = indexes[~face_adj]
            data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            true_features = torch.cat((fused_face_features[true_indexes[:, 0]], fused_face_features[true_indexes[:, 1]]), dim=1)
            false_features = torch.cat((fused_face_features[false_indexes[:, 0]], fused_face_features[false_indexes[:, 1]]), dim=1)
            features = torch.cat((true_features, false_features), dim=0)
            pred = self.classifier(features)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred

        return loss, data

# Full continuous VAE
class AutoEncoder_graph_flattened(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_flattened, self).__init__()
        self.dim_shape = 256
        self.dim_latent = 32
        ds = self.dim_shape
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv2d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w h -> ... w h n',),
        )

        self.edge_coords = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2), # 1
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b n -> b n 1"),
            nn.Conv1d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"), # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1), 
            nn.Upsample(scale_factor=2, mode="linear"), # 32
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )

        bd = 1024 # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            Rearrange("b n h w -> b (n h w) 1"),
            nn.Conv1d(1024, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0),
            nn.Conv1d(bd, self.dim_latent, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"), # 1
        )
        self.classifier = nn.Linear(self.dim_latent, 1)
        self.face_attn = Attn_fuser_single(512, 2048)

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                512, 512, 
                heads=1, edge_dim=32,
            ))
            self.graph_face_edge.append(nn.ReLU())

        
        self.time_statics = {}
    
    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        # Classification
        true_features = torch.cat((true_intersection_embedding[:,0], true_intersection_embedding[:,1]), dim=1)
        true_intersection_embedding = self.edge_feature_proj(true_features)
        false_features = torch.cat((false_intersection_embedding[:,0], false_intersection_embedding[:,1]), dim=1)
        false_intersection_embedding = self.edge_feature_proj(false_features)

        pred_true = self.classifier(true_intersection_embedding)
        pred_false = self.classifier(false_intersection_embedding)
        
        gt_true = torch.ones_like(pred_true)
        gt_false = torch.zeros_like(pred_false)
        loss_edge = F.binary_cross_entropy_with_logits(pred_true, gt_true) + \
            F.binary_cross_entropy_with_logits(pred_false, gt_false)
        
        return loss_edge, true_intersection_embedding


    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        x = face_features.reshape(-1, 4*4*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Attn
        x = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=4, w=4)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features,)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            # true_indexes = indexes[face_adj]
            # false_indexes = indexes[~face_adj]
            # data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            # true_features = torch.cat((fused_face_features[true_indexes[:, 0]], fused_face_features[true_indexes[:, 1]]), dim=1)
            # false_features = torch.cat((fused_face_features[false_indexes[:, 0]], fused_face_features[false_indexes[:, 1]]), dim=1)
            # features = torch.cat((true_features, false_features), dim=0)
            # pred = self.classifier(self.edge_feature_proj(features))[...,0]
            # pred = torch.sigmoid(pred) > 0.5
            # data["pred"] = pred

            indexes = indexes.reshape(-1,2)
            feature_pair = torch.cat((fused_face_features[indexes[:,0]], fused_face_features[indexes[:,1]]), dim=1)
            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred
            data["gt"] = face_adj.reshape(-1)

            data["face_features"] = fused_face_features.cpu().numpy()
            data["edge_loss"] = nn.functional.l1_loss(
                pre_edge_coords,
                gt_edge_points,
                reduction="none"
            ).mean(dim=1).mean(dim=1).cpu().numpy()
            data["face_loss"] = nn.functional.l1_loss(
                pre_face_coords,
                v_data["face_points"],
                reduction="none"
            ).mean(dim=1).mean(dim=1).mean(dim=1).cpu().numpy()
        return loss, data


    def decode(self, v_face_features):
        pre_face_coords = self.face_coords_decoder(v_face_features)
        return pre_face_coords

class AutoEncoder_graph_flattened_tiny(AutoEncoder_graph_flattened):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_flattened_tiny, self).__init__(v_conf)
        self.dim_shape = 128
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w h -> ... w h n',),
        )

        self.edge_coords = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=4),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 32
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )

        bd = 1024 # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            Rearrange("b n h w -> b (n h w) 1"),
            nn.Conv1d(dl * 4 * 4 * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, ds * 4, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(ds*4, 1)
        self.face_attn = Attn_fuser_single(dl * 4 * 4, 2048)

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                dl * 4 * 4, dl * 4 * 4, 
                heads=1, edge_dim=ds * 4,
            ))
            self.graph_face_edge.append(nn.ReLU())
        
        self.time_statics = {}
        self.face_coords = torch.compile(self.face_coords, dynamic=True)
        self.face_coords_decoder = torch.compile(self.face_coords_decoder, dynamic=True)
        self.edge_coords = torch.compile(self.edge_coords, dynamic=True)
        self.edge_coords_decoder = torch.compile(self.edge_coords_decoder, dynamic=True)
        self.edge_feature_proj = torch.compile(self.edge_feature_proj, dynamic=True)

class AutoEncoder_graph_flattened_tiny_more_tr(AutoEncoder_graph_flattened_tiny):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_flattened_tiny_more_tr, self).__init__(v_conf)
        self.dim_latent = v_conf["dim_latent"]
        dl = self.dim_latent
        layer = nn.TransformerEncoderLayer(
            dl * 2 * 2, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 12)

        self.face_weights = 1.
        self.edge_weights = 1.
        self.edge_ori_weights = 1.
        self.edge_fea_weights = 1.
        self.classfication_weights = 0.01

    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Autoencoder
        face_features = self.face_coords(v_data["face_points"])
        edge_features = self.edge_coords(v_data["edge_points"])
        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        x = face_features.reshape(-1, 4*4*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Attn
        x = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=4, w=4)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features,)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification * self.classfication_weights
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        ) * self.face_weights
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        ) * self.edge_weights
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        ) * self.edge_ori_weights
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        ) * self.edge_fea_weights
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)
            # true_indexes = indexes[face_adj]
            # false_indexes = indexes[~face_adj]
            # data["gt"] = torch.cat((torch.ones(true_indexes.shape[0]), torch.zeros(false_indexes.shape[0])), dim=0).to(device)
            # true_features = torch.cat((fused_face_features[true_indexes[:, 0]], fused_face_features[true_indexes[:, 1]]), dim=1)
            # false_features = torch.cat((fused_face_features[false_indexes[:, 0]], fused_face_features[false_indexes[:, 1]]), dim=1)
            # features = torch.cat((true_features, false_features), dim=0)
            # pred = self.classifier(self.edge_feature_proj(features))[...,0]
            # pred = torch.sigmoid(pred) > 0.5
            # data["pred"] = pred

            indexes = indexes.reshape(-1,2)
            feature_pair = torch.cat((fused_face_features[indexes[:,0]], fused_face_features[indexes[:,1]]), dim=1)
            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5
            data["pred"] = pred
            data["gt"] = face_adj.reshape(-1)

            data["face_features"] = fused_face_features.cpu().numpy()
            data["edge_loss"] = nn.functional.l1_loss(
                pre_edge_coords,
                gt_edge_points,
                reduction="none"
            ).mean(dim=1).mean(dim=1).cpu().numpy()
            data["face_loss"] = nn.functional.l1_loss(
                pre_face_coords,
                v_data["face_points"],
                reduction="none"
            ).mean(dim=1).mean(dim=1).mean(dim=1).cpu().numpy()
        return loss, data

class AutoEncoder_graph_flattened_plus(AutoEncoder_graph_flattened):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_flattened_plus, self).__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        self.face_conv1 = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=3, stride=1, padding=1),
            Rearrange('... n w h -> ... w h n',),
        )
        self.face_pos_embedding = nn.Parameter(torch.randn(ds, 16, 16))
        self.face_pos_embedding2 = nn.Parameter(torch.randn(2, df))

        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.edge_coords = nn.Sequential(
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )
        self.edge_pos_embedding = nn.Parameter(torch.randn(ds, 16))

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df, 
                heads=1, edge_dim=ds * 2,
            ))
            self.graph_face_edge.append(nn.ReLU())

        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 12)

        bd = 1024 # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, ds * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(ds*2, 1)
        
        self.time_statics = {}
        # self.face_coords = torch.compile(self.face_coords, dynamic=True)
        # self.face_coords_decoder = torch.compile(self.face_coords_decoder, dynamic=True)
        # self.edge_coords = torch.compile(self.edge_coords, dynamic=True)
        # self.edge_coords_decoder = torch.compile(self.edge_coords_decoder, dynamic=True)
        # self.edge_feature_proj = torch.compile(self.edge_feature_proj, dynamic=True)

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        true_intersection_embedding = rearrange(
            true_intersection_embedding,
            'b c n h w -> b c (n h w)', c=2
        )
        false_intersection_embedding = rearrange(
            false_intersection_embedding,
            'b c n h w -> b c (n h w)', c=2
        )

        true_features = true_intersection_embedding + self.face_pos_embedding2[None, :]
        false_features = false_intersection_embedding + self.face_pos_embedding2[None, :]

        true_features = rearrange(true_features, 'b c n -> b (c n) 1')
        false_features = rearrange(false_features, 'b c n -> b (c n) 1')

        true_features = self.edge_feature_proj(true_features)
        false_features = self.edge_feature_proj(false_features)

        pred_true = self.classifier(true_features)
        pred_false = self.classifier(false_features)
        
        gt_true = torch.ones_like(pred_true)
        gt_false = torch.zeros_like(pred_false)
        loss_edge = F.binary_cross_entropy_with_logits(pred_true, gt_true) + \
            F.binary_cross_entropy_with_logits(pred_false, gt_false)
        
        return loss_edge, true_features

    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features.reshape(-1, 2*2*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        x = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=2, w=2)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        
        if v_test:
            data = {}
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = fused_face_features[indexes]
            feature_pair = rearrange(
                feature_pair,
                'b c n h w -> b c (n h w)', c=2
            )
            feature_pair = feature_pair + self.face_pos_embedding2[None, :]
            feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')

            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5

            pre_edge_coords = self.edge_coords_decoder(feature_pair[pred])
            pred_edge_face_connectivity = torch.cat((torch.arange(pre_edge_coords.shape[0], device=device)[:,None], indexes[pred]), dim=1)

            data.update({
                "gt_face_adj": face_adj.cpu().numpy(),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": v_data["edge_points"].cpu().numpy(),
                "gt_face": v_data["face_points"].cpu().numpy(),

                "pred_face_adj": pred.reshape(num_faces, num_faces).cpu().numpy(),
                "pred_edge_face_connectivity": pred_edge_face_connectivity.cpu().numpy(),
                "pred_edge": pre_edge_coords.cpu().numpy(),
                "pred_face": pre_face_coords.cpu().numpy(),

                "face_features": fused_face_features.cpu().numpy()
            })
            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

# Not work
class AutoEncoder_graph_flattened_bigger(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_flattened_bigger, self).__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, 64, kernel_size=1, stride=1, padding=0),
        ) # b c 2 2
        self.face_proj = nn.Linear(ds, dl)
        self.face_deproj = nn.Linear(dl, dl * 2 * 2)
        
        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                ds, ds, 
                heads=1, edge_dim=ds * 2,
            ))
            self.graph_face_edge.append(nn.ReLU())

        layer = nn.TransformerEncoderLayer(
            ds, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 12)

        
    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)
        face_features = rearrange(face_features, 'b n h w -> b (n h w)')

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        x = self.face_attn(x, v_data["attn_mask"])
        lalala_feature = self.face_proj(x)
        x = self.face_deproj(lalala_feature)
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=2, w=2)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        return loss, data

class AutoEncoder_graph_test(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_graph_test, self).__init__(v_conf)
        ds = self.dim_shape
        dl = self.dim_latent
        df = self.df
        
        layer = nn.TransformerEncoderLayer(
            2 * df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.edge_feature_proj1 = Attn_fuser(layer, 12)
        self.edge_feature_proj2 = nn.Linear(df * 2, ds * 2)

        self.time_statics = {}

    def forward(self, v_data, v_test=False, **kwargs):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features.reshape(-1, 2*2*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        x = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(x, 'b (n h w) -> b n h w', h=2, w=2)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        gathered_feature = rearrange(fused_face_features[edge_face_connectivity[:, 1:]], 'b c n h w -> b c (n h w)')
        gathered_feature = gathered_feature + self.face_pos_embedding2[None, :]
        gathered_feature = rearrange(gathered_feature, 'b c n -> b (c n)')
        gathered_feature = self.edge_feature_proj1(gathered_feature, v_data["edge_attn_mask"])
        intersected_edge_feature = self.edge_feature_proj2(gathered_feature)

        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        return loss, data

# Not work
class AutoEncoder_edge_compressed(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_edge_compressed, self).__init__(v_conf)
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = dl * 2 * 2
        df = self.df
        norm = v_conf["norm"]
        self.edge_coords = nn.Sequential(
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=dl, w=2),
            nn.Conv1d(dl, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df, 
                heads=1, edge_dim=dl * 2,
            ))
            self.graph_face_edge.append(nn.ReLU())

        bd = 1024 # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, dl * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(dl*2, 1)
# Not work
class AutoEncoder_edge_compressed1(AutoEncoder_edge_compressed):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_edge_compressed1, self).__init__(v_conf)
        ds = self.dim_shape
        dl = self.dim_latent
        df = self.df
        norm = v_conf["norm"]
        
        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 24)

class AutoEncoder_bigger_decoder(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_bigger_decoder, self).__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        
        dd = 512
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, dd, kernel_size=1, stride=1, padding=0),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(dd, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )
        
class AutoEncoder_more_attn(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_more_attn, self).__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        
        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 28)

class AutoEncoder_bigger(AutoEncoder_graph_flattened_plus):
    def __init__(self,
                v_conf,
                ):
        super(AutoEncoder_bigger, self).__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        
        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 28)

        dd = 512
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, dd, kernel_size=1, stride=1, padding=0),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(dd, dd, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(dd, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )

class AutoEncoder_context(AutoEncoder_graph_flattened_plus):
    def __init__(self, v_conf):
        super().__init__(v_conf)

        layer = nn.TransformerDecoderLayer(self.df, 8, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, 8)

        self.global_feature = nn.Parameter(torch.randn((1, self.df), dtype=torch.float32))

        bd = 1024
        norm = v_conf["norm"]
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(self.df * 3, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, self.dim_shape * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )


    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature, v_gf):
        true_gf = v_gf[v_edge_face_connectivity[:, 1]]
        false_gf = v_gf[v_zero_positions[:, 0]]
        gf = torch.cat((true_gf, false_gf), dim=0).unsqueeze(2)
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
        id_false_start = true_intersection_embedding.shape[0]
        intersection_embedding = rearrange(
            intersection_embedding,
            'b c n h w -> b c (n h w)', c=2
        )

        features = intersection_embedding + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = torch.cat((features, gf), dim=1)
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)
        
        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)
        
        return loss_edge, features[:id_false_start]


    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features.reshape(-1, 2*2*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        fused_face_features = self.face_attn(x, v_data["attn_mask"])
        # Global
        bs = v_data["num_face_record"].shape[0]
        max_faces = v_data["num_face_record"].max()
        gf = self.global_feature.repeat(bs,1).unsqueeze(1)
        # face_batched = torch.ones((bs, max_faces, self.df), dtype=self.global_feature.dtype, device=fused_face_features.device)
        # # Build the batched face features according to the mask
        # face_batched[v_data["valid_mask"]] *= fused_face_features
        # face_batched[torch.logical_not(v_data["valid_mask"])] *= 0.
        # gf = self.cross_attn(tgt=gf, memory=face_batched, memory_key_padding_mask=torch.logical_not(v_data["valid_mask"]),)[:,0]
        gf = gf[:,0]
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)

        fused_face_features = rearrange(fused_face_features, 'b (n h w) -> b n h w', h=2, w=2)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
            gf
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        
        if v_test:
            data = {}
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = fused_face_features[indexes]
            feature_pair = rearrange(
                feature_pair,
                'b c n h w -> b c (n h w)', c=2
            )
            feature_pair = feature_pair + self.face_pos_embedding2[None, :]
            feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
            feature_pair = torch.cat((feature_pair, gf[indexes[:,0]][:,:,None]), dim=1)
            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5

            pre_edge_coords = self.edge_coords_decoder(feature_pair[pred])
            pred_edge_face_connectivity = torch.cat((torch.arange(pre_edge_coords.shape[0], device=device)[:,None], indexes[pred]), dim=1)

            data.update({
                "gt_face_adj": face_adj.cpu().numpy(),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": v_data["edge_points"].cpu().numpy(),
                "gt_face": v_data["face_points"].cpu().numpy(),

                "pred_face_adj": pred.reshape(num_faces, num_faces).cpu().numpy(),
                "pred_edge_face_connectivity": pred_edge_face_connectivity.cpu().numpy(),
                "pred_edge": pre_edge_coords.cpu().numpy(),
                "pred_face": pre_face_coords.cpu().numpy(),

                "face_features": fused_face_features.cpu().numpy()
            })
            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

class AutoEncoder_context_larger_encoder(AutoEncoder_context):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        ds = self.dim_shape
        dl = self.dim_latent
        norm = v_conf["norm"]
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=3, stride=1, padding=1),
            Rearrange('... n w h -> ... w h n',),
        )
        self.edge_coords = nn.Sequential(
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm), 
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm), 
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n',),
        )

class AutoEncoder_context_larger_attn1(AutoEncoder_context):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        df = self.df
        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 24)

class AutoEncoder_context_larger_attn2(AutoEncoder_context):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        df = self.df
        layer = nn.TransformerEncoderLayer(
            768, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.p_embed1 = nn.Linear(df, 768)
        self.face_attn = Attn_fuser(layer, 12)
        self.p_embed2 = nn.Linear(768, df)


    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features.reshape(-1, 2*2*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        fused_face_features = self.p_embed1(x)
        fused_face_features = self.face_attn(fused_face_features, v_data["attn_mask"])
        fused_face_features = self.p_embed2(fused_face_features)
        # Global
        bs = v_data["num_face_record"].shape[0]
        max_faces = v_data["num_face_record"].max()
        gf = self.global_feature.repeat(bs,1).unsqueeze(1)
        # face_batched = torch.ones((bs, max_faces, self.df), dtype=self.global_feature.dtype, device=fused_face_features.device)
        # # Build the batched face features according to the mask
        # face_batched[v_data["valid_mask"]] *= fused_face_features
        # face_batched[torch.logical_not(v_data["valid_mask"])] *= 0.
        # gf = self.cross_attn(tgt=gf, memory=face_batched, memory_key_padding_mask=torch.logical_not(v_data["valid_mask"]),)[:,0]
        gf = gf[:,0]
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)

        fused_face_features = rearrange(fused_face_features, 'b (n h w) -> b n h w', h=2, w=2)
        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
            gf
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        
        if v_test:
            data = {}
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = fused_face_features[indexes]
            feature_pair = rearrange(
                feature_pair,
                'b c n h w -> b c (n h w)', c=2
            )
            feature_pair = feature_pair + self.face_pos_embedding2[None, :]
            feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
            feature_pair = torch.cat((feature_pair, gf[indexes[:,0]][:,:,None]), dim=1)
            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5

            pre_edge_coords = self.edge_coords_decoder(feature_pair[pred])
            pred_edge_face_connectivity = torch.cat((torch.arange(pre_edge_coords.shape[0], device=device)[:,None], indexes[pred]), dim=1)

            data.update({
                "gt_face_adj": face_adj.cpu().numpy(),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": v_data["edge_points"].cpu().numpy(),
                "gt_face": v_data["face_points"].cpu().numpy(),

                "pred_face_adj": pred.reshape(num_faces, num_faces).cpu().numpy(),
                "pred_edge_face_connectivity": pred_edge_face_connectivity.cpu().numpy(),
                "pred_edge": pre_edge_coords.cpu().numpy(),
                "pred_face": pre_face_coords.cpu().numpy(),

                "face_features": fused_face_features.cpu().numpy()
            })
            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

class AutoEncoder_context_fsq(AutoEncoder_context):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        # self.quantizer = FSQ(
        #     dim = self.dim_latent * 2 * 2,
        #     levels = [8, 8, 8, 5, 5, 5],
        # )
        self.quantizer = ResidualFSQ(
            dim = self.dim_latent * 2 * 2,
            levels = [8, 8, 8, 5, 5, 5],
            num_quantizers = 4
        )

    def code(self, fused_face_features):
        fused_face_features = rearrange(fused_face_features, 'b n c -> b 1 (n c)')
        fused_face_features, indices = self.quantizer(fused_face_features)
        fused_face_features = rearrange(fused_face_features, 'b 1 (n c) -> b n c', c=self.dim_latent)
        code_book_loss = None
        return fused_face_features, indices, code_book_loss

    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]
        timer = add_timer(self.time_statics, "encode", timer)

        # Face graph
        x = face_features.reshape(-1, 2*2*self.dim_latent)
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        fused_face_features = self.face_attn(x, v_data["attn_mask"])
        fused_face_features = rearrange(fused_face_features, 'b (n h w) -> b n (h w)', h=2, w=2)
        fused_face_features = rearrange(fused_face_features, 'b n c -> b c n')
        fused_face_features, indices, code_book_loss = self.code(fused_face_features)
        fused_face_features = rearrange(fused_face_features, 'b (h w) n -> b n h w', h=2, w=2)
        
        # Global
        bs = v_data["num_face_record"].shape[0]
        max_faces = v_data["num_face_record"].max()
        gf = self.global_feature.repeat(bs,1).unsqueeze(1)
        # face_batched = torch.ones((bs, max_faces, self.df), dtype=self.global_feature.dtype, device=fused_face_features.device)
        # # Build the batched face features according to the mask
        # face_batched[v_data["valid_mask"]] *= fused_face_features
        # face_batched[torch.logical_not(v_data["valid_mask"])] *= 0.
        # gf = self.cross_attn(tgt=gf, memory=face_batched, memory_key_padding_mask=torch.logical_not(v_data["valid_mask"]),)[:,0]
        gf = gf[:,0]
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)

        timer = add_timer(self.time_statics, "graph", timer)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)
        timer = add_timer(self.time_statics, "normal decoding", timer)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity, 
            v_data["zero_positions"], 
            fused_face_features, 
            gf
        )
        timer = add_timer(self.time_statics, "intersection", timer)

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)
        timer = add_timer(self.time_statics, "intersection decoding", timer)

        # Loss
        loss={}
        if code_book_loss is not None:
            loss["code_book"] = code_book_loss
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords_ori"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        timer = add_timer(self.time_statics, "loss", timer)

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords
        
        if v_test:
            data = {}
            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            feature_pair = fused_face_features[indexes]
            feature_pair = rearrange(
                feature_pair,
                'b c n h w -> b c (n h w)', c=2
            )
            feature_pair = feature_pair + self.face_pos_embedding2[None, :]
            feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
            feature_pair = torch.cat((feature_pair, gf[indexes[:,0]][:,:,None]), dim=1)
            feature_pair = self.edge_feature_proj(feature_pair)
            pred = self.classifier(feature_pair)[...,0]
            pred = torch.sigmoid(pred) > 0.5

            pre_edge_coords = self.edge_coords_decoder(feature_pair[pred])
            pred_edge_face_connectivity = torch.cat((torch.arange(pre_edge_coords.shape[0], device=device)[:,None], indexes[pred]), dim=1)

            data.update({
                "gt_face_adj": face_adj.cpu().numpy(),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": v_data["edge_points"].cpu().numpy(),
                "gt_face": v_data["face_points"].cpu().numpy(),

                "pred_face_adj": pred.reshape(num_faces, num_faces).cpu().numpy(),
                "pred_edge_face_connectivity": pred_edge_face_connectivity.cpu().numpy(),
                "pred_edge": pre_edge_coords.cpu().numpy(),
                "pred_face": pre_face_coords.cpu().numpy(),

                "face_features": fused_face_features.cpu().numpy()
            })
            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

class AutoEncoder_context_lfq(AutoEncoder_context_fsq):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        # self.quantizer = FSQ(
        #     dim = self.dim_latent,
        #     levels = [8, 8, 8, 5, 5, 5],
        # )
        self.quantizer = ResidualLFQ(
            dim = self.dim_latent * 2 * 2,
            num_quantizers = 4,
            codebook_size = 16384,
        )

    def code(self, fused_face_features):
        fused_face_features = rearrange(fused_face_features, 'b n c -> b 1 (n c)')
        fused_face_features, indices, code_book_loss = self.quantizer(fused_face_features)
        fused_face_features = rearrange(fused_face_features, 'b 1 (n c) -> b n c', c=self.dim_latent)
        return fused_face_features, indices, code_book_loss.mean()

class AutoEncoder_context_vq(AutoEncoder_context_lfq):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        # self.quantizer = FSQ(
        #     dim = self.dim_latent,
        #     levels = [8, 8, 8, 5, 5, 5],
        # )
        self.quantizer = ResidualVQ(
            dim = self.dim_latent * 2 * 2,
            num_quantizers = 4,
            codebook_dim = 16,
            codebook_size = 16384,
            shared_codebook = True,
        )

    def code(self, fused_face_features):
        fused_face_features = rearrange(fused_face_features, 'b n c -> b 1 (n c)')
        fused_face_features, indices, code_book_loss = self.quantizer(fused_face_features)
        fused_face_features = rearrange(fused_face_features, 'b 1 (n c) -> b n c', c=self.dim_latent)
        return fused_face_features, indices, code_book_loss.mean()
# Continuous
class AutoEncoder_pure(nn.Module):
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
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.face_coords = nn.Sequential(
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # 4
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((3, 3)), # 2
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        )
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(size=(4, 4), mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=3),
        ) 
        
        self.time_statics = {}

    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = self.face_coords(face_features)

        pre_face_coords = self.face_coords_decoder(face_features)

        # Loss
        loss={}
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        if v_test:
            data["pred_face"] = pre_face_coords.detach().cpu().numpy()
            data["gt_face"] = v_data["face_points"].detach().cpu().numpy()

        return loss, data

# Continuous with center and points loss
class AutoEncoder_pure2(AutoEncoder_pure):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        self.face_points_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(size=(4, 4), mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=3),
        )
        self.face_center_scale_decoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )
        
        self.time_statics = {}


    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        bottleneck_feature = self.face_coords(face_features)

        face_points = self.face_points_decoder(bottleneck_feature)
        face_center_scale = self.face_center_scale_decoder(bottleneck_feature)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]
        pred_face = denormalize_coord(face_points, face_center, face_scale)

        # Loss
        loss={}
        loss["face_coords"] = nn.functional.l1_loss(
            pred_face,
            v_data["face_points"]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        if v_test:
            data["gt_face"] = v_data["face_points"].cpu().numpy()
            data["pred_face"] = pred_face.cpu().numpy()

        return loss, data

# Continuous with center and individual loss
class AutoEncoder_pure3(AutoEncoder_pure2):
    def __init__(self, v_conf):
        super().__init__(v_conf)

    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        bottleneck_feature = self.face_coords(face_features)

        face_points = self.face_points_decoder(bottleneck_feature)
        face_center_scale = self.face_center_scale_decoder(bottleneck_feature)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        # Loss
        loss={}
        loss["face_coords_norm"] = nn.functional.l1_loss(
            face_points,
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
        loss["total_loss"] = sum(loss.values())

        data = {}
        if v_test:
            data["gt_face"] = v_data["face_points"].cpu().numpy()
            pred_face = denormalize_coord(face_points,face_center,face_scale)
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face,
                v_data["face_points"]
            )
            data["pred_face"] = pred_face.cpu().numpy()

        return loss, data
    
# Discrete with individual loss
class AutoEncoder_pure4(AutoEncoder_pure):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        discrete_dim = 256
        self.face_points_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(size=(4, 4), mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ds, 3 * 256, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c d) w h -> ... w h c d',c=3),
        )
        self.face_center_scale_decoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ds, 3 * 2 * 256, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c d n) w h -> ... (w h c) d n', c=3, n=2),
        )
        self.time_statics = {}

    def forward(self, v_data, v_test=False):
        timer = time.time()
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        bottleneck_feature = self.face_coords(face_features)

        face_points = self.face_points_decoder(bottleneck_feature)
        face_center_scale = self.face_center_scale_decoder(bottleneck_feature)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]
        # Loss
        loss={}
        loss["discrete_face_points"] = nn.functional.cross_entropy(
            face_points.reshape(-1, 256),
            v_data["face_points_discrete"].reshape(-1)
        )
        loss["discrete_face_center"] = nn.functional.cross_entropy(
            face_center.reshape(-1, 256),
            v_data["face_center_discrete"].reshape(-1)
        )
        loss["discrete_face_scale"] = nn.functional.cross_entropy(
            face_scale.reshape(-1, 256),
            v_data["face_scale_discrete"].reshape(-1)
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        if v_test:
            face_points = face_points.argmax(dim=-1)
            face_center = face_center.argmax(dim=-1)
            face_scale = face_scale.argmax(dim=-1)
            pred_face = denormalize_coord(*continuous_coord(face_points, face_center, face_scale, 256))
            data["gt_face"] = denormalize_coord(*continuous_coord(
                v_data["face_points_discrete"], 
                v_data["face_center_discrete"], 
                v_data["face_scale_discrete"], 256)).cpu().numpy()
            loss["face_coords"] = nn.functional.l1_loss(
                pred_face,
                v_data["face_points"]
            )
            data["pred_face"]=pred_face.cpu().numpy()

        return loss, data
    

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
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 4, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
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
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 4, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 2, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df, 
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())
        
        bd = 768 # bottlenek_dim
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
        self.classifier = nn.Linear(df*2, 1)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"), # 4
            nn.Conv2d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"), # 8
            nn.Conv2d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"), # 16
            nn.Conv2d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=3),
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
            nn.Upsample(scale_factor=2, mode="linear"), # 4
            nn.Conv1d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="linear"), # 8
            nn.Conv1d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="linear"), # 16
            nn.Conv1d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=3),
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
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.Linear(self.df*2, self.df*2),
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
            return v_fused_face_features, torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0,0])

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
        edge_index=edge_face_connectivity[:, 1:].permute(1,0)
        edge_attr=edge_features[edge_face_connectivity[:, 0]]
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
        edge_scale = torch.sigmoid(edge_center_scale[..., 1])  * 2

        edge_points_local1 = self.edge_points_decoder(edge_features)
        edge_center_scale1 = self.edge_center_scale_decoder(edge_features)
        edge_center1 = edge_center_scale1[..., 0]
        edge_scale1 = torch.sigmoid(edge_center_scale1[..., 1]) * 2
        self.times["Decoder"] += time.time() - timer
        timer = time.time()

        # Loss
        loss={}
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
            
            data["face_features"] = face_z
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

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1,2).to(device)
        feature_pair = v_face_features[indexes]

        feature_pair = feature_pair + self.face_pos_embedding2[None, :]
        feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
        feature_pair = self.edge_feature_proj(feature_pair)
        pred = self.classifier(feature_pair)[...,0]
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

        pred_edge_face_connectivity = torch.cat((torch.arange(pred_edge_points.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        return {
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }
    

class AutoEncoder_0925(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = 3
        # self.in_channels = v_conf["in_channels"]

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
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 4, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds // 2, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
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
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds // 4, ds // 4, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 4, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 2, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df, 
                heads=1, edge_dim=df * 2,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())
        
        bd = 768 # bottlenek_dim
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
        self.classifier = nn.Linear(df*2, 1)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ds, ds // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ds // 2, ds // 2, kernel_size=2, stride=2),
            nn.Conv2d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ds // 4, ds // 4, kernel_size=2, stride=2),
            nn.Conv2d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ds // 8, ds // 8, kernel_size=2, stride=2),
            nn.Conv2d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=3),
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
            nn.Conv1d(ds // 2, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 2, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds // 4, ds // 4, kernel_size=2, stride=2),
            nn.Conv1d(ds // 4, ds // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 4, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds // 8, ds // 8, kernel_size=2, stride=2),
            nn.Conv1d(ds // 8, ds // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds // 8, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=3),
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
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.LayerNorm(self.df*2),
                nn.Linear(self.df*2, self.df*2),
            )

        self.times = {
            "Encoder": 0,
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
            return v_fused_face_features, torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0,0])

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        return fused_face_features, kl_loss

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        loss, recon_data = self.loss(v_data, encoding_result)
        data = {}
        if v_test:
            pred_data = self.inference(encoding_result["face_z"])
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
                v_data["face_points"][..., :self.in_channels]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord(recon_data["edge_points_local"], recon_data["edge_center"], recon_data["edge_scale"]),
                v_data["edge_points"][..., :self.in_channels][v_data["edge_face_connectivity"][:, 0]]
            )
            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord(recon_data["edge_points_local1"], recon_data["edge_center1"], recon_data["edge_scale1"]),
                v_data["edge_points"][..., :self.in_channels]
            )

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1,2).to(device)
        feature_pair = v_face_features[indexes]

        feature_pair = feature_pair + self.face_pos_embedding2[None, :]
        feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
        feature_pair = self.edge_feature_proj(feature_pair)
        pred = self.classifier(feature_pair)[...,0]
        pred_labels = torch.sigmoid(pred) > 0.5
        
        edge_points_local = self.edge_points_decoder(feature_pair[pred_labels])
        edge_center_scale = self.edge_center_scale_decoder(feature_pair[pred_labels])
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]
        pred_edge_points = denormalize_coord(edge_points_local, edge_center, edge_scale)

        face_points_local = self.face_points_decoder(v_face_features)
        face_center_scale = self.face_center_scale_decoder(v_face_features)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]
        pred_face_points = denormalize_coord(face_points_local, face_center, face_scale)

        pred_edge_face_connectivity = torch.cat((torch.arange(pred_edge_points.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        return {
            "face_features": v_face_features,
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_face_adj_prob": torch.sigmoid(pred).reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }

    def encode(self, v_data, v_test):
        # torch.cuda.synchronize()
        # Encoder
        # timer = time.time()
        face_features = self.face_conv1(v_data["face_points"][...,:self.in_channels])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"][...,:self.in_channels])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)
        # timer = self.profile_time(timer, "Encoder")

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

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=x.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features, index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z)
        # timer = self.profile_time(timer, "Fuser")

        face_z, kl_loss = self.sample(face_z, v_is_test=v_test)
        # timer = self.profile_time(timer, "Sample")
        return {
            "face_z": face_z,
            "kl_loss": kl_loss,
            "edge_features": edge_features,
        }

    def loss(self, v_data, v_encoding_result):
        face_z = v_encoding_result["face_z"]
        edge_face_connectivity = v_data["edge_face_connectivity"]
        edge_features = v_encoding_result["edge_features"]
        kl_loss = v_encoding_result["kl_loss"]

        # Intersection
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            face_z,
        )
        # timer = self.profile_time(timer, "Intersection")

        face_points_local = self.face_points_decoder(face_z)
        face_center_scale = self.face_center_scale_decoder(face_z)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_points_decoder(intersected_edge_feature)
        edge_center_scale = self.edge_center_scale_decoder(intersected_edge_feature)
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]

        edge_points_local1 = self.edge_points_decoder(edge_features)
        edge_center_scale1 = self.edge_center_scale_decoder(edge_features)
        edge_center1 = edge_center_scale1[..., 0]
        edge_scale1 = edge_center_scale1[..., 1]
        # timer = self.profile_time(timer, "Decoder")

        # Loss
        loss={}
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
        # timer = self.profile_time(timer, "Loss")

        recon_data = {}
        recon_data["edge_points_local"] = edge_points_local
        recon_data["edge_center"] = edge_center
        recon_data["edge_scale"] = edge_scale

        recon_data["edge_points_local1"] = edge_points_local1
        recon_data["edge_center1"] = edge_center1
        recon_data["edge_scale1"] = edge_scale1
        return loss, recon_data
    

class AutoEncoder_context_KL(nn.Module):
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
            nn.Conv2d(3, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        )  # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=3, stride=1, padding=1),
            Rearrange('... n w h -> ... w h n', ),
        )
        self.face_pos_embedding = nn.Parameter(torch.randn(ds, 16, 16))
        self.face_pos_embedding2 = nn.Parameter(torch.randn(2, df))

        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(3, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.edge_coords = nn.Sequential(
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )  # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"),  # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"),  # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Upsample(scale_factor=2, mode="linear"),  # 16
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n', ),
        )
        self.edge_pos_embedding = nn.Parameter(torch.randn(ds, 16))

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=ds * 2,
            ))
            self.graph_face_edge.append(nn.ReLU())

        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 12)

        bd = 1024  # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 3, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, ds * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(ds * 2, 1)

        layer = nn.TransformerDecoderLayer(self.df, 8, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, 8)

        self.global_feature = nn.Parameter(torch.randn((1, self.df), dtype=torch.float32))

        self.gaussian_proj = nn.Linear(self.df, self.df * 2)
        self.gaussian_weights = 1e-6
        # self.gaussian_weights = v_conf["gaussian_weights"]

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature, v_gf):
        true_gf = v_gf[v_edge_face_connectivity[:, 1]]
        false_gf = v_gf[v_zero_positions[:, 0]]
        gf = torch.cat((true_gf, false_gf), dim=0).unsqueeze(2)
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
        id_false_start = true_intersection_embedding.shape[0]
        intersection_embedding = rearrange(
            intersection_embedding,
            'b c n h w -> b c (n h w)', c=2
        )

        features = intersection_embedding + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = torch.cat((features, gf), dim=1)
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)

        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

        return loss_edge, features[:id_false_start]

    def forward(self, v_data, v_test=False):
        # Encoder
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)

        edge_face_connectivity = v_data["edge_face_connectivity"]

        # Face graph
        x = face_features.reshape(-1, 2 * 2 * self.dim_latent)
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        fused_face_features = self.face_attn(x, v_data["attn_mask"])
        fused_face_features_gau = self.gaussian_proj(fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)

        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if v_test:
            fused_face_features = mean
        else:
            fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights

        # Global
        gf = self.global_feature.repeat((v_data["num_face_record"], 1))

        fused_face_features = rearrange(fused_face_features, 'b (n h w) -> b n h w', h=2, w=2)

        pre_face_coords = self.face_coords_decoder(fused_face_features)
        pre_edge_coords1 = self.edge_coords_decoder(edge_features)

        # Intersection
        gt_edge_points = v_data["edge_points"][edge_face_connectivity[:, 0]]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            fused_face_features,
            gf
        )

        pre_edge_coords = self.edge_coords_decoder(intersected_edge_feature)

        # Loss
        loss = {}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"]
        )
        loss["edge_coords"] = nn.functional.l1_loss(
            pre_edge_coords,
            gt_edge_points
        )
        loss["edge_coords1"] = nn.functional.l1_loss(
            pre_edge_coords1,
            v_data["edge_points"]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["kl_loss"] = kl_loss
        loss["total_loss"] = sum(loss.values())

        data = {}
        data["recon_faces"] = pre_face_coords
        data["recon_edges"] = pre_edge_coords

        if v_test:
            data = {}
            fused_face_features = rearrange(fused_face_features, 'b n h w -> b (n h w)', h=2, w=2)
            recon_data = self.inference(fused_face_features)

            device = pre_face_coords.device
            num_faces = v_data["face_points"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True

            data.update({
                "gt_face_adj": face_adj,
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": v_data["edge_points"].cpu().numpy(),
                "gt_face": v_data["face_points"].cpu().numpy(),

                "pred_face_adj": recon_data["pred_face_adj"],
                "pred_edge_face_connectivity": recon_data["pred_edge_face_connectivity"],
                "pred_face_adj_prob": recon_data["pred_face_adj_prob"],
                "pred_edge": recon_data["pred_edge"],
                "pred_face": recon_data["pred_face"],

                "face_features": recon_data["face_features"]
            })
            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1, 2).to(device)
        feature_pair = v_face_features[indexes]

        gf = self.global_feature.repeat(feature_pair.shape[0], 1).unsqueeze(-1)

        features = feature_pair + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = torch.cat((features, gf), dim=1)
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)[:, 0]
        pred_labels = torch.sigmoid(pred) > 0.5

        fused_face_features = rearrange(v_face_features, 'b (n h w) -> b n h w', h=2, w=2)
        pred_face_points = self.face_coords_decoder(fused_face_features)
        pred_edge_points = self.edge_coords_decoder(features[pred_labels])

        pred_edge_face_connectivity = torch.cat(
            (torch.arange(pred_edge_points.shape[0], device=device)[:, None], indexes[pred_labels]), dim=1)
        return {
            "face_features": v_face_features,
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_face_adj_prob": torch.sigmoid(pred).reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }

# Modified version of AutoEncoder_context_KL
class AutoEncoder_0929(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = 3
        # self.in_channels = v_conf["in_channels"]

        self.face_conv1 = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        )  # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=3, stride=1, padding=1),
            Rearrange('... n w h -> ... w h n', ),
        )
        self.face_center_scale_decoder = nn.Sequential(
            Rearrange("b n -> b n 1 1"),
            nn.Conv2d(dl * 2 * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )
        self.face_pos_embedding = nn.Parameter(torch.randn(ds, 16, 16))
        self.face_pos_embedding2 = nn.Parameter(torch.randn(2, df))

        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.edge_coords = nn.Sequential(
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )  # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=ds, w=2),
            nn.Conv1d(ds, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n', ),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            Rearrange("b n-> b n 1"),
            nn.Conv1d(ds * 2, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w -> ... (w c) n', c=3, n=2),
        )
        self.edge_pos_embedding = nn.Parameter(torch.randn(ds, 16))

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                df, df,
                heads=1, edge_dim=ds * 2,
            ))
            self.graph_face_edge.append(nn.ReLU())

        layer = nn.TransformerEncoderLayer(
            df, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = Attn_fuser(layer, 12)

        bd = 1024  # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, ds * 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(ds * 2, 1)

        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Linear(self.df, self.df * 2)
        else:
            self.gaussian_proj = nn.Linear(self.df, self.df)
        self.with_sigmoid = v_conf["sigmoid"]

        self.times = {
            "Encoder": 0,
            "Fuse": 0,
            "Intersection": 0,
            "Decoder": 0,
            "Loss": 0,
        }

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
        id_false_start = true_intersection_embedding.shape[0]
        intersection_embedding = rearrange(
            intersection_embedding,
            'b c n h w -> b c (n h w)', c=2
        )

        features = intersection_embedding + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)

        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

        return loss_edge, features[:id_false_start]

    def forward(self, v_data, v_test=False):
        # torch.cuda.synchronize()
        # timer = time.time()
        # Encoder
        num_faces = v_data["face_points"].shape[0]
        face_points = v_data["face_points"][:num_faces]
        gt_face_points_local = v_data["face_points_norm"]
        num_edges = v_data["edge_points"].shape[0]
        edge_points = v_data["edge_points"]
        gt_edge_points_local = v_data["edge_points_norm"]

        face_features = self.face_conv1(face_points[...,:self.in_channels])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(edge_points[...,:self.in_channels])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)
        # timer = profile_time(self.times, "Encoder", timer)

        # Face graph
        edge_face_connectivity = v_data["edge_face_connectivity"]
        x = face_features.reshape(-1, 2 * 2 * self.dim_latent)
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_features[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)

        # Face attn
        fused_face_features = self.face_attn(x, v_data["attn_mask"])
        fused_face_features_gau = self.gaussian_proj(fused_face_features)
        if self.gaussian_weights > 0:
            fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)

            mean = fused_face_features_gau[:, :, 0]
            logvar = fused_face_features_gau[:, :, 1]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            if v_test:
                fused_face_features = mean
            else:
                fused_face_features = eps.mul(std).add_(mean)
            kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        else:
            if self.with_sigmoid:
                fused_face_features = torch.sigmoid(fused_face_features_gau) * 2 - 1
            else:
                fused_face_features = fused_face_features_gau

        # Global
        # timer = profile_time(self.times, "Fuse", timer)
        fused_face_features = rearrange(fused_face_features, 'b (n h w) -> b n h w', h=2, w=2)

        # Intersection
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            fused_face_features
        )
        # timer = profile_time(self.times, "Intersection", timer)

        face_points_local = self.face_coords_decoder(fused_face_features)
        face_center_scale = self.face_center_scale_decoder(fused_face_features.reshape(-1, self.df))
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_coords_decoder(intersected_edge_feature)
        edge_center_scale = self.edge_center_scale_decoder(intersected_edge_feature)
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]

        edge_points_local1 = self.edge_coords_decoder(edge_features)
        edge_center_scale1 = self.edge_center_scale_decoder(edge_features)
        edge_center1 = edge_center_scale1[..., 0]
        edge_scale1 = edge_center_scale1[..., 1]
        # timer = profile_time(self.times, "Decoder", timer)

        # Loss
        loss = {}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords_norm"] = nn.functional.l1_loss(
            face_points_local,
            gt_face_points_local
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
            gt_edge_points_local
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
            gt_edge_points_local[edge_face_connectivity[:, 0]]
        )
        loss["edge_center"] = nn.functional.l1_loss(
            edge_center,
            v_data["edge_center"][edge_face_connectivity[:, 0]]
        )
        loss["edge_scale"] = nn.functional.l1_loss(
            edge_scale,
            v_data["edge_scale"][edge_face_connectivity[:, 0]]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = kl_loss
        loss["total_loss"] = sum(loss.values())
        # timer = profile_time(self.times, "Loss", timer)

        data = {}

        if v_test:
            data = {}
            fused_face_features = rearrange(fused_face_features, 'b n h w -> b (n h w)', h=2, w=2)
            recon_data = self.inference(fused_face_features)

            device = v_data["face_points"].device
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True

            data.update({
                "gt_face_adj": face_adj.reshape(-1),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": edge_points.cpu().numpy(),
                "gt_face": face_points.cpu().numpy(),

                "pred_face_adj": recon_data["pred_face_adj"],
                "pred_face_adj_prob": recon_data["pred_face_adj_prob"],
                "pred_edge_face_connectivity": recon_data["pred_edge_face_connectivity"],
                "pred_edge": recon_data["pred_edge"],
                "pred_face": recon_data["pred_face"],

                "face_features": recon_data["face_features"]
            })

            loss["face_coords"] = nn.functional.l1_loss(
                data["pred_face"],
                face_points[...,:self.in_channels]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local, edge_center, edge_scale),
                edge_points[v_data["edge_face_connectivity"][:, 0]][...,:self.in_channels]
            )
            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local1, edge_center1, edge_scale1),
                edge_points[...,:self.in_channels]
            )

            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1, 2).to(device)
        feature_pair = v_face_features[indexes]

        features = feature_pair + self.face_pos_embedding2[None, :]
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)[:, 0]
        pred_labels = torch.sigmoid(pred) > 0.5

        fused_face_features = rearrange(v_face_features, 'b (n h w) -> b n h w', h=2, w=2)

        face_points_local = self.face_coords_decoder(fused_face_features)
        face_center_scale = self.face_center_scale_decoder(fused_face_features.reshape(-1, self.df))
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_coords_decoder(features[pred_labels])
        edge_center_scale = self.edge_center_scale_decoder(features[pred_labels])
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]

        pred_face_points = denormalize_coord(face_points_local, face_center, face_scale)
        pred_edge_points = denormalize_coord(edge_points_local, edge_center, edge_scale)

        pred_edge_face_connectivity = torch.cat(
            (torch.arange(pred_edge_points.shape[0], device=device)[:, None], indexes[pred_labels]), dim=1)
        return {
            "face_features": v_face_features,
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_face_adj_prob": torch.sigmoid(pred).reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }

# Add residual after attn
class AutoEncoder_1003(AutoEncoder_0925):
    def __init__(self, v_conf):
        super().__init__(v_conf)
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Linear(self.df, self.df * 2)
        else:
            self.gaussian_proj = nn.Linear(self.df, self.df)
        self.with_sigmoid = v_conf["sigmoid"]

        bd = 768
        layer = nn.TransformerEncoderLayer(
            bd, 8, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 24, nn.LayerNorm(768))


    def sample(self, v_fused_face_features, v_is_test=False):
        if self.gaussian_weights <= 0:
            if self.with_sigmoid:
                return torch.sigmoid(v_fused_face_features) * 2 - 1, torch.zeros_like(v_fused_face_features[0,0])
            else:
                return v_fused_face_features, torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)
        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0,0])

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        return fused_face_features, kl_loss

    def encode(self, v_data, v_test):
        # torch.cuda.synchronize()
        # Encoder
        # timer = time.time()
        face_features = self.face_conv1(v_data["face_points"])
        face_features = face_features + self.face_pos_embedding
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(v_data["edge_points"])
        edge_features = edge_features + self.edge_pos_embedding
        edge_features = self.edge_coords(edge_features)
        # timer = self.profile_time(timer, "Encoder")

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
        x_in = x
        x = self.face_attn_proj_in(x)
        x = self.face_attn(x, v_data["attn_mask"])
        x = self.face_attn_proj_out(x)
        fused_face_features = x + x_in

        # Global
        bs = v_data["num_face_record"].shape[0]
        index = torch.arange(bs, device=x.device).repeat_interleave(v_data["num_face_record"])
        face_z = fused_face_features
        gf = scatter_mean(fused_face_features.detach(), index, dim=0)
        gf = self.global_feature1(gf)
        gf = gf.repeat_interleave(v_data["num_face_record"], dim=0)
        face_z = torch.cat((fused_face_features, gf), dim=1)
        face_z = self.global_feature2(face_z)
        face_z = face_z + fused_face_features
        # timer = self.profile_time(timer, "Fuser")

        face_z, kl_loss = self.sample(face_z, v_is_test=v_test)
        # timer = self.profile_time(timer, "Sample")
        return {
            "face_z": face_z,
            "kl_loss": kl_loss,
            "edge_features": edge_features,
        }


class AutoEncoder_Test(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df
        in_channels = v_conf["in_channels"]
        self.face_conv1 = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.face_coords = nn.Sequential(
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
        )  # b c 4 4
        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=3, stride=1, padding=1),
            Rearrange('... n w h -> ... w h n', ),
        )
        self.face_center_scale_decoder = nn.Sequential(
            Rearrange("b n -> b n 1 1"),
            nn.Conv2d(dl * 2 * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )

        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(in_channels, ds // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.edge_coords = nn.Sequential(
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 8
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 4
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 2
            res_block_1D(ds // 2, ds // 2, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds // 2, ds // 2, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )  # b c 1
        self.edge_coords_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=128, w=2),
            nn.Conv1d(128, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w -> ... w n', ),
        )
        self.edge_center_scale_decoder = nn.Sequential(
            Rearrange("b n-> b n 1"),
            nn.Conv1d(256, ds, kernel_size=1, stride=1, padding=0),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w -> ... (w c) n', c=3, n=2),
        )

        bd = 1024
        self.attn_in = nn.Linear(df, bd)
        self.attn_out = nn.Linear(bd, df)
        layer = nn.TransformerEncoderLayer(
            bd, 8, dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))

        bd = 1024  # bottlenek_dim
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, 256, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(256, 1)

        self.times = {
            "Encoder": 0,
            "Fuse": 0,
            "Intersection": 0,
            "Decoder": 0,
            "Loss": 0,
        }

    def intersection(self, v_edge_face_connectivity, v_zero_positions, v_face_feature):
        true_intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]
        false_intersection_embedding = v_face_feature[v_zero_positions]

        intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
        id_false_start = true_intersection_embedding.shape[0]
        intersection_embedding = rearrange(
            intersection_embedding,
            'b n c -> b (n c) 1'
        )

        features = self.edge_feature_proj(intersection_embedding)
        pred = self.classifier(features)

        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

        return loss_edge, features[:id_false_start]

    def forward(self, v_data, v_test=False):
        # torch.cuda.synchronize()
        # timer = time.time()
        # Encoder
        num_faces = v_data["face_points"].shape[0]
        face_points = v_data["face_points"][:num_faces]
        gt_face_points_local = v_data["face_points_norm"]
        num_edges = v_data["edge_points"].shape[0]
        edge_points = v_data["edge_points"]
        gt_edge_points_local = v_data["edge_points_norm"]

        face_features = self.face_conv1(face_points)
        face_features = self.face_coords(face_features)

        edge_features = self.edge_conv1(edge_points)
        edge_features = self.edge_coords(edge_features)
        # timer = profile_time(self.times, "Encoder", timer)

        face_features = rearrange(face_features, 'b n h w -> b (n h w)', h=2, w=2)
        face_features = self.attn_in(face_features)
        fused_face_features = self.face_attn(face_features, v_data["attn_mask"])
        fused_face_features = self.attn_out(fused_face_features)
        # fused_face_features = face_features

        # Global
        # timer = profile_time(self.times, "Fuse", timer)
        fused_face_features = torch.sigmoid(fused_face_features) * 2 - 1

        # Intersection
        edge_face_connectivity = v_data["edge_face_connectivity"]
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            fused_face_features
        )
        # timer = profile_time(self.times, "Intersection", timer)

        face_points_local = self.face_coords_decoder(rearrange(fused_face_features, 'b (n h w) -> b n h w', h=2, w=2))
        face_center_scale = self.face_center_scale_decoder(fused_face_features)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_coords_decoder(intersected_edge_feature)
        edge_center_scale = self.edge_center_scale_decoder(intersected_edge_feature)
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]

        edge_points_local1 = self.edge_coords_decoder(edge_features)
        edge_center_scale1 = self.edge_center_scale_decoder(edge_features)
        edge_center1 = edge_center_scale1[..., 0]
        edge_scale1 = edge_center_scale1[..., 1]
        # timer = profile_time(self.times, "Decoder", timer)

        # Loss
        loss = {}
        loss["edge_classification"] = loss_edge_classification * 0.1
        loss["face_coords_norm"] = nn.functional.l1_loss(
            face_points_local,
            gt_face_points_local
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
            gt_edge_points_local
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
            gt_edge_points_local[edge_face_connectivity[:, 0]]
        )
        loss["edge_center"] = nn.functional.l1_loss(
            edge_center,
            v_data["edge_center"][edge_face_connectivity[:, 0]]
        )
        loss["edge_scale"] = nn.functional.l1_loss(
            edge_scale,
            v_data["edge_scale"][edge_face_connectivity[:, 0]]
        )
        loss["edge_feature_loss"] = nn.functional.l1_loss(
            intersected_edge_feature,
            edge_features[edge_face_connectivity[:, 0]]
        )
        loss["total_loss"] = sum(loss.values())
        # timer = profile_time(self.times, "Loss", timer)

        data = {}

        if v_test:
            data = {}
            recon_data = self.inference(fused_face_features)

            device = v_data["face_points"].device
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True

            data.update({
                "gt_face_adj": face_adj.reshape(-1),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].cpu().numpy(),
                "gt_edge": edge_points.cpu().numpy(),
                "gt_face": face_points.cpu().numpy(),

                "pred_face_adj": recon_data["pred_face_adj"],
                "pred_edge_face_connectivity": recon_data["pred_edge_face_connectivity"],
                "pred_edge": recon_data["pred_edge"],
                "pred_face": recon_data["pred_face"],

                "face_features": recon_data["face_features"]
            })

            loss["face_coords"] = nn.functional.l1_loss(
                data["pred_face"],
                face_points[..., :-3]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local, edge_center, edge_scale),
                edge_points[v_data["edge_face_connectivity"][:, 0]][..., :-3]
            )
            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord(edge_points_local1, edge_center1, edge_scale1),
                edge_points[..., :-3]
            )

            data["edge_loss"] = loss["edge_coords"].cpu().numpy()
            data["face_loss"] = loss["face_coords"].cpu().numpy()

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1, 2).to(device)
        feature_pair = v_face_features[indexes]

        features = rearrange(feature_pair, 'b c n -> b (c n) 1')
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)[:, 0]
        pred_labels = torch.sigmoid(pred) > 0.5

        fused_face_features = rearrange(v_face_features, 'b (n h w) -> b n h w', h=2, w=2)

        face_points_local = self.face_coords_decoder(fused_face_features)
        face_center_scale = self.face_center_scale_decoder(fused_face_features.reshape(-1, self.df))
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_coords_decoder(features[pred_labels])
        edge_center_scale = self.edge_center_scale_decoder(features[pred_labels])
        edge_center = edge_center_scale[..., 0]
        edge_scale = edge_center_scale[..., 1]

        pred_face_points = denormalize_coord(face_points_local, face_center, face_scale)
        pred_edge_points = denormalize_coord(edge_points_local, edge_center, edge_scale)

        pred_edge_face_connectivity = torch.cat(
            (torch.arange(pred_edge_points.shape[0], device=device)[:, None], indexes[pred_labels]), dim=1)
        return {
            "face_features": v_face_features,
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_face_adj_prob": torch.sigmoid(pred).reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }


class AutoEncoder_1008(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.in_channels = v_conf["in_channels"]
        self.face_conv1 = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=4, stride=4), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )
        self.edge_conv1 = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=4, stride=4), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, df, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        ) # b c 1

        in_channels_face = self.df + 6
        bd = 768
        self.attn_in = nn.Linear(in_channels_face, bd)
        layer = nn.TransformerEncoderLayer(
            bd, 12, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))
        self.attn_out = nn.Linear(bd, df)

        in_channels_edge = df * 2 + 6
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(df * 2, bd, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, in_channels_edge, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )
        self.classifier = nn.Linear(in_channels_edge, 1)

        # Decoder
        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=3),
        )
        self.face_center_scale_decoder = nn.Sequential(
            Rearrange("b n -> b n 1 1"),
            nn.Conv2d(dl * 2 * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )
        
        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=df, w=2),
            nn.Conv1d(df, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            nn.Conv1d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            nn.Conv1d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            nn.Conv1d(ds, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(ds, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=3),
        )
      
        self.gaussian_weights = v_conf["gaussian_weights"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df*2),
                nn.LeakyReLU(),
                nn.LayerNorm(self.df*2),
                nn.Linear(self.df*2, self.df*2),
            )
        else:
            self.gaussian_proj = nn.Linear(self.df, self.df)
        self.with_sigmoid = v_conf["sigmoid"]

        self.times = {
            "Encoder": 0,
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

        features = intersection_embedding
        features = rearrange(features, 'b c n -> b (c n) 1')
        features = self.edge_feature_proj(features)
        pred = self.classifier(features)

        gt_labels = torch.ones_like(pred)
        gt_labels[id_false_start:] = 0
        loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)
        
        return loss_edge, features[:id_false_start]

    def sample(self, v_fused_face_features, v_is_test=False):
        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)

        if self.gaussian_weights <= 0:
            if self.with_sigmoid:
                return torch.sigmoid(fused_face_features_gau), torch.zeros_like(v_fused_face_features[0,0])
            else:
                return fused_face_features_gau, torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0,0])

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        return fused_face_features, kl_loss

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        loss, recon_data = self.loss(v_data, encoding_result)
        data = {}
        if v_test:
            pred_data = self.inference(encoding_result["face_z"])
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
                v_data["face_points"][..., :3]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                denormalize_coord(recon_data["edge_points_local"], recon_data["edge_center"], recon_data["edge_scale"]),
                v_data["edge_points"][..., :3][v_data["edge_face_connectivity"][:, 0]]
            )
            loss["edge_coords1"] = nn.functional.l1_loss(
                denormalize_coord(recon_data["edge_points_local1"], v_data["edge_center"], v_data["edge_scale"]),
                v_data["edge_points"][..., :3]
            )

        return loss, data

    def inference(self, v_face_features):
        device = v_face_features.device
        num_faces = v_face_features.shape[0]
        indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

        indexes = indexes.reshape(-1,2).to(device)
        feature_pair = v_face_features[indexes]

        feature_pair = feature_pair
        feature_pair = rearrange(feature_pair, 'b c n -> b (c n) 1')
        feature_pair = self.edge_feature_proj(feature_pair)
        pred = self.classifier(feature_pair)[...,0]
        pred_labels = torch.sigmoid(pred) > 0.5
        
        edge_points_local = self.edge_points_decoder(feature_pair[pred_labels][..., :self.df*2])
        edge_center = feature_pair[pred_labels][..., self.df*2:self.df*2+3]
        edge_scale = feature_pair[pred_labels][..., self.df*2+3:]
        pred_edge_points = denormalize_coord(edge_points_local, edge_center, edge_scale)

        face_points_local = self.face_points_decoder(v_face_features)
        face_center_scale = self.face_center_scale_decoder(v_face_features)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]
        pred_face_points = denormalize_coord(face_points_local, face_center, face_scale)

        pred_edge_face_connectivity = torch.cat((torch.arange(pred_edge_points.shape[0], device=device)[:,None], indexes[pred_labels]), dim=1)
        return {
            "face_features": v_face_features,
            "pred_face_adj": pred_labels.reshape(-1),
            "pred_face_adj_prob": torch.sigmoid(pred).reshape(-1),
            "pred_edge_face_connectivity": pred_edge_face_connectivity,
            "pred_face": pred_face_points,
            "pred_edge": pred_edge_points,
        }

    def encode(self, v_data, v_test):
        # torch.cuda.synchronize()
        # Encoder
        # timer = time.time()
        face_features1 = self.face_conv1(v_data["face_points_norm"])
        face_features = torch.cat((face_features1, v_data["face_center"], v_data["face_scale"]), dim=1)

        edge_features1 = self.edge_conv1(v_data["edge_points_norm"])
        edge_features = torch.cat((edge_features1, v_data["edge_center"], v_data["edge_scale"]), dim=1)
        # timer = self.profile_time(timer, "Encoder")

        # Fuser
        edge_face_connectivity = v_data["edge_face_connectivity"]
        # Face graph

        # Face attn
        face_features = self.attn_in(face_features)
        fused_face_features = self.face_attn(face_features, v_data["attn_mask"])
        fused_face_features = self.attn_out(fused_face_features)
        # timer = self.profile_time(timer, "Fuser")

        face_z, kl_loss = self.sample(fused_face_features, v_is_test=v_test)
        # timer = self.profile_time(timer, "Sample")
        return {
            "face_z": face_z,
            "kl_loss": kl_loss,
            "edge_features": edge_features,
        }

    def loss(self, v_data, v_encoding_result):
        face_z = v_encoding_result["face_z"]
        edge_face_connectivity = v_data["edge_face_connectivity"]
        edge_features = v_encoding_result["edge_features"]
        kl_loss = v_encoding_result["kl_loss"]

        # Intersection
        loss_edge_classification, intersected_edge_feature = self.intersection(
            edge_face_connectivity,
            v_data["zero_positions"],
            face_z,
        )
        # timer = self.profile_time(timer, "Intersection")

        face_points_local = self.face_points_decoder(face_z)
        face_center_scale = self.face_center_scale_decoder(face_z)
        face_center = face_center_scale[..., 0]
        face_scale = face_center_scale[..., 1]

        edge_points_local = self.edge_points_decoder(intersected_edge_feature[..., :self.df*2])
        edge_center = intersected_edge_feature[..., self.df*2:self.df*2+3]
        edge_scale = intersected_edge_feature[..., self.df*2+3:]

        edge_points_local1 = self.edge_points_decoder(edge_features[..., :self.df*2])
        # timer = self.profile_time(timer, "Decoder")

        # Loss
        loss={}
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
        # timer = self.profile_time(timer, "Loss")

        recon_data = {}
        recon_data["edge_points_local"] = edge_points_local
        recon_data["edge_center"] = edge_center
        recon_data["edge_scale"] = edge_scale

        recon_data["edge_points_local1"] = edge_points_local1
        return loss, recon_data
    

class EncoderDecoder(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        ds = self.dim_shape
        self.dim_latent = v_conf["dim_latent"]
        dl = self.dim_latent
        norm = v_conf["norm"]

        # Encoder
        self.in_channels = v_conf["in_channels"]
        self.face_conv = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n h w -> b (n h w)")
        )

        self.edge_conv = nn.Sequential(
            Rearrange('b w n -> b n w'),
            nn.Conv1d(self.in_channels, ds, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 8
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 4
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.MaxPool1d(kernel_size=2, stride=2), # 2
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(ds, dl, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )

        self.face_points_decoder = nn.Sequential(
            Rearrange("b (n h w) -> b n h w", h=2, w=2),
            nn.Conv2d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose2d(ds, ds, kernel_size=2, stride=2),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_2D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv2d(ds, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c',c=self.in_channels),
        )

                
        self.edge_points_decoder = nn.Sequential(
            Rearrange("b (n w)-> b n w", n=dl, w=2),
            nn.Conv1d(dl, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.ConvTranspose1d(ds, ds, kernel_size=2, stride=2),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            res_block_1D(ds, ds, ks=3, st=1, pa=1, norm=norm),
            nn.Conv1d(ds, self.in_channels, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w -> ... w c',c=self.in_channels),
        )

    def forward(self, v_data, v_test=False):
        face_features = self.face_conv(v_data["face_norm"][...,:self.in_channels])
        edge_features = self.edge_conv(v_data["edge_norm"][...,:self.in_channels])

        pred_face = self.face_points_decoder(face_features)
        pred_edge = self.edge_points_decoder(edge_features)

        loss = {}
        loss["face_norm"] = nn.functional.l1_loss(
            pred_face,
            v_data["face_norm"][...,:self.in_channels]
        )
        loss["edge_norm"] = nn.functional.l1_loss(
            pred_edge,
            v_data["edge_norm"][...,:self.in_channels]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        if v_test:
            pred_face = denormalize_coord2(pred_face, v_data["face_bbox"])
            pred_edge = denormalize_coord2(pred_edge, v_data["edge_bbox"])

            gt_face = denormalize_coord2(v_data["face_norm"], v_data["face_bbox"])
            gt_edge = denormalize_coord2(v_data["edge_norm"], v_data["edge_bbox"])
            data["pred_face"] = pred_face.cpu().numpy()
            data["pred_edge"] = pred_edge.cpu().numpy()
            data["gt_face"] = gt_face.cpu().numpy()
            data["gt_edge"] = gt_edge.cpu().numpy()

        return loss, data


class AutoEncoder_1012(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        norm = v_conf["norm"]
        ds = self.dim_shape
        dl = self.dim_latent
        self.df = self.dim_latent * 2 * 2
        df = self.df

        self.encoderdecoder = EncoderDecoder(v_conf)
        if "encoderdecoder_weight" in v_conf and v_conf["encoderdecoder_weight"] is not None:
            print("Load ae weight from ", v_conf["encoderdecoder_weight"])
            weights = (torch.load(v_conf["encoderdecoder_weight"])["state_dict"])
            weights = {k.replace("model.", ""): v for k, v in weights.items()}
            self.encoderdecoder.load_state_dict(weights)
            for params in self.encoderdecoder.parameters():
                params.requires_grad = False
            self.encoderdecoder.eval()

        self.face_in = nn.Linear(self.df + 6, 256)
        self.edge_in = nn.Linear(dl * 2 + 6, 256)

        self.graph_face_edge = nn.ModuleList()
        for i in range(5):
            self.graph_face_edge.append(GATv2Conv(
                256, 256, 
                heads=1, edge_dim=256,
            ))
            self.graph_face_edge.append(nn.LeakyReLU())

        bd = 768
        self.attn_in = nn.Linear(256, bd)
        layer = nn.TransformerEncoderLayer(
            bd, 12, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm(bd))
        self.attn_out = nn.Linear(bd, df)

        self.gaussian_weights = v_conf["gaussian_weights"]
        self.with_sigmoid = v_conf["sigmoid"]
        if self.gaussian_weights > 0:
            self.gaussian_proj = nn.Linear(self.df, self.df*2)
        else:
            self.gaussian_proj = nn.Sequential(
                nn.Linear(self.df, self.df),
                nn.Sigmoid() if self.with_sigmoid else nn.Identity()
            )

        # Decoder
        self.attn_in2 = nn.Linear(df, bd)
        layer2 = nn.TransformerEncoderLayer(
            bd, 12, dim_feedforward=2048, dropout=0.1, 
            batch_first=True, norm_first=True)
        self.face_attn2 = nn.TransformerEncoder(layer2, 8, nn.LayerNorm(bd))
        self.attn_out2 = nn.Linear(bd, df + 6)

        edge_out_channels = dl * 2 + 6 + 1 # Geo + bbox + labels
        self.edge_feature_proj = nn.Sequential(
            nn.Conv1d(2 * bd, bd, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            res_block_1D(bd, bd, ks=1, st=1, pa=0, norm=norm),
            nn.Conv1d(bd, edge_out_channels, kernel_size=1, stride=1, padding=0),
            Rearrange("b n w -> b (n w)"),
        )

        self.face_center_scale_decoder = nn.Sequential(
            Rearrange("b n -> b n 1 1"),
            nn.Conv2d(dl * 2 * 2, ds, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            res_block_2D(ds, ds, ks=1, st=1, pa=0, norm=norm),
            nn.Conv2d(ds, 3 * 2, kernel_size=1, stride=1, padding=0),
            Rearrange('... (c n) w h -> ... (w h c) n', c=3, n=2),
        )

        self.times = {
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
        }
        self.cd_computer = ChamferDistance()

    def profile_time(self, timer, key):
        torch.cuda.synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def sample(self, v_fused_face_features, v_is_test=False):
        fused_face_features_gau = self.gaussian_proj(v_fused_face_features)

        if self.gaussian_weights <= 0:
            return fused_face_features_gau, torch.zeros_like(v_fused_face_features[0,0])

        fused_face_features_gau = fused_face_features_gau.reshape(-1, self.df, 2)
        mean = fused_face_features_gau[:, :, 0]
        logvar = fused_face_features_gau[:, :, 1]

        if v_is_test:
            return mean, torch.zeros_like(v_fused_face_features[0,0])

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        fused_face_features = eps.mul(std).add_(mean)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * self.gaussian_weights
        return fused_face_features, kl_loss

    def decode(self, encoding_result, v_data=None):
        # Face attn
        face_z = encoding_result["face_z"]
        face_z = self.attn_in2(face_z)
        if v_data is not None:
            attn_mask = v_data["attn_mask"]
        else:
            attn_mask = torch.zeros((face_z.shape[0], face_z.shape[0]), dtype=face_z.dtype, device=face_z.device)
        face_z = self.face_attn2(face_z, attn_mask)
        face_out = self.attn_out2(face_z)

        pred_face_points = self.encoderdecoder.face_points_decoder(face_out[..., :self.df])
        pred_face_bbox = face_out[..., self.df:]

        if v_data is not None: # Training
            true_intersection_embedding = face_z[v_data["edge_face_connectivity"][:, 1:]]
            false_intersection_embedding = face_z[v_data["zero_positions"]]
            intersection_embedding = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
            id_false_start = true_intersection_embedding.shape[0]

            edge_features = intersection_embedding
            edge_features = rearrange(edge_features, 'b c n -> b (c n) 1')
            edge_features = self.edge_feature_proj(edge_features)
            probability = edge_features[..., -1:]

            gt_labels = torch.ones_like(probability)
            gt_labels[id_false_start:] = 0
            loss_edge_classification = F.binary_cross_entropy_with_logits(probability, gt_labels)

            edge_features = edge_features[:id_false_start]
        else:
            device = face_z.device
            num_faces = face_z.shape[0]
            indexes = torch.stack(torch.meshgrid(torch.arange(num_faces), torch.arange(num_faces), indexing="ij"), dim=2)

            indexes = indexes.reshape(-1,2).to(device)
            edge_features = face_z[indexes]
            edge_features = rearrange(edge_features, 'b c n -> b (c n) 1')
            edge_features = self.edge_feature_proj(edge_features)
            probability = edge_features[..., -1:]

            edge_features = edge_features[torch.sigmoid(probability[...,0]) > 0.5]
            loss_edge_classification = None

        pred_edge_points = self.encoderdecoder.edge_points_decoder(edge_features[..., :self.dim_latent*2])
        pred_edge_bbox = edge_features[..., self.dim_latent*2:-1]

        return {
            "pred_face_points": pred_face_points,
            "pred_face_bbox": pred_face_bbox,
            "pred_edge_points": pred_edge_points,
            "pred_edge_bbox": pred_edge_bbox,
            "loss_edge_classification": loss_edge_classification,
            "probability": probability,
        }

    def encode(self, v_data, v_test):
        # torch.cuda.synchronize()
        # Encoder
        # timer = time.time()
        face_feature = self.encoderdecoder.face_conv(v_data["face_norm"])
        edge_feature = self.encoderdecoder.edge_conv(v_data["edge_norm"])

        face_feature = torch.cat((face_feature, v_data["face_bbox"]), dim=-1)
        edge_feature = torch.cat((edge_feature, v_data["edge_bbox"]), dim=-1)

        face_feature = self.face_in(face_feature)
        edge_feature = self.edge_in(edge_feature)
        # timer = self.profile_time(timer, "Encoder")

        # Fuser
        edge_face_connectivity = v_data["edge_face_connectivity"]
        # Face graph
        x = face_feature
        edge_index = edge_face_connectivity[:, 1:].permute(1, 0)
        edge_attr = edge_feature[edge_face_connectivity[:, 0]]
        for layer in self.graph_face_edge:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        face_features = x

        # Face attn
        face_features = self.attn_in(face_features)
        fused_face_features = self.face_attn(face_features, v_data["attn_mask"])
        fused_face_features = self.attn_out(fused_face_features)
        # timer = self.profile_time(timer, "Fuser")

        face_z, kl_loss = self.sample(fused_face_features, v_is_test=v_test)
        # timer = self.profile_time(timer, "Sample")
        return {
            "face_z": face_z,
            "kl_loss": kl_loss,
            "edge_features": edge_feature,
        }

    def loss(self, v_data, decoding_result):
        # Loss
        loss={}
        loss["edge_classification"] = decoding_result["loss_edge_classification"] * 0.1
        loss["face_coords_norm"] = nn.functional.l1_loss(
            decoding_result["pred_face_points"],
            v_data["face_norm"]
        )
        loss["face_bbox"] = nn.functional.l1_loss(
            decoding_result["pred_face_bbox"],
            v_data["face_bbox"]
        )
        id_edges = v_data["edge_face_connectivity"][:, 0]
        edge_coords_norm1 = nn.functional.l1_loss(
            decoding_result["pred_edge_points"],
            v_data["edge_norm"][id_edges]
        )
        edge_coords_norm2 = nn.functional.l1_loss(
           torch.flip(decoding_result["pred_edge_points"], dims=[1]),
            v_data["edge_norm"][id_edges]
        )
        loss["edge_coords_norm"] = edge_coords_norm1 if edge_coords_norm1 < edge_coords_norm2 else edge_coords_norm2
        loss["edge_center"] = nn.functional.l1_loss(
            decoding_result["pred_edge_bbox"],
            v_data["edge_bbox"][id_edges]
        )
        if self.gaussian_weights > 0:
            loss["kl_loss"] = decoding_result["kl_loss"]
        loss["total_loss"] = sum(loss.values())
        # timer = self.profile_time(timer, "Loss")

        return loss
    
    def forward(self, v_data, v_test=False):
        encoding_result = self.encode(v_data, v_test)
        decoding_result = self.decode(encoding_result, v_data)
        if "kl_loss" in encoding_result:
            decoding_result["kl_loss"] = encoding_result["kl_loss"]
        loss = self.loss(v_data, decoding_result)
        data = {}
        if v_test:
            pred_face = denormalize_coord2(decoding_result["pred_face_points"], decoding_result["pred_face_bbox"])
            pred_edge = denormalize_coord2(decoding_result["pred_edge_points"], decoding_result["pred_edge_bbox"])

            gt_face = denormalize_coord2(v_data["face_norm"], v_data["face_bbox"])
            gt_edge = denormalize_coord2(v_data["edge_norm"], v_data["edge_bbox"])

            loss["face_coords"] = nn.functional.l1_loss(
                pred_face[..., :3],
                gt_face[..., :3]
            )
            loss["edge_coords"] = nn.functional.l1_loss(
                pred_edge[..., :3],
                gt_edge[..., :3][v_data["edge_face_connectivity"][:, 0]]
            )

            encoding_result = {
                "face_z": encoding_result["face_z"],
            }
            decoding_result = self.decode(encoding_result)
            pred_face = denormalize_coord2(decoding_result["pred_face_points"], decoding_result["pred_face_bbox"])
            pred_edge = denormalize_coord2(decoding_result["pred_edge_points"], decoding_result["pred_edge_bbox"])

            num_faces = encoding_result["face_z"].shape[0]
            face_adj = torch.zeros((num_faces, num_faces), dtype=bool, device=loss["total_loss"].device)
            conn = v_data["edge_face_connectivity"]
            face_adj[conn[:, 1], conn[:, 2]] = True

            data = {
                "gt_face_adj": face_adj.reshape(-1),
                "pred_face_adj": (torch.sigmoid(decoding_result["probability"]) > 0.5).reshape(-1),
                
                "gt_face": gt_face.detach().cpu().numpy(),
                "gt_edge": gt_edge.detach().cpu().numpy(),
                "gt_edge_face_connectivity": v_data["edge_face_connectivity"].detach().cpu().numpy(),

                "pred_face": pred_face.detach().cpu().numpy(),
                "pred_edge": pred_edge.detach().cpu().numpy(),
                "pred_edge_face_connectivity": pred_edge.detach().cpu().numpy(),
            }

        return loss, data