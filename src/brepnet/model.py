import time
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

from torch_geometric.nn import GATv2Conv

def add_timer(time_statics, v_attr, timer):
    if v_attr not in time_statics:
        time_statics[v_attr] = 0.
    time_statics[v_attr] += time.time() - timer
    return time.time()

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