import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, reduce

class res_block_1D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1):
        super(res_block_1D, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x = x + self.conv(x)
        x = rearrange(x, '... c h -> ... h c')
        x = self.norm(x)
        x = rearrange(x, '... h c -> ... c h')
        return self.act(x)


class res_block_2D(nn.Module):
    def __init__(self, dim_in, dim_out, ks=3, st=1, pa=1, norm=nn.LayerNorm):
        super(res_block_2D, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=st, padding=pa)
        self.act = nn.ReLU()
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        x = x + self.conv(x)
        x = rearrange(x, '... c h w -> ... h w c')
        x = self.norm(x)
        x = rearrange(x, '... h w c -> ... c h w')
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


# Full continuous VAE
class AutoEncoder_base(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_base, self).__init__()
        self.dim_shape = 128
        self.dim_latent = 32
        self.time_statics = [0 for _ in range(10)]

        ds = self.dim_shape
        self.face_coords = nn.Sequential(
            Rearrange('b h w n -> b n h w'),
            nn.Conv2d(3, ds, kernel_size=5, stride=1, padding=2),
            res_block_2D(ds, ds, ks=5, st=1, pa=2),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=1, st=1, pa=0),
            nn.Conv2d(ds, self.dim_latent, kernel_size=1, stride=1, padding=0),
        ) # b c 4 4

        self.face_coords_decoder = nn.Sequential(
            nn.Conv2d(self.dim_latent, ds, kernel_size=1, stride=1, padding=0),
            res_block_2D(ds, ds, ks=1, st=1, pa=0),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(ds, ds, ks=5, st=1, pa=2),
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
        self.num_max_items = 500

    def intersection(self, v_edge_face_connectivity, v_face_adj, v_face_feature, v_face_mask):
        # True intersection
        intersection_embedding = v_face_feature[v_edge_face_connectivity[:, 1:]]

        # Construct features for false intersection
        face_adj = v_face_adj.clone()
        face_adj[v_face_adj == 0] = 1
        face_adj[v_face_adj == 1] = 0
        torch.diagonal(face_adj, dim1=1, dim2=2).fill_(1)

        face_embeddings = v_face_feature.new_zeros((*v_face_mask.shape, v_face_feature.shape[1], v_face_feature.shape[2], v_face_feature.shape[3]))
        face_embeddings = face_embeddings.masked_scatter(rearrange(v_face_mask, '... -> ... 1 1 1'), v_face_feature)

        zero_positions = (face_adj == 1).nonzero()
        face_embeddings1_idx = zero_positions[:, [0, 1]]
        face_embeddings2_idx = zero_positions[:, [0, 2]]

        if self.num_max_items is not None and face_embeddings1_idx.shape[0] > self.num_max_items:
            indices = torch.randperm(face_embeddings1_idx.shape[0])[:self.num_max_items]
        else:
            indices = torch.arange(face_embeddings1_idx.shape[0])

        # False intersection
        face_embeddings1 = face_embeddings[face_embeddings1_idx[indices, 0], face_embeddings1_idx[indices, 1], :]
        face_embeddings2 = face_embeddings[face_embeddings2_idx[indices, 0], face_embeddings2_idx[indices, 1], :]
        
        # Classification
        pred_true = self.classifier(torch.cat((intersection_embedding[:,0], intersection_embedding[:,1]), dim=1))
        pred_false = self.classifier(torch.cat((face_embeddings1, face_embeddings2), dim=1))
        
        gt_true = torch.ones_like(pred_true)
        gt_false = torch.zeros_like(pred_false)
        loss_edge = F.binary_cross_entropy_with_logits(pred_true, gt_true) + \
            F.binary_cross_entropy_with_logits(pred_false, gt_false)
        return loss_edge

    def forward(self, v_data, v_test=False, **kwargs):
        prepare_connectivity(v_data)
        face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        face_features = self.face_coords(v_data["face_points"][face_mask])
        pre_face_coords = self.face_coords_decoder(face_features,)

        loss_edge_classification = self.intersection(
            v_data["edge_face_connectivity"], 
            v_data["face_adj"], 
            face_features, 
            face_mask
        )

        # Loss
        loss={}
        loss["edge_classification"] = loss_edge_classification
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"][face_mask]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        data["recon_faces"] = pre_face_coords

        if v_test:
            batch_size = v_data["face_points"].shape[0]
            id_accumulate = face_mask.sum(dim=1).cumsum(dim=0)
            id_accumulate = torch.cat((torch.zeros_like(id_accumulate[0:1]), id_accumulate), dim=0)
            
            data["pred"] = []
            data["gt"] = []
            for i in range(batch_size):
                num_valid_faces = face_mask[i].sum() 
                face_feature_gathered = face_features[id_accumulate[i]:id_accumulate[i+1]]

                features = []
                gt = []
                for j in range(num_valid_faces):
                    for k in range(num_valid_faces):
                        if j == k:
                            continue
                        features.append(torch.cat((face_feature_gathered[j], face_feature_gathered[k]), dim=0))
                        gt.append(v_data["face_adj"][i,j,k])
                features = torch.stack(features, dim=0)
                gt = torch.stack(gt, dim=0)
                pred = self.classifier(features)
                pred = torch.sigmoid(pred) > 0.5
                data["pred"].append(pred)
                data["gt"].append(gt)
                pass
            pass

        return loss, data

