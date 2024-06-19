import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn
from torch_geometric.nn import SAGEConv, GATv2Conv

from src.img2brep.model_fuser import Attn_fuser_single, Attn_fuser_cross
import torch.nn.functional as F


def get_attn_mask(v_mask):
    b, n = v_mask.shape
    batch_indices = torch.arange(b, device=v_mask.device).unsqueeze(1).repeat(1, n)
    batch_indices = batch_indices[v_mask]
    attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
    return attn_mask


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


class SAGE_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels_list, normalize=False, project=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        in_channels_layer = in_channels

        for idx, out_channels in enumerate(out_channels_list):
            self.layers.append(SAGEConv(in_channels_layer, out_channels, normalize=normalize, project=project))
            if idx != len(out_channels_list) - 1:
                norm = nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(out_channels)
                )
                self.layers.append(norm)
            in_channels_layer = out_channels

    def forward(self, x, edge_index, **kwargs):
        for layer in self.layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GAT_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels_list, out_channels, edge_dim=None, num_heads=8, concat=False,
                 negative_slope=0.2, fill_value='mean', dropout=0.0, bias=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        in_channels_layer = in_channels

        for idx, out_channels in enumerate(out_channels_list):
            self.layers.append(
                GATv2Conv(in_channels_layer, out_channels, edge_dim=edge_dim, heads=num_heads,
                          concat=concat, negative_slope=negative_slope, fill_value=fill_value,
                          dropout=dropout, bias=bias))

            self.layers.append(nn.LayerNorm(out_channels * num_heads if concat else out_channels))
            self.layers.append(nn.ReLU())
            in_channels_layer = out_channels

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        for layer in self.layers:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr) + x
            else:
                x = layer(x)
        return x


class Separate_encoder(nn.Module):
    def __init__(self, dim_shape=256, dim_latent=8, **kwargs):
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
        self.edge_proj = nn.Linear(dim_shape, dim_latent)

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
        self.vertex_proj = nn.Linear(dim_shape, dim_latent)

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
        edge_features = self.edge_proj(edge_features)

        # Vertex
        vertex_mask = (v_data["vertex_points"] != -1).all(dim=-1)
        vertex_features = self.vertex_coords(v_data["vertex_points"][vertex_mask])

        vertex_attn_mask = get_attn_mask(vertex_mask)
        vertex_features = self.vertex_fuser(vertex_features, vertex_attn_mask)
        vertex_features = self.vertex_proj(vertex_features)

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


class Fused_encoder(Separate_encoder):
    def __init__(self, dim_shape=256, dim_latent=8, **kwargs):
        super().__init__(dim_shape, dim_latent)
        self.fuser_edges_to_faces = Attn_fuser_cross(self.dim_latent)

    def forward(self, v_data):
        prepare_connectivity(v_data)
        face_mask = (v_data["face_points"] != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        face_features = self.face_coords(v_data["face_points"][face_mask])

        face_attn_mask = get_attn_mask(face_mask)
        face_features = self.face_fuser(face_features, face_attn_mask)

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

        face_features = self.face_proj(face_features)
        edge_features = self.edge_proj(edge_features)
        vertex_features = self.vertex_proj(vertex_features)

        # Fusing
        face_edge_embeddings = self.fuser_edges_to_faces(
            v_embeddings1=edge_features,
            v_embeddings2=face_features,
            v_connectivity1_to_2=v_data["edge_face_connectivity"],
            v_attn_mask=face_attn_mask
        )

        return {
            "face_features": face_edge_embeddings,
            "edge_features": edge_features,
            "vertex_features": vertex_features,
            "face_mask": face_mask,
            "edge_mask": edge_mask,
            "vertex_mask": vertex_mask,
        }
