from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch_geometric.nn import SAGEConv, GATv2Conv


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



################### Encoder

# 11137M FLOPS and 3719680 parameters
class Continuous_encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = 256
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1, stride=1, padding=2),
            res_block_2D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.vertex_encoder = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

    def encode_face(self, v_data):
        face_coords = v_data["face_points"]
        face_mask = (face_coords != -1).all(dim=-1).all(dim=-1).all(dim=-1)
        return self.face_encoder(face_coords[face_mask].permute(0, 3, 1, 2)), face_mask

    def encode_edge(self, v_data):
        edge_coords = v_data["edge_points"]
        edge_mask = (edge_coords != -1).all(dim=-1).all(dim=-1)
        return self.edge_encoder(edge_coords[edge_mask].permute(0, 2, 1)), edge_mask

    def encode_vertex(self, v_data):
        vertex_coords = v_data["vertex_points"]
        vertex_mask = (vertex_coords != -1).all(dim=-1)
        return self.vertex_encoder(vertex_coords[vertex_mask][:,:,None])[...,0], vertex_mask

    def forward(self, v_data):
        face_coords, face_mask = self.encode_face(v_data)
        edge_coords, edge_mask = self.encode_edge(v_data)
        vertex_coords, vertex_mask = self.encode_vertex(v_data)
        return face_coords, edge_coords, vertex_coords, face_mask, edge_mask, vertex_mask


# 57196M FLOPS and 4982400 parameters
class Discrete_encoder(Continuous_encoder):
    def __init__(self,
                 dim=256,
                 bbox_discrete_dim=64,
                 coor_discrete_dim=64,
                 ):
        super().__init__()
        hidden_dim = dim
        self.bbox_embedding = nn.Embedding(bbox_discrete_dim - 1, 64)
        self.coords_embedding = nn.Embedding(coor_discrete_dim - 1, 64)

        self.bbox_encoder = nn.Sequential(
            nn.Linear(6 * 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(hidden_dim),
            Rearrange('b h w c -> b c h w'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.face_fuser = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Conv1d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
            Rearrange('b c h -> b h c'),
            nn.LayerNorm(hidden_dim),
            Rearrange('b h c -> b c h'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_fuser = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.vertex_encoder = nn.Sequential(
            Rearrange('b c -> b c 1'),
            nn.Conv1d(3 * 64, hidden_dim, kernel_size=1, stride=1, padding=0),
            Rearrange('b c h -> b h c'),
            nn.LayerNorm(hidden_dim),
            Rearrange('b h c -> b c h'),
            nn.ReLU(),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

    def encode_face(self, v_data):
        face_coords = v_data["discrete_face_points"]
        face_bbox = v_data["discrete_face_bboxes"]
        face_mask = (face_coords != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        flatten_face_coords = face_coords[face_mask]
        flatten_face_bbox = face_bbox[face_mask]

        face_coords = rearrange(self.coords_embedding(flatten_face_coords), "... h w c d -> ... (c d) h w")
        face_bbox = rearrange(self.bbox_embedding(flatten_face_bbox), "... l c d -> ... l (c d)")

        face_coords_features = self.face_encoder(face_coords)
        face_bbox_features = self.bbox_encoder(face_bbox)

        face_features = self.face_fuser(face_coords_features + face_bbox_features)

        return face_features, face_mask

    def encode_edge(self, v_data):
        edge_coords = v_data["discrete_edge_points"]
        edge_bbox = v_data["discrete_edge_bboxes"]
        edge_mask = (edge_coords != -1).all(dim=-1).all(dim=-1)

        flatten_edge_coords = edge_coords[edge_mask]
        flatten_edge_bbox = edge_bbox[edge_mask]

        edge_coords = rearrange(self.coords_embedding(flatten_edge_coords), "... h c d -> ... (c d) h")
        edge_bbox = rearrange(self.bbox_embedding(flatten_edge_bbox), "... l c d -> ... l (c d)")

        edge_coords_features = self.edge_encoder(edge_coords)
        edge_bbox_features = self.bbox_encoder(edge_bbox)

        edge_features = self.edge_fuser(edge_coords_features + edge_bbox_features)

        return edge_features, edge_mask

    def encode_vertex(self, v_data):
        vertex_coords = v_data["discrete_vertex_points"]
        vertex_mask = (vertex_coords != -1).all(dim=-1)

        flatten_vertex_coords = vertex_coords[vertex_mask]

        vertex_coords = rearrange(self.coords_embedding(flatten_vertex_coords), "... h d c -> ... h (d c)")
        vertex_coords_features = self.vertex_encoder(vertex_coords)[..., 0]
        return vertex_coords_features, vertex_mask
