import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F


class Intersector(nn.Module):
    def __init__(self, num_max_items=None):
        super().__init__()
        self.num_max_items = num_max_items

    def prepare_vertex_data(self, edge_features, v_vertex_edge_connectivity, v_edge_adj, v_edge_mask):
        intersection_embedding = edge_features[v_vertex_edge_connectivity[:, 1:]]

        edge_adj = v_edge_adj.clone()
        edge_adj[v_edge_adj == 0] = 1
        edge_adj[v_edge_adj == 1] = 0
        torch.diagonal(edge_adj, dim1=1, dim2=2).fill_(1)

        edge_embeddings = edge_features.new_zeros((*v_edge_mask.shape, edge_features.shape[-1]))
        edge_embeddings = edge_embeddings.masked_scatter(rearrange(v_edge_mask, '... -> ... 1'), edge_features)

        zero_positions = (edge_adj == 1).nonzero()
        edge_embeddings1_idx = zero_positions[:, [0, 1]]
        edge_embeddings2_idx = zero_positions[:, [0, 2]]

        if self.num_max_items is not None and edge_embeddings1_idx.shape[0] > self.num_max_items:
            indices = torch.randperm(edge_embeddings1_idx.shape[0])[:self.num_max_items]
        else:
            indices = torch.arange(edge_embeddings1_idx.shape[0])

        edge_embeddings1 = edge_embeddings[edge_embeddings1_idx[indices, 0], edge_embeddings1_idx[indices, 1], :]
        edge_embeddings2 = edge_embeddings[edge_embeddings2_idx[indices, 0], edge_embeddings2_idx[indices, 1], :]
        null_intersection = torch.stack([edge_embeddings1, edge_embeddings2], dim=1)

        return intersection_embedding, null_intersection

    def prepare_edge_data(self, v_face_embeddings, v_edge_face_connectivity, v_face_adj, v_face_mask):
        # True intersection
        intersection_embedding = v_face_embeddings[v_edge_face_connectivity[:, 1:]]

        # Construct features for false intersection
        face_adj = v_face_adj.clone()
        face_adj[v_face_adj == 0] = 1
        face_adj[v_face_adj == 1] = 0
        torch.diagonal(face_adj, dim1=1, dim2=2).fill_(1)

        face_embeddings = v_face_embeddings.new_zeros((*v_face_mask.shape, v_face_embeddings.shape[-1]))
        face_embeddings = face_embeddings.masked_scatter(rearrange(v_face_mask, '... -> ... 1'), v_face_embeddings)

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
        null_intersection_embedding = torch.stack([face_embeddings1, face_embeddings2], dim=1)
        return intersection_embedding, null_intersection_embedding

    def forward(self,
                sampled_face_feature, v_data,v_encoder_result
                ):
        return

    def loss(self, edge_features, edge_null_features, vertex_features, vertex_null_features):
        return 0

    def inference(self, v_features, v_type):
        return

    def inference_label(self, v_features):
        return torch.cosine_similarity(v_features,
                                       self.null_intersection, dim=-1)

class Attn_intersector_classifier(Intersector):
    def __init__(self, num_max_items=None, dim=256):
        super().__init__(num_max_items)
        self.edge_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerDecoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0.1, batch_first=True),
        ])
        self.vertex_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerDecoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0.1, batch_first=True),
        ])

        self.vertex_token = nn.Parameter(torch.rand(dim))
        self.edge_token = nn.Parameter(torch.rand(dim))

        self.classifier = nn.Linear(dim, 1)
        self.position_embedding = nn.Embedding(2, dim)

    def inference(self, v_features, v_type):
        pos_encoding = self.position_embedding(torch.arange(
            v_features.shape[1],
            device=v_features.device
        )[None, :].repeat(v_features.shape[0], 1))
        v_features = v_features + pos_encoding
        if v_type == "edge":
            x = self.edge_token[None, None].repeat(v_features.shape[0], 1, 1)
        else:
            x = self.vertex_token[None, None].repeat(v_features.shape[0], 1, 1)
        for layer in self.edge_layers if v_type == "edge" else self.vertex_layers:
            x = layer(x, v_features)
        features = x[:, 0]
        return features

    def forward(self,
                sampled_face_feature, v_data,v_encoder_result
                ):
        gathered_edges, null_gathered_edges = self.prepare_edge_data(
            sampled_face_feature,
            v_data["edge_face_connectivity"],
            v_data["face_adj"],
            v_encoder_result["face_mask"])
        edge_features = self.inference(gathered_edges, "edge")
        edge_null_features = self.inference(null_gathered_edges, "edge")

        gathered_vertices, null_gathered_vertices = self.prepare_vertex_data(
            edge_features,
            v_data["vertex_edge_connectivity"],
            v_data["edge_adj"],
            v_encoder_result["edge_mask"]
        )
        vertex_features = self.inference(gathered_vertices, "vertex")
        vertex_null_features = self.inference(null_gathered_vertices, "vertex")

        return edge_features, edge_null_features, vertex_features, vertex_null_features

    def loss(self, edge_features, edge_null_features, vertex_features, vertex_null_features):
        intersection_feature = torch.cat([edge_features, edge_null_features])
        gt_label = torch.cat([torch.ones_like(edge_features[:, 0]),
                              torch.zeros_like(edge_null_features[:, 0])])
        loss_edge = F.binary_cross_entropy_with_logits(
            self.classifier(intersection_feature), gt_label[:, None])

        intersection_feature = torch.cat([vertex_features, vertex_null_features])
        gt_label = torch.cat([torch.ones_like(vertex_features[:, 0]),
                              torch.zeros_like(vertex_null_features[:, 0])])
        loss_vertex = F.binary_cross_entropy_with_logits(
            self.classifier(intersection_feature), gt_label[:, None])

        return loss_edge, loss_vertex

    def inference_label(self, v_features):
        return torch.sigmoid(self.classifier(v_features))[:, 0] > 0.5
