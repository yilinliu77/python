import torch
from torch import nn


# Cross attention layer
class Attn_fuser_cross(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.atten = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, 8, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerDecoderLayer(dim, 8, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerDecoderLayer(dim, 8, dim_feedforward=dim, dropout=0.1, batch_first=True),
        ])
        pass

    def forward(self, v_embeddings1, v_embeddings2, v_connectivity1_to_2, v_attn_mask):
        num1, _ = v_embeddings1.shape
        num2, _ = v_embeddings2.shape
        attn_mask = torch.ones(
            num2, num1, device=v_embeddings1.device, dtype=torch.bool
        )
        attn_mask[v_connectivity1_to_2[:, 1], v_connectivity1_to_2[:, 0]] = False
        attn_mask[v_connectivity1_to_2[:, 2], v_connectivity1_to_2[:, 0]] = False

        x = v_embeddings2
        for layer in self.atten:
            x = layer(
                tgt=x,
                memory=v_embeddings1,
                tgt_mask=v_attn_mask,
                memory_mask=attn_mask,
            )
        return x


# Self attention layer
class Attn_fuser_single(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim
        self.atten = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True),
        ])

    def forward(self, v_embedding, v_attn_mask):
        x = v_embedding
        for layer in self.atten:
            x = layer(x, src_mask=v_attn_mask)
        return x
