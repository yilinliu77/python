import torch
from torch import nn

# Cross attention layer
class Attn_fuser_cross(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.atten = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
        ])
        pass

    def forward(self, v_embeddings1, v_embeddings2, v_connectivity1_to_2):
        num1, _ = v_embeddings1.shape
        num2, _ = v_embeddings2.shape
        attn_mask = torch.ones(
            num2, num1, device=v_embeddings1.device, dtype=torch.bool
        )
        attn_mask[v_connectivity1_to_2[:, 1], v_connectivity1_to_2[:, 0]] = False
        attn_mask[v_connectivity1_to_2[:, 2], v_connectivity1_to_2[:, 0]] = False

        x = v_embeddings2
        for layer in self.atten:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(
                    query=x,
                    key=v_embeddings1,
                    value=v_embeddings1,
                    attn_mask=attn_mask,
                )
                x = x + out

        return x

# Self attention layer
class Attn_fuser_single(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.atten = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 2, 0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
            nn.MultiheadAttention(hidden_dim, 2, 0.1, batch_first=True),
            nn.LayerNorm(hidden_dim),
        ])

    def forward(self, v_embedding, v_mask):
        B, _ = v_mask.shape
        L, _ = v_embedding.shape
        attn_mask = v_embedding.new_ones(L, L, device=v_embedding.device, dtype=torch.bool)
        num_valid = v_mask.long().sum(dim=1)
        num_valid = torch.cat((torch.zeros_like(num_valid[:1]), num_valid.cumsum(dim=0)))
        for i in range(num_valid.shape[0] - 1):
            attn_mask[num_valid[i]:num_valid[i + 1], num_valid[i]:num_valid[i + 1]] = 0

        x = v_embedding
        for layer in self.atten:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                out, weights = layer(x, x, x, attn_mask=attn_mask, need_weights=True)
                x = x + out
        return x
