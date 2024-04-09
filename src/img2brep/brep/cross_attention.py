import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, embedding1, embedding2):
        B, N, _ = embedding1.shape

        # Project to queries, keys, and values
        qkv1 = self.qkv_proj(embedding1).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv2 = self.qkv_proj(embedding2).reshape(B, N, 3, self.num_heads, self.head_dim)

        q1, k1, v1 = qkv1.unbind(dim=2)
        q2, k2, v2 = qkv2.unbind(dim=2)

        # Calculate attention using embedding1 as queries and embedding2 as keys
        attn_weights_1 = torch.einsum('bnhd,bshd->bhns', q1, k2) * self.scale
        attn_weights_1 = F.softmax(attn_weights_1, dim=-1)
        attn_output_1 = torch.einsum('bhns,bshd->bnhd', attn_weights_1, v2).reshape(B, N, self.dim)

        # Calculate attention using embedding2 as queries and embedding1 as keys
        attn_weights_2 = torch.einsum('bnhd,bshd->bhns', q2, k1) * self.scale
        attn_weights_2 = F.softmax(attn_weights_2, dim=-1)
        attn_output_2 = torch.einsum('bhns,bshd->bnhd', attn_weights_2, v1).reshape(B, N, self.dim)

        # Skip connections and layer normalization
        output1 = embedding1 + self.out_proj(attn_output_1)
        output1 = self.ln(output1)

        output2 = embedding2 + self.out_proj(attn_output_2)
        output2 = self.ln(output2)

        return output1, output2


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads) for _ in range(num_layers)
            ])

    def forward(self, embedding1, embedding2):
        for layer in self.layers[:-1]:  # Apply all but the last layer
            embedding1, embedding2 = layer(embedding1, embedding2)

        # Apply the last layer and combine the results
        final_output1, final_output2 = self.layers[-1](embedding1, embedding2)
        combined_output = final_output1 * final_output2  # Element-wise multiplication

        return combined_output
