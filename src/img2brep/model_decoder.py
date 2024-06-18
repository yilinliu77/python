import torch
from einops.layers.torch import Rearrange
from torch import nn

from src.img2brep.model_encoder import res_block_1D, res_block_2D


class Small_decoder(nn.Module):
    def __init__(self,
                 dim_shape=256,
                 dim_latent=8,
                 **kwargs
                 ):
        super(Small_decoder, self).__init__()
        self.vertex_coords_decoder = nn.Sequential(
            Rearrange('b c -> b c 1'),
            nn.Conv1d(dim_latent, dim_shape, kernel_size=1, stride=1, padding=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            nn.Conv1d(dim_shape, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c 1-> ... c', ),
        )

        self.edge_coords_decoder = nn.Sequential(
            Rearrange('b c -> b c 1'),
            nn.Conv1d(dim_latent, dim_shape, kernel_size=1, stride=1, padding=0),
            res_block_1D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.Conv1d(dim_shape, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w-> ... w n', ),
        )


        # ================== Decoder ==================
        self.face_coords_decoder = nn.Sequential(
            Rearrange('b c -> b c 1 1'),
            nn.Conv2d(dim_latent, dim_shape, kernel_size=1, stride=1, padding=0),
            res_block_2D(dim_shape, dim_shape, ks=1, st=1, pa=0),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(dim_shape, dim_shape, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(dim_shape, dim_shape, ks=5, st=1, pa=2),
            nn.Conv2d(dim_shape, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... n w h -> ... w h n',),
        )

    def forward(self, v_face_embeddings, v_edge_embeddings, v_vertex_features):
        pre_face_coords = self.face_coords_decoder(v_face_embeddings)
        pre_edge_coords = self.edge_coords_decoder(v_edge_embeddings)
        pre_vertex_coords = self.vertex_coords_decoder(v_vertex_features)
        return pre_face_coords, pre_edge_coords, pre_vertex_coords

