import importlib
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import SpatialNorm, AttentionProcessor, AttnProcessor
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv
from tqdm import tqdm
from vector_quantize_pytorch import ResidualLFQ, VectorQuantize, ResidualVQ, FSQ, ResidualFSQ

import pytorch_lightning as pl

from shared.common_utils import *
from src.img2brep.brep.common import *
from src.img2brep.brep.model_encoder import GAT_GraphConv, SAGE_GraphConv, res_block_1D, res_block_2D
from src.img2brep.brep.model_fuser import Attn_fuser_cross, Attn_fuser_single

import open3d as o3d


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def get_attn_mask(v_mask):
    b, n = v_mask.shape
    batch_indices = torch.arange(b, device=v_mask.device).unsqueeze(1).repeat(1, n)
    batch_indices = batch_indices[v_mask]
    attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
    return attn_mask


class AutoEncoder(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.with_quantization = v_conf["with_quantization"]
        self.finetune_decoder = v_conf["finetune_decoder"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_encoder')
        self.encoder = getattr(mod, v_conf["encoder"])(
            self.dim_shape,
            bbox_discrete_dim=v_conf["bbox_discrete_dim"],
            coor_discrete_dim=v_conf["coor_discrete_dim"],
        )
        self.vertices_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )
        self.edges_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )
        self.faces_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )

        # ================== GCN to distribute features across primitives ==================
        if v_conf["graphconv"] == "GAT":
            GraphConv = GAT_GraphConv
        else:
            GraphConv = SAGE_GraphConv
        encoder_dims_through_depth_edges = [self.dim_latent for _ in range(4)]
        encoder_dims_through_depth_faces = [self.dim_latent for _ in range(4)]
        self.gcn_on_edges = GraphConv(self.dim_latent, encoder_dims_through_depth_edges,
                                      self.dim_latent)
        self.gcn_on_faces = GraphConv(self.dim_latent, encoder_dims_through_depth_faces,
                                      self.dim_latent, edge_dim=self.dim_latent)

        # ================== self attention to aggregate features ==================
        self.vertex_fuser = Attn_fuser_single(self.dim_latent)
        self.edge_fuser = Attn_fuser_single(self.dim_latent)
        self.face_fuser = Attn_fuser_single(self.dim_latent)

        # ================== cross attention to aggregate features ==================
        self.fuser_vertices_to_edges = Attn_fuser_cross(self.dim_latent)
        self.fuser_edges_to_faces = Attn_fuser_cross(self.dim_latent)

        # ================== Intersection ==================
        mod = importlib.import_module('src.img2brep.brep.model_intersector')
        self.intersector = getattr(mod, v_conf["intersector"])(500, self.dim_latent)

        # ================== Decoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_decoder')
        self.decoder = getattr(mod, v_conf["decoder"])(
            dim_in=self.dim_latent,
            hidden_dim=v_conf["dim_decoder"],
            bbox_discrete_dim=v_conf["bbox_discrete_dim"],
            coor_discrete_dim=v_conf["coor_discrete_dim"],
        )

        self.intersection_face_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.intersection_edge_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.intersection_vertex_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.frozen_models = []
        # ================== Quantization ==================
        self.with_quantization = v_conf["with_quantization"]
        if self.with_quantization:
            if False:
                self.quantizer = VectorQuantize(
                    dim=self.dim_latent,
                    codebook_dim=32,  # a number of papers have shown smaller codebook dimension to be acceptable
                    heads=8,  # number of heads to vector quantize, codebook shared across all heads
                    separate_codebook_per_head=True,
                    # whether to have a separate codebook per head. False would mean 1 shared codebook
                    codebook_size=8196,
                    accept_image_fmap=False
                )
                self.quantizer_proj = nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1),
                    nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1),
                )
                self.frozen_models = []
            else:
                self.quantizer_out = nn.Sequential(
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Sigmoid(),
                )

                # self.quantizer = ResidualVQ(
                #     dim=self.dim_latent,
                #     codebook_dim=32,
                #     num_quantizers=8,
                #     codebook_size=2048,
                # )

                layer = nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1)
                self.quantizer_in = nn.Sequential(
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Sigmoid(),
                )

                self.quantizer = ResidualFSQ(
                    num_quantizers=16,
                    dim=self.dim_latent,
                    num_codebooks=1,
                    levels=[8, 8, 8, 5, 5, 5],
                )

                # self.quantizer = ResidualLFQ(
                #     num_quantizers=4,
                #     dim=self.dim_latent,
                #     codebook_size=16384,
                #     num_codebooks=1,
                # )

                self.frozen_models = [
                    # self.encoder, self.vertices_proj, self.edges_proj, self.faces_proj,
                    # self.gcn_on_edges, self.gcn_on_faces, self.edge_fuser, self.face_fuser,
                    # self.fuser_vertices_to_edges, self.fuser_edges_to_faces,
                    # self.decoder, self.intersector
                ]

        # ================== Freeze models ==================
        # for model in self.frozen_models:
        #     model.eval()
        #     for param in model.parameters():
        #         param.requires_grad = False

    # Inference (B * num_faces * num_features)
    # Pad features are all zeros
    # B==1 currently
    def inference(self, v_face_embeddings, return_topology=False):
        # Use face to intersect edges
        B = v_face_embeddings.shape[0]
        device = v_face_embeddings.device
        assert B == 1
        num_faces = v_face_embeddings.shape[1]
        face_idx = torch.stack(torch.meshgrid(
            torch.arange(num_faces), torch.arange(num_faces), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_face_features = v_face_embeddings[0, face_idx]

        edge_features = self.intersector.inference(gathered_face_features, "edge")
        edge_intersection_mask = self.intersector.inference_label(edge_features)
        edge_features = edge_features[edge_intersection_mask]
        num_edges = edge_features.shape[0]

        # Use edge to intersect vertices
        edge_idx = torch.stack(torch.meshgrid(
            torch.arange(num_edges), torch.arange(num_edges), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_edge_features = edge_features[edge_idx]

        if gathered_edge_features.shape[0] < 64 * 64:
            vertex_features = self.intersector.inference(gathered_edge_features, "vertex")
            vertex_intersection_mask = self.intersector.inference_label(vertex_features)
            vertex_features = vertex_features[vertex_intersection_mask]
        else:
            vertex_features = gathered_edge_features.new_zeros(0, gathered_edge_features.shape[-1])

        # Decode
        recon_data = self.decoder(
            v_face_embeddings.view(-1, v_face_embeddings.shape[-1]),
            edge_features,
            vertex_features,
        )
        recon_faces, recon_edges, recon_vertices = self.decoder.inference(recon_data)
        if return_topology:
            face_edge_connectivity = torch.cat((
                torch.arange(num_edges, device=device)[:, None], face_idx[edge_intersection_mask],), dim=1)
            edge_vertex_connectivity = torch.cat((
                torch.arange(vertex_features.shape[0], device=device)[:, None], edge_idx[vertex_intersection_mask],),
                dim=1)
            return recon_vertices, recon_edges, recon_faces, face_edge_connectivity, edge_vertex_connectivity
        return recon_vertices, recon_edges, recon_faces

    def encode(self, v_data):
        # ================== Encode the edge and face points ==================
        face_embeddings, edge_embeddings, vertex_embeddings, face_mask, edge_mask, vertex_mask = self.encoder(v_data)
        face_embeddings = self.faces_proj(face_embeddings)
        edge_embeddings = self.edges_proj(edge_embeddings)
        vertex_embeddings = self.vertices_proj(vertex_embeddings)

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

        edge_face_connectivity = v_data["edge_face_connectivity"].clone()
        edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
        # Solve the edge_face_connectivity: last two dimension (id_face)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        # Solve the edge_face_connectivity: first dimension (id_edge)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]

        # ================== Self-attention on vertices ==================
        b, n = vertex_mask.shape
        batch_indices = torch.arange(b, device=vertex_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[vertex_mask]
        vertex_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        atten_vertex_embeddings = self.vertex_fuser(vertex_embeddings, vertex_attn_mask)

        # ================== Fuse vertex features to the corresponding edges ==================
        b, n = edge_mask.shape
        batch_indices = torch.arange(b, device=edge_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[edge_mask]
        edge_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        edge_vertex_embeddings = self.fuser_vertices_to_edges(
            v_embeddings1=atten_vertex_embeddings,
            v_embeddings2=edge_embeddings,
            v_connectivity1_to_2=vertex_edge_connectivity,
            v_attn_mask=edge_attn_mask
        )

        # ================== GCN and self-attention on edges ==================
        edge_embeddings_gcn = self.gcn_on_edges(edge_vertex_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_attn_mask)

        # ================== fuse edges features to the corresponding faces ==================
        b, n = face_mask.shape
        batch_indices = torch.arange(b, device=face_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[face_mask]
        face_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        face_edge_embeddings = self.fuser_edges_to_faces(
            v_connectivity1_to_2=edge_face_connectivity,
            v_embeddings1=atten_edge_embeddings,
            v_embeddings2=face_embeddings,
            v_attn_mask=face_attn_mask,
        )

        # ================== GCN and self-attention on faces  ==================
        face_edge_embeddings_gcn = self.gcn_on_faces(face_edge_embeddings,
                                                     edge_face_connectivity[..., 1:].permute(1, 0),
                                                     edge_attr=atten_edge_embeddings[edge_face_connectivity[..., 0]])

        atten_face_embeddings = self.face_fuser(face_edge_embeddings_gcn,
                                                face_attn_mask)  # This is the true latent

        atten_vertex_embeddings = torch.sigmoid(atten_vertex_embeddings)
        atten_edge_embeddings = torch.sigmoid(atten_edge_embeddings)
        atten_face_embeddings = torch.sigmoid(atten_face_embeddings)

        v_data["edge_face_connectivity"] = edge_face_connectivity
        v_data["vertex_edge_connectivity"] = vertex_edge_connectivity
        v_data["face_mask"] = face_mask
        v_data["edge_mask"] = edge_mask
        v_data["vertex_mask"] = vertex_mask
        v_data["face_attn_mask"] = face_attn_mask
        v_data["edge_attn_mask"] = edge_attn_mask
        v_data["face_embeddings"] = face_embeddings
        v_data["edge_embeddings"] = edge_embeddings
        v_data["vertex_embeddings"] = vertex_embeddings
        v_data["atten_face_embeddings"] = atten_face_embeddings
        v_data["atten_edge_embeddings"] = atten_edge_embeddings
        v_data["atten_vertex_embeddings"] = atten_vertex_embeddings
        return

    def decode(self, v_data=None):
        # ================== Intersection  ==================
        recon_data = {}
        if v_data is not None:
            face_adj = v_data["face_adj"]
            edge_adj = v_data["edge_adj"]
            inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
                v_data["atten_face_embeddings"],
                v_data["atten_edge_embeddings"],
                v_data["edge_face_connectivity"],
                v_data["vertex_edge_connectivity"],
                face_adj, v_data["face_mask"],
                edge_adj, v_data["edge_mask"],
            )

        else:
            raise

        # Recover the shape features
        recon_data["proj_face_features"] = self.intersection_face_decoder(v_data["atten_face_embeddings"][..., None])[
            ..., 0]
        recon_data["proj_edge_features"] = self.intersection_edge_decoder(inter_edge_features[..., None])[..., 0]
        recon_data["proj_vertex_features"] = self.intersection_vertex_decoder(inter_vertex_features[..., None])[..., 0]

        recon_data["inter_edge_null_features"] = inter_edge_null_features
        recon_data["inter_vertex_null_features"] = inter_vertex_null_features
        recon_data["inter_edge_features"] = inter_edge_features
        recon_data["inter_vertex_features"] = inter_vertex_features

        # Decode with intersection feature
        recon_data.update(self.decoder(
            recon_data["proj_face_features"],
            recon_data["proj_edge_features"],
            recon_data["proj_vertex_features"],
        ))
        return recon_data

    def loss(self, v_data, v_recon_data):
        atten_edge_embeddings = v_data["atten_edge_embeddings"]
        atten_vertex_embeddings = v_data["atten_vertex_embeddings"]
        edge_mask = v_data["edge_mask"]
        vertex_mask = v_data["vertex_mask"]
        face_mask = v_data["face_mask"]

        used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
        used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]

        # ================== Normal Decoding  ==================
        vertex_data = self.decoder.decode_vertex(v_data["vertex_embeddings"])
        edge_data = self.decoder.decode_edge(v_data["edge_embeddings"])
        face_data = self.decoder.decode_face(v_data["face_embeddings"])

        loss = {}
        # Loss for predicting discrete points from the intersection features
        loss.update(self.decoder.loss(
            v_recon_data, v_data, face_mask,
            edge_mask, used_edge_indexes,
            vertex_mask, used_vertex_indexes
        ))

        # Loss for classifying the intersection features
        loss_edge, loss_vertex = self.intersector.loss(
            v_recon_data["inter_edge_features"], v_recon_data["inter_edge_null_features"],
            v_recon_data["inter_vertex_features"], v_recon_data["inter_vertex_null_features"]
        )
        loss.update({"intersection_edge": loss_edge})
        loss.update({"intersection_vertex": loss_vertex})

        # Loss for normal decoding edges
        loss_edge = self.decoder.loss_edge(
            edge_data, v_data, edge_mask,
            torch.arange(atten_edge_embeddings.shape[0]))
        for key in loss_edge:
            loss[key + "1"] = loss_edge[key]

        # Loss for normal decoding vertices
        loss_vertex = self.decoder.loss_vertex(
            vertex_data, v_data, vertex_mask,
            torch.arange(atten_vertex_embeddings.shape[0]))
        for key in loss_vertex:
            loss[key + "1"] = loss_vertex[key]

        # Loss for normal decoding faces
        loss_face = self.decoder.loss_face(
            face_data, v_data, face_mask)
        for key in loss_face:
            loss[key + "1"] = loss_face[key]

        loss["face_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_face_features"], v_data["face_embeddings"], reduction='mean')
        loss["edge_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_edge_features"], v_data["edge_embeddings"][used_edge_indexes], reduction='mean')
        loss["vertex_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_vertex_features"], v_data["vertex_embeddings"][used_vertex_indexes], reduction='mean')
        loss["inter_edge_l2"] = nn.functional.mse_loss(
            v_recon_data["inter_edge_features"], v_data["atten_edge_embeddings"], reduction='mean')
        loss["total_loss"] = sum(loss.values())
        return loss

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        self.encode(v_data)

        loss = {}
        # ================== Quantization  ==================
        if self.with_quantization:
            quantized_face_embeddings, indices = self.quantizer(v_data["atten_face_embeddings"][:, None])
            quantized_face_embeddings = quantized_face_embeddings[:, 0]
            indices = indices[:, 0]
            true_face_embeddings = self.quantizer_out(quantized_face_embeddings.unsqueeze(2))[..., 0]
            v_data["atten_face_embeddings"] = true_face_embeddings
            # loss["quantization_internal"] = quantized_loss.mean()
        else:
            indices = None

        recon_data = self.decode(v_data)

        loss.update(self.loss(
            v_data, recon_data
        ))
        # Compute model size and flops
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.encoder(v_data)
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.decoder(atten_face_edge_embeddings, intersected_edge_features)
        data = {}
        if return_recon:
            used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
            used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]
            face_mask = v_data["face_mask"]
            edge_mask = v_data["edge_mask"]
            vertex_mask = v_data["vertex_mask"]
            recon_face, recon_edges, recon_vertices = self.decoder.inference(recon_data)
            recon_face_full = recon_face.new_zeros(v_data["face_points"].shape)
            recon_face_full = recon_face_full.masked_scatter(rearrange(face_mask, '... -> ... 1 1 1'), recon_face)
            recon_face_full[~face_mask] = -1

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[used_edge_indexes] = recon_edges
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            recon_vertex_full = recon_vertex_full.masked_scatter(rearrange(vertex_mask, '... -> ... 1'), recon_vertices)
            recon_vertex_full[~vertex_mask] = -1

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.mse_loss(
                data["recon_faces"], v_data["face_points"], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            true_recon_edge_loss = nn.functional.mse_loss(
                recon_edges, v_data["edge_points"][edge_mask][used_edge_indexes], reduction='mean')
            loss["true_recon_edge"] = true_recon_edge_loss
            true_recon_vertex_loss = nn.functional.mse_loss(
                recon_vertices, v_data["vertex_points"][vertex_mask][used_vertex_indexes], reduction='mean')
            loss["true_recon_vertex"] = true_recon_vertex_loss

        if return_face_features:
            face_embeddings_return = true_face_embeddings.new_zeros(
                (*face_mask.shape, true_face_embeddings.shape[-1]))
            face_embeddings_return = face_embeddings_return.masked_scatter(
                rearrange(face_mask, '... -> ... 1'), true_face_embeddings)
            data["face_embeddings"] = face_embeddings_return

            if self.with_quantization:
                quantized_face_indices_return = indices.new_zeros(
                    (*face_mask.shape, indices.shape[-1]))
                quantized_face_indices_return = quantized_face_indices_return.masked_scatter(
                    rearrange(face_mask, '... -> ... 1'), indices)
                data["quantized_face_indices"] = quantized_face_indices_return
        return loss, data


class res_block_linear(nn.Module):
    def __init__(self, v_in, v_out):
        super(res_block_linear, self).__init__()
        self.conv = nn.Linear(v_in, v_out)
        self.act = nn.ReLU()
        if v_in != v_out:
            self.conv2 = nn.Linear(v_in, v_out)
        else:
            self.conv2 = None

    def forward(self, x):
        out = self.act(self.conv(x))
        if self.conv2 is not None:
            out = out + self.conv2(x)
        return out


class AutoEncoder2(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.with_quantization = v_conf["with_quantization"]
        self.finetune_decoder = v_conf["finetune_decoder"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        hidden_dim = self.dim_shape
        bbox_discrete_dim = v_conf["bbox_discrete_dim"]
        coor_discrete_dim = v_conf["coor_discrete_dim"]
        self.bbox_embedding = nn.Embedding(bbox_discrete_dim - 1, 64)
        self.coords_embedding = nn.Embedding(coor_discrete_dim - 1, 64)

        self.bbox_encoder = nn.Sequential(
            res_block_linear(6 * 64, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim // 2),
            res_block_linear(hidden_dim // 2, hidden_dim // 2),
            res_block_linear(hidden_dim // 2, hidden_dim // 4),
            res_block_linear(hidden_dim // 4, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            res_block_2D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),

            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            res_block_2D(hidden_dim // 2, hidden_dim // 2, ks=7, st=1, pa=3),

            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.face_fuser = nn.Sequential(
            res_block_linear(hidden_dim // 2, hidden_dim // 2),
            res_block_linear(hidden_dim // 2, hidden_dim // 4),
            res_block_linear(hidden_dim // 4, hidden_dim // 4),
            res_block_linear(hidden_dim // 4, hidden_dim // 8),
            res_block_linear(hidden_dim // 8, hidden_dim // 8),
            nn.Linear(hidden_dim // 8, hidden_dim // 8)
        )

        # ================== Quantization ==================
        self.quantizer_in = nn.Sequential(
            res_block_linear(hidden_dim // 8, hidden_dim // 8),
            nn.Linear(hidden_dim // 8, 6),
        )

        self.quantizer = ResidualFSQ(
            num_quantizers=4,
            dim=6,
            num_codebooks=1,
            levels=[8, 8, 8, 5, 5, 5],
        )

        self.quantizer_out = nn.Sequential(
            res_block_linear(6, hidden_dim // 8),
            res_block_linear(hidden_dim // 8, hidden_dim // 8),
            res_block_linear(hidden_dim // 8, hidden_dim // 4),
            res_block_linear(hidden_dim // 4, hidden_dim // 4),
            res_block_linear(hidden_dim // 4, hidden_dim // 2),
            res_block_linear(hidden_dim // 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ================== Decoder ==================
        self.bd = bbox_discrete_dim - 1  # discrete_dim
        self.cd = coor_discrete_dim - 1  # discrete_dim

        self.bbox_decoder = nn.Sequential(
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim * 2),
            res_block_linear(hidden_dim * 2, hidden_dim * 4),
            res_block_linear(hidden_dim * 4, hidden_dim * 4),
            res_block_linear(hidden_dim * 4, hidden_dim * 8),
            res_block_linear(hidden_dim * 8, hidden_dim * 8),
            res_block_linear(hidden_dim * 8, 6 * self.bd),
            Rearrange('...(p c) -> ... p c', p=6, c=self.bd),
        )

        # For faces
        self.coords_decoder = nn.Sequential(
            nn.Linear(self.dim_latent, hidden_dim),
            Rearrange('... c -> ... c 1 1'),

            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),

            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),

            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),

            res_block_2D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.Upsample(size=(20, 20), mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),

            nn.Conv2d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) w h -> ... w h p c', p=3, c=self.cd),
        )

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        face_coords = v_data["discrete_face_points"]
        face_bbox = v_data["discrete_face_bboxes"]
        face_mask = (face_coords != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        flatten_face_coords = face_coords[face_mask]
        flatten_face_bbox = face_bbox[face_mask]

        face_coords = rearrange(self.coords_embedding(flatten_face_coords), "... h w c d -> ... (c d) h w")
        face_bbox = rearrange(self.bbox_embedding(flatten_face_bbox), "... l c d -> ... l (c d)")

        face_coords_features = self.face_encoder(face_coords)
        face_bbox_features = self.bbox_encoder(face_bbox)

        face_features = self.face_fuser(torch.cat((face_coords_features, face_bbox_features), dim=1))
        face_features = self.quantizer_in(face_features)

        loss = {}
        if self.with_quantization:
            # FSQ with attn
            # attn_mask = get_attn_mask(face_mask)
            # quantized_features, indices = self.quantizer(face_embeddings[:,None])
            # quantized_features = quantized_features[:,0]
            # indices = indices[:,0]
            # quantized_features = self.quantizer_out(quantized_features, attn_mask)
            # true_face_latent = self.quantizer_out2(quantized_features)

            # # LFQ with resnet
            # quantized_features, indices, quantization_loss = self.quantizer(face_embeddings[:,None])
            # quantized_features = quantized_features[:,0]
            # indices = indices[:,0]
            # true_face_latent = self.quantizer_out(quantized_features[...,None])[...,0]
            # loss["quantization_internal"] = quantization_loss.mean()

            # FSQ with encoder
            # attn_mask = get_attn_mask(face_mask)
            quantized_features, indices = self.quantizer(face_features[:, None])
            true_face_latent = quantized_features[:, 0]
            indices = indices[:, 0]
            # true_face_latent = quantized_features
        else:
            true_face_latent = face_features
        true_face_latent = self.quantizer_out(true_face_latent)

        # Decode with normal feature
        face_bbox_logits = self.bbox_decoder(true_face_latent, )
        face_coords_logits = self.coords_decoder(true_face_latent, )
        # =============================== Loss for normal decoding ===============================
        gt_face_bbox = v_data["discrete_face_bboxes"][face_mask]
        gt_face_coords = v_data["discrete_face_points"][face_mask]

        loss_face_coords = nn.functional.cross_entropy(face_coords_logits.flatten(0, -2),
                                                       gt_face_coords.flatten())
        loss_face_bbox = nn.functional.cross_entropy(face_bbox_logits.flatten(0, -2),
                                                     gt_face_bbox.flatten())
        loss["total_loss"] = (loss_face_coords + loss_face_bbox).sum()

        data = {}
        if return_recon:
            # used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
            # used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]

            bbox_shifts = (self.bd + 1) // 2 - 1
            coord_shifts = (self.cd + 1) // 2 - 1

            face_bbox = (face_bbox_logits.argmax(dim=-1) - bbox_shifts) / bbox_shifts
            face_center = (face_bbox[:, 3:] + face_bbox[:, :3]) / 2
            face_length = (face_bbox[:, 3:] - face_bbox[:, :3])
            face_coords = (face_coords_logits.argmax(dim=-1) - coord_shifts) / coord_shifts / 2
            face_coords = face_coords * face_length[:, None, None] + face_center[:, None, None]

            recon_face_full = face_coords.new_zeros(v_data["face_points"].shape)
            recon_face_full = recon_face_full.masked_scatter(rearrange(face_mask, '... -> ... 1 1 1'), face_coords)
            recon_face_full[~face_mask] = -1

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.mse_loss(
                recon_face_full, v_data["face_points"], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
        return loss, data


def init_code(v_quantizer, v_codebook_size):
    print("Pre-compute the codebook")
    training_root = Path(r"G:/Dataset/img2brep/0501_512_512_woquantized_2attn_big_389_training")
    features = []
    for file in tqdm(os.listdir("G:/Dataset/img2brep/0501_512_512_woquantized_2attn_big_389_training")):
        feature = np.load(training_root / file, allow_pickle=True)
        features.append(feature)
    features = np.concatenate(features, axis=0)
    pass
    # KMEANS
    kmeans = MiniBatchKMeans(n_clusters=v_codebook_size, random_state=0).fit(features)
    v_quantizer.layers[0]._codebook = nn.Parameter(torch.tensor(kmeans.cluster_centers_).float())


class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """

    def __init__(self, n_embed, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-0.04, 0.04)
        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        B, C = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)

        # DeepMind def does not do this but I find I have to... ;\
        if False and self.training and self.data_initialized.item() == 0:
            print('running kmeans!!')  # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        dist = (
                z.pow(2).sum(1, keepdim=True)
                - 2 * z @ self.embed.weight.t()
                + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.view(B)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind)  # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z + (z_q - z).detach()  # noop in forward pass, straight-through gradient estimator in backward pass
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class Face_vq_fitting(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(Face_vq_fitting, self).__init__()

        hidden_dim = 512

        self.quantizer_in = nn.Sequential(
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ================== Quantization ==================
        self.quantizer = ResidualFSQ(
            dim=hidden_dim,
            levels=[2] * 16,
            num_quantizers=4,
            num_codebooks=2,
            keep_num_codebooks_dim=True,
        )
        # self.quantizer = ResidualVQ(
        #     dim=192,
        #     codebook_dim=192,
        #     num_quantizers=4,
        #     codebook_size=self.codebook_size,
        #
        #     shared_codebook=True,
        #     quantize_dropout=False,
        #
        #     kmeans_init=False,
        #     kmeans_iters=10,
        #     sync_kmeans=True,
        #     use_cosine_sim=False,
        #     threshold_ema_dead_code=0,
        #     stochastic_sample_codes=False,
        #     sample_codebook_temp=1.,
        #     straight_through=False,
        #     reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
        # )
        # init_code(self.quantizer, self.codebook_size)

        self.quantizer_out = nn.Sequential(
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            res_block_linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        face_feature = v_data
        face_mask = (face_feature != 0).all(dim=-1)

        face_feature = face_feature[face_mask]
        encoded_face_feature = self.quantizer_in(face_feature)
        true_face_latent, indices = self.quantizer(encoded_face_feature[:, None])
        true_face_latent = true_face_latent[:, 0]
        indices = indices[:, 0]
        true_face_latent = self.quantizer_out(true_face_latent)

        loss = {}
        loss["quantization_l2"] = F.mse_loss(true_face_latent, face_feature, reduction='mean')
        loss["total_loss"] = sum(loss.values())

        return loss, {}


class AutoEncoder3(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder3, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.with_quantization = v_conf["with_quantization"]
        self.finetune_decoder = v_conf["finetune_decoder"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        hidden_dim = 6 if self.with_quantization else 3

        self.encoder = AutoencoderKL(in_channels=3,
                                     out_channels=3,
                                     down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                       'DownEncoderBlock2D'],
                                     up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                     'UpDecoderBlock2D'],
                                     block_out_channels=[128, 256, 512, 512],
                                     layers_per_block=2,
                                     act_fn='silu',
                                     latent_channels=hidden_dim,
                                     norm_num_groups=32,
                                     sample_size=512,
                                     )

        # ================== Quantization ==================
        if self.with_quantization:
            self.quantizer = FSQ(
                dim=6,
                num_codebooks=1,
                levels=[8, 8, 8, 5, 5, 5],
            )

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        face_points = v_data["face_points"]
        face_mask = (face_points != -1).all(dim=-1).all(dim=-1).all(dim=-1)

        flatten_face_points = face_points[face_mask].permute(0, 3, 1, 2)
        posterior = self.encoder.encode(flatten_face_points).latent_dist

        loss = {}
        if self.with_quantization:
            face_features = posterior.mean
            true_face_latent, indices = self.quantizer(face_features)
            # true_face_latent = quantized_features[:, 0]
            # indices = indices[:, 0]
            # true_face_latent = quantized_features
            # true_face_latent = self.quantizer_out(true_face_latent)
        else:
            true_face_latent = posterior.sample()

        # Decode with normal feature
        predicted_face_points = self.encoder.decode(true_face_latent).sample
        # =============================== Loss for normal decoding ===============================
        loss["face_points"] = nn.functional.l1_loss(flatten_face_points, predicted_face_points)
        if self.with_quantization:
            loss["total_loss"] = loss["face_points"]
        else:
            loss["kl"] = posterior.kl().mean()
            loss["total_loss"] = loss["face_points"] + loss["kl"] * 1e-6

        data = {}
        if return_recon:
            # used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
            # used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]
            recon_face_full = predicted_face_points.new_zeros(v_data["face_points"].shape)
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), predicted_face_points.permute(0, 2, 3, 1))
            recon_face_full[~face_mask] = -1

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.mse_loss(
                recon_face_full, v_data["face_points"], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            data["recon_faces"] = recon_face_full
        return loss, data


from diffusers.models.unets.unet_1d_blocks import ResConvBlock, SelfAttention1d, get_down_block, get_up_block, \
    Upsample1d
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states


class Encoder1D(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock1D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv1d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv1d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )

                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)[0]

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder1D(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpBlock1D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            # sample = sample.to(upscale_dtype)
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL1D(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            scaling_factor: float = 0.18215,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv1d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
            self,
            sample: torch.FloatTensor,
            sample_posterior: bool = False,  # True
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


class AutoEncoder4(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder4, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.with_quantization = v_conf["with_quantization"]
        self.finetune_decoder = v_conf["finetune_decoder"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_encoder')
        self.encoder = getattr(mod, v_conf["encoder"])(
            dim=self.dim_shape,
            bbox_discrete_dim=v_conf["bbox_discrete_dim"],
            coor_discrete_dim=v_conf["coor_discrete_dim"],
        )
        self.vertices_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )
        self.edges_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )
        self.faces_proj = nn.Sequential(
            nn.Linear(self.dim_shape, self.dim_latent),
            nn.Sigmoid()
        )

        # ================== GCN to distribute features across primitives ==================
        if v_conf["graphconv"] == "GAT":
            GraphConv = GAT_GraphConv
        else:
            GraphConv = SAGE_GraphConv
        encoder_dims_through_depth_edges = [self.dim_latent for _ in range(4)]
        encoder_dims_through_depth_faces = [self.dim_latent for _ in range(4)]
        self.gcn_on_edges = GraphConv(self.dim_latent, encoder_dims_through_depth_edges,
                                      self.dim_latent)
        self.gcn_on_faces = GraphConv(self.dim_latent, encoder_dims_through_depth_faces,
                                      self.dim_latent, edge_dim=self.dim_latent)

        # ================== self attention to aggregate features ==================
        self.vertex_fuser = Attn_fuser_single(self.dim_latent)
        self.edge_fuser = Attn_fuser_single(self.dim_latent)
        self.face_fuser = Attn_fuser_single(self.dim_latent)

        # ================== cross attention to aggregate features ==================
        self.fuser_vertices_to_edges = Attn_fuser_cross(self.dim_latent)
        self.fuser_edges_to_faces = Attn_fuser_cross(self.dim_latent)

        # ================== Intersection ==================
        mod = importlib.import_module('src.img2brep.brep.model_intersector')
        self.intersector = getattr(mod, v_conf["intersector"])(500, self.dim_latent)

        # ================== Decoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_decoder')
        self.decoder = getattr(mod, v_conf["decoder"])(
            dim_in=self.dim_latent,
            hidden_dim=v_conf["dim_decoder"],
            bbox_discrete_dim=v_conf["bbox_discrete_dim"],
            coor_discrete_dim=v_conf["coor_discrete_dim"],
        )

        self.intersection_face_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.intersection_edge_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.intersection_vertex_decoder = nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.frozen_models = []
        # ================== Quantization ==================
        self.with_quantization = v_conf["with_quantization"]
        if self.with_quantization:
            if False:
                self.quantizer = VectorQuantize(
                    dim=self.dim_latent,
                    codebook_dim=32,  # a number of papers have shown smaller codebook dimension to be acceptable
                    heads=8,  # number of heads to vector quantize, codebook shared across all heads
                    separate_codebook_per_head=True,
                    # whether to have a separate codebook per head. False would mean 1 shared codebook
                    codebook_size=8196,
                    accept_image_fmap=False
                )
                self.quantizer_proj = nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1),
                    nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1),
                )
                self.frozen_models = []
            else:
                self.quantizer_out = nn.Sequential(
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Sigmoid(),
                )

                # self.quantizer = ResidualVQ(
                #     dim=self.dim_latent,
                #     codebook_dim=32,
                #     num_quantizers=8,
                #     codebook_size=2048,
                # )

                layer = nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, batch_first=True, dropout=0.1)
                self.quantizer_in = nn.Sequential(
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
                    nn.Sigmoid(),
                )

                self.quantizer = ResidualFSQ(
                    num_quantizers=16,
                    dim=self.dim_latent,
                    num_codebooks=1,
                    levels=[8, 8, 8, 5, 5, 5],
                )

                # self.quantizer = ResidualLFQ(
                #     num_quantizers=4,
                #     dim=self.dim_latent,
                #     codebook_size=16384,
                #     num_codebooks=1,
                # )

                self.frozen_models = [
                    # self.encoder, self.vertices_proj, self.edges_proj, self.faces_proj,
                    # self.gcn_on_edges, self.gcn_on_faces, self.edge_fuser, self.face_fuser,
                    # self.fuser_vertices_to_edges, self.fuser_edges_to_faces,
                    # self.decoder, self.intersector
                ]

        # ================== Freeze models ==================
        # for model in self.frozen_models:
        #     model.eval()
        #     for param in model.parameters():
        #         param.requires_grad = False

    # Inference (B * num_faces * num_features)
    # Pad features are all zeros
    # B==1 currently
    def inference(self, v_face_embeddings, return_topology=False):
        # Use face to intersect edges
        B = v_face_embeddings.shape[0]
        device = v_face_embeddings.device
        assert B == 1
        num_faces = v_face_embeddings.shape[1]
        face_idx = torch.stack(torch.meshgrid(
            torch.arange(num_faces), torch.arange(num_faces), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_face_features = v_face_embeddings[0, face_idx]

        edge_features = self.intersector.inference(gathered_face_features, "edge")
        edge_intersection_mask = self.intersector.inference_label(edge_features)
        edge_features = edge_features[edge_intersection_mask]
        num_edges = edge_features.shape[0]

        # Use edge to intersect vertices
        edge_idx = torch.stack(torch.meshgrid(
            torch.arange(num_edges), torch.arange(num_edges), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_edge_features = edge_features[edge_idx]

        if gathered_edge_features.shape[0] < 64 * 64:
            vertex_features = self.intersector.inference(gathered_edge_features, "vertex")
            vertex_intersection_mask = self.intersector.inference_label(vertex_features)
            vertex_features = vertex_features[vertex_intersection_mask]
        else:
            vertex_features = gathered_edge_features.new_zeros(0, gathered_edge_features.shape[-1])

        # Decode
        recon_data = self.decoder(
            v_face_embeddings.view(-1, v_face_embeddings.shape[-1]),
            edge_features,
            vertex_features,
        )
        recon_faces, recon_edges, recon_vertices = self.decoder.inference(recon_data)
        if return_topology:
            face_edge_connectivity = torch.cat((
                torch.arange(num_edges, device=device)[:, None], face_idx[edge_intersection_mask],), dim=1)
            edge_vertex_connectivity = torch.cat((
                torch.arange(vertex_features.shape[0], device=device)[:, None], edge_idx[vertex_intersection_mask],),
                dim=1)
            return recon_vertices, recon_edges, recon_faces, face_edge_connectivity, edge_vertex_connectivity
        return recon_vertices, recon_edges, recon_faces

    def encode(self, v_data):
        # ================== Encode the edge and face points ==================
        face_embeddings, edge_embeddings, vertex_embeddings, face_mask, edge_mask, vertex_mask = self.encoder(v_data)
        face_embeddings = self.faces_proj(face_embeddings)
        edge_embeddings = self.edges_proj(edge_embeddings)
        vertex_embeddings = self.vertices_proj(vertex_embeddings)

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

        edge_face_connectivity = v_data["edge_face_connectivity"].clone()
        edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
        # Solve the edge_face_connectivity: last two dimension (id_face)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        # Solve the edge_face_connectivity: first dimension (id_edge)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]

        # ================== Self-attention on vertices ==================
        b, n = vertex_mask.shape
        batch_indices = torch.arange(b, device=vertex_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[vertex_mask]
        vertex_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        atten_vertex_embeddings = self.vertex_fuser(vertex_embeddings, vertex_attn_mask)

        # ================== Fuse vertex features to the corresponding edges ==================
        b, n = edge_mask.shape
        batch_indices = torch.arange(b, device=edge_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[edge_mask]
        edge_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        edge_vertex_embeddings = self.fuser_vertices_to_edges(
            v_embeddings1=atten_vertex_embeddings,
            v_embeddings2=edge_embeddings,
            v_connectivity1_to_2=vertex_edge_connectivity,
            v_attn_mask=edge_attn_mask
        )

        # ================== GCN and self-attention on edges ==================
        edge_embeddings_gcn = self.gcn_on_edges(edge_vertex_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_attn_mask)

        # ================== fuse edges features to the corresponding faces ==================
        b, n = face_mask.shape
        batch_indices = torch.arange(b, device=face_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[face_mask]
        face_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        face_edge_embeddings = self.fuser_edges_to_faces(
            v_connectivity1_to_2=edge_face_connectivity,
            v_embeddings1=atten_edge_embeddings,
            v_embeddings2=face_embeddings,
            v_attn_mask=face_attn_mask,
        )

        # ================== GCN and self-attention on faces  ==================
        face_edge_embeddings_gcn = self.gcn_on_faces(face_edge_embeddings,
                                                     edge_face_connectivity[..., 1:].permute(1, 0),
                                                     edge_attr=atten_edge_embeddings[edge_face_connectivity[..., 0]])

        atten_face_embeddings = self.face_fuser(face_edge_embeddings_gcn,
                                                face_attn_mask)  # This is the true latent

        atten_vertex_embeddings = torch.sigmoid(atten_vertex_embeddings)
        atten_edge_embeddings = torch.sigmoid(atten_edge_embeddings)
        atten_face_embeddings = torch.sigmoid(atten_face_embeddings)

        v_data["edge_face_connectivity"] = edge_face_connectivity
        v_data["vertex_edge_connectivity"] = vertex_edge_connectivity
        v_data["face_mask"] = face_mask
        v_data["edge_mask"] = edge_mask
        v_data["vertex_mask"] = vertex_mask
        v_data["face_attn_mask"] = face_attn_mask
        v_data["edge_attn_mask"] = edge_attn_mask
        v_data["face_embeddings"] = face_embeddings
        v_data["edge_embeddings"] = edge_embeddings
        v_data["vertex_embeddings"] = vertex_embeddings
        v_data["atten_face_embeddings"] = atten_face_embeddings
        v_data["atten_edge_embeddings"] = atten_edge_embeddings
        v_data["atten_vertex_embeddings"] = atten_vertex_embeddings
        return

    def decode(self, v_data=None):
        # ================== Intersection  ==================
        recon_data = {}
        if v_data is not None:
            face_adj = v_data["face_adj"]
            edge_adj = v_data["edge_adj"]
            inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
                v_data["atten_face_embeddings"],
                v_data["atten_edge_embeddings"],
                v_data["edge_face_connectivity"],
                v_data["vertex_edge_connectivity"],
                face_adj, v_data["face_mask"],
                edge_adj, v_data["edge_mask"],
            )

        else:
            raise

        # Recover the shape features
        recon_data["proj_face_features"] = self.intersection_face_decoder(
            v_data["atten_face_embeddings"][..., None])[..., 0]
        recon_data["proj_edge_features"] = self.intersection_edge_decoder(inter_edge_features[..., None])[..., 0]
        recon_data["proj_vertex_features"] = self.intersection_vertex_decoder(inter_vertex_features[..., None])[..., 0]

        recon_data["inter_edge_null_features"] = inter_edge_null_features
        recon_data["inter_vertex_null_features"] = inter_vertex_null_features
        recon_data["inter_edge_features"] = inter_edge_features
        recon_data["inter_vertex_features"] = inter_vertex_features

        # Decode with intersection feature
        recon_data.update(self.decoder(
            recon_data["proj_face_features"],
            recon_data["proj_edge_features"],
            recon_data["proj_vertex_features"],
        ))
        return recon_data

    def loss(self, v_data, v_recon_data):
        atten_edge_embeddings = v_data["atten_edge_embeddings"]
        atten_vertex_embeddings = v_data["atten_vertex_embeddings"]
        edge_mask = v_data["edge_mask"]
        vertex_mask = v_data["vertex_mask"]
        face_mask = v_data["face_mask"]

        used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
        used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]

        # ================== Normal Decoding  ==================
        vertex_data = self.decoder.decode_vertex(v_data["vertex_embeddings"])
        edge_data = self.decoder.decode_edge(v_data["edge_embeddings"])
        face_data = self.decoder.decode_face(v_data["face_embeddings"])

        loss = {}
        # Loss for predicting discrete points from the intersection features
        loss.update(self.decoder.loss(
            v_recon_data, v_data, face_mask,
            edge_mask, used_edge_indexes,
            vertex_mask, used_vertex_indexes
        ))

        # Loss for classifying the intersection features
        loss_edge, loss_vertex = self.intersector.loss(
            v_recon_data["inter_edge_features"], v_recon_data["inter_edge_null_features"],
            v_recon_data["inter_vertex_features"], v_recon_data["inter_vertex_null_features"]
        )
        loss.update({"intersection_edge": loss_edge})
        loss.update({"intersection_vertex": loss_vertex})

        # Loss for normal decoding edges
        loss_edge = self.decoder.loss_edge(
            edge_data, v_data, edge_mask,
            torch.arange(atten_edge_embeddings.shape[0]))
        for key in loss_edge:
            loss[key + "1"] = loss_edge[key]

        # Loss for normal decoding vertices
        loss_vertex = self.decoder.loss_vertex(
            vertex_data, v_data, vertex_mask,
            torch.arange(atten_vertex_embeddings.shape[0]))
        for key in loss_vertex:
            loss[key + "1"] = loss_vertex[key]

        # Loss for normal decoding faces
        loss_face = self.decoder.loss_face(
            face_data, v_data, face_mask)
        for key in loss_face:
            loss[key + "1"] = loss_face[key]

        loss["face_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_face_features"], v_data["face_embeddings"], reduction='mean')
        loss["edge_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_edge_features"], v_data["edge_embeddings"][used_edge_indexes], reduction='mean')
        loss["vertex_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_vertex_features"], v_data["vertex_embeddings"][used_vertex_indexes], reduction='mean')
        # loss["inter_edge_l2"] = nn.functional.mse_loss(
        #     v_recon_data["inter_edge_features"], v_data["atten_edge_embeddings"], reduction='mean')
        loss["total_loss"] = sum(loss.values())
        return loss

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        self.encode(v_data)

        loss = {}
        # ================== Quantization  ==================
        if self.with_quantization:
            quantized_face_embeddings, indices = self.quantizer(v_data["atten_face_embeddings"][:, None])
            quantized_face_embeddings = quantized_face_embeddings[:, 0]
            indices = indices[:, 0]
            true_face_embeddings = self.quantizer_out(quantized_face_embeddings.unsqueeze(2))[..., 0]
            v_data["atten_face_embeddings"] = true_face_embeddings
            # loss["quantization_internal"] = quantized_loss.mean()
        else:
            indices = None

        recon_data = self.decode(v_data)

        loss.update(self.loss(
            v_data, recon_data
        ))
        # Compute model size and flops
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.encoder(v_data)
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.decoder(atten_face_edge_embeddings, intersected_edge_features)
        data = {}
        if return_recon:
            used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
            used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]
            face_mask = v_data["face_mask"]
            edge_mask = v_data["edge_mask"]
            vertex_mask = v_data["vertex_mask"]
            recon_face, recon_edges, recon_vertices = self.decoder.inference(recon_data)
            recon_face_full = recon_face.new_zeros(v_data["face_points"].shape)
            recon_face_full = recon_face_full.masked_scatter(rearrange(face_mask, '... -> ... 1 1 1'), recon_face)
            recon_face_full[~face_mask] = -1

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[used_edge_indexes] = recon_edges
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            recon_vertex_full = recon_vertex_full.masked_scatter(rearrange(vertex_mask, '... -> ... 1'), recon_vertices)
            recon_vertex_full[~vertex_mask] = -1

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.mse_loss(
                data["recon_faces"], v_data["face_points"], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            true_recon_edge_loss = nn.functional.mse_loss(
                recon_edges, v_data["edge_points"][edge_mask][used_edge_indexes], reduction='mean')
            loss["true_recon_edge"] = true_recon_edge_loss
            true_recon_vertex_loss = nn.functional.mse_loss(
                recon_vertices, v_data["vertex_points"][vertex_mask][used_vertex_indexes], reduction='mean')
            loss["true_recon_vertex"] = true_recon_vertex_loss

        if return_face_features:
            face_embeddings_return = true_face_embeddings.new_zeros(
                (*face_mask.shape, true_face_embeddings.shape[-1]))
            face_embeddings_return = face_embeddings_return.masked_scatter(
                rearrange(face_mask, '... -> ... 1'), true_face_embeddings)
            data["face_embeddings"] = face_embeddings_return

            if self.with_quantization:
                quantized_face_indices_return = indices.new_zeros(
                    (*face_mask.shape, indices.shape[-1]))
                quantized_face_indices_return = quantized_face_indices_return.masked_scatter(
                    rearrange(face_mask, '... -> ... 1'), indices)
                data["quantized_face_indices"] = quantized_face_indices_return
        return loss, data


class AutoEncoder5(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder5, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.with_quantization = v_conf["with_quantization"]
        self.finetune_decoder = v_conf["finetune_decoder"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        self.bd = v_conf["bbox_discrete_dim"] - 1
        self.cd = v_conf["coor_discrete_dim"] - 1
        self.bbox_embedding = nn.Embedding(self.bd, 64)
        self.coords_embedding = nn.Embedding(self.cd, 64)

        hidden_dim = 256
        self.face_coords = nn.Sequential(
            nn.Embedding(v_conf["coor_discrete_dim"] - 1, 64),
            Rearrange('b h w n c -> b (n c) h w'),
            nn.Conv2d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
            res_block_2D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=1),
        )
        self.face_bbox = nn.Sequential(
            nn.Embedding(self.bd, 64),
            Rearrange('b n c-> b (n c) 1'),
            nn.Conv1d(6 * 64, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

        self.face_fuser = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_2D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_2D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_2D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

        self.edge_bbox = nn.Sequential(
            nn.Embedding(self.bd, 64),
            Rearrange('b n c-> b (n c) 1'),
            nn.Conv1d(6 * 64, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )
        self.edge_coords = nn.Sequential(
            nn.Embedding(self.cd, 64),
            Rearrange('b h n c -> b (n c) h'),
            nn.Conv1d(3 * 64, hidden_dim, kernel_size=7, stride=1, padding=3),
            res_block_1D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=1),
        )

        self.edge_fuser = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

        self.vertices_encoder = nn.Sequential(
            nn.Embedding(self.cd, 64),
            Rearrange('b n c -> b (n c) 1'),
            nn.Conv1d(3 * 64, hidden_dim, kernel_size=1, stride=1, padding=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
        )

        # ================== Decoder ==================
        self.face_bbox_decoder = nn.Sequential(
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            Rearrange('... 1 1 -> ... (1 1)'),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, 6 * self.bd, kernel_size=1, stride=1, padding=0),
            Rearrange('...(p c) 1-> ... p c', p=6, c=self.bd),
        )

        self.face_coords_decoder = nn.Sequential(
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.Conv2d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) w h -> ... w h p c', p=3, c=self.cd),
        )

        self.edge_bbox_decoder = nn.Sequential(
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.MaxPool1d(kernel_size=4, stride=4),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, 6 * self.bd, kernel_size=1, stride=1, padding=0),
            Rearrange('...(p c) 1-> ... p c', p=6, c=self.bd),
        )

        self.edge_coords_decoder = nn.Sequential(
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim, ks=3, st=1, pa=1),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim, ks=5, st=1, pa=2),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim, ks=7, st=1, pa=3),
            nn.Conv1d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) w-> ... w p c', p=3, c=self.cd),
        )

        self.vertex_coords_decoder = nn.Sequential(
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) 1-> ... (1 p) c', p=3, c=self.cd),
        )

        self.frozen_models = []
        # ================== Quantization ==================
        self.with_quantization = v_conf["with_quantization"]
        self.with_vae = v_conf["with_vae"]
        if self.with_quantization:
            self.quantizer_in = nn.Sequential(
                nn.Conv2d(hidden_dim, 6, kernel_size=1, stride=1, padding=0),
            )
            self.quantizer = FSQ(
                dim=6,
                num_codebooks=1,
                levels=[8, 8, 8, 5, 5, 5],
            )
            self.quantizer_out = nn.Sequential(
                nn.Conv2d(6, hidden_dim, kernel_size=1, stride=1, padding=0),
            )
        elif self.with_vae:
            self.vae_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, 6, kernel_size=1, stride=1, padding=0),
            )
            self.vae_out = nn.Sequential(
                nn.Conv2d(3, hidden_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Identity()
            )

            self.bottleneck = nn.Sequential(
                nn.Conv2d(hidden_dim, 6, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(6, hidden_dim, kernel_size=1, stride=1, padding=0),
            )

    # Inference (B * num_faces * num_features)
    # Pad features are all zeros
    # B==1 currently
    def inference(self, v_face_embeddings, return_topology=False):
        # Use face to intersect edges
        B = v_face_embeddings.shape[0]
        device = v_face_embeddings.device
        assert B == 1
        num_faces = v_face_embeddings.shape[1]
        face_idx = torch.stack(torch.meshgrid(
            torch.arange(num_faces), torch.arange(num_faces), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_face_features = v_face_embeddings[0, face_idx]

        edge_features = self.intersector.inference(gathered_face_features, "edge")
        edge_intersection_mask = self.intersector.inference_label(edge_features)
        edge_features = edge_features[edge_intersection_mask]
        num_edges = edge_features.shape[0]

        # Use edge to intersect vertices
        edge_idx = torch.stack(torch.meshgrid(
            torch.arange(num_edges), torch.arange(num_edges), indexing="xy"), dim=2
        ).reshape(-1, 2).to(device)
        gathered_edge_features = edge_features[edge_idx]

        if gathered_edge_features.shape[0] < 64 * 64:
            vertex_features = self.intersector.inference(gathered_edge_features, "vertex")
            vertex_intersection_mask = self.intersector.inference_label(vertex_features)
            vertex_features = vertex_features[vertex_intersection_mask]
        else:
            vertex_features = gathered_edge_features.new_zeros(0, gathered_edge_features.shape[-1])

        # Decode
        recon_data = self.decoder(
            v_face_embeddings.view(-1, v_face_embeddings.shape[-1]),
            edge_features,
            vertex_features,
        )
        recon_faces, recon_edges, recon_vertices = self.decoder.inference(recon_data)
        if return_topology:
            face_edge_connectivity = torch.cat((
                torch.arange(num_edges, device=device)[:, None], face_idx[edge_intersection_mask],), dim=1)
            edge_vertex_connectivity = torch.cat((
                torch.arange(vertex_features.shape[0], device=device)[:, None], edge_idx[vertex_intersection_mask],),
                dim=1)
            return recon_vertices, recon_edges, recon_faces, face_edge_connectivity, edge_vertex_connectivity
        return recon_vertices, recon_edges, recon_faces

    def encode(self, v_data):
        # ================== Encode the edge and face points ==================
        face_embeddings, edge_embeddings, vertex_embeddings, face_mask, edge_mask, vertex_mask = self.encoder(v_data)
        face_embeddings = self.faces_proj(face_embeddings)
        edge_embeddings = self.edges_proj(edge_embeddings)
        vertex_embeddings = self.vertices_proj(vertex_embeddings)

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

        edge_face_connectivity = v_data["edge_face_connectivity"].clone()
        edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
        # Solve the edge_face_connectivity: last two dimension (id_face)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        # Solve the edge_face_connectivity: first dimension (id_edge)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]

        # ================== Self-attention on vertices ==================
        b, n = vertex_mask.shape
        batch_indices = torch.arange(b, device=vertex_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[vertex_mask]
        vertex_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        atten_vertex_embeddings = self.vertex_fuser(vertex_embeddings, vertex_attn_mask)

        # ================== Fuse vertex features to the corresponding edges ==================
        b, n = edge_mask.shape
        batch_indices = torch.arange(b, device=edge_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[edge_mask]
        edge_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        edge_vertex_embeddings = self.fuser_vertices_to_edges(
            v_embeddings1=atten_vertex_embeddings,
            v_embeddings2=edge_embeddings,
            v_connectivity1_to_2=vertex_edge_connectivity,
            v_attn_mask=edge_attn_mask
        )

        # ================== GCN and self-attention on edges ==================
        edge_embeddings_gcn = self.gcn_on_edges(edge_vertex_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_attn_mask)

        # ================== fuse edges features to the corresponding faces ==================
        b, n = face_mask.shape
        batch_indices = torch.arange(b, device=face_mask.device).unsqueeze(1).repeat(1, n)
        batch_indices = batch_indices[face_mask]
        face_attn_mask = ~(batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1))
        face_edge_embeddings = self.fuser_edges_to_faces(
            v_connectivity1_to_2=edge_face_connectivity,
            v_embeddings1=atten_edge_embeddings,
            v_embeddings2=face_embeddings,
            v_attn_mask=face_attn_mask,
        )

        # ================== GCN and self-attention on faces  ==================
        face_edge_embeddings_gcn = self.gcn_on_faces(face_edge_embeddings,
                                                     edge_face_connectivity[..., 1:].permute(1, 0),
                                                     edge_attr=atten_edge_embeddings[edge_face_connectivity[..., 0]])

        atten_face_embeddings = self.face_fuser(face_edge_embeddings_gcn,
                                                face_attn_mask)  # This is the true latent

        atten_vertex_embeddings = torch.sigmoid(atten_vertex_embeddings)
        atten_edge_embeddings = torch.sigmoid(atten_edge_embeddings)
        atten_face_embeddings = torch.sigmoid(atten_face_embeddings)

        v_data["edge_face_connectivity"] = edge_face_connectivity
        v_data["vertex_edge_connectivity"] = vertex_edge_connectivity
        v_data["face_mask"] = face_mask
        v_data["edge_mask"] = edge_mask
        v_data["vertex_mask"] = vertex_mask
        v_data["face_attn_mask"] = face_attn_mask
        v_data["edge_attn_mask"] = edge_attn_mask
        v_data["face_embeddings"] = face_embeddings
        v_data["edge_embeddings"] = edge_embeddings
        v_data["vertex_embeddings"] = vertex_embeddings
        v_data["atten_face_embeddings"] = atten_face_embeddings
        v_data["atten_edge_embeddings"] = atten_edge_embeddings
        v_data["atten_vertex_embeddings"] = atten_vertex_embeddings
        return

    def decode(self, v_data=None):
        # ================== Intersection  ==================
        recon_data = {}
        if v_data is not None:
            face_adj = v_data["face_adj"]
            edge_adj = v_data["edge_adj"]
            inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
                v_data["atten_face_embeddings"],
                v_data["atten_edge_embeddings"],
                v_data["edge_face_connectivity"],
                v_data["vertex_edge_connectivity"],
                face_adj, v_data["face_mask"],
                edge_adj, v_data["edge_mask"],
            )

        else:
            raise

        # Recover the shape features
        recon_data["proj_face_features"] = self.intersection_face_decoder(
            v_data["atten_face_embeddings"][..., None])[..., 0]
        recon_data["proj_edge_features"] = self.intersection_edge_decoder(inter_edge_features[..., None])[..., 0]
        recon_data["proj_vertex_features"] = self.intersection_vertex_decoder(inter_vertex_features[..., None])[..., 0]

        recon_data["inter_edge_null_features"] = inter_edge_null_features
        recon_data["inter_vertex_null_features"] = inter_vertex_null_features
        recon_data["inter_edge_features"] = inter_edge_features
        recon_data["inter_vertex_features"] = inter_vertex_features

        # Decode with intersection feature
        recon_data.update(self.decoder(
            recon_data["proj_face_features"],
            recon_data["proj_edge_features"],
            recon_data["proj_vertex_features"],
        ))
        return recon_data

    def loss(self, v_data, v_recon_data):
        atten_edge_embeddings = v_data["atten_edge_embeddings"]
        atten_vertex_embeddings = v_data["atten_vertex_embeddings"]
        edge_mask = v_data["edge_mask"]
        vertex_mask = v_data["vertex_mask"]
        face_mask = v_data["face_mask"]

        used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
        used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]

        # ================== Normal Decoding  ==================
        vertex_data = self.decoder.decode_vertex(v_data["vertex_embeddings"])
        edge_data = self.decoder.decode_edge(v_data["edge_embeddings"])
        face_data = self.decoder.decode_face(v_data["face_embeddings"])

        loss = {}
        # Loss for predicting discrete points from the intersection features
        loss.update(self.decoder.loss(
            v_recon_data, v_data, face_mask,
            edge_mask, used_edge_indexes,
            vertex_mask, used_vertex_indexes
        ))

        # Loss for classifying the intersection features
        loss_edge, loss_vertex = self.intersector.loss(
            v_recon_data["inter_edge_features"], v_recon_data["inter_edge_null_features"],
            v_recon_data["inter_vertex_features"], v_recon_data["inter_vertex_null_features"]
        )
        loss.update({"intersection_edge": loss_edge})
        loss.update({"intersection_vertex": loss_vertex})

        # Loss for normal decoding edges
        loss_edge = self.decoder.loss_edge(
            edge_data, v_data, edge_mask,
            torch.arange(atten_edge_embeddings.shape[0]))
        for key in loss_edge:
            loss[key + "1"] = loss_edge[key]

        # Loss for normal decoding vertices
        loss_vertex = self.decoder.loss_vertex(
            vertex_data, v_data, vertex_mask,
            torch.arange(atten_vertex_embeddings.shape[0]))
        for key in loss_vertex:
            loss[key + "1"] = loss_vertex[key]

        # Loss for normal decoding faces
        loss_face = self.decoder.loss_face(
            face_data, v_data, face_mask)
        for key in loss_face:
            loss[key + "1"] = loss_face[key]

        loss["face_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_face_features"], v_data["face_embeddings"], reduction='mean')
        loss["edge_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_edge_features"], v_data["edge_embeddings"][used_edge_indexes], reduction='mean')
        loss["vertex_l2"] = nn.functional.mse_loss(
            v_recon_data["proj_vertex_features"], v_data["vertex_embeddings"][used_vertex_indexes], reduction='mean')
        # loss["inter_edge_l2"] = nn.functional.mse_loss(
        #     v_recon_data["inter_edge_features"], v_data["atten_edge_embeddings"], reduction='mean')
        loss["total_loss"] = sum(loss.values())
        return loss

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        face_mask = (v_data["discrete_face_bboxes"]!=-1).all(dim=-1)
        face_bbox = self.face_bbox(v_data["discrete_face_bboxes"][face_mask])
        face_coords = self.face_coords(v_data["discrete_face_points"][face_mask])
        face_features = self.face_fuser(face_bbox[...,None] + face_coords)

        edge_mask = (v_data["discrete_edge_bboxes"]!=-1).all(dim=-1)
        edge_bbox = self.edge_bbox(v_data["discrete_edge_bboxes"][edge_mask])
        edge_coords = self.edge_coords(v_data["discrete_edge_points"][edge_mask])
        edge_features = self.edge_fuser(edge_bbox + edge_coords)

        vertex_mask = (v_data["discrete_vertex_points"]!=-1).all(dim=-1)
        vertex_features = self.vertices_encoder(v_data["discrete_vertex_points"][vertex_mask])

        loss = {}
        # ================== Bottleneck  ==================
        if self.with_quantization:
            proj_face_features = self.quantizer_in(face_features)
            quan_face_features, indices = self.quantizer(proj_face_features)
            face_features_plus = self.quantizer_out(quan_face_features)
        elif self.with_vae:
            mean, logvar = self.vae_proj(face_features).chunk(2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sampled_feature = eps.mul(std).add_(mean)
            face_features_plus = self.vae_out(sampled_feature)
            loss["kl"] = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * 1e-6
        else:
            face_features_plus = self.bottleneck(face_features)

        pre_face_bbox = self.face_bbox_decoder(face_features_plus)
        pre_face_coords = self.face_coords_decoder(face_features_plus)
        pre_edge_bbox = self.edge_bbox_decoder(edge_features)
        pre_edge_coords = self.edge_coords_decoder(edge_features)
        pre_vertex_coords = self.vertex_coords_decoder(vertex_features)

        loss["face_bbox1"] = nn.functional.cross_entropy(
            pre_face_bbox.flatten(0, -2), v_data["discrete_face_bboxes"][face_mask].flatten())
        loss["face_coords1"] = nn.functional.cross_entropy(
            pre_face_coords.flatten(0, -2), v_data["discrete_face_points"][face_mask].flatten())
        loss["edge_bbox1"] = nn.functional.cross_entropy(
            pre_edge_bbox.flatten(0, -2), v_data["discrete_edge_bboxes"][edge_mask].flatten())
        loss["edge_coords1"] = nn.functional.cross_entropy(
            pre_edge_coords.flatten(0, -2), v_data["discrete_edge_points"][edge_mask].flatten())
        loss["vertex_coords1"] = nn.functional.cross_entropy(
            pre_vertex_coords.flatten(0, -2), v_data["discrete_vertex_points"][vertex_mask].flatten())

        loss["total_loss"] = sum(loss.values())

        # Compute model size and flops
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.encoder(v_data)
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.decoder(atten_face_edge_embeddings, intersected_edge_features)
        data = {}
        if return_recon:
            bbox_shifts = (self.bd + 1) // 2 - 1
            coord_shifts = (self.cd + 1) // 2 - 1

            face_bbox = (pre_face_bbox.argmax(dim=-1) - bbox_shifts) / bbox_shifts
            face_center = (face_bbox[:, 3:] + face_bbox[:, :3]) / 2
            face_length = (face_bbox[:, 3:] - face_bbox[:, :3])
            face_coords = (pre_face_coords.argmax(dim=-1) - coord_shifts) / coord_shifts / 2
            face_coords = face_coords * face_length[:, None, None] + face_center[:, None, None]

            edge_bbox = (pre_edge_bbox.argmax(dim=-1) - bbox_shifts) / bbox_shifts
            edge_center = (edge_bbox[:, 3:] + edge_bbox[:, :3]) / 2
            edge_length = (edge_bbox[:, 3:] - edge_bbox[:, :3])
            edge_coords = (pre_edge_coords.argmax(dim=-1) - coord_shifts) / coord_shifts / 2
            edge_coords = edge_coords * edge_length[:, None] + edge_center[:, None]

            vertex_coords = (pre_vertex_coords.argmax(dim=-1) - coord_shifts) / coord_shifts

            used_edge_indexes = torch.arange(edge_coords.shape[0], device=edge_coords.device)
            used_vertex_indexes = torch.arange(vertex_coords.shape[0], device=vertex_coords.device)

            recon_face_full = -torch.ones_like(v_data["face_points"])
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), face_coords)

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[used_edge_indexes] = edge_coords
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            recon_vertex_full = recon_vertex_full.masked_scatter(
                rearrange(vertex_mask, '... -> ... 1'), vertex_coords)

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.l1_loss(
                face_coords, v_data["face_points"][face_mask], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            true_recon_edge_loss = nn.functional.l1_loss(
                edge_coords, v_data["edge_points"][edge_mask][used_edge_indexes], reduction='mean')
            loss["true_recon_edge"] = true_recon_edge_loss
            true_recon_vertex_loss = nn.functional.l1_loss(
                vertex_coords, v_data["vertex_points"][vertex_mask][used_vertex_indexes], reduction='mean')
            loss["true_recon_vertex"] = true_recon_vertex_loss

        return loss, data