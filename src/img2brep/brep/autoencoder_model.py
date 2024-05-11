import importlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv
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


class AutoEncoder1(nn.Module):
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

        face_features = self.face_fuser(torch.cat((face_coords_features, face_bbox_features),dim=1))
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
        loss["total_loss"] = (loss_face_coords+loss_face_bbox).sum()

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
