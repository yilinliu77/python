import importlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv
from vector_quantize_pytorch import ResidualLFQ, VectorQuantize, ResidualVQ

import pytorch_lightning as pl

from shared.common_utils import *
from src.img2brep.brep.common import *
from src.img2brep.brep.model_encoder import GAT_GraphConv, SAGE_GraphConv, res_block_1D
from src.img2brep.brep.model_fuser import Attn_fuser_cross, Attn_fuser_single

import open3d as o3d


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


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
        self.intersection_edge_decoder=nn.Sequential(
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            res_block_1D(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Conv1d(self.dim_latent, self.dim_latent, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.intersection_vertex_decoder=nn.Sequential(
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

                self.quantizer = ResidualVQ(
                    dim=self.dim_latent,
                    codebook_dim=32,
                    num_quantizers=8,
                    codebook_size=2048,
                )

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
        recon_data["proj_face_features"] = self.intersection_face_decoder(v_data["atten_face_embeddings"][...,None])[...,0]
        recon_data["proj_edge_features"] = self.intersection_edge_decoder(inter_edge_features[...,None])[...,0]
        recon_data["proj_vertex_features"] = self.intersection_vertex_decoder(inter_vertex_features[...,None])[...,0]

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
            quantized_face_embeddings, indices, quantized_loss = self.quantizer(v_data["atten_face_embeddings"])
            true_face_embeddings = self.quantizer_out(quantized_face_embeddings.unsqueeze(2))[...,0]
            v_data["atten_face_embeddings"] = true_face_embeddings
            loss["quantization_internal"] = quantized_loss.mean()
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

                self.quantizer = ResidualVQ(
                    dim=self.dim_latent,
                    codebook_dim=32,
                    num_quantizers=8,
                    codebook_size=2048,
                )

                self.frozen_models = [
                    # self.encoder, self.vertices_proj, self.edges_proj, self.faces_proj,
                    # self.gcn_on_edges, self.gcn_on_faces, self.edge_fuser, self.face_fuser,
                    # self.fuser_vertices_to_edges, self.fuser_edges_to_faces,
                    # self.decoder, self.intersector
                ]

        # ================== VAE ==================
        self.with_vae = v_conf["with_vae"]
        if self.with_vae:
            layer = nn.TransformerEncoderLayer(d_model=self.dim_latent * 2, nhead=8, dim_feedforward=512,
                                               batch_first=True, dropout=0.1)
            self.vae = nn.Sequential(
                nn.Linear(self.dim_latent, self.dim_latent * 2),
                nn.GELU(),
                nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(self.dim_latent * 2)),
                nn.Linear(self.dim_latent * 2, self.dim_latent * 2),
            )
            self.vae_weight = v_conf["vae_weight"]
            layer = nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, dim_feedforward=512,
                                               batch_first=True, dropout=0.1)
            self.vae_proj = nn.Sequential(
                nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(self.dim_latent)),
                nn.Linear(self.dim_latent, self.dim_latent),
            )
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

        encoding_result = {}

        encoding_result["face_mask"] = face_mask
        encoding_result["edge_mask"] = edge_mask
        encoding_result["vertex_mask"] = vertex_mask
        encoding_result["face_embeddings"] = face_embeddings
        encoding_result["edge_embeddings"] = edge_embeddings
        encoding_result["vertex_embeddings"] = vertex_embeddings
        return encoding_result

    def loss(self, v_data, encoding_result, exchange_result, intersections, intersection_recon_data,):
        edge_mask = encoding_result["edge_mask"]
        vertex_mask = encoding_result["vertex_mask"]
        face_mask = encoding_result["face_mask"]

        used_edge_indexes = exchange_result["edge_face_connectivity"][..., 0]
        used_vertex_indexes = exchange_result["vertex_edge_connectivity"][..., 0]

        loss = {}
        # Loss for predicting discrete points from the intersection features
        loss.update(self.decoder.loss(
            intersection_recon_data, v_data, face_mask,
            edge_mask, used_edge_indexes,
            vertex_mask, used_vertex_indexes
        ))

        # Loss for classifying the intersection features
        loss_edge, loss_vertex = self.intersector.loss(
            intersections["inter_edge_features"], intersections["inter_edge_null_features"],
            intersections["inter_vertex_features"], intersections["inter_vertex_null_features"]
        )
        loss.update({"intersection_edge": loss_edge})
        loss.update({"intersection_vertex": loss_vertex})

        # Loss for l2 distance from the quantized face features and normal face features
        loss["face_l2"] = nn.functional.mse_loss(
            true_face_embeddings, atten_face_edge_embeddings, reduction='mean')
        # Loss for l2 distance from the intersection edge features and normal edge features
        loss["edge_l2"] = nn.functional.mse_loss(
            inter_edge_features, atten_edge_embeddings[used_edge_indexes], reduction='mean')
        # Loss for l2 distance from the intersection vertex features and normal vertex features
        loss["vertex_l2"] = nn.functional.mse_loss(
            inter_vertex_features, atten_vertex_embeddings[used_vertex_indexes], reduction='mean')
        loss["total_loss"] = sum(loss.values())
        return loss

    def loss_decoding(self, v_recon_data, v_data, v_encoding_result):
        face_mask = v_encoding_result["face_mask"]
        edge_mask = v_encoding_result["edge_mask"]
        vertex_mask = v_encoding_result["vertex_mask"]
        # Loss for normal decoding edges
        loss_edge = self.decoder.loss_edge(
            v_recon_data, v_data, edge_mask,
            torch.arange(v_recon_data["edge_bbox_logits"].shape[0]))

        # Loss for normal decoding vertices
        loss_vertex = self.decoder.loss_vertex(
            v_recon_data, v_data, vertex_mask,
            torch.arange(v_recon_data["vertex_coords_logits"].shape[0]))

        # Loss for normal decoding faces
        loss_face = self.decoder.loss_face(
            v_recon_data, v_data, face_mask)

        loss = {}
        loss.update(loss_edge)
        loss.update(loss_vertex)
        loss.update(loss_face)

        return loss

    def intersection(self, true_face_embeddings, v_data):
        recon_data = {}
        if v_data is not None:
            face_adj = v_data["face_adj"]
            edge_adj = v_data["edge_adj"]
            inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
                true_face_embeddings,
                v_data["edge_face_connectivity"],
                v_data["vertex_edge_connectivity"],
                face_adj, v_data["face_mask"],
                edge_adj, v_data["edge_mask"],
            )
            recon_data["inter_edge_null_features"] = inter_edge_null_features
            recon_data["inter_vertex_null_features"] = inter_vertex_null_features
            recon_data["inter_edge_features"] = inter_edge_features
            recon_data["inter_vertex_features"] = inter_vertex_features
        else:
            raise
        return recon_data

    def feature_exchange(self, v_data, v_encoding_result):
        face_mask = v_encoding_result["face_mask"]
        edge_mask = v_encoding_result["edge_mask"]
        vertex_mask = v_encoding_result["vertex_mask"]

        vertex_embeddings = v_encoding_result["vertex_embeddings"]
        edge_embeddings = v_encoding_result["edge_embeddings"]
        face_embeddings = v_encoding_result["face_embeddings"]

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

        exchange_result = {}

        exchange_result["edge_face_connectivity"] = edge_face_connectivity
        exchange_result["vertex_edge_connectivity"] = vertex_edge_connectivity
        exchange_result["atten_vertex_embeddings"] = atten_vertex_embeddings
        exchange_result["atten_edge_embeddings"] = atten_edge_embeddings
        exchange_result["atten_face_embeddings"] = atten_face_embeddings
        return exchange_result

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        encoding_result = self.encode(v_data)
        exchange_result = self.feature_exchange(v_data, encoding_result)

        if self.with_quantization:
            face_embeddings = exchange_result['atten_face_embeddings']
            quantized_features, indices, quantization_loss = self.quantizer(face_embeddings)
            quantized_features = self.quantizer_out(quantized_features[...,None])[...,0]
            exchange_result["atten_face_embeddings"] = quantized_features

        # Decode with normal feature
        normal_recon_data = self.decoder(
            encoding_result["face_embeddings"],
            encoding_result["edge_embeddings"],
            encoding_result["vertex_embeddings"],
        )
        normal_loss = self.loss_decoding(normal_recon_data, v_data, encoding_result)

        true_face_embedding = exchange_result["atten_face_embeddings"]
        # Intersection
        intersection_topologies = {}
        intersection_topologies["face_adj"] = v_data["face_adj"]
        intersection_topologies["edge_adj"] = v_data["edge_adj"]
        intersection_topologies["edge_mask"] = encoding_result["edge_mask"]
        intersection_topologies["face_mask"] = encoding_result["face_mask"]
        intersection_topologies["edge_face_connectivity"] = exchange_result["edge_face_connectivity"]
        intersection_topologies["vertex_edge_connectivity"] = exchange_result["vertex_edge_connectivity"]
        intersections = self.intersection(true_face_embedding, intersection_topologies, )

        intersection_recon_data = self.decoder(
            encoding_result["face_embeddings"],
            encoding_result["edge_embeddings"],
            encoding_result["vertex_embeddings"],
        )
        intersection_recon_loss = self.loss(
            v_data, encoding_result, exchange_result, intersections, intersection_recon_data
        )



        if return_loss:

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
            # used_edge_indexes = v_data["edge_face_connectivity"][..., 0]
            # used_vertex_indexes = v_data["vertex_edge_connectivity"][..., 0]
            used_edge_indexes = torch.arange(recon_data["edge_bbox_logits"].shape[0])
            used_vertex_indexes = torch.arange(recon_data["vertex_coords_logits"].shape[0])

            face_mask = encoding_result["face_mask"]
            edge_mask = encoding_result["edge_mask"]
            vertex_mask = encoding_result["vertex_mask"]
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