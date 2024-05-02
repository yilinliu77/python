import importlib

import numpy as np
import torch
# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv
from vector_quantize_pytorch import ResidualLFQ, VectorQuantize, ResidualVQ

from src.img2brep.brep.common import *
from src.img2brep.brep.model_encoder import GAT_GraphConv, SAGE_GraphConv
from src.img2brep.brep.model_fuser import Attn_fuser_cross, Attn_fuser_single


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
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_encoder')
        self.encoder = getattr(mod, v_conf["encoder"])(
            self.dim_shape,
            bbox_discrete_dim=v_conf["bbox_discrete_dim"],
            coor_discrete_dim=v_conf["coor_discrete_dim"],
        )
        self.vertices_proj = nn.Linear(self.dim_shape, self.dim_latent)
        self.edges_proj = nn.Linear(self.dim_shape, self.dim_latent)
        self.faces_proj = nn.Linear(self.dim_shape, self.dim_latent)

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
                # self.quantizer = VectorQuantize(
                #     dim=self.dim_latent,
                #     codebook_dim=32,  # a number of papers have shown smaller codebook dimension to be acceptable
                #     heads=8,  # number of heads to vector quantize, codebook shared across all heads
                #     separate_codebook_per_head=True,
                #     # whether to have a separate codebook per head. False would mean 1 shared codebook
                #     codebook_size=8196,
                #     accept_image_fmap=False
                # )

                self.quantizer = ResidualVQ(
                    dim=self.dim_latent,
                    codebook_dim=32,
                    num_quantizers=8,
                    quantize_dropout=True,

                    # separate_codebook_per_head=True,
                    codebook_size=16384,
                )

                layer = nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, dim_feedforward=256,
                                                batch_first=True, dropout=0.1)
                self.quantizer_proj = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(self.dim_latent))
                self.quantizer_proj2 = nn.Linear(self.dim_latent, self.dim_latent)

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

        if gathered_edge_features.shape[0] < 64*64:
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
                torch.arange(num_edges,device=device)[:,None], face_idx[edge_intersection_mask], ),dim=1)
            edge_vertex_connectivity = torch.cat((
                torch.arange(vertex_features.shape[0],device=device)[:,None], edge_idx[vertex_intersection_mask], ),
                dim=1)
            return recon_vertices, recon_edges, recon_faces, face_edge_connectivity, edge_vertex_connectivity
        return recon_vertices, recon_edges, recon_faces

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        # ================== Encode the edge and face points ==================
        face_embeddings, edge_embeddings, vertex_embeddings, face_mask, edge_mask, vertex_mask = self.encoder(v_data)
        face_embeddings = self.faces_proj(face_embeddings)
        face_embeddings = torch.sigmoid(face_embeddings)
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

        # ================== Fuse vertex features to the corresponding edges ==================
        L = edge_embeddings.shape[0]
        edge_attn_mask = edge_embeddings.new_ones(L, L, dtype=torch.bool)
        num_valid = edge_mask.long().sum(dim=1)
        num_valid = torch.cat((torch.zeros_like(num_valid[:1]), num_valid.cumsum(dim=0)))
        for i in range(num_valid.shape[0] - 1):
            edge_attn_mask[num_valid[i]:num_valid[i + 1], num_valid[i]:num_valid[i + 1]] = 0

        edge_vertex_embeddings = self.fuser_vertices_to_edges(
            v_embeddings1=vertex_embeddings,
            v_embeddings2=edge_embeddings,
            v_connectivity1_to_2=vertex_edge_connectivity,
            v_attn_mask=edge_attn_mask
        )

        # ================== GCN and self-attention on edges ==================
        edge_embeddings_gcn = self.gcn_on_edges(edge_vertex_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_attn_mask)

        # ================== fuse edges features to the corresponding faces ==================
        L = face_embeddings.shape[0]
        face_attn_mask = face_embeddings.new_ones(L, L, dtype=torch.bool)
        num_valid = face_mask.long().sum(dim=1)
        num_valid = torch.cat((torch.zeros_like(num_valid[:1]), num_valid.cumsum(dim=0)))
        for i in range(num_valid.shape[0] - 1):
            face_attn_mask[num_valid[i]:num_valid[i + 1], num_valid[i]:num_valid[i + 1]] = 0
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

        atten_face_edge_embeddings = self.face_fuser(face_edge_embeddings_gcn,
                                                     face_attn_mask)  # This is the true latent
        atten_face_edge_embeddings = torch.sigmoid(atten_face_edge_embeddings)

        if self.with_quantization and self.with_vae:
            raise NotImplementedError
        elif self.with_quantization:
            # ================== Quantization  ==================
            quantized_face_embeddings, indices, quantized_loss = self.quantizer(atten_face_edge_embeddings[:, None])
            quantized_face_embeddings = quantized_face_embeddings[:, 0]
            indices = indices[:, 0]
            true_face_embeddings = self.quantizer_proj(quantized_face_embeddings, face_attn_mask)
            true_face_embeddings = self.quantizer_proj2(true_face_embeddings)
            true_face_embeddings = torch.sigmoid(true_face_embeddings)
        elif self.with_vae:
            vae_face_embeddings = self.vae(atten_face_edge_embeddings)
            # Sampling
            mean = vae_face_embeddings[..., :self.dim_latent]
            logvar = vae_face_embeddings[..., self.dim_latent:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sampled_embeddings = mean + eps * std * self.vae_weight
            # Proj
            true_face_embeddings = self.vae_proj(sampled_embeddings)
            true_face_embeddings = torch.sigmoid(true_face_embeddings)
            indices = None
            vae_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        else:
            true_face_embeddings = atten_face_edge_embeddings
            indices = None

        # ================== Intersection  ==================
        face_adj = v_data["face_adj"]
        edge_adj = v_data["edge_adj"]
        inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
            true_face_embeddings,
            edge_face_connectivity,
            vertex_edge_connectivity,
            face_adj, face_mask,
            edge_adj, edge_mask,
        )

        # ================== Decoding  ==================
        vertex_data = self.decoder.decode_vertex(vertex_embeddings)  # Normal decoding vertex
        edge_data = self.decoder.decode_edge(edge_embeddings)  # Normal decoding edges
        face_data = self.decoder.decode_face(face_embeddings)  # Normal decoding faces
        # Decode with intersection feature
        recon_data = self.decoder(
            true_face_embeddings,
            inter_edge_features,
            inter_vertex_features,
        )

        loss = {}
        data = {}
        # Return
        used_edge_indexes = edge_face_connectivity[..., 0]
        used_vertex_indexes = vertex_edge_connectivity[..., 0]
        if return_loss:
            # Loss for predicting discrete points from the intersection features
            loss.update(self.decoder.loss(
                recon_data, v_data, face_mask,
                edge_mask, used_edge_indexes,
                vertex_mask, used_vertex_indexes
            ))

            # Loss for classifying the intersection features
            loss_edge, loss_vertex = self.intersector.loss(
                inter_edge_features, inter_edge_null_features,
                inter_vertex_features, inter_vertex_null_features
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
                torch.arange(vertex_embeddings.shape[0]))
            for key in loss_vertex:
                loss[key + "1"] = loss_vertex[key]

            # Loss for normal decoding faces
            loss_face = self.decoder.loss_face(
                face_data, v_data, face_mask)
            for key in loss_face:
                loss[key + "1"] = loss_face[key]

            # Loss for l2 distance from the quantized face features and normal face features
            loss["face_l2"] = nn.functional.mse_loss(
                true_face_embeddings, face_embeddings, reduction='mean')
            # Loss for l2 distance from the intersection edge features and normal edge features
            loss["edge_l2"] = nn.functional.mse_loss(
                inter_edge_features, edge_embeddings[used_edge_indexes], reduction='mean')
            # Loss for l2 distance from the intersection vertex features and normal vertex features
            loss["vertex_l2"] = nn.functional.mse_loss(
                inter_vertex_features, vertex_embeddings[used_vertex_indexes], reduction='mean')
            if self.with_quantization:
                loss["quantization"] = quantized_loss.mean()
            if self.with_vae:
                loss["vae"] = vae_loss
            loss["total_loss"] = sum(loss.values())

        # Compute model size and flops
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.encoder(v_data)
        # counter = FlopCounterMode(depth=999)
        # with counter:
        #     self.decoder(atten_face_edge_embeddings, intersected_edge_features)

        # Return

        # Construct the full points using the mask
        # recon_data["face_coords_logits"] = nn.functional.one_hot(
        #     v_data["discrete_face_points"][face_mask], self.decoder.cd)
        # recon_data["face_bbox_logits"] = nn.functional.one_hot(
        #     v_data["discrete_face_bboxes"][face_mask], self.decoder.bd)
        # recon_data["edge_coords_logits"] = nn.functional.one_hot(
        #     v_data["discrete_edge_points"][edge_mask], self.decoder.cd)
        # recon_data["edge_bbox_logits"] = nn.functional.one_hot(
        #     v_data["discrete_edge_bboxes"][edge_mask], self.decoder.bd)
        if return_recon:
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
