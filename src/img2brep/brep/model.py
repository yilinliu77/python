import importlib

# from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.nn import SAGEConv, GATv2Conv

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
                 max_length=100,
                 dim_codebook_edge=256,
                 dim_codebook_face=256,
                 encoder_dims_through_depth_edges=(256, 256, 256, 256),
                 encoder_dims_through_depth_faces=(256, 256, 256, 256),
                 ):
        super(AutoEncoder, self).__init__()
        self.max_length = max_length
        self.dim_codebook_edge = dim_codebook_edge
        self.dim_codebook_face = dim_codebook_face
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        # ================== Convolutional encoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_encoder')
        self.encoder = getattr(mod, v_conf["encoder"])(
                bbox_discrete_dim=v_conf["bbox_discrete_dim"],
                coor_discrete_dim=v_conf["coor_discrete_dim"],
                )

        # ================== GCN to distribute features across primitives ==================
        if v_conf["graphconv"] == "GAT":
            GraphConv = GAT_GraphConv
        else:
            GraphConv = SAGE_GraphConv
        self.gcn_on_edges = GraphConv(dim_codebook_edge, encoder_dims_through_depth_edges)
        self.gcn_on_faces = GraphConv(dim_codebook_face, encoder_dims_through_depth_faces, edge_dim=dim_codebook_edge)

        # ================== self attention to aggregate features ==================
        self.edge_fuser = Attn_fuser_single()
        self.face_fuser = Attn_fuser_single()

        # ================== cross attention to aggregate features ==================
        self.fuser_edges_to_faces = Attn_fuser_cross()
        self.fuser_vertices_to_edges = Attn_fuser_cross()

        # ================== Intersection ==================
        mod = importlib.import_module('src.img2brep.brep.model_intersector')
        self.intersector = getattr(mod, v_conf["intersector"])(500)

        # ================== Decoder ==================
        mod = importlib.import_module('src.img2brep.brep.model_decoder')
        self.decoder = getattr(mod, v_conf["decoder"])(
                dim_codebook_edge=dim_codebook_edge,
                dim_codebook_face=dim_codebook_face,
                resnet_dropout=0.0,
                bbox_discrete_dim=v_conf["bbox_discrete_dim"],
                coor_discrete_dim=v_conf["coor_discrete_dim"],
                )

    # Inference (B * num_faces * num_features)
    # Pad features are all zeros
    # B==1 currently
    def inference(self, v_face_embeddings):
        # Use face to intersect edges
        B = v_face_embeddings.shape[0]
        assert B == 1
        num_faces = v_face_embeddings.shape[1]
        idx = torch.stack(torch.meshgrid(
                torch.arange(num_faces), torch.arange(num_faces), indexing="xy"), dim=2).reshape(-1, 2)
        gathered_face_features = v_face_embeddings[:, idx]
        gathered_face_mask = gathered_face_features.all(dim=-1).all(dim=-1)

        edge_features = self.intersector.inference(gathered_face_features[gathered_face_mask], "edge")
        edge_intersection_mask = self.intersector.inference_label(edge_features)

        # Use edge to intersect vertices
        num_edges = idx.shape[0]
        idx = torch.stack(torch.meshgrid(
                torch.arange(num_edges), torch.arange(num_edges), indexing="xy"), dim=2).reshape(-1, 2)
        edge_features_full = edge_features.new_zeros(B, num_edges, edge_features.shape[-1])
        edge_features_full[gathered_face_mask] = edge_features
        gathered_edge_features = edge_features_full[:, idx]
        gathered_edge_mask = gathered_edge_features.all(dim=-1).all(dim=-1)

        vertex_features = self.intersector.inference(gathered_edge_features[gathered_edge_mask], "vertex")
        vertex_intersection_mask = self.intersector.inference_label(vertex_features)

        # Decode
        recon_data = self.decoder(
                v_face_embeddings.view(-1, v_face_embeddings.shape[-1]),
                edge_features[edge_intersection_mask],
                vertex_features[vertex_intersection_mask],
                )
        recon_faces, recon_edges, recon_vertices = self.decoder.inference(recon_data)
        return recon_vertices, recon_edges, recon_faces

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        # ================== Encode the edge and face points ==================
        face_embeddings, edge_embeddings, vertex_embeddings, face_mask, edge_mask, vertex_mask = self.encoder(v_data)

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
        edge_vertex_embeddings = self.fuser_vertices_to_edges(
                v_embeddings1=vertex_embeddings,
                v_embeddings2=edge_embeddings,
                v_connectivity1_to_2=vertex_edge_connectivity
                )

        # ================== GCN and self-attention on edges ==================
        edge_embeddings_gcn = self.gcn_on_edges(edge_vertex_embeddings,
                                                vertex_edge_connectivity[..., 1:].permute(1, 0))
        atten_edge_embeddings = self.edge_fuser(edge_embeddings_gcn, edge_mask)

        # ================== fuse edges features to the corresponding faces ==================
        pass
        # face_edge_loop = v_data["face_edge_loop"].clone()
        # original_1 = face_edge_loop == -1
        # original_2 = face_edge_loop == -2
        # face_edge_loop += edge_index_offsets[:, None, None]
        # face_edge_loop[original_1] = -1
        # face_edge_loop[original_2] = -2
        face_edge_embeddings = self.fuser_edges_to_faces(
                v_connectivity1_to_2=edge_face_connectivity,
                v_embeddings1=atten_edge_embeddings,
                v_embeddings2=face_embeddings
                )

        # ================== GCN and self-attention on faces  ==================
        face_edge_embeddings_gcn = self.gcn_on_faces(face_edge_embeddings,
                                                     edge_face_connectivity[..., 1:].permute(1, 0),
                                                     edge_attr=atten_edge_embeddings[edge_face_connectivity[..., 0]])

        atten_face_edge_embeddings = self.face_fuser(face_edge_embeddings_gcn, face_mask)  # This is the true latent

        # ================== Intersection  ==================
        face_adj = v_data["face_adj"]
        edge_adj = v_data["edge_adj"]
        inter_edge_features, inter_edge_null_features, inter_vertex_features, inter_vertex_null_features = self.intersector(
                atten_face_edge_embeddings,
                edge_face_connectivity,
                vertex_edge_connectivity,
                face_adj, face_mask,
                edge_adj, edge_mask,
                )

        # ================== Decoding  ==================
        vertex_data = self.decoder.decode_vertex(vertex_embeddings)  # Normal decoding vertex
        edge_data = self.decoder.decode_edge(atten_edge_embeddings)  # Normal decoding edges
        # Decode with intersection feature
        recon_data = self.decoder(
                atten_face_edge_embeddings,
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

            # Loss for l2 distance from the intersection edge features and normal edge features
            loss["edge_l2"] = nn.functional.mse_loss(
                    inter_edge_features, atten_edge_embeddings[used_edge_indexes], reduction='mean')

            # Loss for l2 distance from the intersection vertex features and normal vertex features
            loss["vertex_l2"] = nn.functional.mse_loss(
                    inter_vertex_features, vertex_embeddings[used_vertex_indexes], reduction='mean')

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
            true_recon_face_loss = nn.functional.mse_loss(data["recon_faces"], v_data["face_points"], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            true_recon_edge_loss = nn.functional.mse_loss(
                    recon_edges, v_data["edge_points"][edge_mask][used_edge_indexes], reduction='mean')
            loss["true_recon_edge"] = true_recon_edge_loss
            true_recon_vertex_loss = nn.functional.mse_loss(
                    recon_vertices, v_data["vertex_points"][vertex_mask][used_vertex_indexes], reduction='mean')
            loss["true_recon_vertex"] = true_recon_vertex_loss

        if return_face_features:
            face_embeddings_return = atten_face_edge_embeddings.new_zeros(
                    (*face_mask.shape, atten_face_edge_embeddings.shape[-1]))
            face_embeddings_return = face_embeddings_return.masked_scatter(
                    rearrange(face_mask, '... -> ... 1'), atten_face_edge_embeddings)
            data["face_embeddings"] = face_embeddings_return

        return loss, data
