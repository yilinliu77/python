import importlib
from src.img2brep.brep.common import *
from src.img2brep.model_encoder import prepare_connectivity


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


# Full continuous VAE
class AutoEncoder_base(nn.Module):
    def __init__(self,
                 v_conf,
                 ):
        super(AutoEncoder_base, self).__init__()
        self.dim_shape = v_conf["dim_shape"]
        self.dim_latent = v_conf["dim_latent"]
        self.pad_id = -1

        self.time_statics = [0 for _ in range(10)]

        encoder = importlib.import_module('src.img2brep.model_encoder')
        self.encoder = getattr(encoder, v_conf["encoder"])(
            self.dim_shape, self.dim_latent, v_conf
        )

        decoder = importlib.import_module('src.img2brep.model_decoder')
        self.decoder = getattr(decoder, v_conf["decoder"])(
            self.dim_shape, self.dim_latent
        )

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        encoder_result = self.encoder(v_data)
        face_mask = encoder_result["face_mask"]
        edge_mask = encoder_result["edge_mask"]
        vertex_mask = encoder_result["vertex_mask"]

        sampled_face_feature, kl_loss = self.vae_sample(encoder_result["face_features"])

        pre_face_coords, pre_edge_coords, pre_vertex_coords = self.decoder(
            sampled_face_feature,
            encoder_result["edge_features"],
            encoder_result["vertex_features"]
        )

        # Loss
        loss={}
        loss["kl"] = kl_loss
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"][face_mask]
        )
        loss["edge_coords_orig"] = nn.functional.l1_loss(
            pre_edge_coords,
            v_data["edge_points"][edge_mask]
        )
        loss["vertex_coords_orig"] = nn.functional.l1_loss(
            pre_vertex_coords,
            v_data["vertex_points"][vertex_mask]
        )
        loss["total_loss"] = sum(loss.values())

        data = {}
        if return_recon:
            recon_face_full = -torch.ones_like(v_data["face_points"])
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), pre_face_coords.to(recon_face_full.dtype))

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            recon_edge_full = recon_edge_full.masked_scatter(
                rearrange(edge_mask, '... -> ... 1 1'), pre_edge_coords.to(recon_face_full.dtype))

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            recon_vertex_full = recon_vertex_full.masked_scatter(
                rearrange(vertex_mask, '... -> ... 1'), pre_vertex_coords.to(recon_face_full.dtype))

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.l1_loss(
                pre_face_coords, v_data["face_points"][face_mask], reduction='mean')
            true_recon_edge_loss = nn.functional.l1_loss(
                pre_edge_coords, v_data["edge_points"][edge_mask], reduction='mean')
            true_recon_vertex_loss = nn.functional.l1_loss(
                pre_vertex_coords, v_data["vertex_points"][vertex_mask], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            loss["true_recon_edge"] = true_recon_edge_loss
            loss["true_recon_vertex"] = true_recon_vertex_loss

        return loss, data

    def vae_sample(self, v_face_feature):
        mean = v_face_feature[:, :8]
        logvar = v_face_feature[:, 8:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_face_feature = eps.mul(std).add_(mean)
        return sampled_face_feature, (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) * 1e-6


# Full continuous VAE with intersection
class AutoEncoder_inter(AutoEncoder_base):
    def __init__(self,
                 v_conf,
                 ):
        super().__init__(v_conf)
        # ================== Intersection ==================
        mod = importlib.import_module('src.img2brep.model_intersector')
        self.intersector = getattr(mod, v_conf["intersector"])(500, self.dim_latent)



    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        encoder_result = self.encoder(v_data)
        face_mask, edge_mask, vertex_mask = \
            encoder_result["face_mask"], encoder_result["edge_mask"], encoder_result["vertex_mask"]

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
        v_data["vertex_edge_connectivity"] = vertex_edge_connectivity
        vertex_index = vertex_edge_connectivity[..., 0]

        edge_face_connectivity = v_data["edge_face_connectivity"].clone()
        edge_face_connectivity_valid = (edge_face_connectivity != -1).all(dim=-1)
        # Solve the edge_face_connectivity: last two dimension (id_face)
        edge_face_connectivity[..., 1:] += face_index_offsets[:, None, None]
        # Solve the edge_face_connectivity: first dimension (id_edge)
        edge_face_connectivity[..., 0] += edge_index_offsets[:, None]
        edge_face_connectivity = edge_face_connectivity[edge_face_connectivity_valid]
        v_data["edge_face_connectivity"] = edge_face_connectivity
        edge_index = edge_face_connectivity[..., 0]

        sampled_face_feature, kl_loss = self.vae_sample(encoder_result["face_features"])

        edge_features, edge_null_features, vertex_features, vertex_null_features = self.intersector(
            sampled_face_feature, v_data, encoder_result)

        # Decode using original features
        pre_face_coords, pre_edge_coords_orig, pre_vertex_coords_orig = self.decoder(
            sampled_face_feature,
            encoder_result["edge_features"],
            encoder_result["vertex_features"]
        )

        # Decode using intersection features
        pre_edge_coords_inter = self.decoder.edge_coords_decoder(edge_features)
        pre_vertex_coords_inter = self.decoder.vertex_coords_decoder(vertex_features)

        # Loss
        loss = {}
        loss["kl"] = kl_loss
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"][face_mask]
        )
        loss["edge_coords_orig"] = nn.functional.l1_loss(
            pre_edge_coords_orig,
            v_data["edge_points"][edge_mask]
        )
        loss["vertex_coords_orig"] = nn.functional.l1_loss(
            pre_vertex_coords_orig,
            v_data["vertex_points"][vertex_mask]
        )

        # Intersection
        loss["edge_coords_inter"] = nn.functional.l1_loss(
            pre_edge_coords_inter,
            v_data["edge_points"][edge_mask][edge_index]
        )
        loss["vertex_coords_inter"] = nn.functional.l1_loss(
            pre_vertex_coords_inter,
            v_data["vertex_points"][vertex_mask][vertex_index]
        )
        loss_edge, loss_vertex = self.intersector.loss(
            edge_features, edge_null_features, vertex_features, vertex_null_features)
        loss["vertex_classifier"] = loss_vertex
        loss["edge_classifier"] = loss_edge
        loss["total_loss"] = sum(loss.values())

        data = {}
        if return_recon:
            dtype = v_data["face_points"].dtype
            recon_face_full = -torch.ones_like(v_data["face_points"])
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), pre_face_coords.to(dtype))

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[edge_index] = pre_edge_coords_inter.to(dtype)
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            bbb = recon_vertex_full[vertex_mask].clone()
            bbb[vertex_index] = pre_vertex_coords_inter.to(dtype)
            recon_vertex_full[vertex_mask] = bbb

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.l1_loss(
                pre_face_coords, v_data["face_points"][face_mask], reduction='mean')
            true_recon_edge_loss = nn.functional.l1_loss(
                pre_edge_coords_inter, v_data["edge_points"][edge_mask][edge_index], reduction='mean')
            true_recon_vertex_loss = nn.functional.l1_loss(
                pre_vertex_coords_inter, v_data["vertex_points"][vertex_mask][vertex_index], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            loss["true_recon_edge"] = true_recon_edge_loss
            loss["true_recon_vertex"] = true_recon_vertex_loss

        return loss, data


# Full continuous VAE with intersection
class AutoEncoder_inter_edgel2(AutoEncoder_inter):
    def __init__(self,
                 v_conf,
                 ):
        super().__init__(v_conf)

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        prepare_connectivity(v_data)
        encoder_result = self.encoder(v_data)
        face_mask, edge_mask, vertex_mask = \
            encoder_result["face_mask"], encoder_result["edge_mask"], encoder_result["vertex_mask"]

        edge_index = v_data["edge_face_connectivity"][..., 0]
        vertex_index = v_data["vertex_edge_connectivity"][..., 0]

        sampled_face_feature, kl_loss = self.vae_sample(encoder_result["face_features"])

        edge_features, edge_null_features, vertex_features, vertex_null_features = self.intersector(
            sampled_face_feature, v_data, encoder_result)

        # Decode using original features
        pre_face_coords, pre_edge_coords_orig, pre_vertex_coords_orig = self.decoder(
            sampled_face_feature,
            encoder_result["edge_features"],
            encoder_result["vertex_features"]
        )

        # Decode using intersection features
        pre_edge_coords_inter = self.decoder.edge_coords_decoder(edge_features)
        pre_vertex_coords_inter = self.decoder.vertex_coords_decoder(vertex_features)

        # Loss
        loss = {}
        loss["kl"] = kl_loss
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"][face_mask]
        )
        loss["edge_coords_orig"] = nn.functional.l1_loss(
            pre_edge_coords_orig,
            v_data["edge_points"][edge_mask]
        )
        loss["vertex_coords_orig"] = nn.functional.l1_loss(
            pre_vertex_coords_orig,
            v_data["vertex_points"][vertex_mask]
        )

        # Intersection
        loss["edge_coords_inter"] = nn.functional.l1_loss(
            pre_edge_coords_inter,
            v_data["edge_points"][edge_mask][edge_index]
        )
        loss["vertex_coords_inter"] = nn.functional.l1_loss(
            pre_vertex_coords_inter,
            v_data["vertex_points"][vertex_mask][vertex_index]
        )
        loss_edge, loss_vertex = self.intersector.loss(
            edge_features, edge_null_features, vertex_features, vertex_null_features)
        loss["vertex_classifier"] = loss_vertex
        loss["edge_classifier"] = loss_edge

        loss["edge_l2"] = nn.functional.mse_loss(
            edge_features,
            encoder_result["edge_features"][edge_index].detach()
        )

        loss["total_loss"] = sum(loss.values())

        data = {}
        if return_recon:
            dtype = v_data["face_points"].dtype
            recon_face_full = -torch.ones_like(v_data["face_points"])
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), pre_face_coords.to(dtype))

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[edge_index] = pre_edge_coords_inter.to(dtype)
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            bbb = recon_vertex_full[vertex_mask].clone()
            bbb[vertex_index] = pre_vertex_coords_inter.to(dtype)
            recon_vertex_full[vertex_mask] = bbb

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.l1_loss(
                pre_face_coords, v_data["face_points"][face_mask], reduction='mean')
            true_recon_edge_loss = nn.functional.l1_loss(
                pre_edge_coords_inter, v_data["edge_points"][edge_mask][edge_index], reduction='mean')
            true_recon_vertex_loss = nn.functional.l1_loss(
                pre_vertex_coords_inter, v_data["vertex_points"][vertex_mask][vertex_index], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            loss["true_recon_edge"] = true_recon_edge_loss
            loss["true_recon_vertex"] = true_recon_vertex_loss

        return loss, data



# Full continuous VAE with intersection
class AutoEncoder_inter_fused(AutoEncoder_base):
    def __init__(self,
                 v_conf,
                 ):
        super().__init__(v_conf)
        # ================== Intersection ==================
        mod = importlib.import_module('src.img2brep.model_intersector')
        self.intersector = getattr(mod, v_conf["intersector"])(500, self.dim_latent)

    def forward(self, v_data,
                return_recon=False,
                return_loss=True,
                return_face_features=False,
                return_true_loss=False,
                **kwargs):
        encoder_result = self.encoder(v_data)
        face_mask, edge_mask, vertex_mask = \
            encoder_result["face_mask"], encoder_result["edge_mask"], encoder_result["vertex_mask"]

        edge_index = v_data["edge_face_connectivity"][..., 0]
        vertex_index = v_data["vertex_edge_connectivity"][..., 0]

        sampled_face_feature, kl_loss = self.vae_sample(encoder_result["face_features"])

        edge_features, edge_null_features, vertex_features, vertex_null_features = self.intersector(
            sampled_face_feature, v_data, encoder_result)

        # Decode using original features
        pre_face_coords, pre_edge_coords_orig, pre_vertex_coords_orig = self.decoder(
            sampled_face_feature,
            encoder_result["edge_features"],
            encoder_result["vertex_features"]
        )

        # Decode using intersection features
        pre_edge_coords_inter = self.decoder.edge_coords_decoder(edge_features)
        pre_vertex_coords_inter = self.decoder.vertex_coords_decoder(vertex_features)

        # Loss
        loss = {}
        loss["kl"] = kl_loss
        loss["face_coords"] = nn.functional.l1_loss(
            pre_face_coords,
            v_data["face_points"][face_mask]
        )
        loss["edge_coords_orig"] = nn.functional.l1_loss(
            pre_edge_coords_orig,
            v_data["edge_points"][edge_mask]
        )
        loss["vertex_coords_orig"] = nn.functional.l1_loss(
            pre_vertex_coords_orig,
            v_data["vertex_points"][vertex_mask]
        )

        # Intersection
        loss["edge_coords_inter"] = nn.functional.l1_loss(
            pre_edge_coords_inter,
            v_data["edge_points"][edge_mask][edge_index]
        )
        loss["vertex_coords_inter"] = nn.functional.l1_loss(
            pre_vertex_coords_inter,
            v_data["vertex_points"][vertex_mask][vertex_index]
        )
        loss_edge, loss_vertex = self.intersector.loss(
            edge_features, edge_null_features, vertex_features, vertex_null_features)
        loss["vertex_classifier"] = loss_vertex
        loss["edge_classifier"] = loss_edge
        
        loss["edge_l2"] = nn.functional.mse_loss(
            edge_features,
            encoder_result["edge_features"][edge_index].detach()
        )
        
        loss["total_loss"] = sum(loss.values())

        data = {}
        if return_recon:
            dtype = v_data["face_points"].dtype
            recon_face_full = -torch.ones_like(v_data["face_points"])
            recon_face_full = recon_face_full.masked_scatter(
                rearrange(face_mask, '... -> ... 1 1 1'), pre_face_coords.to(dtype))

            recon_edge_full = -torch.ones_like(v_data["edge_points"])
            bbb = recon_edge_full[edge_mask].clone()
            bbb[edge_index] = pre_edge_coords_inter.to(dtype)
            recon_edge_full[edge_mask] = bbb

            recon_vertex_full = -torch.ones_like(v_data["vertex_points"])
            bbb = recon_vertex_full[vertex_mask].clone()
            bbb[vertex_index] = pre_vertex_coords_inter.to(dtype)
            recon_vertex_full[vertex_mask] = bbb

            data["recon_faces"] = recon_face_full
            data["recon_edges"] = recon_edge_full
            data["recon_vertices"] = recon_vertex_full

        if return_true_loss:
            if not return_recon:
                raise
            # Compute the true loss with the continuous points
            true_recon_face_loss = nn.functional.l1_loss(
                pre_face_coords, v_data["face_points"][face_mask], reduction='mean')
            true_recon_edge_loss = nn.functional.l1_loss(
                pre_edge_coords_inter, v_data["edge_points"][edge_mask][edge_index], reduction='mean')
            true_recon_vertex_loss = nn.functional.l1_loss(
                pre_vertex_coords_inter, v_data["vertex_points"][vertex_mask][vertex_index], reduction='mean')
            loss["true_recon_face"] = true_recon_face_loss
            loss["true_recon_edge"] = true_recon_edge_loss
            loss["true_recon_vertex"] = true_recon_vertex_loss

        return loss, data

