import torch
from einops.layers.torch import Rearrange
from torch import nn

from src.img2brep.brep.common import gaussian_blur_1d
from src.img2brep.brep.model_encoder import res_block_1D, res_block_2D


################### Decoder

# 43134M FLOPS and 6300678 parameters
class Small_decoder(nn.Module):
    def __init__(self,
                 dim_in,
                 hidden_dim=256,
                 **kwargs
                 ):
        super(Small_decoder, self).__init__()
        # For vertex
        self.vertex_decoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1'),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv1d(256, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c 1 -> ... c', c=3),
        )

        # For edges
        self.edge_decoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1'),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim, 5, 1, 2),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim, 5, 1, 2),
            nn.Upsample(size=20, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c v -> ... v c', c=3),
        )

        # For faces
        self.face_decoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1 1'),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, 5, 1, 2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim, 5, 1, 2),
            nn.Upsample(size=(20, 20), mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=1, stride=1, padding=0),
            Rearrange('... c w h -> ... w h c', c=3),
        )

    def forward(self, v_face_embeddings, v_edge_embeddings, v_vertex_features):
        recon_vertices = self.decode_vertex(v_vertex_features)
        recon_edges = self.decode_edge(v_edge_embeddings)
        recon_faces = self.decode_face(v_face_embeddings)
        recon_vertices.update(recon_edges)
        recon_vertices.update(recon_faces)
        return recon_vertices

    def decode_vertex(self, v_vertex_features):
        return {"vertex_coords": self.vertex_decoder(v_vertex_features)[..., 0]}

    def decode_edge(self, v_edge_embeddings):
        return {"edge_coords": self.edge_decoder(v_edge_embeddings)}

    def decode_face(self, v_face_embeddings):
        return {"face_coords": self.face_decoder(v_face_embeddings)}

    def inference(self, v_data):
        return v_data["face_coords"], v_data["edge_coords"], v_data["vertex_coords"]

    def loss_vertex(self, v_pred, v_data, v_vertex_mask, v_used_vertex_indexes):
        gt_vertex = v_data["vertex_points"][v_vertex_mask][v_used_vertex_indexes]

        loss_vertex_coords = nn.functional.mse_loss(
            gt_vertex,
            v_pred["vertex_coords"],
            reduction='mean')

        return {
            "vertex_coords": loss_vertex_coords,
        }

    def loss_edge(self, v_pred, v_data, v_edge_mask, v_used_edge_indexes):
        gt_edge = v_data["edge_points"][v_edge_mask][v_used_edge_indexes]

        loss_edge_coords = nn.functional.mse_loss(
            gt_edge,
            v_pred["edge_coords"],
            reduction='mean')

        return {
            "edge_coords": loss_edge_coords,
        }

    def loss(self, v_pred, v_data, v_face_mask,
             v_edge_mask, v_used_edge_indexes,
             v_vertex_mask, v_used_vertex_indexes,
             ):
        loss_edge = self.loss_edge(v_pred, v_data, v_edge_mask, v_used_edge_indexes)
        loss_vertex = self.loss_vertex(v_pred, v_data, v_vertex_mask, v_used_vertex_indexes)

        gt_face = v_data["face_points"][v_face_mask]
        loss_face_coords = nn.functional.mse_loss(
            gt_face,
            v_pred["face_coords"],
            reduction='mean')

        loss_result = {
            "face_coords": loss_face_coords,
        }
        loss_result.update(loss_edge)
        loss_result.update(loss_vertex)
        loss_result.update({
            "total_loss": loss_face_coords + loss_edge["edge_coords"] + loss_vertex["vertex_coords"],
        })
        return loss_result


# 26897M FLOPS and 6271200 parameters
class Discrete_decoder(Small_decoder):
    def __init__(self,
                 dim_in,
                 hidden_dim=256,
                 bbox_discrete_dim=64,
                 coor_discrete_dim=64,
                 ):
        super(Discrete_decoder, self).__init__(dim_in, hidden_dim)
        self.bd = bbox_discrete_dim - 1  # discrete_dim
        self.cd = coor_discrete_dim - 1  # discrete_dim

        self.bbox_decoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1'),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            res_block_1D(hidden_dim, hidden_dim, ks=1, st=1, pa=0),
            nn.Conv1d(hidden_dim, 6 * self.bd, kernel_size=1, stride=1, padding=0),
            Rearrange('...(p c) 1-> ... p c', p=6, c=self.bd),
        )

        # For faces
        self.face_coords = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1 1'),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            res_block_2D(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) w h -> ... w h p c', p=3, c=self.cd),
        )

        # For edges
        self.edge_coords = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1'),
            nn.Upsample(scale_factor=4, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode="linear"),
            res_block_1D(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) w -> ... w p c', p=3, c=self.cd),
        )

        # For vertex
        self.vertex_coords = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            Rearrange('... c -> ... c 1'),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            res_block_1D(hidden_dim, hidden_dim, 1, 1, 0),
            nn.Conv1d(hidden_dim, 3 * self.cd, kernel_size=1, stride=1, padding=0),
            Rearrange('... (p c) 1 -> ... p c', p=3, c=self.cd),
        )

    def decode_face(self, v_face_embeddings):
        face_coords_logits = self.face_coords(v_face_embeddings)
        face_bbox_logits = self.bbox_decoder(v_face_embeddings)
        return {
            "face_coords_logits": face_coords_logits,
            "face_bbox_logits": face_bbox_logits,
        }

    def decode_edge(self, v_edge_embeddings):
        edge_coords_logits = self.edge_coords(v_edge_embeddings)
        edge_bbox_logits = self.bbox_decoder(v_edge_embeddings)
        return {
            "edge_coords_logits": edge_coords_logits,
            "edge_bbox_logits": edge_bbox_logits,
        }

    def decode_vertex(self, v_vertex_embeddings):
        vertex_coords_logits = self.vertex_coords(v_vertex_embeddings)
        return {
            "vertex_coords_logits": vertex_coords_logits,
        }

    def cross_entropy_loss(self, pred, gt, is_blur=False):
        if not is_blur:
            return nn.functional.cross_entropy(pred, gt, reduction='mean')
        else:
            pred_log_prob = pred.log_softmax(dim=-1)
            gt = nn.functional.one_hot(gt, num_classes=pred.shape[-1])
            gt = gaussian_blur_1d(gt.float(), sigma=0.3, guassian_kernel_width=5)
            return -torch.sum(gt * pred_log_prob) / pred.shape[0]

    def cross_entropy_loss_with_blur(self, pred, gt, bin_smooth_blur_sigma=0.3, guassian_kernel_width=5):
        pred_log_prob = pred.log_softmax(dim=-1)
        gt = nn.functional.one_hot(gt, num_classes=pred.shape[-1])
        gt = gaussian_blur_1d(gt.float(), bin_smooth_blur_sigma, guassian_kernel_width)
        return -torch.sum(gt * pred_log_prob) / pred.shape[0]

    def loss_vertex(self, v_pred, v_data, v_vertex_mask, v_used_vertex_indexes):
        gt_vertex = v_data["discrete_vertex_points"][v_vertex_mask][v_used_vertex_indexes]

        loss_vertex_coords = self.cross_entropy_loss(v_pred["vertex_coords_logits"].flatten(0, -2),
                                                     gt_vertex.flatten())
        return {
            "vertex_coords": loss_vertex_coords,
        }

    def loss_edge(self, v_pred, v_data, v_edge_mask, v_used_edge_indexes):
        gt_edge_bbox = v_data["discrete_edge_bboxes"][v_edge_mask][v_used_edge_indexes]
        gt_edge_coords = v_data["discrete_edge_points"][v_edge_mask][v_used_edge_indexes]

        loss_edge_coords = self.cross_entropy_loss(v_pred["edge_coords_logits"].flatten(0, -2),
                                                   gt_edge_coords.flatten())
        loss_edge_bbox = self.cross_entropy_loss(v_pred["edge_bbox_logits"].flatten(0, -2),
                                                 gt_edge_bbox.flatten())

        return {
            "edge_coords": loss_edge_coords,
            "edge_bbox": loss_edge_bbox,
        }

    def loss_face(self, v_pred, v_data, v_face_mask):
        gt_face_bbox = v_data["discrete_face_bboxes"][v_face_mask]
        gt_face_coords = v_data["discrete_face_points"][v_face_mask]

        loss_face_coords = self.cross_entropy_loss(v_pred["face_coords_logits"].flatten(0, -2),
                                                   gt_face_coords.flatten())
        loss_face_bbox = self.cross_entropy_loss(v_pred["face_bbox_logits"].flatten(0, -2),
                                                 gt_face_bbox.flatten())

        return {
            "face_coords": loss_face_coords,
            "face_bbox": loss_face_bbox,
        }

    def loss(self, v_pred, v_data, v_face_mask,
             v_edge_mask, v_used_edge_indexes,
             v_vertex_mask, v_used_vertex_indexes,
             ):
        loss_vertex = self.loss_vertex(v_pred, v_data, v_vertex_mask, v_used_vertex_indexes)
        loss_edge = self.loss_edge(v_pred, v_data, v_edge_mask, v_used_edge_indexes)
        loss_face = self.loss_face(v_pred, v_data, v_face_mask)

        loss_face.update(loss_vertex)
        loss_face.update(loss_edge)

        return loss_face

    def inference(self, v_data):
        bbox_shifts = (self.bd + 1) // 2 - 1
        coord_shifts = (self.cd + 1) // 2 - 1

        face_bbox = (v_data["face_bbox_logits"].argmax(dim=-1) - bbox_shifts) / bbox_shifts
        face_center = (face_bbox[:, 3:] + face_bbox[:, :3]) / 2
        face_length = (face_bbox[:, 3:] - face_bbox[:, :3])
        face_coords = (v_data["face_coords_logits"].argmax(dim=-1) - coord_shifts) / coord_shifts / 2
        face_coords = face_coords * face_length[:, None, None] + face_center[:, None, None]

        edge_bbox = (v_data["edge_bbox_logits"].argmax(dim=-1) - bbox_shifts) / bbox_shifts
        edge_center = (edge_bbox[:, 3:] + edge_bbox[:, :3]) / 2
        edge_length = (edge_bbox[:, 3:] - edge_bbox[:, :3])
        edge_coords = (v_data["edge_coords_logits"].argmax(dim=-1) - coord_shifts) / coord_shifts / 2
        edge_coords = edge_coords * edge_length[:, None] + edge_center[:, None]

        vertex_coords = (v_data["vertex_coords_logits"].argmax(dim=-1) - coord_shifts) / coord_shifts

        return face_coords, edge_coords, vertex_coords
