import torch

from torch import nn

from src.img2brep.brep.model import AutoEncoder


class AutoregressiveModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=256,
                 num_head=8,
                 num_decoder_layers=6,
                 dropout=0.1,
                 ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        autoencoder_model_path = v_conf["checkpoint_autoencoder"]

        if autoencoder_model_path is not None:
            print(f"Loading autoencoder checkpoint from {autoencoder_model_path}")
            state_dict = torch.load(autoencoder_model_path)["state_dict"]
            state_dict = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k}
            self.autoencoder.load_state_dict(state_dict, strict=True)
        else:
            print("No autoencoder model found. Using random pameters.")

        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token = nn.Parameter(torch.randn(dim))

        self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=dim,
                                           nhead=num_head,
                                           dim_feedforward=dim,
                                           dropout=dropout,
                                           batch_first=True),
                num_layers=num_decoder_layers,
                )

        # Load model
        pass

    def forward(self, v_data, only_return_loss=False, only_return_recon=False):
        sample_points_faces = v_data["sample_points_faces"]
        sample_points_edges = v_data["sample_points_lines"]
        sample_points_vertices = v_data["sample_points_vertices"]

        v_face_edge_loop = v_data["face_edge_loop"]
        face_adj = v_data["face_adj"]
        v_edge_face_connectivity = v_data["edge_face_connectivity"]
        v_vertex_edge_connectivity = v_data["vertex_edge_connectivity"]

        face_idx_sequence = v_data["face_idx_sequence"]

        loss, recon_data = self.autoencoder(v_data, only_return_loss=False)

        face_embeddings = recon_data["face_embeddings"]

        # pad sos and eos
        face_embeddings = torch.cat([
            self.sos_token[None, None, :].repeat(face_embeddings.shape[0], 1, 1),
            face_embeddings,
            torch.zeros_like(self.sos_token)[None, None, :].repeat(face_embeddings.shape[0], 1, 1)], dim=1)

        face_embeddings_mask = ~(face_embeddings == 0).all(dim=-1)

        eos_pos = torch.stack([torch.arange(face_embeddings.shape[0]).to(face_embeddings_mask.device),
                               face_embeddings_mask.sum(dim=-1)]).T

        face_embeddings[eos_pos[:, 0], eos_pos[:, 1]] = self.eos_token
        face_embeddings_mask[eos_pos[:, 0], eos_pos[:, 1]] = True

        gen_square_subsequent_mask = nn.Transformer.generate_square_subsequent_mask

        prediction = self.decoder(
                tgt=face_embeddings[:, :-1],
                tgt_is_causal=True,
                tgt_mask=gen_square_subsequent_mask(face_embeddings.shape[1] - 1, device=face_embeddings.device),
                tgt_key_padding_mask=(~face_embeddings_mask[:, :-1]),

                memory=face_embeddings[:, :-1],
                memory_is_causal=True,
                memory_mask=gen_square_subsequent_mask(face_embeddings.shape[1] - 1, device=face_embeddings.device),
                memory_key_padding_mask=~face_embeddings_mask[:, :-1],
                )

        if only_return_recon:
            return prediction

        prediction[~face_embeddings_mask[:, 1:]] = 0

        mse_loss = nn.functional.mse_loss(prediction, face_embeddings[:, 1:])

        total_loss = mse_loss

        loss = {
            "total_loss": total_loss,
            }

        if only_return_loss:
            return loss

        return loss, prediction

    def loss(self, v_prediction, v_data):
        pass
