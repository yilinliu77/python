import torch

from torch import nn
from tqdm import tqdm

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

        self.linear_is_eos = nn.Linear(dim, 1)

        # Load model
        pass

    # face_embeddings: (batch, max_seq_len, dim),
    def forward_on_embedding(self, face_embeddings, only_return_loss=False, only_return_recon=False):
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

        eos_pos_offset = face_embeddings_mask[:, 1:].sum(dim=-1) - 1

        batch_idx = torch.arange(face_embeddings_mask.shape[0]).to(face_embeddings_mask.device)

        true_eos_pred = prediction[batch_idx, eos_pos_offset, :]

        not_eos_idx_samples = (torch.rand_like(eos_pos_offset.float()) * eos_pos_offset.float()).int()

        not_eos_pred = prediction[batch_idx, not_eos_idx_samples, :]

        eos_pred = torch.cat([true_eos_pred, not_eos_pred], dim=0)

        eos_pred_prob = torch.sigmoid(self.linear_is_eos(eos_pred))

        eos_pred_gt = torch.cat([torch.ones(true_eos_pred.shape[0]), torch.zeros(not_eos_pred.shape[0])]).to(
                eos_pred_prob.device)

        eos_classifier_loss = nn.functional.binary_cross_entropy(eos_pred_prob, eos_pred_gt.unsqueeze(1))

        mse_loss = nn.functional.mse_loss(prediction, face_embeddings[:, 1:])

        total_loss = mse_loss + eos_classifier_loss

        loss = {
            "total_loss"         : total_loss,
            "mse_loss"           : mse_loss,
            "eos_classifier_loss": eos_classifier_loss,
            }

        if only_return_loss:
            return loss

        return loss, prediction

    def prepare_face_embedding(self, v_data):
        loss, recon_data = self.autoencoder(v_data, only_return_loss=False)
        face_embeddings = recon_data["face_embeddings"]
        return face_embeddings

    def forward(self, v_data, **kwargs):
        face_embeddings = self.prepare_face_embedding(v_data)
        face_idx_sequence = v_data["face_idx_sequence"]
        face_idx_sequence_mask = face_idx_sequence != -1
        face_idx_sequence[~face_idx_sequence_mask] = 0

        face_embeddings = torch.gather(face_embeddings, 1,
                                       face_idx_sequence[:, :, None].repeat(1, 1, face_embeddings.shape[-1]))
        face_embeddings[~face_idx_sequence_mask] = 0

        return self.forward_on_embedding(face_embeddings, **kwargs)

    def generate(self, init_token=None, batch_size=2, max_length=1000, temperature=1.0, ):
        if init_token is None:
            init_token = self.sos_token[None, None, :].repeat(batch_size, 1, 1)
        memory_list = [init_token]

        is_eos = torch.zeros(batch_size).bool().to(init_token.device)

        for idx in tqdm(range(max_length)):
            if is_eos.all():
                break
            next_token = self.decoder(tgt=memory_list[idx], memory=torch.cat(memory_list[0:idx + 1], dim=1))
            is_eos_c = torch.sigmoid(self.linear_is_eos(next_token.squeeze(1))) > 0.5
            is_eos = torch.logical_or(is_eos, is_eos_c.squeeze(1))
            next_token[is_eos, :, :] = 0
            memory_list.append(next_token)

        gen_face_embeddings = torch.cat(memory_list[1:], dim=1)

        recon_edges, recon_faces = self.autoencoder.inference(gen_face_embeddings)

        return recon_edges, recon_faces
