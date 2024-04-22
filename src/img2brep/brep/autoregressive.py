import torch
from einops import rearrange

from torch import nn
from tqdm import tqdm

from src.img2brep.brep.model import AutoEncoder
from x_transformers import ContinuousTransformerWrapper, Decoder, ContinuousAutoregressiveWrapper


class EOS_Classifier(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True),
            nn.LayerNorm(dim),
            nn.ReLU(),

            nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True),
            nn.LayerNorm(dim),
            nn.ReLU(),

            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),

            nn.Sigmoid(),
            ])

    def forward(self, x, eos_token):
        eos_token = eos_token[None, None, :].repeat(x.shape[0], x.shape[1], 1)
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                out, weights = layer(query=x,
                                     key=eos_token,
                                     value=x)
                x = x + out
            else:
                x = layer(x)
        return x

    def inferrence(self, x, eos_token):
        return self.forward(x, eos_token) > 0.5


class GaussianDistributions(nn.Module):
    def __init__(self, embedding_dim):
        super(GaussianDistributions, self).__init__()
        self.mean = nn.Parameter(torch.randn(embedding_dim))
        self.log_std = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, num_samples):
        std = torch.exp(self.log_std)
        eps = torch.randn(num_samples, self.mean.size(0)).to(self.mean.device)
        samples = self.mean + eps * std
        return samples


class ConditionGaussianDistributions(nn.Module):
    def __init__(self, embedding_dim, condition_dim=None):
        super(ConditionGaussianDistributions, self).__init__()
        self.mean = nn.Parameter(torch.randn(embedding_dim))
        self.log_std = nn.Parameter(torch.zeros(embedding_dim))

        self.condition_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=condition_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(condition_dim),
            nn.MultiheadAttention(embed_dim=condition_dim, num_heads=2, dropout=0.1, batch_first=True),
            nn.LayerNorm(condition_dim),
            ])

    def condition_forward(self, condition_embeddings):
        condition_embeddings_mask = ~(condition_embeddings == 0).all(dim=-1)

        for layer in self.condition_attention:
            if isinstance(layer, nn.MultiheadAttention):
                out, weights = layer(key=condition_embeddings, value=condition_embeddings, query=condition_embeddings)
                condition_embeddings = condition_embeddings + out
            else:
                condition_embeddings = layer(condition_embeddings)

        condition_embeddings[~condition_embeddings_mask] = torch.nan

        return condition_embeddings.nanmean(dim=1)

    def forward(self, num_samples, condition_embeddings=None):
        if condition_embeddings is None:
            noise = torch.randn(num_samples, self.mean.size(0)).to(self.mean.device)
        else:
            assert condition_embeddings.shape[0] == num_samples
            noise = self.condition_forward(condition_embeddings)

        std = torch.exp(self.log_std)
        eps = torch.randn(num_samples, self.mean.size(0)).to(self.mean.device)
        samples = self.mean + eps * std
        return samples + noise


class AutoregressiveModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=256,
                 num_head=8,
                 num_decoder_layers=6,
                 dropout=0.0,
                 ):
        super().__init__()
        self.autoencoder = AutoEncoder(v_conf)
        self.autoencoder.eval()

        autoencoder_model_path = v_conf["checkpoint_autoencoder"]

        if autoencoder_model_path is not None:
            print(f"Loading autoencoder checkpoint from {autoencoder_model_path}")
            state_dict = torch.load(autoencoder_model_path)["state_dict"]
            state_dict = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k}
            self.autoencoder.load_state_dict(state_dict, strict=True)
        else:
            print("No autoencoder model found. Using random pameters.")

        self.sos_token_condition = ConditionGaussianDistributions(embedding_dim=dim, condition_dim=dim)
        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token = nn.Parameter(torch.randn(dim))

        self.TransformerWrapper = ContinuousTransformerWrapper(
                dim_in=dim,
                dim_out=dim,
                max_seq_len=1024,
                attn_layers=Decoder(
                        dim=512,
                        depth=12,
                        heads=8
                        )
                )

        self.AutoregressiveWrapper = ContinuousAutoregressiveWrapper(
                net=self.TransformerWrapper,
                pad_value=0,
                loss_fn=nn.MSELoss(reduction='none')
                )

        # predict if the token is eos
        self.is_eos_classifier = EOS_Classifier(dim=dim)

        # Load model
        pass

    # face_embeddings: (batch, max_seq_len, dim),
    def forward_on_embedding(self, face_embeddings, only_return_loss=False, only_return_recon=False):
        # pad sos and eos
        # sos_token_noise = torch.randn(face_embeddings.shape[0], 1, face_embeddings.shape[-1]).to(face_embeddings.device)
        sos_token_noise = self.sos_token_condition(face_embeddings.shape[0], face_embeddings).unsqueeze(1)

        face_embeddings = torch.cat([
            # self.sos_token[None, None, :].repeat(face_embeddings.shape[0], 1, 1),
            self.sos_token + sos_token_noise,
            face_embeddings,
            torch.zeros_like(self.eos_token)[None, None, :].repeat(face_embeddings.shape[0], 1, 1)], dim=1)

        face_embeddings_mask = ~(face_embeddings == 0).all(dim=-1)

        batch_idx = torch.arange(face_embeddings.shape[0]).to(face_embeddings_mask.device)
        eos_pos = face_embeddings_mask.sum(dim=-1)
        face_embeddings[batch_idx, eos_pos] = self.eos_token
        face_embeddings_mask[batch_idx, eos_pos] = True

        gen_square_subsequent_mask = nn.Transformer.generate_square_subsequent_mask

        mask = gen_square_subsequent_mask(face_embeddings.shape[1] - 1, device=face_embeddings.device)

        prediction, mse_loss = self.AutoregressiveWrapper(x=face_embeddings, mask=face_embeddings_mask)

        if only_return_recon:
            return prediction

        # nn.functional.mse_loss(prediction, face_embeddings[:, 1:], reduction='none')[
        #     face_embeddings_mask[:, :-1]].mean()

        # prediction[~face_embeddings_mask[:, 1:]] = 0

        eos_pos_offset = face_embeddings_mask[:, 1:].sum(dim=-1) - 1

        batch_idx = torch.arange(face_embeddings_mask.shape[0]).to(face_embeddings_mask.device)

        true_eos_pred = prediction[batch_idx, eos_pos_offset, :]

        if True:
            not_eos_mask = face_embeddings_mask[:, 1:].clone()
            not_eos_mask[batch_idx, eos_pos_offset] = False
            not_eos_pred = prediction[not_eos_mask]
        else:
            not_eos_idx_samples = (torch.rand_like(eos_pos_offset.float()) * eos_pos_offset.float()).int()
            not_eos_pred = prediction[batch_idx, not_eos_idx_samples, :]

        eos_pred = torch.cat([true_eos_pred, not_eos_pred], dim=0)

        eos_pred_prob = self.is_eos_classifier(eos_pred.unsqueeze(1), self.eos_token)

        eos_pred_gt = torch.cat([torch.ones(true_eos_pred.shape[0]), torch.zeros(not_eos_pred.shape[0])]).to(
                eos_pred_prob.device)

        eos_classifier_loss = nn.functional.binary_cross_entropy(eos_pred_prob.squeeze(1, 2), eos_pred_gt)

        # mse_loss = nn.functional.mse_loss(prediction, face_embeddings[:, 1:])

        total_loss = mse_loss + eos_classifier_loss
        # total_loss = mse_loss

        loss = {
            "total_loss"         : total_loss,
            "mse_loss"           : mse_loss,
            "eos_classifier_loss": eos_classifier_loss,
            }

        if only_return_loss:
            return loss

        return loss, prediction

    def forward_on_embedding_1(self, face_embeddings, only_return_loss=False, only_return_recon=False):
        # pad sos and eos
        sos_token_noise = torch.randn(face_embeddings.shape[0], 1, face_embeddings.shape[-1]).to(face_embeddings.device)

        face_embeddings = torch.cat([
            self.sos_token + sos_token_noise,
            face_embeddings,
            self.eos_token[None, None, :].repeat(face_embeddings.shape[0], 1, 1)], dim=1)

        face_embeddings_mask = ~(face_embeddings == 0).all(dim=-1)
        face_embeddings_offsets = face_embeddings_mask.sum(dim=-1)

        face_embeddings_flatten = face_embeddings[face_embeddings_mask]

        face_embeddings_idx_pair = torch.stack(
                [torch.arange(0, face_embeddings_flatten.shape[0] - 1),
                 torch.arange(1, face_embeddings_flatten.shape[0])], dim=1).to(face_embeddings.device)

        face_embeddings_idx_pair_mask = torch.ones(face_embeddings_idx_pair.shape[0], dtype=torch.bool).to(
                face_embeddings.device)
        eos2sos_offset = (torch.cumsum(face_embeddings_offsets, dim=0) - 1)[0: -1]
        face_embeddings_idx_pair_mask[eos2sos_offset] = False

        face_embeddings_idx_pair = face_embeddings_idx_pair[face_embeddings_idx_pair_mask]

        face_embeddings_pair = face_embeddings_flatten[face_embeddings_idx_pair]

        next_embeddings, mse_loss = self.AutoregressiveWrapper(x=face_embeddings_pair)

        if only_return_recon:
            return next_embeddings

        true_eos_mask = torch.zeros(next_embeddings.shape[0], dtype=torch.bool).to(next_embeddings.device)
        true_eos_mask[eos2sos_offset - 1] = True
        true_eos_mask[-1] = True

        eos_pred_prob = self.is_eos_classifier(next_embeddings, self.eos_token)

        eos_classifier_loss = nn.functional.binary_cross_entropy(eos_pred_prob.squeeze(1, 2), true_eos_mask.float())

        # mse_loss = nn.functional.mse_loss(next_embeddings, face_embeddings_pair[:, 1, :].unsqueeze(1))

        total_loss = mse_loss + eos_classifier_loss

        loss = {
            "total_loss"         : total_loss,
            "mse_loss"           : mse_loss,
            "eos_classifier_loss": eos_classifier_loss,
            }

        if only_return_loss:
            return loss

        return loss, next_embeddings

    @torch.no_grad()
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

    @torch.no_grad()
    def generate(self, init_token=None, batch_size=4, max_length=1000, temperature=1.0, face_embeddings=None):
        if face_embeddings is not None:
            batch_size = face_embeddings.shape[0]
        if init_token is None:
            sos_token_noise = self.sos_token_condition(batch_size, face_embeddings).unsqueeze(1)
            init_token = self.sos_token + sos_token_noise

        gen_face_embeddings = self.AutoregressiveWrapper.generate(init_token, max_length)

        b, n, d = gen_face_embeddings.shape

        is_eos = self.is_eos_classifier(rearrange(gen_face_embeddings, 'b n d -> (b n) 1 d'),
                                        self.eos_token)

        is_eos = is_eos.view(b, n, 1).squeeze(2)

        first_eos = is_eos.argmax(dim=1)

        for i in range(gen_face_embeddings.shape[0]):
            gen_face_embeddings[i, first_eos[i]:, :] = 0

        gen_face_embeddings = gen_face_embeddings.contiguous()

        recon_edges, recon_faces = self.autoencoder.inference(gen_face_embeddings)

        return recon_edges, recon_faces
