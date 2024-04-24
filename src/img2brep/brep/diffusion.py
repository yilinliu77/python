import torch
from denoising_diffusion_pytorch import GaussianDiffusion, GaussianDiffusion1D, Unet1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import SinusoidalPosEmb, default
from einops import rearrange

from torch import nn
from tqdm import tqdm

from src.img2brep.brep.model import AutoEncoder
from x_transformers import ContinuousTransformerWrapper, Decoder, ContinuousAutoregressiveWrapper



class Diff_transformer(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        # determine dimensions

        if self.self_condition:
            self.project_in = nn.Linear(256 * 2, dim)
        else:
            self.project_in = nn.Linear(256, dim)

        self.atten = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
            nn.TransformerEncoderLayer(d_model=dim, nhead=12, dim_feedforward=dim, dropout=0.1, batch_first=True),
        ])


        # time embeddings
        time_dim = dim
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.final_mlp = nn.Linear(dim, 256)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.project_in(x.permute(0,2,1))
        # x = x.permute(0,2,1)
        r = x.clone()
        t = self.time_mlp(time)
        for layer in self.atten:
            x = layer(x + t[:,None,:])
        x = self.final_mlp(x + r)
        x = x.permute(0,2,1)
        return x


class DiffusionModel(nn.Module):
    def __init__(self,
                 v_conf,
                 dim=256,
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

        self.num_max_faces = v_conf["num_max_faces"]

        # model = Unet1D(
        #     dim=256,
        #     channels=256,
        #     dim_mults=(1, 1, 2, 2),
        # )
        dim = 768
        self.model = Diff_transformer(
            dim=dim,
            channels=256,
            self_condition=False,
        )

        # from diffusers.schedulers import DDPMScheduler, DDIMScheduler
        # self.diffusion = DDIMScheduler()

        self.diffusion = GaussianDiffusion1D(
            self.model,
            seq_length=self.num_max_faces,
            timesteps=1000,
            auto_normalize=False,
            # objective = 'pred_v',
        )

        pass

    # face_embeddings: (batch, max_seq_len, dim),
    def forward_on_embedding(self, v_face_embeddings, only_return_loss=False, only_return_recon=False):
        B, num_face, dim = v_face_embeddings.shape
        zero_flag = (v_face_embeddings==0).all(dim=-1)
        num_valid = (~zero_flag).sum(dim=1)
        # face_embeddings = self.mapper(v_face_embeddings)
        face_embeddings = torch.sigmoid(v_face_embeddings)
        # face_embeddings[zero_flag] = 0

        idx = torch.randperm(self.num_max_faces, device=face_embeddings.device)[None,:].repeat(B,1)
        idx = idx % num_valid[:,None]

        padded_face_embeddings = torch.gather(face_embeddings, 1, idx[:,:,None].repeat(1,1,dim))
        # padded_face_embeddings = face_embeddings
        # noise = torch.randn(padded_face_embeddings.shape, device=padded_face_embeddings.device)
        # timesteps = torch.randint(
        #     0, 1000, (B,), device=padded_face_embeddings.device,
        #     dtype=torch.int64
        # )
        # noisy_images = self.diffusion.add_noise(padded_face_embeddings, noise, timesteps)
        # noise_pred = self.model(noisy_images.permute(0,2,1), timesteps).permute(0,2,1)
        # loss = nn.functional.mse_loss(noise_pred, noise)

        loss = self.diffusion(padded_face_embeddings.permute(0,2,1))

        loss = {
            "total_loss"         : loss,
            }

        return loss, {}

    @torch.no_grad()
    def prepare_face_embedding(self, v_data):
        loss, recon_data = self.autoencoder(v_data, return_face_features=True)
        face_embeddings = recon_data["face_embeddings"]
        return face_embeddings

    def forward(self, v_data, **kwargs):
        face_embeddings = v_data
        # face_embeddings = self.prepare_face_embedding(v_data)
        return self.forward_on_embedding(face_embeddings, **kwargs)

    @torch.no_grad()
    def generate(self, batch_size=1):
        samples = self.diffusion.sample(batch_size).permute(0,2,1)
        face_embeddings = -torch.log(torch.clamp_min(1 / samples - 1, 1e-4))
        recon_vertices, recon_edges, recon_faces = self.autoencoder.inference(face_embeddings)
        return recon_vertices, recon_edges, recon_faces
