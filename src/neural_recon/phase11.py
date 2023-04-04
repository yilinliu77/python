import mcubes
import tinycudann as tcnn
import numpy as np
import PIL.Image
import open3d as o3d
import torch
from torch import nn
import cv2

import math
import platform
import shutil
import sys, os

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

from shared.img_torch_tools import print_model_size
from src.neural_recon.Image_dataset import Image_dataset
from src.neural_recon.colmap_io import read_dataset
from src.neural_recon.phase1 import NGPModel


class NGPModel1(nn.Module):
    def __init__(self):
        super(NGPModel1, self).__init__()

        # Define models
        self.n_frequencies = 32
        self.n_layer = 4
        self.model1 = tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "Frequency",
            "n_frequencies": self.n_frequencies
        }, dtype=torch.float32)
        assert self.model1.n_output_dims % self.n_layer == 0
        self.num_freq_per_layer = self.n_frequencies // self.n_layer

        self.model2 = nn.ModuleList()
        for i in range(self.n_layer):
            self.model2.append(tcnn.Network(
                n_input_dims=self.model1.n_output_dims // self.n_layer * (i + 1),
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 16,
                }))

    def forward(self, v_pixel_pos):
        bs = v_pixel_pos.shape[0]
        nf = self.num_freq_per_layer
        features = self.model1(v_pixel_pos)  # (batch_size, 2 * n_frequencies * 2)
        features = features.reshape(bs, 2, -1, 2)
        predicted_grays = []
        accumulated_pos_encoding = None
        for i_layer in range(self.n_layer):
            feature_layer = features[:, :, nf * i_layer:nf * i_layer + nf, :].reshape(bs, -1)
            if accumulated_pos_encoding is not None:
                feature_layer = torch.cat((accumulated_pos_encoding, feature_layer), dim=1)
            accumulated_pos_encoding = feature_layer
            predicted_gray = self.model2[i_layer](accumulated_pos_encoding)
            predicted_grays.append(predicted_gray)
        return predicted_grays


class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class NGPModel2(nn.Module):
    def __init__(self):
        super(NGPModel2, self).__init__()

        # Define models
        self.n_frequencies = 12
        self.n_layer = 4
        self.num_freq_per_layer = self.n_frequencies // self.n_layer
        v_hidden_dim = 256

        self.model2 = nn.ModuleList()
        for i in range(self.n_layer):
            self.model2.append(
                nn.Sequential(
                    torch.nn.Linear(2 * 2 * self.num_freq_per_layer * (i + 1), v_hidden_dim),
                    torch.nn.ReLU(),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    ResidualBlock(
                        nn.Sequential(
                            torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                            torch.nn.ReLU()
                        )
                    ),
                    torch.nn.Linear(v_hidden_dim, 1)
                )
            )

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, v_pixel_pos):
        bs = v_pixel_pos.shape[0]
        nf = self.num_freq_per_layer
        features = self.positional_encoding(v_pixel_pos, self.n_frequencies)  # (batch_size, 2 * n_frequencies * 2)
        features = features.reshape(bs, 2, 2, -1)
        predicted_grays = []
        accumulated_pos_encoding = None
        for i_layer in range(self.n_layer):
            feature_layer = features[:, :, :, nf * i_layer:nf * i_layer + nf].reshape(bs, -1)
            if accumulated_pos_encoding is not None:
                feature_layer = torch.cat((accumulated_pos_encoding, feature_layer), dim=1)
            accumulated_pos_encoding = feature_layer
            predicted_gray = self.model2[i_layer](accumulated_pos_encoding)
            predicted_grays.append(predicted_gray)
        return predicted_grays


class NGPModel3(nn.Module):
    def __init__(self):
        super(NGPModel3, self).__init__()

        # Define models
        self.n_frequencies = 24
        self.n_layer = 4
        self.num_freq_per_layer = self.n_frequencies // self.n_layer
        v_hidden_dim = 256
        v_hidden_num = 8

        self.model2 = nn.ModuleList()
        for i in range(self.n_layer):
            model = []
            model.append(nn.Linear(2 * 2 * self.num_freq_per_layer * (i + 1), v_hidden_dim))
            model.append(nn.LeakyReLU())
            for j in range(v_hidden_num):
                model.append(ResidualBlock(
                    nn.Sequential(
                        torch.nn.Linear(v_hidden_dim, v_hidden_dim),
                        torch.nn.LeakyReLU()
                    )
                ))
            model.append(nn.Linear(v_hidden_dim, 1))
            self.model2.append(nn.Sequential(*model))

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, v_pixel_pos):
        bs = v_pixel_pos.shape[0]
        nf = self.num_freq_per_layer
        features = self.positional_encoding(v_pixel_pos, self.n_frequencies)  # (batch_size, 2 * n_frequencies * 2)
        features = features.reshape(bs, 2, 2, -1, )
        predicted_grays = []
        accumulated_pos_encoding = None
        for i_layer in range(self.n_layer):
            feature_layer = features[:, :, :, nf * i_layer:nf * i_layer + nf].reshape(bs, -1)
            if accumulated_pos_encoding is not None:
                feature_layer = torch.cat((accumulated_pos_encoding, feature_layer), dim=1)
            accumulated_pos_encoding = feature_layer
            predicted_gray = self.model2[i_layer](accumulated_pos_encoding)
            predicted_grays.append(predicted_gray)
        return predicted_grays


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


class MFNBase(nn.Module):
    def __init__(self, hidden_size, out_size, n_layers, weight_scale,
                 bias=True, output_act=False):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )

        self.output_linear = nn.Linear(hidden_size, out_size)

        self.output_act = output_act

        self.linear.apply(mfn_weights_init)
        self.output_linear.apply(mfn_weights_init)

    def forward(self, model_input):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}
        coords = input_dict['coords']

        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return {'model_in': input_dict, 'model_out': {'output': out}}


class FourierLayer(nn.Module):

    def __init__(self, in_features, out_features, weight_scale, quantization_interval=2 * np.pi):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        r = 2 * weight_scale[0] / quantization_interval
        assert math.isclose(r, round(r)), \
            'weight_scale should be divisible by quantization interval'

        # sample discrete uniform distribution of frequencies
        for i in range(self.linear.weight.data.shape[1]):
            init = torch.randint_like(self.linear.weight.data[:, i],
                                      0, int(2 * weight_scale[i] / quantization_interval) + 1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class MultiscaleBACON(MFNBase):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size,
                 hidden_layers=3,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 frequency=(128, 128),
                 quantization_interval=2 * np.pi,
                 centered=True,
                 is_sdf=False,
                 input_scales=None,
                 output_layers=None,
                 reuse_filters=False):

        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.centered = centered
        self.is_sdf = is_sdf
        self.frequency = frequency
        self.output_layers = output_layers
        self.reuse_filters = reuse_filters
        self.stop_after = None

        # we need to multiply by this to be able to fit the signal
        if input_scales is None:
            input_scale = [round((np.pi * freq / (hidden_layers + 1))
                                 / quantization_interval) * quantization_interval for freq in frequency]

            self.filters = nn.ModuleList([
                FourierLayer(in_size, hidden_size, input_scale,
                             quantization_interval=quantization_interval)
                for i in range(hidden_layers + 1)])
        else:
            if len(input_scales) != hidden_layers + 1:
                raise ValueError('require n+1 scales for n hidden_layers')
            input_scale = [[round((np.pi * freq * scale) / quantization_interval) * quantization_interval
                            for freq in frequency] for scale in input_scales]

            self.filters = nn.ModuleList([
                FourierLayer(in_size, hidden_size, input_scale[i],
                             quantization_interval=quantization_interval)
                for i in range(hidden_layers + 1)])

        # linear layers to extract intermediate outputs
        self.output_linear = nn.ModuleList([nn.Linear(hidden_size, out_size) for i in range(len(self.filters))])
        self.output_linear.apply(mfn_weights_init)

        # if outputs layers is None, output at every possible layer
        if self.output_layers is None:
            self.output_layers = np.arange(1, len(self.filters))

        print(self)

    def layer_forward(self, coords, filter_outputs, specified_layers,
                      get_feature, continue_layer, continue_feature):
        """ for multiscale SDF extraction """

        # hardcode the 8 layer network that we use for all sdf experiments
        filter_ind_dict = [2, 2, 2, 4, 4, 6, 6, 8, 8]
        outputs = []

        if continue_feature is None:
            assert (continue_layer == 0)
            out = self.filters[filter_ind_dict[0]](coords)
            filter_output_dict = {filter_ind_dict[0]: out}
        else:
            out = continue_feature
            filter_output_dict = {}

        for i in range(continue_layer + 1, len(self.filters)):
            if filter_ind_dict[i] not in filter_output_dict.keys():
                filter_output_dict[filter_ind_dict[i]] = self.filters[filter_ind_dict[i]](coords)
            out = filter_output_dict[filter_ind_dict[i]] * self.linear[i - 1](out)

            if i in self.output_layers and i == specified_layers:
                if get_feature:
                    outputs.append([self.output_linear[i](out), out])
                else:
                    outputs.append(self.output_linear[i](out))
                return outputs

        return outputs

    def forward(self, model_input, specified_layers=None, get_feature=False,
                continue_layer=0, continue_feature=None):

        if self.is_sdf:
            model_input = {key: input.clone().detach().requires_grad_(True)
                           for key, input in model_input.items()}

        if 'coords' in model_input:
            coords = model_input['coords']
        elif 'ray_samples' in model_input:
            coords = model_input['ray_samples']

        outputs = []
        if self.reuse_filters:

            # which layers to reuse
            if len(self.filters) < 9:
                filter_outputs = 2 * [self.filters[0](coords), ] + \
                                 (len(self.filters) - 2) * [self.filters[-1](coords), ]
            else:
                filter_outputs = 3 * [self.filters[2](coords), ] + \
                                 2 * [self.filters[4](coords), ] + \
                                 2 * [self.filters[6](coords), ] + \
                                 2 * [self.filters[8](coords), ]

            # multiscale sdf extractions (evaluate only some layers)
            if specified_layers is not None:
                outputs = self.layer_forward(coords, filter_outputs, specified_layers,
                                             get_feature, continue_layer, continue_feature)

            # evaluate all layers
            else:
                out = filter_outputs[0]
                for i in range(1, len(self.filters)):
                    out = filter_outputs[i] * self.linear[i - 1](out)

                    if i in self.output_layers:
                        outputs.append(self.output_linear[i](out))
                        if self.stop_after is not None and len(outputs) > self.stop_after:
                            break

        # no layer reuse
        else:
            out = self.filters[0](coords)
            for i in range(1, len(self.filters)):
                out = self.filters[i](coords) * self.linear[i - 1](out)

                if i in self.output_layers:
                    outputs.append(self.output_linear[i](out))
                    if self.stop_after is not None and len(outputs) > self.stop_after:
                        break

        if self.is_sdf:  # convert dtype
            return {'model_in': model_input['coords'],
                    'model_out': outputs}  # outputs is a list of tensors

        return {'model_in': model_input, 'model_out': {'output': outputs}}


class Phase11(pl.LightningModule):
    def __init__(self, hparams, v_img_path):
        super(Phase11, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.img = cv2.cvtColor(cv2.imread(v_img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
        self.img_size = self.hydra_conf["dataset"]["img_size"]
        self.img = cv2.resize(self.img, self.img_size, interpolation=cv2.INTER_AREA)[:, :, None]
        self.img_name = os.path.basename(v_img_path).split(".")[0]
        self.log_root = os.path.join(self.hydra_conf["trainer"]["output"], self.img_name)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(os.path.join(self.log_root, "imgs"), exist_ok=True)

        f = getattr(sys.modules[__name__], self.hydra_conf["model"]["model_name"])
        # self.model = f()
        self.model = MultiscaleBACON(
            2, 256,
            out_size=256,
            hidden_layers=4,
            bias=True,
            frequency=(6000, 4000),
            quantization_interval=2 * np.pi,
            input_scales=[0.125, 0.125, 0.25, 0.25, 0.25],
            output_layers=[1, 2, 4],
            reuse_filters=False
        )
        print_model_size(self.model)
        # torch.set_float32_matmul_precision('medium')

    def train_dataloader(self):
        self.train_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "training",
            self.hydra_conf["dataset"]["sampling_strategy"],
            self.hydra_conf["dataset"]["query_strategy"],
        )
        return DataLoader(self.train_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=True,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def val_dataloader(self):
        self.valid_dataset = Image_dataset(
            self.img,
            self.hydra_conf["dataset"]["num_sample"],
            self.hydra_conf["trainer"]["batch_size"],
            "validation",
            self.hydra_conf["dataset"]["sampling_strategy"],
            self.hydra_conf["dataset"]["query_strategy"],
        )
        return DataLoader(self.valid_dataset,
                          batch_size=1,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=True,
                          persistent_workers=True if self.num_worker > 0 else 0
                          )

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            # "scheduler": StepLR(optimizer, 100, 0.5),
            # "scheduler": ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5),
            # "frequency": self.hydra_conf["trainer"]["check_val_every_n_epoch"]
            # "monitor": "Validation_Loss",
            # },
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        pixel_pos = batch[0][0]
        gt_gray = batch[1][0]
        batch_size = gt_gray.shape[0]
        predicted_gray = self.model(pixel_pos)
        loss = torch.stack([F.l1_loss(prediction, gt_gray) for prediction in predicted_gray]).mean()

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_pos = batch[0][0]
        gt_gray = batch[1][0]
        batch_size = gt_gray.shape[0]
        predicted_gray = self.model(pixel_pos)
        predicted_gray = self.model(pixel_pos.unsqueeze(0))
        loss = torch.stack([F.l1_loss(prediction, gt_gray) for prediction in predicted_gray]).mean()

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)
        return predicted_gray

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        n_layer = len(result[0])
        predicted_imgs = [torch.cat([item[i] for item in result]).cpu().numpy() for i in range(n_layer)]
        predicted_imgs = [
            np.clip(item, 0, 1).reshape([self.img.shape[0], self.img.shape[1], 1]) for item in predicted_imgs]
        predicted_imgs = [(item * 255).astype(np.uint8) for item in predicted_imgs]
        for i in range(n_layer):
            img = cv2.cvtColor(predicted_imgs[i], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(self.log_root, "imgs/{}_{}.png".format(self.trainer.current_epoch, i)),
                        img)
            self.trainer.logger.experiment.add_image("Image/{}".format(i),
                                                     img, self.trainer.current_epoch, dataformats="HWC")


@hydra.main(config_name="phase11_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # Read dataset and bounds
    bounds_min = np.array((-50, -50, -10), dtype=np.float32)
    bounds_max = np.array((250, 200, 60), dtype=np.float32)
    scene_bounds = np.array([bounds_min, bounds_max])
    imgs, world_points = read_dataset(v_cfg["dataset"]["colmap_dir"],
                                      scene_bounds)

    for id_img in range(len(imgs)):
        img_name = os.path.basename(imgs[id_img].img_path).split(".")[0]
        if img_name not in v_cfg["dataset"]["target_img"]:
            continue
        model = Phase11(v_cfg, imgs[id_img].img_path)
        trainer = Trainer(
            accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
            devices=v_cfg["trainer"].gpu, enable_model_summary=False,
            max_epochs=v_cfg["trainer"]["max_epoch"],
            # num_sanity_val_steps=2,
            # precision=16,
            reload_dataloaders_every_n_epochs=v_cfg["trainer"]["reload_dataloaders_every_n_epochs"],
            check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
            default_root_dir=log_dir,
        )
        trainer.fit(model)


if __name__ == '__main__':
    main()
