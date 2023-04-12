from collections import OrderedDict

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
from src.neural_recon.Image_dataset import Image_dataset, Images_dataset
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


class MultiscaleBACON_(MFNBase):
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

        # print(self)

    def forward(self, coords, specified_layers=None, get_feature=False,
                continue_layer=0, continue_feature=None):

        outputs = []
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
            if i in self.output_layers:
                outputs.append(self.output_linear[i](out))
                if self.stop_after is not None and len(outputs) > self.stop_after:
                    break
        return outputs


class MultiscaleBACON(nn.Module):
    def __init__(self):
        super(MultiscaleBACON, self).__init__()
        input_scales = [1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 4]
        output_layers = [2, 4, 6, 8, 10, 12, 14, 15]

        self.model = MultiscaleBACON_(
            2, 256,
            out_size=1,
            hidden_layers=16,
            bias=True,
            frequency=(6000, 4000),
            quantization_interval=2 * np.pi,
            # input_scales=input_scales,
            output_layers=output_layers,
            reuse_filters=False
        )

    def forward(self, v_data):
        return self.model(v_data - 0.5)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren_(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class Siren(nn.Module):
    def __init__(self):
        super(Siren, self).__init__()
        self.model = Siren_(
            in_features=2,
            out_features=1,
            hidden_features=256,
            hidden_layers=12,
            outermost_linear=True
        )

    def forward(self, v_data):
        return [self.model(v_data * 2 - 1)[0], ]


class Siren2(nn.Module):
    def __init__(self, v_img_names, first_omega_0=30, hidden_omega_0=30.):
        super(Siren2, self).__init__()

        self.n_frequencies = 12
        self.hidden_features = 1024
        self.n_hidden_layers = 4

        self.encoder = nn.ModuleDict()
        for img_name in v_img_names:
            self.encoder[img_name] = nn.ModuleList()
            pos_encoding1 = SineLayer(
                self.n_frequencies * 4,
                512,
                is_first=True, omega_0=first_omega_0
            )
            self.encoder[img_name].append(pos_encoding1)

            pos_encoding2 = SineLayer(
                2,
                512,
                is_first=True, omega_0=first_omega_0
            )
            self.encoder[img_name].append(pos_encoding2)

            self.net = []
            for i in range(self.n_hidden_layers):
                self.net.append(SineLayer(self.hidden_features, self.hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))

            self.encoder[img_name].append(nn.Sequential(*self.net))
        self.final_linear = nn.Linear(self.hidden_features, 1)

        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_features) / hidden_omega_0,
                                            np.sqrt(6 / self.hidden_features) / hidden_omega_0)

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, v_name, coords):
        pos1 = self.encoder[v_name][0](self.positional_encoding(coords, self.n_frequencies))
        coords = coords * 2 - 1
        pos2 = self.encoder[v_name][1](coords)
        features = self.encoder[v_name][2](torch.cat((pos1, pos2), dim=1))
        output = self.final_linear(features)
        return features, output


class Siren3(nn.Module):
    def __init__(self, first_omega_0=30, hidden_omega_0=30.):
        super(Siren3, self).__init__()

        self.n_frequencies = 12
        self.hidden_features = 256
        self.n_hidden_layers = 6

        self.pos_encoding1 = SineLayer(
            self.n_frequencies * 4,
            128,
            is_first=True, omega_0=first_omega_0
        )

        self.pos_encoding2 = SineLayer(
            2,
            128,
            is_first=True, omega_0=first_omega_0
        )

        self.net = []
        for i in range(self.n_hidden_layers):
            self.net.append(SineLayer(self.hidden_features, self.hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        final_linear = nn.Linear(self.hidden_features, 1)

        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / self.hidden_features) / hidden_omega_0,
                                         np.sqrt(6 / self.hidden_features) / hidden_omega_0)

        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, coords):
        pos1 = self.pos_encoding1(self.positional_encoding(coords, self.n_frequencies))
        coords = coords * 2 - 1
        pos2 = self.pos_encoding2(coords)
        output = self.net(torch.cat((pos1, pos2), dim=1))
        return [output]



class Phase12(pl.LightningModule):
    def __init__(self, hparams, v_imgs):
        super(Phase12, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]

        self.imgs = {}
        for img_name in v_imgs:
            img = cv2.cvtColor(cv2.imread(v_imgs[img_name], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
            os.makedirs(os.path.join(self.log_root, img_name), exist_ok=True)
            cv2.imwrite(os.path.join(self.log_root, img_name, "gt.png"), img)
            self.imgs[img_name] = img

        f = getattr(sys.modules[__name__], self.hydra_conf["model"]["model_name"])
        self.model = f([item for item in self.imgs])
        print_model_size(self.model)
        # torch.set_float32_matmul_precision('medium')

    def train_dataloader(self):
        self.train_dataset = Images_dataset(
            self.imgs,
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
        self.valid_dataset = Images_dataset(
            self.imgs,
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
        img_name = batch[0][0]
        pixel_pos = batch[1][0]
        gt_gray = batch[2][0]
        batch_size = gt_gray.shape[0]

        predicted_gray = self.model(img_name, pixel_pos)

        loss = F.l1_loss(predicted_gray[:,0], gt_gray)

        self.log("Training_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        img_name = batch[0][0]
        pixel_pos = batch[1][0]
        gt_gray = batch[2][0]
        batch_size = gt_gray.shape[0]

        predicted_gray = self.model(img_name, pixel_pos)

        loss = F.l1_loss(predicted_gray[:,0], gt_gray)

        self.log("Validation_Loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 batch_size=batch_size)
        return img_name, predicted_gray

    def validation_epoch_end(self, result) -> None:
        if self.trainer.sanity_checking:
            return
        for img_name in self.imgs:
            predicted_imgs = torch.cat([item[1] for item in result if item[0] == img_name], dim=0)
            predicted_imgs = np.clip(predicted_imgs.cpu().numpy(), 0, 1).reshape(
                [self.imgs[img_name].shape[0], self.imgs[img_name].shape[1], 1])
            predicted_imgs = (predicted_imgs * 255).astype(np.uint8)
            img = cv2.cvtColor(predicted_imgs, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(self.log_root, img_name,"{}.png".format(self.trainer.current_epoch)),
                        img)
            self.trainer.logger.experiment.add_image("Image/{}".format(img_name),
                                                     img, self.trainer.current_epoch, dataformats="HWC")


@hydra.main(config_name="phase11_img.yaml", config_path="../../configs/neural_recon/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    # Read dataset and bounds
    img_cache_name = "output/img_field_test/img_cache.npy"
    if os.path.exists(img_cache_name):
        print("Found cache ", img_cache_name)
        imgs, points_3d = np.load(img_cache_name, allow_pickle=True)
    else:
        print("Dosen't find cache, read raw img data")
        bound_min = np.array((-40, -40, -5))
        bound_max = np.array((130, 150, 60))
        bounds_center = (bound_min + bound_max) / 2
        bounds_size = (bound_max - bound_min).max()
        imgs, points_3d = read_dataset(v_cfg["dataset"]["colmap_dir"],
                                       [bound_min,
                                        bound_max]
                                       )
        np.save(img_cache_name[:-4], np.asarray([imgs, points_3d], dtype=object))
        print("Save cache to ", img_cache_name)

    imgs = {img.img_name:img.img_path  for img in imgs[1:3]}

    model = Phase12(v_cfg, imgs)
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
