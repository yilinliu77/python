import sys, os

sys.path.append("thirdparty/sdf_computer/build/")
import pysdf

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

import mcubes
import numpy as np
import open3d as o3d

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

from src.neural_bsp.bak.bspt import get_mesh_watertight

def write_ply_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(polygons)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


class BSP_dataset_single_scene(torch.utils.data.Dataset):
    def __init__(self, v_data, v_batch_size, v_training_mode):
        super(BSP_dataset_single_scene, self).__init__()
        self.points = torch.from_numpy(v_data["points"]) - 0.5
        self.points = torch.cat((self.points, torch.ones_like(self.points[:, 0:1])), dim=1)
        self.sdfs = torch.from_numpy(v_data["sdfs"])

        mask = self.sdfs > 0
        self.sdfs[mask] = 0
        self.sdfs[~mask] = 1

        self.mode = v_training_mode
        self.batch_size = v_batch_size
        pass

    def __len__(self):
        return 10000 if self.mode=="training" else 1

    def __getitem__(self, idx):
        which_point = torch.randint(0, self.points.shape[0], (self.batch_size,))
        return self.points[which_point], self.sdfs[which_point]


class generator(nn.Module):
    def __init__(self, phase, p_dim=4096, c_dim=256):
        super(generator, self).__init__()
        self.phase = phase
        self.p_dim = p_dim
        self.c_dim = c_dim
        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        concave_layer_weights = torch.zeros((self.c_dim, 1))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        self.concave_layer_weights = nn.Parameter(concave_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

    def forward(self, points, plane_m, convex_mask=None, is_training=False):
        if self.phase == 0:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)
            h2 = torch.clamp(1 - h2, min=0, max=1)

            # level 3
            h3 = torch.matmul(h2, self.concave_layer_weights)
            h3 = torch.clamp(h3, min=0, max=1)

            return h2, h3, self.convex_layer_weights, self.concave_layer_weights
        elif self.phase == 1 or self.phase == 2:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights > 0.01).float())

            # level 3
            if convex_mask is None:
                h3 = torch.min(h2, dim=2, keepdim=True)[0]
            else:
                h3 = torch.min(h2 + convex_mask, dim=2, keepdim=True)[0]

            return h2, h3, self.convex_layer_weights, self.concave_layer_weights
        elif self.phase == 3 or self.phase == 4:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)

            # level 3
            if convex_mask is None:
                h3 = torch.min(h2, dim=2, keepdim=True)[0]
            else:
                h3 = torch.min(h2 + convex_mask, dim=2, keepdim=True)[0]

            return h2, h3, self.convex_layer_weights, self.concave_layer_weights


class encoder(nn.Module):
    def __init__(self, ef_dim=32):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 8, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias, 0)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias, 0)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs, is_training=False):
        d_1 = self.conv_1(inputs)
        d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

        d_2 = self.conv_2(d_1)
        d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)

        d_3 = self.conv_3(d_2)
        d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

        d_4 = self.conv_4(d_3)
        d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.ef_dim * 8)
        d_5 = torch.sigmoid(d_5)

        return d_5


class decoder(nn.Module):
    def __init__(self, ef_dim=32, p_dim=4096):
        super(decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.linear_1 = nn.Linear(self.ef_dim * 8, self.ef_dim * 16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim * 16, self.ef_dim * 32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim * 32, self.ef_dim * 64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim * 64, self.p_dim * 4, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, inputs, is_training=False):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4


class Base_model(nn.Module):
    def __init__(self, v_phase=0, p_dim=4096, c_dim=256):
        super(Base_model, self).__init__()
        self.phase = v_phase
        # self.decoder = decoder()
        plane_m = torch.rand((4,p_dim), dtype=torch.float32)
        # plane_m = plane_m / 100
        self.plane_m = nn.Parameter(plane_m)
        self.generator = generator(self.phase, p_dim, c_dim)

        self.test_resolution = 64
        self.test_coords = torch.stack(torch.meshgrid(
            torch.linspace(-0.5, 0.5, self.test_resolution),
            torch.linspace(-0.5, 0.5, self.test_resolution),
            torch.linspace(-0.5, 0.5, self.test_resolution), indexing='xy'
        ), dim=3).reshape(-1, 3)
        self.test_coords = torch.cat((self.test_coords, torch.ones_like(self.test_coords[:, 0:1])), dim=1)

    def forward(self, v_data, v_training=False):
        points, sdfs = v_data
        plane_m = self.plane_m.unsqueeze(0)
        net_out_convexes, net_out, convex_layer_weights, concave_layer_weights = self.generator(points, plane_m, )

        return None, plane_m, net_out_convexes, net_out, convex_layer_weights, concave_layer_weights

    def loss(self, v_predictions, v_input):
        points, sdfs = v_input
        z_vector, plane_m, net_out_convexes, net_out, convex_layer_weights, concave_layer_weights = v_predictions
        if self.phase == 0:
            # phase 0 continuous for better convergence
            # L_recon + L_W + L_T
            # net_out_convexes - network output (convex layer), the last dim is the number of convexes
            # net_out - network output (final output)
            # point_value - ground truth inside-outside value for each point
            # convex_layer_weights - connections T
            # concave_layer_weights - auxiliary weights W
            loss_sp = torch.mean((sdfs - net_out) ** 2)
            loss = loss_sp + \
                   torch.sum(torch.abs(concave_layer_weights - 1)) + \
                   (torch.sum(torch.clamp(convex_layer_weights - 1, min=0) - torch.clamp(convex_layer_weights, max=0)))
            return loss_sp, loss
        elif self.phase == 1:
            # phase 1 hard discrete for bsp
            # L_recon
            loss_sp = torch.mean((1 - sdfs) * (1 - torch.clamp(net_out, max=1)) + \
                                 sdfs * (torch.clamp(net_out, min=0)))
            loss = loss_sp
            return loss_sp, loss
        elif self.phase == 2:
            # phase 2 hard discrete for bsp with L_overlap
            # L_recon + L_overlap
            loss_sp = torch.mean((1 - sdfs) * (1 - torch.clamp(net_out, max=1)) + \
                                 sdfs * (torch.clamp(net_out, min=0)))
            G2_inside = (net_out_convexes < 0.01).float()
            bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True) > 1).float()
            loss = loss_sp - torch.mean(net_out_convexes * sdfs * bmask)
            return loss_sp, loss
        elif self.phase == 3:
            # phase 3 soft discrete for bsp
            # L_recon + L_T
            # soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
            loss_sp = torch.mean((1 - sdfs) * (1 - torch.clamp(net_out, max=1)) + \
                                 sdfs * (torch.clamp(net_out, min=0)))
            loss = loss_sp + \
                   torch.sum((convex_layer_weights < 0.01).float() * torch.abs(convex_layer_weights)) + \
                   torch.sum((convex_layer_weights >= 0.01).float() * torch.abs(convex_layer_weights - 1))
            return loss_sp, loss
        elif self.phase == 4:
            # phase 4 soft discrete for bsp with L_overlap
            # L_recon + L_T + L_overlap
            # soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
            loss_sp = torch.mean((1 - sdfs) * (1 - torch.clamp(net_out, max=1)) + \
                                 sdfs * (torch.clamp(net_out, min=0)))
            G2_inside = (net_out_convexes < 0.01).float()
            bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True) > 1).float()
            loss = loss_sp + \
                   torch.sum((convex_layer_weights < 0.01).float() * torch.abs(convex_layer_weights)) + \
                   torch.sum((convex_layer_weights >= 0.01).float() * torch.abs(convex_layer_weights - 1)) - \
                   torch.mean(net_out_convexes * sdfs * bmask)
            return loss_sp, loss
        return

    def extract_mc(self, v_plane_m, v_output_file, v_batch_size=10000):
        points = self.test_coords.to(v_plane_m.device)
        num_batch = self.test_coords.shape[0] // v_batch_size + 1
        net_outs = []
        for i in range(num_batch):
            id_start = i * v_batch_size
            id_end = min((i + 1) * v_batch_size, self.test_coords.shape[0])
            if id_start >= self.test_coords.shape[0]:
                break
            _, net_out, _, _ = self.generator(points[None, id_start:id_end], v_plane_m, )
            net_outs.append(net_out.cpu().numpy())
        net_outs = np.concatenate(net_outs, axis=1)[0, :, 0]
        net_outs = net_outs.reshape(self.test_resolution, self.test_resolution, self.test_resolution)
        vertices, triangles = mcubes.marching_cubes(net_outs, 0.5)
        if vertices.shape[0] == 0:
            return
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d.io.write_triangle_mesh(v_output_file, mesh)

    def extract_fi(self, v_plane_m, v_output_file, v_batch_size=10000):
        plane_m = v_plane_m.cpu().numpy()
        points = self.test_coords.to(v_plane_m.device)
        num_batch = self.test_coords.shape[0] // v_batch_size + 1
        net_out_convexes = []
        net_outs = []
        dim_convex = self.generator.c_dim
        dim_plane = self.generator.p_dim
        w2 = self.generator.convex_layer_weights.detach().cpu().numpy()
        for i in range(num_batch):
            id_start = i * v_batch_size
            id_end = min((i + 1) * v_batch_size, self.test_coords.shape[0])
            if id_start >= self.test_coords.shape[0]:
                break
            net_out_convex, net_out, _, _ = self.generator(points[None, id_start:id_end], v_plane_m, )
            net_out_convexes.append(net_out_convex.cpu().numpy())
            net_outs.append(net_out.cpu().numpy())
        net_outs = np.concatenate(net_outs, axis=1)[0, :, 0]
        net_out_convexes = np.concatenate(net_out_convexes, axis=1)[0, :]
        net_outs = net_outs.reshape(self.test_resolution, self.test_resolution, self.test_resolution)
        net_out_convexes = net_out_convexes.reshape(self.test_resolution, self.test_resolution, self.test_resolution,
                                                    dim_convex)

        # Extract
        bsp_convex_list = []
        net_out_convexes = net_out_convexes < 0.01
        net_out_convexes_sum = np.sum(net_out_convexes, axis=3)
        unused_convex = np.ones([dim_convex], np.float32)
        for i in range(dim_convex):
            slice_i = net_out_convexes[:, :, :, i]
            if np.max(slice_i) > 0:  # if one voxel is inside a convex
                if np.min(
                        net_out_convexes_sum - slice_i * 2) >= 0:  # if this convex is redundant, i.e. the convex is inside the shape
                    net_out_convexes_sum = net_out_convexes_sum - slice_i
                else:
                    box = []
                    for j in range(dim_plane):
                        if w2[j, i] > 0.01:
                            a = -plane_m[0, 0, j]
                            b = -plane_m[0, 1, j]
                            c = -plane_m[0, 2, j]
                            d = -plane_m[0, 3, j]
                            box.append([a, b, c, d])
                    if len(box) > 0:
                        bsp_convex_list.append(np.array(box, np.float32))
                        unused_convex[i] = 0

        if len(bsp_convex_list) == 0:
            return
        # convert bspt to mesh
        # vertices, polygons = get_mesh(bsp_convex_list)
        # use the following alternative to merge nearby vertices to get watertight meshes
        vertices, polygons = get_mesh_watertight(bsp_convex_list)
        write_ply_polygon(v_output_file, vertices, polygons)
        pass


class Scene_phase(pl.LightningModule):
    def __init__(self, hparams, v_data):
        super(Scene_phase, self).__init__()
        self.hydra_conf = hparams
        self.learning_rate = self.hydra_conf["trainer"]["learning_rate"]
        self.batch_size = self.hydra_conf["trainer"]["batch_size"]
        self.num_worker = self.hydra_conf["trainer"]["num_worker"]
        self.save_hyperparameters(hparams)

        self.log_root = self.hydra_conf["trainer"]["output"]
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.data = v_data
        self.phase = self.hydra_conf["model"]["phase"]
        self.model = Base_model(self.phase,
                                self.hydra_conf["model"]["pdim"],
                                self.hydra_conf["model"]["cdim"])

        # Used for visualizing during the training
        self.viz_data = {}

    def train_dataloader(self):
        self.train_dataset = BSP_dataset_single_scene(
            self.data,
            self.batch_size,
            "training",
        )
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                          num_workers=self.hydra_conf["trainer"]["num_worker"],
                          pin_memory=True,
                          persistent_workers=True if self.hydra_conf["trainer"]["num_worker"] > 0 else False)

    def val_dataloader(self):
        self.valid_dataset = BSP_dataset_single_scene(
            self.data,
            self.batch_size,
            "validation"
        )
        return DataLoader(self.valid_dataset, batch_size=1, num_workers=0)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, )

        return {
            'optimizer': optimizer,
            'monitor': 'Validation_Loss'
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, True)
        loss_sp, total_loss = self.model.loss(outputs, batch)

        self.log("Training_Loss", total_loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[0])
        self.log("Training_Loss_SP", loss_sp.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[0])

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch, False)
        loss_sp, total_loss = self.model.loss(outputs, batch)

        self.log("Validation_Loss", total_loss.detach(), prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 sync_dist=True,
                 batch_size=batch[0].shape[0])

        if self.global_rank == 0 and batch_idx == 0:
            self.viz_data["planes"] = outputs[1][0:1]
            # self.viz_data["gt"] = batch[2][0,0]

        return total_loss

    def validation_epoch_end(self, result) -> None:
        if self.global_rank != 0:
            return

        # if self.trainer.sanity_checking:
        #     return

        if self.phase == 0:
            self.model.extract_mc(self.viz_data["planes"],
                                  os.path.join(self.log_root, "{}.ply".format(self.current_epoch)),
                                  10000)
        else:
            self.model.extract_fi(self.viz_data["planes"],
                                  os.path.join(self.log_root, "{}.ply".format(self.current_epoch)),
                                  10000)


def prepare_dataset(v_root, v_rebuild_data = False):
    cache_file_name = "output/neural_bsp/bsp_data.npy"
    if os.path.exists(cache_file_name) and not v_rebuild_data:
        print("Start to read cache: {}".format(cache_file_name))
        data = np.load(cache_file_name,allow_pickle=True).item()
        print("{} points and values;".format(
            data["points"].shape[0],
        ))

    else:
        print("Start to read dataset: {}".format(v_root))

        mesh = o3d.io.read_triangle_mesh(v_root)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        sdf_computer = pysdf.PYSDF_computer()
        sdf_computer.setup_mesh(vertices[faces], True)
        scale = sdf_computer.m_scale
        center = sdf_computer.m_center
        sdf = sdf_computer.compute_sdf(0, int(2e8), int(1e8), False)

        sampled_points = sdf[:,:3].astype(np.float32)
        sdf = sdf[:,3].astype(np.float32)

        print("Done")
        print("{} points and values;".format(
            sampled_points.shape[0],
        ))
        os.makedirs("output/neural_bsp",exist_ok=True)
        data = {
            "points": sampled_points,
            "sdfs": sdf,
            "scale": np.insert(center,3, scale)
        }
        np.save(cache_file_name,data,allow_pickle=True)
    return data


@hydra.main(config_name="optimized_scene.yaml", config_path="../../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    print(OmegaConf.to_yaml(v_cfg))
    data = prepare_dataset(
        v_cfg["dataset"]["root"],
        v_cfg["dataset"]["rebuild_data"],
    )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])

    model = Scene_phase(v_cfg, data)

    trainer = Trainer(
        accelerator='gpu' if v_cfg["trainer"].gpu != 0 else None,
        # strategy = "ddp",
        devices=v_cfg["trainer"].gpu, enable_model_summary=False,
        max_epochs=int(1e8),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        default_root_dir=log_dir,
        # precision=16,
        # gradient_clip_val=0.5
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
