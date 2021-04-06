import sys

sys.path.append("/")
from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import numpy as np



class Benchmark(pl.LightningModule):
    def __init__(self, hparams):
        super(Benchmark, self).__init__()
        self.hparams = hparams

        if self.hparams.model == 0:
            self.model = None
            self.dataset_builder = None

        # self.test_loss = pl.metrics.Accuracy()

    def forward(self, v_data):
        point_inside = self.model(v_data)

        return point_inside

    def train_dataloader(self):
        dataset_path = self.hparams.train_dataset.split(";")
        dataset = []
        for item in dataset_path:
            dataset.append(self.dataset_builder(item,self.hparams, True))
        self.train_dataset = torch.utils.data.ConcatDataset(dataset)
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        dataset_path = self.hparams.valid_dataset.split(";")
        dataset = []
        for item in dataset_path:
            dataset.append(self.dataset_builder(item,self.hparams, False))
        self.valid_dataset = torch.utils.data.ConcatDataset(dataset)
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          drop_last=True,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        self.test_dataset = self.dataset_builder(self.hparams.test_dataset)
        # self.cd_calculator = dist_chamfer_3D.chamfer_3DDist()
        self.model.prepare_test_query_points(v_coordinate_size=self.hparams.coordinate_size)
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_worker,
                          pin_memory=True,
                          )

    def configure_optimizers(self):
        if self.hparams.svr:
            optimizer = Adam(self.model.img_encoder.parameters(), lr=self.hparams.learning_rate,)
        else:
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate,)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': ReduceLROnPlateau(optimizer, patience=10, verbose=True, min_lr=1e-7, threshold=1e-6,
            #                                   factor=0.5),
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        data = batch

        prediction = self.forward(data)
        loss_sp, loss_total = self.model.loss(prediction, data)

        self.log("training_loss", loss_total, prog_bar=True, on_epoch=True)
        self.log("training_points", loss_sp, prog_bar=True, on_epoch=True)

        return {
            'loss': loss_total,
        }

    def validation_step(self, batch, batch_idx):
        data = batch

        prediction = self.forward(data)
        loss_sp, loss_total = self.model.loss(prediction, data)
        return {
            'val_loss': loss_total,
            'val_point_loss': loss_sp,
        }

    def validation_epoch_end(self, outputs):
        avg_loss_total = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_point = torch.stack([x['val_point_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss_total.item(), prog_bar=True, on_epoch=True)
        self.log("val_point_loss", avg_loss_point.item(), prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data = batch

        prediction_training_data = self.forward(data)

        return {

        }

    def test_epoch_end(self, outputs):
        if self.hparams.save_points:
            np.save("temp/points.npy", self.model.test_coords.cpu().numpy())

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        chamfer_distance = np.stack([x['chamfer_distance'] for x in outputs]).mean()
        skip_nums = np.sum([x['skip'] for x in outputs])
        loss_svr = torch.stack([x['loss_svr'] for x in outputs]).mean()

        return {
            "chamfer_distance": chamfer_distance * 1000,
            "avg_err_points": loss,
            "skip": skip_nums,
            "loss_svr": loss_svr,
        }


def get_model_arguments():
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-gp', '--gpu', type=int, default=1)
    parser.add_argument('-nw', '--num_worker', type=int, default=0)
    parser.add_argument('-tr', '--train_dataset', type=str)
    parser.add_argument('-va', '--valid_dataset', type=str)
    parser.add_argument('-te', '--test_dataset', type=str)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    #
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--model', type=int, default=0)
    parser.add_argument('--use_keypoints', action='store_true')
    parser.add_argument('--svr', action='store_true')
    #
    parser.add_argument('--num_attention_layer', type=int, default=3)
    parser.add_argument('--hidden_dim_attention_layer', type=int, default=512)
    parser.add_argument('--dropout_attention_attention_layer', type=float, default=0)
    parser.add_argument('--num_head_attention_layer', type=int, default=1)
    # Test options
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--coordinate_size', type=int, default=64)
    parser.add_argument('--uniform_sample', action='store_true')
    parser.add_argument('--save_points', action='store_true')

    #
    parser.add_argument('-p', '--plane_dim', type=int, default=4096)
    parser.add_argument('-c', '--convex_dim', type=int, default=256)

    return parser


if __name__ == '__main__':
    seed_everything(0)
    # set_start_method('spawn')
    parser = get_model_arguments()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    early_stop_callback = EarlyStopping(
        patience=100,
        monitor="val_loss"
    )

    trainer = Trainer(gpus=args.gpu, weights_summary=None,
                      distributed_backend="ddp" if args.gpu > 1 else None,
                      # early_stop_callback=early_stop_callback,
                      auto_lr_find="learning_rate" if args.auto_lr_find else False,
                      max_epochs=5000,
                      )

    model = Benchmark(args)
    if args.resume_from_checkpoint is not None:
        model.load_state_dict(torch.load(args.resume_from_checkpoint)["state_dict"], strict=True)
    if args.auto_lr_find:
        trainer.tune(model)
        print(model.hparams.learning_rate)
    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)
