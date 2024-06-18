import sys

sys.path.append('../../../')
import os.path

import hydra
# import matplotlib
# matplotlib.use('Agg')
from omegaconf import DictConfig, OmegaConf

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric import seed_everything

from src.img2brep.autoencoder import TrainAutoEncoder


@hydra.main(config_name="train_brepgen.yaml", config_path="../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    if v_cfg["trainer"]["spawn"] is True:
        torch.multiprocessing.set_start_method("spawn")

    mc = ModelCheckpoint(monitor="Validation_Loss", save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = TrainAutoEncoder(v_cfg)
    exp_name = v_cfg["trainer"]["exp_name"]
    logger = TensorBoardLogger(
        log_dir,
        name="autoencoder" if exp_name is None else "autoencoder")

    trainer = Trainer(
        default_root_dir=log_dir,
        logger=logger,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true" if v_cfg["trainer"].gpu > 1 else "auto",
        # strategy="auto",
        devices=v_cfg["trainer"].gpu,
        log_every_n_steps=25,
        enable_model_summary=False,
        callbacks=[mc, lr_monitor],
        max_epochs=int(v_cfg["trainer"]["max_epochs"]),
        # max_epochs=2,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=v_cfg["trainer"]["check_val_every_n_epoch"],
        precision=v_cfg["trainer"]["accelerator"],
        gradient_clip_val=0.5,
        # accumulate_grad_batches=1,
        # profiler=SimpleProfiler(dirpath=log_dir, filename="profiler.txt"),
    )

    if v_cfg["trainer"].resume_from_checkpoint is not None and v_cfg["trainer"].resume_from_checkpoint != "none":
        print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")
        state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
        # state_dict = {k: v for k, v in ckpt["state_dict"].items() if "autoencoder" not in k and "quantizer" not in k}
        # state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    if v_cfg["trainer"].evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
