from src.img2brep.brep.autoregressive import AutoregressiveModel
from src.img2brep.brep.dataset import *
import hydra
from omegaconf import DictConfig


@hydra.main(config_name="train_brepgen.yaml", config_path="../../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    log_root = Path(v_cfg["trainer"]["output"])

    # load model
    autoregressiveModel = AutoregressiveModel()

    print(f"Resuming from {v_cfg['trainer'].resume_from_checkpoint}")

    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]

    state_dict = {k[12:]: v for k, v in state_dict.items() if 'autoencoder' in k}

    autoencoder.load_state_dict(state_dict, strict=True)

    data_set = Auotoencoder_Dataset("validation", v_cfg["dataset"], )
    dataloader = DataLoader(data_set,
                            batch_size=v_cfg["trainer"]["batch_size"],
                            collate_fn=Auotoencoder_Dataset.collate_fn,
                            num_workers=v_cfg["trainer"]["num_worker"],
                            # pin_memory=True,
                            persistent_workers=True if v_cfg["trainer"]["num_worker"] > 0 else False,
                            prefetch_factor=2 if v_cfg["trainer"]["num_worker"] > 0 else None,
                            )

    batch_size = v_cfg["trainer"]["batch_size"]
