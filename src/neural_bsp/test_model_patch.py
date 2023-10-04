import importlib
import os.path
from pathlib import Path

import hydra
import numpy as np
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from shared.common_utils import export_point_cloud, sigmoid
from src.neural_bsp.abc_hdf5_dataset import ABC_dataset_test_mesh, ABC_dataset_test_pc


@hydra.main(config_name="test_model_patch.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    output_root = os.path.join(log_dir, v_cfg["trainer"]["output"])

    mod = importlib.import_module('src.neural_bsp.model')
    model = getattr(mod, v_cfg["model"]["model_name"])(
        v_cfg["model"]
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)

    test_list = v_cfg["dataset"]["test_list"]
    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    ps = 32  # Patch size
    warp = ps // 4

    for model_name in test_list:
        print("Start testing {}".format(model_name))
        name = Path(model_name).stem
        dataset = ABC_dataset_test_pc(
            model_name,
            v_cfg["trainer"]["batch_size"],
            v_cfg["dataset"]["v_output_features"],
            res
        )

        features = []
        flags = []
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            feat = [torch.from_numpy(data[0]).cuda(), None]

            prediction = model(feat, False)
            features.append(data[0])
            flags.append(prediction.detach().cpu().numpy())

        features = np.concatenate(features, axis=0)
        flags = np.concatenate(flags, axis=0)

        flags = flags.reshape(
            (-1, 15, 15, 15, 16, 16, 16)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(240, 240, 240)
        flags = np.pad(flags, 8, mode="constant", constant_values=0)
        flags = sigmoid(flags) > 0.5

        v_resolution = 256
        query_points = np.meshgrid(np.arange(v_resolution), np.arange(v_resolution), np.arange(v_resolution),
                                   indexing="ij")
        query_points = np.stack(query_points, axis=3) / (v_resolution - 1)
        query_points = (query_points * 2 - 1).astype(np.float32).reshape(-1, 3)
        export_point_cloud(os.path.join(output_root, "{}_pc.ply".format(name)),
                           query_points[flags.reshape(-1)])

        feat_data = dataset.feat_data
        predicted_labels = flags.astype(np.ubyte)
        gradients_and_udf = np.concatenate((feat_data[..., 1:4], feat_data[..., 0:1]), axis=-1)

        np.save(os.path.join(output_root, "{}_feat.npy".format(name)), gradients_and_udf)
        np.save(os.path.join(output_root, "{}_pred.npy".format(name)), predicted_labels)


if __name__ == '__main__':
    main()
