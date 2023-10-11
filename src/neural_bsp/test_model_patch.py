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
from src.neural_bsp.abc_hdf5_dataset import ABC_test_mesh, ABC_test_pc, generate_coords


@hydra.main(config_name="test_model_patch.yaml", config_path="../../configs/neural_bsp/", version_base="1.1")
def main(v_cfg: DictConfig):
    # Predefined variables

    seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(v_cfg))
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    output_root = os.path.join(log_dir, v_cfg["trainer"]["output"])

    test_list = v_cfg["dataset"]["test_list"] # List of models to test
    threshold = v_cfg["model"]["test_threshold"]
    res = v_cfg["model"]["test_resolution"]
    query_points = generate_coords(res).reshape(-1,3)

    # Load model and dataset
    dataset_module = importlib.import_module('src.neural_bsp.abc_hdf5_dataset')
    dataset_name = getattr(dataset_module, v_cfg["dataset"]["dataset_name"])

    mod = importlib.import_module('src.neural_bsp.model')
    model = getattr(mod, v_cfg["model"]["model_name"])(
        v_cfg["model"]
    )
    model.cuda()
    state_dict = torch.load(v_cfg["trainer"].resume_from_checkpoint)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=True)

    # Start inference
    for model_name in test_list:
        print("Start testing {}".format(model_name))
        name = Path(model_name).stem

        dataset = dataset_name(
            model_name,
            v_cfg["trainer"]["batch_size"],
            v_cfg["dataset"]["v_output_features"],
            res,
            output_root,
        )

        features = []
        flags = []
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            feat = [(torch.from_numpy(data[0]).cuda().unsqueeze(0), torch.zeros(data[0].shape[0])), None]

            prediction = model(feat, False)
            features.append(data[0])
            flags.append(prediction.detach().cpu().numpy()[0])

        features = np.concatenate(features, axis=0)
        num_features = features.shape[1]
        features_ = np.transpose(features[:,:, 8:24, 8:24, 8:24], (0,2,3,4,1))
        features_ = features_.reshape(
            (-1, 15, 15, 15, 16, 16, 16, num_features)).transpose((0,1, 4, 2, 5, 3, 6, 7)).reshape(240, 240, 240, num_features)
        features = np.ones((res,res,res, num_features), dtype=np.float32) * 999
        features[8:-8, 8:-8, 8:-8, :] = features_

        flags = np.concatenate(flags, axis=0)
        flags = flags.reshape(
            (-1, 15, 15, 15, 16, 16, 16)).transpose((0, 1, 4, 2, 5, 3, 6)).reshape(240, 240, 240)
        flags = np.pad(flags, 8, mode="constant", constant_values=0)
        flags = (sigmoid(flags) > threshold).reshape(-1)

        # valid_points = query_points[flags]
        valid_points = query_points[np.logical_and(flags, (features[..., 0] < 0.1).reshape(-1))]

        export_point_cloud(os.path.join(output_root, "{}_pc.ply".format(name)),
                           valid_points)

        feat_data = dataset.feat_data
        predicted_labels = flags.astype(np.ubyte).reshape(res,res,res)
        gradients_and_udf = np.concatenate((feat_data[..., 1:4], feat_data[..., 0:1]), axis=-1)

        np.save(os.path.join(output_root, "{}_feat.npy".format(name)), gradients_and_udf)
        np.save(os.path.join(output_root, "{}_pred.npy".format(name)), predicted_labels)


if __name__ == '__main__':
    main()
