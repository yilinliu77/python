from src.img2brep.brep.model import AutoEncoder
from src.img2brep.brep.dataset import *
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="train_brepgen.yaml", config_path="../../../configs/img2brep/", version_base="1.1")
def main(v_cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    v_cfg["trainer"]["output"] = os.path.join(log_dir, v_cfg["trainer"]["output"])
    log_root = Path(v_cfg["trainer"]["output"])

    # load model
    autoencoder = AutoEncoder()

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

    for batch_idx, batch in enumerate(dataloader):
        data = batch
        with torch.no_grad():
            recon_edges, recon_edge_mask, recon_faces, recon_face_mask = autoencoder(data, is_inference=True)

        for v_idx in range(batch_size):
            folder_name = data_set.data_folders[batch_idx * batch_size + v_idx].split("/")[-1]
            print(f"Processing {folder_name}")

            gt_edges = data["sample_points_lines"][v_idx].cpu().numpy()
            gt_faces = data["sample_points_faces"][v_idx].cpu().numpy()

            recon_edges = recon_edges[v_idx].cpu().numpy()
            recon_edge_mask = recon_edge_mask[v_idx].cpu().numpy()
            recon_faces = recon_faces[v_idx].cpu().numpy()
            recon_face_mask = recon_face_mask[v_idx].cpu().numpy()

            valid_flag = (gt_edges != -1).all(axis=-1).all(axis=-1)
            gt_edges = gt_edges[valid_flag]
            recon_edges = recon_edges[recon_edge_mask]

            valid_flag = (gt_faces != -1).all(axis=-1).all(axis=-1).all(axis=-1)
            gt_faces = gt_faces[valid_flag]
            recon_faces = recon_faces[recon_face_mask]

            edge_points = np.concatenate((gt_edges, recon_edges), axis=0).reshape(-1, 3)
            edge_colors = np.concatenate(
                    (np.repeat(np.array([[255, 0, 0]], dtype=np.uint8), gt_edges.shape[0] * 20, axis=0),
                     np.repeat(np.array([[0, 255, 0]], dtype=np.uint8), recon_edges.shape[0] * 20, axis=0)), axis=0)

            face_points = np.concatenate((gt_faces, recon_faces), axis=0).reshape(-1, 3)
            face_colors = np.concatenate(
                    (np.repeat(np.array([[0, 0, 255]], dtype=np.uint8), gt_faces.shape[0] * 400, axis=0),
                     np.repeat(np.array([[255, 255, 0]], dtype=np.uint8), recon_faces.shape[0] * 400, axis=0)), axis=0)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(edge_points)
            pc.colors = o3d.utility.Vector3dVector(edge_colors / 255.0)
            o3d.io.write_point_cloud(str(log_root / (folder_name + "_viz_edges.ply")), pc)
            pc.points = o3d.utility.Vector3dVector(face_points)
            pc.colors = o3d.utility.Vector3dVector(face_colors / 255.0)
            o3d.io.write_point_cloud(str(log_root / (folder_name + "_viz_faces.ply")), pc)
            pass


if __name__ == "__main__":
    main()
