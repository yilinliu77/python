import os
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
import ray

root = Path(r"/mnt/d/yilin/img2brep/deepcad_730_imgs_npz_v1")

@ray.remote(num_gpus=0.5)
def worker(folders, v_id):
    torch.set_float32_matmul_precision("medium")
    img_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    img_model.eval()
    img_model.cuda()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        # with torch.autocast(dtype=torch.float16, device_type='cuda'):
        for prefix in tqdm(folders, disable=(v_id != 0)):
            data = np.load(root / prefix / "data.npz")
            transformed_imgs = [transform(item) for item in data["svr_imgs"]] + [transform(item) for item in data["sketch_imgs"]] + [transform(item) for item in data["mvr_imgs"]]
            data = torch.stack(transformed_imgs, dim=0).cuda()
            feature = img_model(data).cpu().numpy()
            np.save(root / prefix / "img_feature_dinov2", feature)

if __name__ == '__main__':
    ray.init(
        num_gpus=8,
    )

    folders = list(root.glob("*"))
    folders.sort()

    folders2 = []
    for folder in folders:
        if not os.path.exists(folder / "img_feature_dinov2.npy"):
            folders2.append(folder.name)
    folders = folders2
    
    print(f"Total {len(folders)} folders to process.")
    results = []
    for i in range(16):
        i_start = i * len(folders) // 16
        i_end = (i + 1) * len(folders) // 16
        results.append(worker.remote(folders[i_start:i_end], i))
    ray.get(results)
