import os
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
import ray

root = Path(r"/mnt/d/deepcad_cond_v0")

@ray.remote(num_gpus=1)
def worker(folders):

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
        for prefix in folders:
            data = np.load(root / prefix / "imgs.npz")["imgs"]
            transformed_imgs = [transform(item) for item in data]
            data = torch.stack(transformed_imgs, dim=0).cuda()
            feature = img_model(data).cpu().numpy()
            np.save(root / prefix / "img_feature_dinov2", feature)

if __name__ == '__main__':
    ray.init(
        num_gpus=8,
    )

    folders = list(root.glob("*"))
    folders.sort()

    results = []
    for i in range(8):
        i_start = i * len(folders) // 8
        i_end = (i + 1) * len(folders) // 8
        results.append(worker.remote(folders[i_start:i_end]))
    ray.get(results)
