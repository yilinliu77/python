import os
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T

root = Path(r"/mnt/d/deepcad_train_v6")

if __name__ == '__main__':
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

    files = list(root.glob("*"))
    with torch.no_grad():
        for prefix in tqdm(os.listdir(root)):
            if not os.path.isdir(root / prefix):
                continue
            if not os.path.exists(root / prefix / "data.npz"):
                shutil.rmtree(root / prefix)
                continue
            data = np.load(root / prefix / "data.npz")["imgs"]
            transformed_imgs = [transform(item) for item in data]
            data = torch.stack(transformed_imgs, dim=0).cuda()
            feature = img_model(data).cpu().numpy()
            np.save(root / prefix / "img_feature_dinov2", feature)
