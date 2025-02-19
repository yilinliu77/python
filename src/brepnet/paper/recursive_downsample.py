from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

target_size = (1500, 1500)
root = Path(r"D:/brepnet/paper_imgs/txt_results/text2cad")

if __name__ == '__main__1':
    files = [file for file in root.rglob('gt.png') if file.is_file()]
    for file in tqdm(files):
        img = cv2.imread(str(file))
        cv2.imwrite(file.parent / (file.stem + '_backup.png'), img)
        img = cv2.resize(img, target_size)
        filename = file.parent / (file.stem + '.png')
        cv2.imwrite(filename, img)

if __name__ == '__main__':
    files = [file for file in root.rglob('*gt.png') if file.is_file()]
    for file in tqdm(files):
        img = cv2.imread(str(file))
        # Remove background to make it transparent
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[np.all(img == (255, 255, 255, 255), axis=-1)] = (255, 255, 255, 0)
        filename = file.parent / (file.stem + '_downsampled.png')
        # cv2.imwrite(filename, img)
        cv2.imwrite(file, img)