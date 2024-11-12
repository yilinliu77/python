import numpy as np
from pathlib import Path
import os, torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
from src.brepnet.dataset import normalize_coord

if __name__ == "__main__":
    root = Path(r"/mnt/d/abc_v0")
    pred_root = Path(r"/mnt/d/test_pure")
    folder_list = [x for x in os.listdir(pred_root)]
    folder_list.sort()

    loss = []
    loss2 = []
    length = []
    length_edge = []
    for folder in tqdm(folder_list[:1000]):
        data = np.load(root/folder/"data.npz")
        pred_data = np.load(pred_root/folder/"data.npz")
        loss_face = np.abs(pred_data["pred_face"][...,:3] - pred_data["gt_face"][...,:3]).mean(axis=-1).mean(axis=-1).mean(axis=-1)

        pred_face = pred_data["pred_face"][...,:3]
        gt_face = pred_data["gt_face"][...,:3]

        gt_center = gt_face.reshape(gt_face.shape[0], -1, 3).mean(axis=1)
        gt_scale = gt_face.reshape(gt_face.shape[0], -1, 3)
        gt_scale = gt_scale.max(axis=1)-gt_scale.min(axis=1)
        gt_scale = np.max(gt_scale, axis=1)
        length_face = gt_scale

        pred_face_norm = (pred_data["pred_face"][...,:3] - gt_center[:,None,None]) / (gt_scale[:,None,None,None] + 1e-6)
        gt_face_norm = (pred_data["gt_face"][...,:3] - gt_center[:,None,None]) / (gt_scale[:,None,None,None] + 1e-6)
        loss_face2 = np.abs(pred_face_norm[...,:3] - gt_face_norm[...,:3]).mean(axis=-1).mean(axis=-1).mean(axis=-1)

        gt_edge = pred_data["gt_edge"][...,:3]
        gt_center = gt_edge.reshape(gt_edge.shape[0], -1, 3).mean(axis=1)
        gt_scale = gt_edge.reshape(gt_edge.shape[0], -1, 3)
        gt_scale = gt_scale.max(axis=1)-gt_scale.min(axis=1)
        gt_scale = np.max(gt_scale, axis=1)
        if gt_scale.min()<5e-4 and gt_scale.min()>1e-4:
            pass
        length_edge.append(gt_scale)

        loss.append(loss_face)
        loss2.append(loss_face2)
        length.append(length_face)
    loss = np.concatenate(loss)
    loss2 = np.concatenate(loss2)
    length = np.concatenate(length)
    length_edge = np.concatenate(length_edge)
    pass

    fig, ax = plt.subplots(3, 2)
    ax[0,0].hist(loss, bins=10, range=(0,0.02))
    ax[0,1].hist(length, bins=10, range=(0,4))
    ax[1,0].scatter(length, loss, s=0.5, edgecolor='none')
    ax[1,0].set_xlim(0,4)
    ax[1,0].set_ylim(0,0.1)
    ax[1,1].scatter(length, loss2, s=0.5, edgecolor='none')
    ax[1,1].set_xlim(0,4)
    ax[1,1].set_ylim(0,0.1)
    index = np.digitize(length, bins=np.linspace(0,2,10))
    loss_mean = [loss2[index==i].mean() for i in range(1,10)]
    ax[2,0].plot(np.linspace(0,2,9), loss_mean)
    plt.savefig("length_loss.png",dpi=300)
    ax[2,1].plot(np.linspace(0,2,9), np.log(loss_mean))
    plt.savefig("length_loss.png",dpi=300)
    pass