import h5py, time
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from src.neural_bsp.abc_hdf5_dataset import angle2vector, generate_coords
from src.neural_bsp.my_dataloader import MyDataLoader


def test_hdf5():
    id_items = np.random.randint(0, 244, 1000)
    id_x = np.random.randint(0, 7, 1000)
    id_y = np.random.randint(0, 7, 1000)
    id_z = np.random.randint(0, 7, 1000)

    cur_time = time.time()
    for i in tqdm(range(320)):
        with h5py.File("G:/Dataset/GSP/GSP_v6_200/training.hdf5") as f:
            f["features"][
            id_items[i],
            id_x[i] * 32:(id_x[i] + 1) * 32,
            id_y[i] * 32:(id_y[i] + 1) * 32,
            id_z[i] * 32:(id_z[i] + 1) * 32
            ]
    print(time.time() - cur_time)

    id_items = np.random.randint(0, 244, 1000)
    cur_time = time.time()
    for i in tqdm(range(40)):
        with h5py.File("G:/Dataset/GSP/GSP_v6_200/training.hdf5") as f:
            f["features"][
            id_items[i],
            id_x[i] * 32:(id_x[i] + 1) * 32,
            id_y[i] * 32:(id_y[i] + 1) * 32,
            :
            ]
    print(time.time() - cur_time)

    id_items = np.random.randint(0, 244, 1000)
    cur_time = time.time()
    for i in tqdm(range(5)):
        with h5py.File("G:/Dataset/GSP/GSP_v6_200/training.hdf5") as f:
            f["features"][
            id_items[i],
            id_x[i] * 32:(id_x[i] + 1) * 32,
            :,
            :
            ]
    print(time.time() - cur_time)

def test_hdf52():
    id_items = np.random.randint(0, 244, 1000)
    id_x = np.random.randint(0, 6, 1000)
    id_y = np.random.randint(0, 6, 1000)
    id_z = np.random.randint(0, 6, 1000)

    cur_time = time.time()
    for i in tqdm(range(320)):
        with h5py.File("G:/Dataset/GSP/GSP_v7_10k2/training.h5") as f:
            f["features"][
            id_items[i],
            id_x[i] * 32:(id_x[i] + 1) * 32,
            :
            ]
    print(time.time() - cur_time)

    cur_time = time.time()
    for i in tqdm(range(320)):
        with h5py.File("G:/Dataset/GSP/GSP_v7_10k2/training.h5") as f:
            f["features"][
            id_items[i],
            id_x[i] * 32 + 16:(id_x[i] + 1) * 32 + 16,
            :
            ]
    print(time.time() - cur_time)

    cur_time = time.time()
    for i in tqdm(range(320)):
        with h5py.File("G:/Dataset/GSP/GSP_v7_10k2/training.h5") as f:
            f["features"][
            id_items[i],
            ]
    print(time.time() - cur_time)

def test_hdf53():
    root = "G:/Dataset/GSP/GSP_v7_1k/training.h5"
    id_item = 500
    query_points = generate_coords(256)
    with h5py.File(root) as f:
        udf = f["features"][id_item, :, :, :, 0] / 65535. * 2
        angles = f["features"][id_item, :, :, :, 1:3]
        gradients = angle2vector(angles)
        normal_angles = f["features"][id_item, :, :, :, 3:5]
        normal_vector = angle2vector(normal_angles)

        p = query_points + udf[:,:,:,None] * gradients
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(p.reshape(-1,3))
        pc.normals = o3d.utility.Vector3dVector(normal_vector.reshape(-1,3))
        o3d.io.write_point_cloud("test.ply", pc)

        pc.points = o3d.utility.Vector3dVector(query_points[f["flags"][id_item].astype(bool)].reshape(-1, 3))
        o3d.io.write_point_cloud("test.ply", pc)

        pass
    pass

class ABC(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __getitem__(self, idx):
        time.sleep(0.5)
        return idx

    def __len__(self):
        return 100

def test_dataloader():
    dataset = ABC()
    loader = MyDataLoader(dataset, num_workers=8, batch_size=1, shuffle=True)
    # loader = DataLoader(dataset, num_workers=8, batch_size=1, shuffle=True)

    index = []
    for i in tqdm(loader):
        index.append(i)
    index = torch.cat(index)
    index = torch.sort(index)[0]
    pass

if __name__ == '__main__':
    test_dataloader()
    # test_hdf53()
