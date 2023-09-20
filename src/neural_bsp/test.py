import h5py, time
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    id_items = np.random.randint(0, 244, 1000)
    id_x = np.random.randint(0, 7, 1000)
    id_y = np.random.randint(0, 7, 1000)
    id_z = np.random.randint(0, 7, 1000)

    cur_time= time.time()
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
    cur_time= time.time()
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
    cur_time= time.time()
    for i in tqdm(range(5)):
        with h5py.File("G:/Dataset/GSP/GSP_v6_200/training.hdf5") as f:
            f["features"][
                id_items[i],
                id_x[i] * 32:(id_x[i] + 1) * 32,
                :,
                :
            ]
    print(time.time() - cur_time)
