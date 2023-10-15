import numpy as np
import h5py
import os
from pathlib import Path

from tqdm import tqdm

root_path = Path("c://DATASET/test")
test_shape = 1000

def generate_dataset():
    os.makedirs(root_path, exist_ok=True)
    hdf1_ = root_path/"test_compression.hdf5"
    hdf2_ = root_path/"test_compression_shuffle"
    hdf3_ = root_path/"test_vanilla.hdf5"
    hdf4_ = root_path/"test_typed.hdf5"
    npy_folder = root_path/"npy"

    hdf1 = h5py.File(hdf1_, "w")
    hdf1.create_dataset("features", (test_shape, 512, 32, 32, 32),
                        dtype=np.float16,
                        chunks = (1, 1, 32, 32, 32),
                        compression="lzf")
    hdf2 = h5py.File(hdf2_, "w")
    hdf2.create_dataset("features", (test_shape, 512, 32, 32, 32),
                        dtype=np.float16,
                        chunks = (1, 1, 32, 32, 32),
                        compression="lzf",
                        shuffle=True)
    hdf3 = h5py.File(hdf3_, "w")
    hdf3.create_dataset("features", (test_shape, 512, 32, 32, 32),
                        dtype=np.float16,
                        )
    hdf4 = h5py.File(hdf4_, "w")
    hdf4.create_dataset("features", (test_shape, 512, 32, 32, 32),
                        dtype=np.uint16,
                        )
    os.makedirs(npy_folder, exist_ok=True)

    for i in tqdm(range(test_shape)):
        data = np.random.rand(512,32,32,32).astype(np.float16)
        hdf1["features"][i] = data
        hdf2["features"][i] = data
        hdf3["features"][i] = data
        hdf4["features"][i] = data
        np.save(str(root_path/"npy"/"{}".format(i)), data)
        np.savez_compressed(str(root_path/"npy"/"{}_c".format(i)), data)

    hdf1.close()
    hdf2.close()
    hdf3.close()
    hdf4.close()


def test_write_single():
    hdf1_ = root_path / "test_compression.hdf5"
    hdf2_ = root_path / "test_compression_shuffle"
    hdf3_ = root_path / "test_vanilla.hdf5"
    hdf4_ = root_path / "test_typed.hdf5"
    npy_folder = root_path / "npy"
    length = 100

    hdf1 = h5py.File(hdf1_, "r")
    hdf2 = h5py.File(hdf2_, "r")
    hdf3 = h5py.File(hdf3_, "r")
    hdf4 = h5py.File(hdf4_, "r")

    print("Compressed HDF5 sequential")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.asarray(hdf1["features"][idx, idy:idy+256])

    print("Compressed HDF5 sample")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 511, 256)
        idy = np.sort(idy)
        np.asarray(hdf1["features"][idx, idy])

    print("Typed Compressed HDF5")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.asarray(hdf1["features"][idx, idy:idy+256]).astype(np.float32)

    print("Compressed HDF5 with shuffle")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.asarray(hdf1["features"][idx, idy:idy+256])

    print("Vanilla HDF5")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.asarray(hdf1["features"][idx, idy:idy+256])

    print("Typed HDF5")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.asarray(hdf1["features"][idx, idy:idy+256]).astype(np.float32)

    print("NPY")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.load(str(npy_folder/"{}".format(i))+".npy", "r")[idx, idy:idy+256][:]

    print("Compressed NPY")
    for i in tqdm(range(length)):
        idx = np.random.randint(0, test_shape-1)
        idy = np.random.randint(0, 255)
        np.load(str(npy_folder/"{}".format(i))+".npz", "r")[idx, idy:idy+256][:]

    hdf1.close()
    hdf2.close()
    hdf3.close()
    hdf4.close()

if __name__ == '__main__':
    # generate_dataset()
    test_write_single()
