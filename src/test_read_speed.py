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

    hdf1 = h5py.File(hdf1_, "w")
    hdf2 = h5py.File(hdf2_, "w")
    hdf3 = h5py.File(hdf3_, "w")
    hdf4 = h5py.File(hdf4_, "w")


if __name__ == '__main__':
    # generate_dataset()
    test_write_single()
