import pathlib
import time
from random import shuffle, seed

import numpy as np

import os, sys

from tqdm import tqdm

from multiprocessing import Process, Queue
import h5py

def test():
    root_dir = r"F:\GSP\GSP_v3\training"
    tasks = [os.path.join(root_dir, item) for item in os.listdir(root_dir) if item.endswith("_feat.npy")]
    num_patch = 512

    # Sequential read1
    timer = time.time()
    for item in tqdm(tasks):
        for i in range(num_patch):
            feat = np.load(item, mmap_mode="r")
            a = feat[i].mean()
    print(time.time() - timer)

    # Sequential read2
    timer = time.time()
    for item in tqdm(tasks):
        feat = np.load(item, mmap_mode="r")
        for i in range(num_patch):
            a = feat[i].mean()
    print(time.time() - timer)

    # Random read
    timer = time.time()
    for i in tqdm(range(len(tasks) * num_patch)):
        feat = np.load(os.path.join(root_dir, tasks[i // num_patch]), mmap_mode="r")
        a = feat[i % num_patch].mean()
    print(time.time() - timer)

    # Sequential read hdf5
    timer = time.time()
    with h5py.File("train.hdf5", "r") as f:
        for i in tqdm(range(len(tasks) * num_patch)):
            feat = f["features"][i // num_patch, i % num_patch]
            a = feat.mean()
    print(time.time() - timer)

    return

def reader(v_queue: Queue, v_features, v_flags):
    cur_id = 0
    while cur_id < len(v_features):
        if v_queue.qsize() >= 200:
            time.sleep(1)
            continue
        prefix = pathlib.Path(v_features[cur_id]).name[:-9]
        feat = np.load(v_features[cur_id])
        flag = np.load(v_flags[cur_id])
        v_queue.put((cur_id, prefix, feat, flag))
        cur_id += 1

def writer(v_queue: Queue, v_num_files, v_filename):
    num_block = 100

    output_file = h5py.File(v_filename, "w",)

    output_file.create_dataset("features", shape=(num_block, 512, 32, 32, 32, 3), dtype=np.uint16,
                               maxshape=(v_num_files, 512, 32, 32, 32, 3),
                               chunks=(1, 1, 32, 32, 32, 3),
                               compression="gzip", compression_opts=1,
                               # compression="lzf"
                               )
    output_file.create_dataset("flags", shape=(num_block, 512, 32, 32, 32), dtype=np.uint8,
                               maxshape=(v_num_files, 512, 32, 32, 32),
                               chunks=(1, 1, 32, 32, 32),
                               )
    output_file.create_dataset("names", shape=(v_num_files, ), dtype=int,)

    num_finished = 0
    while True:
        if num_finished + num_block > v_num_files:
            num_block = v_num_files - num_finished

        if v_queue.qsize() < num_block:
            time.sleep(1)

        idx, prefix, feat, flag = [None]*num_block,[None]*num_block,[None]*num_block,[None]*num_block
        for i in range(num_block):
            idx[i], prefix[i], feat[i], flag[i] = v_queue.get()

        prefix = np.stack(prefix, axis=0).astype(np.int32)
        feat = np.stack(feat, axis=0)
        flag = np.stack(flag, axis=0)

        output_file["features"][num_finished:num_finished+num_block] = feat
        output_file["flags"][num_finished:num_finished+num_block] = flag
        output_file["names"][num_finished:num_finished+num_block] = prefix

        target_shape = min(output_file["features"].shape[0] + num_block, v_num_files)
        output_file["features"].resize(target_shape, axis=0)
        output_file["flags"].resize(target_shape, axis=0)

        num_finished+=num_block
        assert num_finished <= v_num_files
        if num_finished == v_num_files:
            break
        if num_finished%(v_num_files//100)==0:
            print("{}/{}".format(num_finished,v_num_files))

    output_file.close()
    return

if __name__ == '__main__':
    # test()
    is_shuffle = True
    root_dir = sys.argv[1]
    prefix = set([item[:-9] for item in os.listdir(root_dir) if item.endswith("_feat.npy")])
    prefix = sorted(list(prefix),key=lambda item:int(item))

    seed(0)
    if is_shuffle:
        shuffle(prefix)

    feature_files = [os.path.join(root_dir, item+"_feat.npy") for item in prefix]
    flag_files = [os.path.join(root_dir, item+"_flag.npy") for item in prefix]

    num_files = len(feature_files)

    timer = time.time()
    q = Queue()
    hdf5_file = sys.argv[2]
    p_reader = Process(target=reader, args=(q, feature_files, flag_files))
    p_writer = Process(target=writer, args=(q, num_files, hdf5_file))
    p_reader.start()
    p_writer.start()

    p_reader.join()
    p_writer.join()
    print(time.time()-timer)

    def single_thread_test():
        output_file = h5py.File("train.hdf5", "w")

        id_consumer = 3
        output_file.create_dataset("features", shape=(id_consumer, 512, 32, 32, 32, 3), dtype=np.uint16,
                                   maxshape=(num_files, 512, 32, 32, 32, 3),
                                   chunks=(1, 1, 32, 32, 32, 3),
                                   compression="gzip", compression_opts=1
                                   )
        output_file.create_dataset("flags", shape=(num_files, 512, 32, 32, 32), dtype=np.uint8,
                                   maxshape=(num_files, 512, 32, 32, 32),
                                   )

        times = [0] * 10
        for idx in tqdm(range(num_files)):
            timer = time.time()
            if id_consumer == 0:
                target_shape = min(output_file["features"].shape[0] + 3, num_files)
                id_consumer += target_shape - output_file["features"].shape[0]
                output_file["features"].resize(target_shape, axis=0)
            times[2] += time.time() - timer

            timer = time.time()
            a = np.load(feature_files[idx])
            b = np.load(flag_files[idx])
            times[0] += time.time() - timer

            timer = time.time()
            output_file["features"][idx] = a
            output_file["flags"][idx] = b
            id_consumer -= 1
            times[1] += time.time() - timer

        output_file.close()
    pass

