import pathlib
import time
from random import shuffle

import numpy as np
import h5py

import os, sys

from tqdm import tqdm

from multiprocessing import Process, Queue

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
    while True:
        if len(v_features) == cur_id+1:
            v_queue.put((-1, None, None, None))
            return

        if v_queue.qsize() > 100:
            continue
        prefix = pathlib.Path(v_features[cur_id]).name[:-9]
        feat = np.load(v_features[cur_id])
        flag = np.load(v_flags[cur_id])
        v_queue.put((cur_id, prefix, feat, flag))
        cur_id += 1

def writer(v_queue: Queue, v_num_files):
    MAX_CONSUMER = 10
    id_consumer = MAX_CONSUMER

    output_file = h5py.File(sys.argv[2], "w")

    output_file.create_dataset("features", shape=(id_consumer, 512, 32, 32, 32, 3), dtype=np.uint16,
                               maxshape=(v_num_files, 512, 32, 32, 32, 3),
                               chunks=(1, 1, 32, 32, 32, 3),
                               compression="gzip", compression_opts=1,
                               # compression="lzf"
                               )
    output_file.create_dataset("flags", shape=(v_num_files, 512, 32, 32, 32), dtype=np.uint8,
                               maxshape=(v_num_files, 512, 32, 32, 32),
                               )
    output_file.create_dataset("names", shape=(v_num_files, ), dtype=int,
                               )

    num_finished = 0
    while True:
        if v_queue.qsize() == 0:
            continue

        if id_consumer == 0:
            target_shape = min(output_file["features"].shape[0] + MAX_CONSUMER, v_num_files)
            id_consumer += target_shape - output_file["features"].shape[0]
            output_file["features"].resize(target_shape, axis=0)

        idx, prefix, feat, flag = v_queue.get()
        if idx==-1:
            break
        output_file["features"][idx] = feat
        output_file["flags"][idx] = flag
        output_file["names"][idx] = int(prefix)
        id_consumer -= 1

        num_finished+=1
        if num_finished%(v_num_files//100)==0:
            print("{}/{}".format(num_finished,v_num_files))

    output_file.close()
    return

if __name__ == '__main__':
    # test()

    is_shuffle = True
    root_dir = sys.argv[1]
    feature_files = [os.path.join(root_dir, item) for item in os.listdir(root_dir) if item.endswith("_feat.npy")]
    flag_files = [os.path.join(root_dir, item) for item in os.listdir(root_dir) if item.endswith("_flag.npy")]

    if is_shuffle:
        shuffle(feature_files)
        shuffle(flag_files)

    num_files = len(feature_files)

    timer = time.time()
    q = Queue()
    p_reader = Process(target=reader, args=(q, feature_files, flag_files))
    p_writer = Process(target=writer, args=(q, num_files))
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

