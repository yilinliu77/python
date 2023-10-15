import pathlib
import time
from random import shuffle, seed
import shutil

import numpy as np

import os, sys

from tqdm import tqdm

from multiprocessing import Process, Queue, Manager
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


def reader(v_queue: Queue, v_features, v_point_features, v_flags, v_points, num_block):
    cur_id = 0
    while cur_id < len(v_features):
        if v_queue.qsize() >= num_block * 4:
            time.sleep(1)
            continue
        prefix = pathlib.Path(v_features[cur_id]).name[:-9]
        feat = np.load(v_features[cur_id])
        flag = np.load(v_flags[cur_id])
        points = np.load(v_points[cur_id])
        point_feat = np.load(v_point_features[cur_id])
        if True:
            os.remove(v_features[cur_id])
            os.remove(v_flags[cur_id])
            os.remove(v_points[cur_id])
            os.remove(v_point_features[cur_id])
        v_queue.put((cur_id, prefix, feat, point_feat, flag, points))
        cur_id += 1
    print("reader finished")


def writer(v_queue: Queue, v_num_files, v_filename, num_block):
    output_file = h5py.File(v_filename, "w", )

    output_file.create_dataset("features", shape=(0, 256, 256, 256, 3), dtype=np.uint16,
                               maxshape=(v_num_files, 256, 256, 256, 3),
                               chunks=(1, 32, 32, 32, 3),
                               shuffle=True,
                               compression="lzf"
                               )
    output_file.create_dataset("point_features", shape=(0, 256, 256, 256, 5), dtype=np.uint16,
                               maxshape=(v_num_files, 256, 256, 256, 5),
                               chunks=(1, 32, 32, 32, 5),
                               shuffle=True,
                               compression="lzf"
                               )
    output_file.create_dataset("flags", shape=(0, 256, 256, 256), dtype=np.uint32,
                               maxshape=(v_num_files, 256, 256, 256),
                               chunks=(1, 32, 32, 32),
                               shuffle=True,
                               compression="lzf"
                               )
    output_file.create_dataset("points", shape=(0, 10000, 6), dtype=np.float32,
                               maxshape=(v_num_files, 10000, 6),
                               chunks=(1, 10000, 6),
                               shuffle=True,
                               compression="lzf"
                               )
    output_file.create_dataset("names", shape=(v_num_files,), dtype=int, )

    num_finished = 0
    progress_bar = tqdm(total=v_num_files)
    while True:
        if num_finished + num_block > v_num_files:
            num_block = v_num_files - num_finished

        if v_queue.qsize() < num_block:
            time.sleep(1)

        idx, prefix, feat, point_feat, flag, points = (
            [None] * num_block, [None] * num_block, [None] * num_block,
            [None] * num_block, [None] * num_block, [None] * num_block)
        for i in range(num_block):
            idx[i], prefix[i], feat[i], point_feat[i], flag[i], points[i] = v_queue.get()

        prefix = np.stack(prefix, axis=0).astype(np.int32)
        feat = np.stack(feat, axis=0)
        point_feat = np.stack(point_feat, axis=0)
        flag = np.stack(flag, axis=0)
        points = np.stack(points, axis=0).astype(np.float32)

        target_shape = min(output_file["features"].shape[0] + num_block, v_num_files)
        output_file["features"].resize(target_shape, axis=0)
        output_file["flags"].resize(target_shape, axis=0)
        output_file["points"].resize(target_shape, axis=0)
        output_file["point_features"].resize(target_shape, axis=0)

        cur_time = time.time()
        output_file["features"][num_finished:num_finished + num_block] = feat
        output_file["point_features"][num_finished:num_finished + num_block] = point_feat
        output_file["flags"][num_finished:num_finished + num_block] = flag
        output_file["points"][num_finished:num_finished + num_block] = points
        output_file["names"][num_finished:num_finished + num_block] = prefix
        # print(time.time() - cur_time)

        num_finished += num_block
        assert num_finished <= v_num_files
        if num_finished == v_num_files:
            break
        progress_bar.update(num_block)

    output_file.close()
    return


if __name__ == '__main__':
    # test()
    is_shuffle = False
    root_dir = sys.argv[1]
    prefix = set([item[:-9] for item in os.listdir(root_dir) if item.endswith("_feat.npy")])
    prefix = sorted(list(prefix), key=lambda item: int(item))

    seed(0)
    if is_shuffle:
        shuffle(prefix)

    feature_files = [os.path.join(root_dir, item + "_feat.npy") for item in prefix]
    flag_files = [os.path.join(root_dir, item + "_flag.npy") for item in prefix]
    point_files = [os.path.join(root_dir, item + "_points.npy") for item in prefix]
    point_feat_files = [os.path.join(root_dir, item + "_pfeat.npy") for item in prefix]

    num_block = 100
    if len(sys.argv) == 4:
        num_block = int(sys.argv[3])

    num_files = len(feature_files)

    timer = time.time()
    q = Manager().Queue()
    # q = Queue()
    hdf5_file = sys.argv[2]
    p_reader = Process(target=reader, args=(q, feature_files, point_feat_files, flag_files, point_files, num_block))
    p_writer = Process(target=writer, args=(q, num_files, hdf5_file, num_block))
    p_reader.start()
    p_writer.start()

    p_reader.join()
    p_writer.join()
    print(time.time() - timer)


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
