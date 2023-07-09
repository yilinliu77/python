import os, time
import queue
from numba.typed import List

import numba
import numpy as np
import torch

# data_root
training_data = r"G:/Dataset/GSP_v1/00000006_d4fe04f0f5f84b52bd4f10e4/data.npy"
resolution = 64


def load_data():
    data = np.load(training_data, allow_pickle=True).item()
    consistent_flags = np.unpackbits(data["consistent_flags"]).reshape((256, 256, 256))
    consistent_flags = torch.max_pool3d(
        torch.from_numpy(consistent_flags)[None, None, :].to(torch.float32), 4, 4).numpy()[0, 0]

    query_points = np.stack(np.meshgrid(
        np.arange(resolution), np.arange(resolution), np.arange(resolution), indexing="ij"),
        axis=3).reshape(-1, 3)
    query_points = ((query_points / (resolution - 1)) * 2 - 1).astype(np.float32)

    return consistent_flags, query_points

@numba.njit
def add_neighbour(cur_pos, queues):
    queues.append(cur_pos + np.array((-1, -1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, -1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, -1, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 0, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 0, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 0, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((-1, 1, 1), dtype=np.int32))

    queues.append(cur_pos + np.array((0, -1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((0, -1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((0, -1, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((0, 0, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((0, 0, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((0, 1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((0, 1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((0, 1, 1), dtype=np.int32))

    queues.append(cur_pos + np.array((1, -1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((1, -1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((1, -1, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 0, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 0, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 0, 1), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 1, -1), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 1, 0), dtype=np.int32))
    queues.append(cur_pos + np.array((1, 1, 1), dtype=np.int32))

@numba.njit
def process(consistent_flags):
    cur_pos = np.random.randint(0, resolution, (3))
    is_visited = consistent_flags.copy()
    clusters = []
    while not is_visited.min():
        if (cur_pos < 0).any() or (cur_pos >= resolution).any() or \
                is_visited[cur_pos[0], cur_pos[1], cur_pos[2]]:
            cur_pos = np.random.randint(0, resolution, (3))
            continue

        queues = List()
        queues.append(cur_pos)
        local_clusters = []
        while len(queues) > 0:
            cur_pos = queues.pop(0)
            if (cur_pos < 0).any() or (cur_pos >= resolution).any() or \
                    is_visited[cur_pos[0], cur_pos[1], cur_pos[2]]:
                continue
            is_visited[cur_pos[0], cur_pos[1], cur_pos[2]] = True
            if consistent_flags[cur_pos[0], cur_pos[1], cur_pos[2]]:
                continue
            local_clusters.append(cur_pos)

            add_neighbour(cur_pos, queues)
        # local_clusters_ = np.concatenate(local_clusters)
        clusters.append(local_clusters)
        cur_pos = np.random.randint(0, resolution, (3))
    return clusters

if __name__ == '__main__':
    consistent_flags, query_points = load_data()

    clusters = process(consistent_flags.astype(bool))

    exit()
