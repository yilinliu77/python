import os, time
import queue

import ray
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
    # open("output/test.txt", "w").write("".join(consistent_flags.reshape(-1).astype(np.int32).astype(str).tolist()))
    # open("output/test.txt", "w").write("".join(consistent_flags.reshape(-1).astype(np.int32).astype(str).tolist()))
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
    log_mind_stone = 0.1
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

        while float(is_visited.sum()) / (is_visited.size) > log_mind_stone:
            print(int(log_mind_stone * 100), "%")
            log_mind_stone += 0.1

    return clusters


@numba.njit
def process_item(resolution, consistent_flags, v_is_visited):
    is_visited = v_is_visited.copy()
    while True:
        cur_pos = np.random.randint(0, resolution, (3))
        if (cur_pos < 0).any() or (cur_pos >= resolution).any() or \
                is_visited[cur_pos[0], cur_pos[1], cur_pos[2]]:
            continue
        break

    clusters = []

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
    clusters.append(local_clusters)

    return clusters, is_visited


@ray.remote
def process_item_wrapper(resolution, consistent_flags, is_visited):
    return process_item(resolution, consistent_flags, is_visited)


if __name__ == '__main__':
    consistent_flags, query_points = load_data()
    consistent_flags = consistent_flags.astype(bool)

    num_cpus = 12
    ray.init(
        num_cpus=num_cpus
    )

    # dummy
    ray.get(process_item_wrapper.remote(resolution, consistent_flags.copy(), consistent_flags.copy()))

    # Start
    is_visited = consistent_flags.copy()
    clusters = []
    log_mind_stone = 0.1
    while True:
        tasks = []
        for i in range(num_cpus):
            tasks.append(process_item_wrapper.remote(resolution, consistent_flags, is_visited))
        results = ray.get(tasks)
        delete_flags = [False] * len(results)
        # Check duplication
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if np.all(results[i][1] == results[j][1]):
                    delete_flags[j] = True
        for idx, item in enumerate(results):
            if delete_flags[idx]:
                continue
            clusters.append(item[0])
            is_visited = np.logical_or(is_visited, item[1])

        if is_visited.min():
            break

        while float(is_visited.sum()) / (is_visited.size) > log_mind_stone:
            print(int(log_mind_stone * 100), "%")
            log_mind_stone += 0.1
    exit()
