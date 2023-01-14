import io
import os
from dataclasses import dataclass

import cv2
import igraph
import math
import networkx as nx
from PIL import Image
from igraph import Graph, EdgeSeq
import numpy as np

from matplotlib import pyplot as plt
import plotly.graph_objects as go
from networkx.drawing.nx_pydot import graphviz_layout

from tqdm import tqdm

root_path = r"D:\Projects\LOD\tree"


@dataclass
class Node:
    parent: int
    left_child: int
    right_child: int
    raw_str: str


def read_tree_networkx(v_path):
    print("Start to read file")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 4 == 0

    num_nodes = len(data) // 4

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = []
    for i_tree in tqdm(range(num_nodes)):
        id_left_child = int(data[i_tree * 4 + 1])
        if id_left_child != -1:
            edges.append((i_tree, id_left_child))

        id_right_child = int(data[i_tree * 4 + 2])
        if id_right_child != -1:
            edges.append((i_tree, id_right_child))
        pass
    G.add_edges_from(edges)
    return G


def read_whole_tree(v_path):
    print("Start to read file")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 4 == 0

    num_nodes = len(data) // 4

    edges = []
    attribute = {"cut": []}
    for i_tree in tqdm(range(num_nodes)):
        id_left_child = int(data[i_tree * 4 + 1])
        if id_left_child != -1:
            edges.append((i_tree, id_left_child))

        id_right_child = int(data[i_tree * 4 + 2])
        if id_right_child != -1:
            edges.append((i_tree, id_right_child))
        attribute["cut"].append(float(data[i_tree * 4 + 3]))
        pass

    G = Graph()  # 2 stands for children number
    G.add_vertices(num_nodes, attribute)
    G.add_edges(edges)
    return G


def _viz_tree(v_graph):
    total_levels = max(v_graph.vs, key=lambda item: item["level"])["level"] + 1

    pos_x = [[] for _ in range(total_levels)]
    pos_y = [[] for _ in range(total_levels)]
    labels = [[] for _ in range(total_levels)]
    sizes = [-1 for _ in range(total_levels)]

    for i in range(len(v_graph.vs)):
        cur_level = v_graph.vs[i]["level"]
        pos_x[cur_level].append(v_graph.vs[i]["pos"][0])
        pos_y[cur_level].append(v_graph.vs[i]["pos"][1])
        labels[cur_level].append(v_graph.vs[i]["cut"])
        sizes[cur_level] = v_graph.vs[i]["size"]

    edge_x = [[] for _ in range(total_levels)]
    edge_y = [[] for _ in range(total_levels)]
    edge_width = [1 for _ in range(total_levels)]

    for edge in v_graph.es:
        cur_level = edge["level"]
        edge_x[cur_level].append(edge["pos1"][0])
        edge_x[cur_level].append(edge["pos2"][0])
        edge_x[cur_level].append(None)
        edge_y[cur_level].append(edge["pos1"][1])
        edge_y[cur_level].append(edge["pos2"][1])
        edge_y[cur_level].append(None)
        edge_width[cur_level] = edge["size"]

    # Visualize
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=50000,
        height=5000,
        paper_bgcolor="LightSteelBlue",
    )

    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=edge_x[i],
                                 y=edge_y[i],
                                 mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=edge_width[i]),
                                 hoverinfo='none'
                                 ))
    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=pos_x[i],
                                 y=pos_y[i],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[i],
                                             color='#6175c1',  # '#DB4551',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=1)
                                             ),
                                 text=labels[i],
                                 hoverinfo='text',
                                 opacity=0.8
                                 ))

    fig_bytes = fig.to_image(format="png")
    img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def viz_whole_tree(v_graph, progress_bar):
    print("Start to export coordinate")
    position = np.asarray(v_graph.layout('tree', root=[0]).coords)
    min_y = position[:, 1].min()
    max_y = position[:, 1].max()
    num_vertices = position.shape[0]

    max_node_size = 200
    size_decreased = 30
    # Nodes
    print("Construct nodes")
    total_levels = int(max_y - min_y + 1)

    for i in range(num_vertices):
        cur_level = int(position[i][1])
        assert cur_level < total_levels

        v_graph.vs[i]["level"] = cur_level
        v_graph.vs[i]["pos"] = [position[i][0], 4 * (max_y - position[i][1])]
        v_graph.vs[i]["size"] = max(max_node_size / pow(1.5, cur_level), 5)

        pass

    print("Construct edges")
    max_edge_size = 10
    for i in range(num_vertices):
        neighbours = v_graph.vs[i].neighbors(1)
        cur_level = v_graph.vs[i]["level"]

        for neighbour in neighbours:
            if neighbour["level"] > v_graph.vs[i]["level"]:
                id_edge = v_graph.get_eid(i, neighbour.index)
                v_graph.es[id_edge]["level"] = cur_level
                v_graph.es[id_edge]["pos1"] = v_graph.vs[i]["pos"]
                v_graph.es[id_edge]["pos2"] = v_graph.vs[neighbour.index]["pos"]
                v_graph.es[id_edge]["size"] = max(max_edge_size / pow(2, cur_level), 1)

    return _viz_tree(v_graph)


def read_tree(v_path, v_graph):
    print("Start to read file")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 5 == 0

    num_nodes = len(data) // 5

    G = Graph()  # 2 stands for children number
    for i_tree in tqdm(range(num_nodes)):
        id_current_node = int(data[i_tree * 5 + 4])
        G.add_vertex(v_graph.vs[id_current_node])
        G.vs[id_current_node]["cut"] = float(data[i_tree * 5 + 3])
        pass

    return G


if __name__ == '__main__':
    files = sorted(os.listdir(root_path), key=lambda item: int(item.split("_")[0]))

    whole_tree = read_whole_tree(files[-1])
    whole_img = viz_whole_tree(whole_tree, False)

    cv2.imwrite("output/viz_bsp/whole.png", whole_img)

    # tree = read_tree(files[0], whole_tree)

    cv2.namedWindow("1", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("1", 2200, 900)
    cv2.imshow("1", whole_img)
    cv2.waitKey(0)

    for file in tqdm(files[:-1]):
        tree = read_tree(files[-1])
