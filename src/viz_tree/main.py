import io
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pickle import dumps, loads
from typing import List

import cv2
import igraph
import math
import networkx as nx
from PIL import Image
from igraph import Graph, EdgeSeq, Vertex
import numpy as np

from matplotlib import pyplot as plt
import plotly.graph_objects as go
from networkx.drawing.nx_pydot import graphviz_layout

from tqdm import tqdm
from multiprocessing import Pool

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
        assert len(data) % 5 == 0

    num_nodes = len(data) // 5

    edges = []
    attribute = {"cut": []}
    for i_tree in tqdm(range(num_nodes)):
        id_left_child = int(data[i_tree * 5 + 2])
        if id_left_child != -1:
            edges.append((i_tree, id_left_child))

        id_right_child = int(data[i_tree * 5 + 3])
        if id_right_child != -1:
            edges.append((i_tree, id_right_child))
        attribute["cut"].append(float(data[i_tree * 5 + 4]))
        pass

    G = Graph()  # 2 stands for children number
    G.add_vertices(num_nodes, attribute)
    G.add_edges(edges)
    return G


def _viz_tree(v_graph, v_boundary, v_is_background):
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
        width=5000,
        height=2000,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#CCCCCC",
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False)
    )

    fig.add_trace(go.Scatter(x=v_boundary[0],
                             y=v_boundary[1],
                             mode='markers',
                             marker=dict(symbol='circle-dot',
                                         size=0,
                                         color='rgb(255,0,0)',  # '#DB4551',
                                         ),
                             opacity=0,
                             ))

    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=edge_x[i],
                                 y=edge_y[i],
                                 mode='lines',
                                 line=dict(color='#CCFF99' if not v_is_background else '#000000', width=edge_width[i]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=pos_x[i],
                                 y=pos_y[i],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[i],
                                             color='#FFCC99' if not v_is_background else '#000000',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[i],
                                 hoverinfo='text',
                                 opacity=1
                                 ))

    if not v_is_background:
        # min_y = min(v_graph.vs, key=lambda item: item["pos"][1])["pos"][1]
        current_y = v_graph.vs[v_graph["id_current_node"]]["pos"][1]
        line_x = [v_boundary[0][0], v_boundary[0][1], None]
        line_y = [current_y, current_y, None]
        fig.add_trace(go.Scatter(x=line_x,
                                 y=line_y,
                                 mode='lines',
                                 line=dict(color='#CCFF99', width=3, dash='dot'),
                                 hoverinfo='none',
                                 opacity=1
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

    max_node_size = 50
    size_decreased = 5
    # Nodes
    print("Construct nodes")
    total_levels = int(max_y - min_y + 1)

    for i in range(num_vertices):
        cur_level = int(position[i][1])
        assert cur_level < total_levels

        v_graph.vs[i]["level"] = cur_level
        v_graph.vs[i]["pos"] = [position[i][0], 4 * (max_y - position[i][1])]
        v_graph.vs[i]["size"] = max(max_node_size / pow(1.5, cur_level), 2)
        v_graph.vs[i]["original_index"] = i

        pass

    print("Construct edges")
    max_edge_size = 8
    for i in range(num_vertices):
        neighbours = v_graph.vs[i].neighbors(1)
        cur_level = v_graph.vs[i]["level"]

        for neighbour in neighbours:
            if neighbour["level"] > v_graph.vs[i]["level"]:
                id_edge = v_graph.get_eid(i, neighbour.index)
                v_graph.es[id_edge]["level"] = cur_level
                v_graph.es[id_edge]["pos1"] = v_graph.vs[i]["pos"]
                v_graph.es[id_edge]["pos2"] = v_graph.vs[neighbour.index]["pos"]
                v_graph.es[id_edge]["size"] = max(max_edge_size / pow(1.25, cur_level), 3)

    pos = np.array([item["pos"] for item in v_graph.vs])
    min_point = pos.min(axis=0) - np.array([50, 20])
    max_point = pos.max(axis=0) + np.array([50, 20])
    boundary = [(min_point[0], max_point[0]), (min_point[1], max_point[1])]
    return _viz_tree(v_graph, boundary, True), boundary


def read_tree(v_path, v_graph):
    print("Start to process intermediate nodes")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 2 == 0

    num_states = len(data) // 2

    graphs = [[[0],[]], ] # # (node, id_original_node)
    graph_states = [(0, 2), ] # (id_new_node, id_original_node)
    for i_tree in tqdm(range(num_states)):
        G = deepcopy(graphs[-1])
        id_original_node = int(data[i_tree * 2 + 0])
        graph_states.append(
            (int(i_tree * 2 + 0), int(i_tree * 2 + 1))
        )
        for neighbor in v_graph.vs[id_original_node].neighbors():
            if neighbor["level"] > v_graph.vs[id_original_node]["level"]:
                # new_vertex = G.add_vertex(1)
                G[0].append(neighbor.index)
                G[1].append((id_original_node,neighbor.index))
        graphs.append(G)

    # root_graph = Graph()
    # root_vertex = root_graph.add_vertex(1)
    # root_vertex.update_attributes(v_graph.vs[0].attributes())
    # root_graph["id_current_node"] = 0
    # graphs: List[Graph] = [root_graph]
    # graph_states = [(0, 2), ]
    #
    # acc_leaf_nodes = [(0, 0), ]
    # for i_tree in tqdm(range(num_states)):
    #     # G = deepcopy(graphs[-1])
    #     G = graphs[-1].copy()
    #     id_original_node = int(data[i_tree * 2 + 0])
    #     graph_states.append(
    #         (int(i_tree * 2 + 0), int(i_tree * 2 + 1))
    #     )
    #
    #     id_current_node = -1
    #     for i in range(len(acc_leaf_nodes)):
    #         if acc_leaf_nodes[i][1] == id_original_node:
    #             id_current_node = acc_leaf_nodes[i][0]
    #             acc_leaf_nodes.pop(i)
    #             break
    #     assert id_current_node != -1
    #     for neighbor in v_graph.vs[id_original_node].neighbors():
    #         if neighbor["level"] > v_graph.vs[id_original_node]["level"]:
    #             new_vertex = G.add_vertex(1)
    #             new_vertex.update_attributes(neighbor.attributes())
    #             new_edge = G.add_edge(id_current_node, new_vertex.index)
    #             new_edge.update_attributes(v_graph.es[v_graph.get_eid(id_original_node, neighbor.index)].attributes())
    #             acc_leaf_nodes.append((new_vertex.index, neighbor.index))
    #             G["id_current_node"] = new_vertex.index
    #     graphs.append(G)
    #     pass

    with open(os.path.join(root_path, "intermediate1.bin"), "wb") as f:
        f.write(dumps(graphs))
    with open(os.path.join(root_path, "intermediate2.bin"), "wb") as f:
        f.write(dumps(graph_states))

    return graphs, graph_states


def write_video():
    accelerate = 1
    with open(r"D:\Projects\LOD\time_index.txt") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 2 == 0
    num_viz = len(data) // 2

    file_list = [item for item in os.listdir("output/viz_bsp/inter_imgs") if item.split(".")[0].isnumeric()]
    file_list = sorted(file_list, key=lambda item: int(item.split(".")[0]))
    whole_img = cv2.imread("output/viz_bsp/whole.png")
    size = (whole_img.shape[1], whole_img.shape[0])
    out = cv2.VideoWriter('output/viz_bsp/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    # for i in tqdm(range(len(file_list))):
    #     img = cv2.imread(os.path.join("output/viz_bsp/", file_list[i]))
    #     out.write(img)
    last_time = 0
    for i in tqdm(range(num_viz)):
        id_cut = int(data[i * 2 + 1])
        time = float(data[i * 2 + 0])
        img = cv2.imread(os.path.join("output/viz_bsp/inter_imgs", file_list[id_cut]))

        for _ in range(max(int((time - last_time) * 15 // accelerate), 1)):
            out.write(img)
        last_time = time
    out.release()


def viz_item(v_args, v_graph:Graph, v_whole_img, v_boundary):
    idx, tree = v_args
    tree,tree_states = tree
    vertices = tree[0]
    edges = tree[1]

    total_levels = max(v_graph.vs, key=lambda item: item["level"])["level"] + 1

    pos_x = [[] for _ in range(total_levels)]
    pos_y = [[] for _ in range(total_levels)]
    labels = [[] for _ in range(total_levels)]
    sizes = [-1 for _ in range(total_levels)]

    for i in range(len(tree[0])):
        cur_level = v_graph.vs[vertices[i]]["level"]
        pos_x[cur_level].append(v_graph.vs[vertices[i]]["pos"][0])
        pos_y[cur_level].append(v_graph.vs[vertices[i]]["pos"][1])
        labels[cur_level].append(v_graph.vs[vertices[i]]["cut"])
        sizes[cur_level] = v_graph.vs[vertices[i]]["size"]

    edge_x = [[] for _ in range(total_levels)]
    edge_y = [[] for _ in range(total_levels)]
    edge_width = [1 for _ in range(total_levels)]

    for edge in edges:
        id_edge = v_graph.get_eid(edge[0], edge[1])
        cur_level = v_graph.es[id_edge]["level"]
        edge_x[cur_level].append(v_graph.es[id_edge]["pos1"][0])
        edge_x[cur_level].append(v_graph.es[id_edge]["pos2"][0])
        edge_x[cur_level].append(None)
        edge_y[cur_level].append(v_graph.es[id_edge]["pos1"][1])
        edge_y[cur_level].append(v_graph.es[id_edge]["pos2"][1])
        edge_y[cur_level].append(None)
        edge_width[cur_level] = v_graph.es[id_edge]["size"]

    # Visualize
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=5000,
        height=2000,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#CCCCCC",
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False)
    )

    fig.add_trace(go.Scatter(x=v_boundary[0],
                             y=v_boundary[1],
                             mode='markers',
                             marker=dict(symbol='circle-dot',
                                         size=0,
                                         color='rgb(255,0,0)',  # '#DB4551',
                                         ),
                             opacity=0,
                             ))

    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=edge_x[i],
                                 y=edge_y[i],
                                 mode='lines',
                                 line=dict(color='#CCFF99', width=edge_width[i]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
    for i in range(total_levels):
        fig.add_trace(go.Scatter(x=pos_x[i],
                                 y=pos_y[i],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[i],
                                             color='#FFCC99',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[i],
                                 hoverinfo='text',
                                 opacity=1
                                 ))

    # min_y = min(v_graph.vs, key=lambda item: item["pos"][1])["pos"][1]
    current_y = v_graph.vs[tree_states[0]]["pos"][1]
    line_x = [v_boundary[0][0], v_boundary[0][1], None]
    line_y = [current_y, current_y, None]
    fig.add_trace(go.Scatter(x=line_x,
                             y=line_y,
                             mode='lines',
                             line=dict(color='#CCFF99', width=10, dash='dot'),
                             hoverinfo='none',
                             opacity=1
                             ))

    fig_bytes = fig.to_image(format="png")
    item_img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.addWeighted(item_img[:, :, :3], 0.8, v_whole_img[:, :, :3], 0.2, 0)
    cv2.imwrite(os.path.join(root_path, "inter_imgs/{}.png".format(idx)), img)


def parent(v_vertex: Vertex):
    neighbours = v_vertex.neighbors()
    for neighbour in neighbours:
        if neighbour["level"] < v_vertex["level"]:
            return neighbour
    raise


def children(v_vertex: Vertex):
    neighbours = v_vertex.neighbors()
    result = []
    for neighbour in neighbours:
        if neighbour["level"] > v_vertex["level"]:
            result.append(neighbour)
    return result


def all_children(v_vertex: Vertex):
    result = children(v_vertex)
    if len(result) == 0:
        return []
    result += all_children(result[0])
    result += all_children(result[1])
    return result


def generate_specific_cut(v_id_cut, v_tree: Graph, v_whole_tree: Graph, v_viewpoint_size=200):
    edges = np.asarray([item.tuple for item in v_tree.es])
    filtered_nodes = np.unique(edges[:, 0])
    total_nodes = np.arange(len(v_tree.vs))
    id_leaf_nodes = np.setxor1d(filtered_nodes, total_nodes)
    sorted_result = sorted(id_leaf_nodes, key=lambda item: v_tree.vs[item]["cut"], reverse=True)

    def _viz_tree_local(v_graph, v_index):
        min_levels = min(v_graph.vs, key=lambda item: item["level"])["level"]
        max_levels = max(v_graph.vs, key=lambda item: item["level"])["level"]

        pos_x = [[], [], []]
        pos_y = [[], [], []]
        labels = [[], [], []]
        sizes = [[], [], []]
        annotations = []

        for i in range(len(v_graph.vs)):
            cur_level = v_graph.vs[i]["level"]
            if v_graph.vs[i]["original_index"] == v_index:
                cur_level = 2
            elif cur_level > min_levels + 1:
                cur_level = 1
            else:
                cur_level = 0
            pos_x[cur_level].append(v_graph.vs[i]["pos"][0])
            pos_y[cur_level].append(v_graph.vs[i]["pos"][1])
            labels[cur_level].append(v_graph.vs[i]["cut"])
            # sizes[cur_level] = v_graph.vs[i]["size"]
            sizes[cur_level] = 40

            annotations.append(dict(
                text="{:.1f}".format(v_graph.vs[i]["cut"]),  # or replace labels with a different list for the text within the circle
                x=v_graph.vs[i]["pos"][0], y=v_graph.vs[i]["pos"][1],
                xref='x1', yref='y1',
                font=dict(color='rgb(250,250,250)', size=10),
                showarrow=False))

        edge_x = [[],[]]
        edge_y = [[],[]]
        edge_width = [[],[]]

        for edge in v_graph.es:
            cur_level = edge["level"]
            if cur_level > min_levels + 1:
                cur_level = 1
            else:
                cur_level = 0
            edge_x[cur_level].append(edge["pos1"][0])
            edge_x[cur_level].append(edge["pos2"][0])
            edge_x[cur_level].append(None)
            edge_y[cur_level].append(edge["pos1"][1])
            edge_y[cur_level].append(edge["pos2"][1])
            edge_y[cur_level].append(None)
            # edge_width[cur_level] = edge["size"]
            edge_width[cur_level] = 5

        # Visualize
        fig = go.Figure()
        fig.update_layout(
            annotations=annotations,
            autosize=False,
            width=1500,
            height=1000,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#CCCCCC",
            xaxis=dict(showgrid=False, visible=False, showticklabels=False),
            yaxis=dict(showgrid=False, visible=False, showticklabels=False)
        )

        fig.add_trace(go.Scatter(x=edge_x[0],
                                 y=edge_y[0],
                                 mode='lines',
                                 line=dict(color='#CCFF99',
                                           width=edge_width[0]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
        fig.add_trace(go.Scatter(x=edge_x[1],
                                 y=edge_y[1],
                                 mode='lines',
                                 line=dict(color='#CCFF99',
                                           width=edge_width[1]),
                                 hoverinfo='none',
                                 opacity=0.4
                                 ))
        fig.add_trace(go.Scatter(x=pos_x[0],
                                 y=pos_y[0],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[0],
                                             color='#FFCC99',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[0],
                                 hoverinfo='text',
                                 opacity=1
                                 ))
        fig.add_trace(go.Scatter(x=pos_x[1],
                                 y=pos_y[1],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[1],
                                             color='#FFCC99',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[1],
                                 hoverinfo='text',
                                 opacity=0.4
                                 ))
        fig.add_trace(go.Scatter(x=pos_x[2],
                                 y=pos_y[2],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[1],
                                             color='#FF0000',
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[2],
                                 hoverinfo='text',
                                 opacity=1
                                 ))
        fig_bytes = fig.to_image(format="png")
        img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        return img

    for i in range(5):
        current_leaf_node = v_tree.vs[sorted_result[i]]
        original_leaf_node = v_whole_tree.vs[current_leaf_node["original_index"]]

        sub_graph_root = parent(original_leaf_node)
        sub_graph = [sub_graph_root] + all_children(sub_graph_root)
        sub_graph = [item for item in sub_graph if abs(item["level"] - original_leaf_node["level"]) < 4]
        sub_graph = v_whole_tree.subgraph(sub_graph)
        img = _viz_tree_local(sub_graph, original_leaf_node["original_index"])
        cv2.imwrite("output/viz_bsp/local_{}.png".format(i),img)
        pass
    pass


if __name__ == '__main__':
    is_visualize_whole_image = False
    is_generate_subtree = False
    is_generate_sub_imgs = True
    is_generate_video = True
    is_generate_specific_cut = True
    id_specific_cut = 100

    pool = Pool(1)
    whole_tree = read_whole_tree(os.path.join(root_path, "tree.txt"))
    whole_img, boundary = viz_whole_tree(whole_tree, False)
    cv2.imwrite(os.path.join(root_path, "whole.txt"), whole_img)

    if is_visualize_whole_image:
        cv2.namedWindow("1", cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("1", 2200, 900)
        cv2.imshow("1", whole_img)
        cv2.waitKey(0)

    if is_generate_subtree:
        tree, tree_states = read_tree(os.path.join(root_path, "order_state.txt"), whole_tree)
    else:
        with open(os.path.join(root_path, "intermediate1.bin"), "rb") as f:
            line = f.read()
            tree = loads(line)
        with open(os.path.join(root_path, "intermediate2.bin"), "rb") as f:
            line = f.read()
            tree_states = loads(line)

    if is_generate_sub_imgs:
        pool.map(partial(viz_item, v_graph = whole_tree, v_whole_img=whole_img, v_boundary=boundary),
                 list(enumerate(zip(tree, tree_states)))[20000:20003])

    # if is_generate_video:
    #     write_video()

    # if is_generate_specific_cut:
    #     generate_specific_cut(id_specific_cut, tree[id_specific_cut], whole_tree)

    pool.close()


