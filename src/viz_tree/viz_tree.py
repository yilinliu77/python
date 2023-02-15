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
from pwlf import pwlf
from scipy.interpolate import KroghInterpolator

from tqdm import tqdm
from multiprocessing import Pool

import scipy.interpolate as interpolate

# root_path = r"D:\Projects\LOD\test_scene\l7_sliding"
# root_path = r"D:\Projects\LOD\test_scene\teaser"
# root_path = r"D:\Projects\LOD\test_scene\church"
root_path = r"D:\Projects\LOD\test_scene\l7_final"

# Teaser
# bg_color = "#FFFFFF"
# mark_color = "#FF9900"
# edge_color = "#699BFF"
# window_width = int(3840)
# window_height = int(540)
# mark_size = 30
# edge_size = 6
# notation_size = 9
# mark_size_decreased = 1.4
# edge_size_decreased = 1.1
# mark_min_size = 3


# Demo
bg_color = "#333333"
mark_color = "#FFFF00"
edge_color = "#244FE1"
window_width = int((2160 - 300))
window_height = int(3840 / 5 * 2)
mark_size = 30
edge_size = 6
notation_size = 9
mark_size_decreased = 1.3
edge_size_decreased = 1.1
mark_min_size = 1


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
        line = f.readline()
        data = line.strip().split(" ")
        assert len(data) % 4 == 0

    num_nodes = len(data) // 4

    edges = []
    id_node_remap = {}
    attribute = {"cut": []}
    for i_tree in tqdm(range(num_nodes)):
        id_current = int(data[i_tree * 4 + 0])
        id_left_child = int(data[i_tree * 4 + 2])
        id_right_child = int(data[i_tree * 4 + 3])

        if id_current not in id_node_remap:
            id_node_remap[id_current] = len(id_node_remap)
        id_current = id_node_remap[id_current]

        if id_left_child != -1:
            if id_left_child not in id_node_remap:
                id_node_remap[id_left_child] = len(id_node_remap)
            id_left_child = id_node_remap[id_left_child]
            edges.append((i_tree, id_left_child))

        if id_right_child != -1:
            if id_right_child not in id_node_remap:
                id_node_remap[id_right_child] = len(id_node_remap)
            id_right_child = id_node_remap[id_right_child]
            edges.append((i_tree, id_right_child))
        attribute["cut"].append(0.)
        pass

    G = Graph()  # 2 stands for children number
    G.add_vertices(num_nodes, attribute)
    G.add_edges(edges)
    return G


def read_whole_non_bsp_tree(v_path):
    print("Start to read file")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = [item.strip().split(" ") for item in f.readlines() if len(item) > 2]

    num_nodes = len(data)

    edges = []
    attribute = {"cut": []}
    for id_node, line in tqdm(enumerate(data)):
        num_child = len(line) - 3
        for i in range(num_child):
            id_child = int(line[3 + i])
            if id_child != -1:
                edges.append((id_node, id_child))

        attribute["cut"].append(float(line[2]))
        pass

    G = Graph()  # 2 stands for children number
    G.add_vertices(num_nodes, attribute)
    G.add_edges(edges)
    return G


def _viz_tree(v_indices, v_states, v_graph, v_boundary, v_is_background, v_is_fixed=False, v_curve_color = "rgb(255,255,0)"):
    total_levels = max(v_graph.vs, key=lambda item: item["level"])["level"] + 1

    vertices = v_indices[0]
    edges = v_indices[1]

    pos_x = [[] for _ in range(total_levels)]
    pos_y = [[] for _ in range(total_levels)]
    labels = [[] for _ in range(total_levels)]
    sizes = [1 for _ in range(total_levels)]

    for i in range(len(vertices)):
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
        width=window_width,
        height=window_height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor=bg_color,
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    if v_is_fixed:
        fig.update_layout(
            xaxis_range=[v_boundary[0][0] - 20, v_boundary[0][1]],
            yaxis_range=[v_boundary[1][0], v_boundary[1][1]]
        )
        # fig.add_trace(go.Scatter(x=v_boundary[0],
        #                          y=v_boundary[1],
        #                          mode='markers',
        #                          marker=dict(symbol='circle-dot',
        #                                      size=0,
        #                                      color='rgb(255,0,0)',  # '#DB4551',
        #                                      ),
        #                          opacity=0,
        #                          ))

    for i in range(total_levels):
        if len(edge_x[i]) == 0:
            continue
        fig.add_trace(go.Scatter(x=edge_x[i],
                                 y=edge_y[i],
                                 mode='lines',
                                 line=dict(color=edge_color, width=edge_width[i]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
    for i in range(total_levels):
        if len(pos_x[i]) == 0:
            continue
        fig.add_trace(go.Scatter(x=pos_x[i],
                                 y=pos_y[i],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[i],
                                             # color=mark_color if not i > 10 else edge_color,
                                             color=mark_color,
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[i],
                                 hoverinfo='text',
                                 opacity=1
                                 ))

    fig.update_traces(
        marker=dict(symbol="square"),
        selector=dict(mode="markers"),
    )

    if not v_is_background and False:
        # min_y = min(v_graph.vs, key=lambda item: item["pos"][1])["pos"][1]
        if v_graph.vs[v_states[0]]["level"] == 0:
            current_y = v_graph.vs[v_states[0]]["pos"][1]
        else:
            item_children = children(v_graph.vs[v_states[0]])
            if len(item_children) != 0:
                current_y = item_children[0]["pos"][1]
            else:
                current_y = v_graph.vs[v_states[0]]["pos"][1]
        min_y = min(v_graph.vs, key=lambda item: item["pos"][1])["pos"][1]
        max_y = max(v_graph.vs, key=lambda item: item["pos"][1])["pos"][1]

        # First horizontal line
        line_x = [v_boundary[0][0] - 50, v_boundary[0][0] - 50, None]
        line_y = [min_y, current_y, None]
        fig.add_trace(go.Scatter(x=line_x,
                                 y=line_y,
                                 mode='lines',
                                 line=dict(color="#474747", width=notation_size),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
        # Second horizontal line
        line_x = [v_boundary[0][0] - 50, v_boundary[0][0] - 50, None]
        line_y = [current_y, max_y, None]
        fig.add_trace(go.Scatter(x=line_x,
                                 y=line_y,
                                 mode='lines',
                                 line=dict(color="#7E7E7E", width=notation_size),
                                 hoverinfo='none',
                                 opacity=1
                                 ))

        # Vertical line
        line_x = [v_boundary[0][0] - 50, v_boundary[0][1] - 50, None]
        line_y = [current_y, current_y, None]
        fig.add_trace(go.Scatter(x=line_x,
                                 y=line_y,
                                 mode='lines',
                                 line=dict(color=mark_color, width=notation_size, dash="dot"),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
        fig.add_trace(go.Scatter(x=[v_boundary[0][0] - 50],
                                 y=[current_y],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=mark_size,
                                             color=mark_color,
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=None,
                                 hoverinfo=None,
                                 opacity=1
                                 ))

    if not v_is_background and len(edges) > 2:
        edges = np.array(edges)
        filtered_nodes = np.unique(edges[:, 0])
        total_nodes = np.asarray(vertices)
        id_leaf_nodes = np.setxor1d(filtered_nodes, total_nodes)
        sorted_result = sorted(id_leaf_nodes, key=lambda item: v_graph.vs[item]["pos"][0], reverse=False)

        if True:
            line_x = []
            line_y = []
            for idx, item in enumerate(sorted_result):
                if len(line_x) > 1 and v_graph.vs[item]["pos"][0] == line_x[-1]:
                    continue
                line_x.append(v_graph.vs[item]["pos"][0])
                line_y.append(v_graph.vs[item]["pos"][1])

            z1 = np.polyfit(line_x, line_y, 3)
            p1 = np.poly1d(z1)

            min_x = min(line_x) - 40
            max_x = max(line_x) + 40
            xx = np.linspace(min_x, max_x, 1000)
            yy = p1(xx)
            curve_x = []
            curve_y = []

            for i in range(xx.shape[0] - 1):
                curve_x.append(xx[i])
                curve_x.append(xx[i + 1])
                curve_x.append(None)
                curve_y.append(yy[i])
                curve_y.append(yy[i + 1])
                curve_y.append(None)

            fig.add_trace(go.Scatter(x=curve_x,
                                     y=curve_y,
                                     mode='lines',
                                     line=dict(color=v_curve_color, width=notation_size),
                                     opacity=1
                                     ))
        else:
            line_x = []
            line_y = []
            for idx, item in enumerate(sorted_result[:-1]):
                line_x.append(v_graph.vs[item]["pos"][0])
                line_x.append(v_graph.vs[sorted_result[idx + 1]]["pos"][0])
                line_x.append(None)
                line_y.append(v_graph.vs[item]["pos"][1])
                line_y.append(v_graph.vs[sorted_result[idx + 1]]["pos"][1])
                line_y.append(None)

            fig.add_trace(go.Scatter(x=line_x,
                                     y=line_y,
                                     mode='lines',
                                     line=dict(color=mark_color, width=notation_size),
                                     opacity=1
                                     ))

    # fig_bytes = fig.to_image(format="png")
    # img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return fig


def viz_whole_tree(v_graph, v_is_fix_pos=False):
    print("Start to export coordinate")
    position = np.asarray(v_graph.layout_reingold_tilford(mode="in", root=[0]).coords)
    min_y = position[:, 1].min()
    max_y = position[:, 1].max()
    num_vertices = position.shape[0]

    max_node_size = mark_size
    # Nodes
    print("Construct nodes")
    total_levels = int(max_y - min_y + 1)

    for i in range(num_vertices):
        cur_level = int(position[i][1])
        assert cur_level < total_levels

        v_graph.vs[i]["level"] = cur_level
        v_graph.vs[i]["pos"] = [position[i][0], (max_y - position[i][1]) * 2]
        v_graph.vs[i]["size"] = max(max_node_size / pow(mark_size_decreased, cur_level), mark_min_size)
        v_graph.vs[i]["original_index"] = i

        pass

    print("Construct edges")
    max_edge_size = edge_size
    for i in range(num_vertices):
        neighbours = v_graph.vs[i].neighbors(1)
        cur_level = v_graph.vs[i]["level"]

        for neighbour in neighbours:
            if neighbour["level"] > v_graph.vs[i]["level"]:
                id_edge = v_graph.get_eid(i, neighbour.index)
                v_graph.es[id_edge]["level"] = cur_level
                v_graph.es[id_edge]["pos1"] = v_graph.vs[i]["pos"]
                v_graph.es[id_edge]["pos2"] = v_graph.vs[neighbour.index]["pos"]
                v_graph.es[id_edge]["size"] = max(max_edge_size / pow(edge_size_decreased, cur_level), 1)

    pos = np.array([item["pos"] for item in v_graph.vs])
    min_point = pos.min(axis=0) - np.array([50, 3])
    max_point = pos.max(axis=0) + np.array([50, 3])
    boundary = [(min_point[0], max_point[0]), (min_point[1], max_point[1])]

    graph = [[], []]
    graph[0] = [item.index for item in v_graph.vs]
    graph[1] = [(item.source, item.target) for item in v_graph.es]

    return _viz_tree(graph, (0, 2), v_graph, boundary, v_is_background=True, v_is_fixed=v_is_fix_pos), boundary


def read_tree(v_path, v_graph):
    print("Start to process intermediate nodes")
    with open(os.path.join(root_path, v_path), "r") as f:
        data = f.readline().strip().split(" ")
        assert len(data) % 2 == 0

    num_states = len(data) // 2

    graphs = [[[0], []], ]  # # (node, id_original_node)
    graph_states = [(0, 3), ]  # (id_new_node, id_original_node)
    for i_tree in tqdm(range(num_states)):
        G = deepcopy(graphs[-1])
        id_original_node = int(data[i_tree * 2 + 0])
        graph_states.append(
            (int(data[i_tree * 2 + 0]), int(data[i_tree * 2 + 1]))
        )
        for neighbor in v_graph.vs[id_original_node].neighbors():
            if neighbor["level"] > v_graph.vs[id_original_node]["level"]:
                G[0].append(neighbor.index)
                G[1].append((id_original_node, neighbor.index))
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

    return graphs, graph_states


def write_video():
    accelerate = 1
    # with open(r"D:\Projects\LOD\time_index.txt") as f:
    #     data = f.readline().strip().split(" ")
    #     assert len(data) % 2 == 0
    # num_viz = len(data) // 2

    file_list = [item for item in os.listdir(os.path.join(root_path, "inter_imgs")) if item.split(".")[0].isnumeric()]
    file_list = sorted(file_list, key=lambda item: int(item.split(".")[0]))
    whole_img = cv2.imread(os.path.join(root_path, "whole.png"))
    size = (whole_img.shape[0], whole_img.shape[1])
    out = cv2.VideoWriter(os.path.join(root_path, "project.mp4"), cv2.VideoWriter_fourcc(*'avc1'), 15, size)

    for i in tqdm(range(len(file_list))):
        img = cv2.imread(os.path.join(root_path, "inter_imgs", file_list[i]))
        out.write(img)
    # last_time = 0
    # for i in tqdm(range(num_viz)):
    #     id_cut = int(data[i * 2 + 1])
    #     time = float(data[i * 2 + 0])
    #     img = cv2.imread(os.path.join("output/viz_bsp/inter_imgs", file_list[id_cut]))
    #
    #     for _ in range(max(int((time - last_time) * 15 // accelerate), 1)):
    #         out.write(img)
    #     last_time = time
    out.release()


def viz_item(v_args, v_graph: Graph, v_whole_img, v_boundary, v_is_fix_pos, v_states_range):
    idx, tree = v_args
    tree, tree_states = tree
    # if tree_states[1] != 1:
    #     return
    id_state = 0
    for item in v_states_range:
        if item > idx:
            break
        id_state+=1
    # curve_color = ["rgb(255,153,0)", "rgb(61,157,230)", "rgb(21,236,87)", "rgb(199,148,234)", "rgb(229,53,43)"]
    # id_state=min(id_state, len(curve_color) - 1)
    img = _viz_tree(tree, tree_states, v_graph, v_boundary, v_is_background=False,
                    v_is_fixed=v_is_fix_pos, v_curve_color = mark_color)
    # if v_is_fix_pos:
    #     img = cv2.addWeighted(img[:, :, :3], 0.8, v_whole_img[:, :, :3], 0.2, 0)

    is_render_pdf = False
    if is_render_pdf:
        img.write_image(os.path.join(root_path, "inter_imgs/{}.pdf".format(idx)))
    else:
        fig_bytes = img.to_image(format="png")
        img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(root_path, "inter_imgs/{}.png".format(idx)), img)


def visualize_whole_image_details(v_tree, v_tree_state, v_whole_tree):
    states = np.array(v_tree_state)
    results = np.where(states[:, 1] == 1)
    id_cut1 = results[0][-2]
    id_cut2 = results[0][-1]
    id_target_vertex = states[id_cut1 + 2][0]

    def _viz_tree_local(v_graph, v_index):
        min_levels = min(v_graph.vs, key=lambda item: item["level"])["level"]
        max_levels = max(v_graph.vs, key=lambda item: item["level"])["level"]

        pos_x = [[], [], []]
        pos_y = [[], [], []]
        labels = [[], [], []]
        sizes = [200, 200, 200]
        annotations = []

        for i in range(len(v_graph.vs)):
            cur_level = v_graph.vs[i]["level"]
            if v_graph.vs[i]["original_index"] == v_index:
                cur_level = 2
            elif cur_level >= min_levels + 1:
                cur_level = 1
            else:
                cur_level = 0
            pos_x[cur_level].append(v_graph.vs[i]["pos"][0])
            pos_y[cur_level].append(v_graph.vs[i]["pos"][1])
            labels[cur_level].append(v_graph.vs[i]["cut"])
            # sizes[cur_level] = v_graph.vs[i]["size"]

        edge_x = [[], []]
        edge_y = [[], []]
        edge_width = [20, 20]

        for edge in v_graph.es:
            cur_level = edge["level"]
            if cur_level >= min_levels + 1:
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

        # Visualize
        fig = go.Figure()
        fig.update_layout(
            autosize=False,
            width=2500,
            height=2500,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor=bg_color,
            xaxis=dict(showgrid=False, visible=False, showticklabels=False),
            yaxis=dict(showgrid=False, visible=False, showticklabels=False),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        fig.add_trace(go.Scatter(x=edge_x[0],
                                 y=edge_y[0],
                                 mode='lines',
                                 line=dict(color="#000000",
                                           width=edge_width[0]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
        if len(edge_x[1]) != 0:
            fig.add_trace(go.Scatter(x=edge_x[1],
                                     y=edge_y[1],
                                     mode='lines',
                                     line=dict(color="#000000",
                                               width=edge_width[1]),
                                     hoverinfo='none',
                                     opacity=1
                                     ))
        fig.add_trace(go.Scatter(x=pos_x[0],
                                 y=pos_y[0],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[0],
                                             color="#000000",
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
                                             color="#000000",
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[1],
                                 hoverinfo='text',
                                 opacity=1
                                 ))
        fig.add_trace(go.Scatter(x=pos_x[2],
                                 y=pos_y[2],
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=sizes[1],
                                             color="#FF6666",
                                             line=dict(
                                                 color='rgb(50,50,50)',
                                                 width=0)
                                             ),
                                 text=labels[2],
                                 hoverinfo='text',
                                 opacity=1
                                 ))
        fig.update_traces(
            marker=dict(symbol="square"),
            selector=dict(mode="markers"),
        )

        return fig

    children_nodes1 = all_children_except(parent(parent(v_whole_tree.vs[id_target_vertex])), v_tree[id_cut1][0])
    children_nodes2 = all_children_except(parent(parent(v_whole_tree.vs[id_target_vertex])), v_tree[id_cut2][0])

    sub1 = v_whole_tree.subgraph(children_nodes1)
    sub2 = v_whole_tree.subgraph(children_nodes2)

    img_total = _viz_tree_local(v_whole_tree, id_target_vertex)
    img1 = _viz_tree_local(sub1, id_target_vertex)
    img1.write_image(os.path.join(root_path, "details/subtree_1.pdf"))
    img1 = img1.to_image(format="png")
    img1 = cv2.imdecode(np.frombuffer(img1, np.uint8), cv2.IMREAD_UNCHANGED)
    img2 = _viz_tree_local(sub2, id_target_vertex)
    img2.write_image(os.path.join(root_path, "details/subtree_2.pdf"))
    img2 = img2.to_image(format="png")
    img2 = cv2.imdecode(np.frombuffer(img2, np.uint8), cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(os.path.join(root_path, "whole_indicate.png"), img_total)
    cv2.imwrite(os.path.join(root_path, "details/subtree_1.png"), img1)
    cv2.imwrite(os.path.join(root_path, "details/subtree_2.png"), img2)

    pass


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


def children_except(v_vertex: Vertex, v_indexes):
    neighbours = v_vertex.neighbors()
    result = []
    for neighbour in neighbours:
        if neighbour["level"] > v_vertex["level"] and neighbour.index in v_indexes:
            result.append(neighbour)
    return result


def all_children_except(v_vertex: Vertex, v_indexes):
    if v_vertex.index not in v_indexes:
        return []
    children_nodes = children_except(v_vertex, v_indexes)
    if len(children_nodes) == 0:
        return []
    results = []
    for item in children_nodes:
        results += all_children_except(item, v_indexes)
    results += children_nodes
    results.append(v_vertex)
    return results


def generate_small_tree(v_tree_path):
    local_bsp_tree = read_whole_tree(v_tree_path)
    print("Start to export coordinate")
    position = np.asarray(local_bsp_tree.layout_reingold_tilford(mode="in", root=[0]).coords)

    pos_x = []
    pos_y = []
    sizes = 50

    for i in range(position.shape[0]):
        pos_x.append(position[i, 0])
        pos_y.append(-position[i, 1])

    edge_x = []
    edge_y = []
    edge_width = 20

    for edge in local_bsp_tree.es:
        edge_x.append(position[edge.source][0])
        edge_x.append(position[edge.target][0])
        edge_x.append(None)
        edge_y.append(-position[edge.source][1])
        edge_y.append(-position[edge.target][1])
        edge_y.append(None)

    # Visualize
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=2500,
        height=2500,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor=bg_color,
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.add_trace(go.Scatter(x=edge_x,
                             y=edge_y,
                             mode='lines',
                             line=dict(color=edge_color, width=edge_width),
                             hoverinfo='none',
                             opacity=1
                             ))
    fig.add_trace(go.Scatter(x=pos_x,
                             y=pos_y,
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=sizes,
                                         color=mark_color,
                                         line=dict(
                                             color='rgb(50,50,50)',
                                             width=0)
                                         ),
                             opacity=1
                             ))

    fig.update_traces(
        marker=dict(symbol="square"),
        selector=dict(mode="markers"),
    )

    fig_bytes = fig.to_image(format="png")
    img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def generate_specific_cut(v_tree: Graph, v_id_node, v_whole_tree: Graph, v_viewpoint_size=200):
    edges = np.asarray(v_tree[1])
    filtered_nodes = np.unique(edges[:, 0])
    total_nodes = np.asarray(v_tree[0])
    id_leaf_nodes = np.setxor1d(filtered_nodes, total_nodes)
    sorted_result = sorted(id_leaf_nodes, key=lambda item: v_whole_tree.vs[item]["cut"], reverse=True)

    def _viz_tree_local(v_graph, v_index):
        min_levels = min(v_graph.vs, key=lambda item: item["level"])["level"]
        max_levels = max(v_graph.vs, key=lambda item: item["level"])["level"]

        pos_x = [[], [], []]
        pos_y = [[], [], []]
        labels = [[], [], []]
        sizes = [200, 200, 200]
        annotations = []

        for i in range(len(v_graph.vs)):
            cur_level = v_graph.vs[i]["level"]
            if v_graph.vs[i]["original_index"] == v_index:
                cur_level = 2
            elif cur_level >= min_levels + 1:
                cur_level = 1
            else:
                cur_level = 0
            pos_x[cur_level].append(v_graph.vs[i]["pos"][0])
            pos_y[cur_level].append(v_graph.vs[i]["pos"][1])
            labels[cur_level].append(v_graph.vs[i]["cut"])
            # sizes[cur_level] = v_graph.vs[i]["size"]

            annotations.append(dict(
                text="{:.1f}".format(v_graph.vs[i]["cut"]),
                # or replace labels with a different list for the text within the circle
                x=v_graph.vs[i]["pos"][0], y=v_graph.vs[i]["pos"][1],
                xref='x1', yref='y1',
                font=dict(
                    color='rgb(250,250,250)' if cur_level != 0 else 'rgb(0,0,0)',
                    size=75
                ),
                showarrow=False))

        edge_x = [[], []]
        edge_y = [[], []]
        edge_width = [20, 20]

        for edge in v_graph.es:
            cur_level = edge["level"]
            if cur_level >= min_levels + 1:
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

        # Visualize
        fig = go.Figure()
        fig.update_layout(
            annotations=annotations,
            autosize=False,
            width=2500,
            height=2500,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor=bg_color,
            xaxis=dict(showgrid=False, visible=False, showticklabels=False),
            yaxis=dict(showgrid=False, visible=False, showticklabels=False),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        fig.add_trace(go.Scatter(x=edge_x[0],
                                 y=edge_y[0],
                                 mode='lines',
                                 line=dict(color=edge_color,
                                           width=edge_width[0]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
        if len(edge_x[1]) != 0:
            fig.add_trace(go.Scatter(x=edge_x[1],
                                     y=edge_y[1],
                                     mode='lines',
                                     line=dict(color=edge_color,
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
                                             color=mark_color,
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
                                             color=mark_color,
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
        fig.update_traces(
            marker=dict(symbol="square"),
            selector=dict(mode="markers"),
        )
        fig_bytes = fig.to_image(format="png")
        img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        return img

    for i in range(5):
        original_leaf_node = v_whole_tree.vs[sorted_result[i]]

        sub_graph_root = parent(original_leaf_node)
        sub_graph = [sub_graph_root] + all_children(sub_graph_root)
        sub_graph = [item for item in sub_graph if abs(item["level"] - original_leaf_node["level"]) < 2]
        sub_graph = v_whole_tree.subgraph(sub_graph)
        img = _viz_tree_local(sub_graph, sorted_result[i])
        cv2.imwrite(os.path.join(root_path, "local_{}.png".format(i)), img)
        pass

    def save_indicate(v_idx, save_name):
        img = visualize_indicate(v_whole_tree, v_idx)
        fig_bytes = img.to_image(format="png")
        img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(save_name, img)

    save_indicate(sorted_result[0], os.path.join(root_path, "local_indicate_{}.png".format(0)))
    save_indicate(sorted_result[1], os.path.join(root_path, "local_indicate_{}.png".format(1)))
    save_indicate(sorted_result[2], os.path.join(root_path, "local_indicate_{}.png".format(2)))
    save_indicate(sorted_result[3], os.path.join(root_path, "local_indicate_{}.png".format(3)))
    save_indicate(sorted_result[4], os.path.join(root_path, "local_indicate_{}.png".format(4)))

    pass


def visualize_indicate(v_graph, v_index):
    min_levels = min(v_graph.vs, key=lambda item: item["level"])["level"]
    max_levels = max(v_graph.vs, key=lambda item: item["level"])["level"]

    marker_size = 10
    edge_size = 1

    pos_x = [[], [], []]
    pos_y = [[], [], []]
    labels = [[], [], []]
    sizes = [marker_size, marker_size, marker_size]
    annotations = []

    for i in range(len(v_graph.vs)):
        cur_level = v_graph.vs[i]["level"]
        if v_graph.vs[i]["original_index"] == v_index:
            cur_level = 2
        elif cur_level >= min_levels + 1:
            cur_level = 1
        else:
            cur_level = 0
        pos_x[cur_level].append(v_graph.vs[i]["pos"][0])
        pos_y[cur_level].append(v_graph.vs[i]["pos"][1])
        labels[cur_level].append(v_graph.vs[i]["cut"])
        # sizes[cur_level] = v_graph.vs[i]["size"]

    edge_x = [[], []]
    edge_y = [[], []]
    edge_width = [edge_size, edge_size]

    for edge in v_graph.es:
        cur_level = edge["level"]
        if cur_level >= min_levels + 1:
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

    # Visualize
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=5000,
        height=300,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor=bg_color,
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.add_trace(go.Scatter(x=edge_x[0],
                             y=edge_y[0],
                             mode='lines',
                             line=dict(color="#000000",
                                       width=edge_width[0]),
                             hoverinfo='none',
                             opacity=1
                             ))
    if len(edge_x[1]) != 0:
        fig.add_trace(go.Scatter(x=edge_x[1],
                                 y=edge_y[1],
                                 mode='lines',
                                 line=dict(color="#000000",
                                           width=edge_width[1]),
                                 hoverinfo='none',
                                 opacity=1
                                 ))
    fig.add_trace(go.Scatter(x=pos_x[0],
                             y=pos_y[0],
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=sizes[0],
                                         color="#000000",
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
                                         color="#000000",
                                         line=dict(
                                             color='rgb(50,50,50)',
                                             width=0)
                                         ),
                             text=labels[1],
                             hoverinfo='text',
                             opacity=1
                             ))
    fig.add_trace(go.Scatter(x=pos_x[2],
                             y=pos_y[2],
                             mode='markers',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=sizes[1],
                                         color="#FF6666",
                                         line=dict(
                                             color='rgb(50,50,50)',
                                             width=0)
                                         ),
                             text=labels[2],
                             hoverinfo='text',
                             opacity=1
                             ))
    fig.update_traces(
        marker=dict(symbol="square"),
        selector=dict(mode="markers"),
    )

    return fig


if __name__ == '__main__':
    if False:
        data = np.loadtxt(r"D:\Documents_tencent\1.csv", delimiter=",")
        data = np.sort(data, axis=0)
        x = data[:, 0]
        time_1 = data[:, 1]
        time_2 = data[:, 2]

        z1 = np.polyfit(x, time_1, deg=3)
        z2 = np.polyfit(x, time_2, deg=3)
        f1 = np.poly1d(z1)
        f2 = np.poly1d(z2)

        xx = np.arange(0, 7000, 1)
        # plt.plot(xx, f1(xx), label='IO-View')
        # plt.plot(xx, f2(xx), label='LOD-Tree')
        # plt.legend()
        # plt.xlabel("Shape numbers (#)")
        # plt.ylabel("Time (s)")
        # plt.show()

        data = np.loadtxt(
            r"D:\Documents_tencent\L7_loss_curve.txt", )
        # par1 = plt.twinx()
        x = np.arange(data.shape[0])
        y = data
        y[y<10e-2] = 0
        plt.ylabel("Summed diff value")
        plt.yscale('log')
        xx = [10, 200, 350, 450]
        plt.plot(x, y)
        plt.scatter(xx, y[xx])
        plt.ylim(0, 3 * 10e5)
        plt.margins(y=1)

        # par2 = plt.twinx()
        # par2.set_ylabel("Hausdorff Error")
        # par2.plot(x, x)

        # par3 = plt.twinx()
        # par3.xaxis.set_visible(False)
        # par3.yaxis.set_visible(False)

        # plt.plot(x, np.log10(y))
        plt.legend()
        plt.xlabel("Steps (#)")

        plt.show()
        pass

    if False:
        img_root = r"D:\Projects\LOD\demo_start"
        file_lists = sorted(os.listdir(img_root),key=lambda item:int(item.split(".")[0]))
        img = cv2.imread(os.path.join(img_root, file_lists[0]))
        out = cv2.VideoWriter(os.path.join(root_path, "demo.mkv"), cv2.VideoWriter_fourcc(*'h264'), 60, (img.shape[1],img.shape[0]))
        for item in tqdm(file_lists):
            img = cv2.imread(os.path.join(img_root, item))
            out.write(img)
        out.release()

    if False:
        import open3d as o3d
        from hausdorff import hausdorff_distance
        root_path = r"D:\Projects\LOD\hausdorff"
        for scene, min_z in [("04", -11), ("07", -19), ("08", -31), ("11", -14)]:
            names = [item for item in os.listdir(os.path.join(root_path, scene)) if item.split(".")[0][-4:] == "mesh" and item[0]!="_"]
            input_mesh = o3d.io.read_triangle_mesh(os.path.join(root_path, scene, scene+"-input.obj"))
            area = input_mesh.get_surface_area()
            num_sample = int(area * 100)
            pc_input = input_mesh.sample_points_uniformly(num_sample)
            ray_scene = o3d.t.geometry.RaycastingScene()
            input_mesh = o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)
            _ = ray_scene.add_triangles(input_mesh)

            # we do not need the geometry ID for mesh
            for name in names:
                output_mesh = o3d.io.read_triangle_mesh(os.path.join(root_path, scene, name))
                pc_output = output_mesh.sample_points_uniformly(num_sample)
                output = np.asarray(pc_output.points)
                output = output[output[:,2]>min_z]
                pc_output.points = o3d.utility.Vector3dVector(output)
                o3d.io.write_point_cloud(os.path.join(root_path, scene, "_"+name+"_samples.ply"), pc_output)
                unsigned_distance = ray_scene.compute_distance(output.astype(np.float32)).numpy()
                print("{}_{}: {}".format(scene,name,unsigned_distance.mean()))

        pass

    is_fix_pos = False

    is_visualize_whole_image = False
    is_visualize_whole_image_details = False
    is_generate_subtree = False
    is_generate_sub_imgs = True
    is_generate_video = False
    is_generate_bsp = False
    is_generate_specific_cut = False

    # Whole tree
    whole_tree = read_whole_non_bsp_tree(os.path.join(root_path, "tree.txt"))
    # whole_tree = read_whole_tree(os.path.join(root_path, "tree.txt"))
    whole_img, boundary = viz_whole_tree(whole_tree, v_is_fix_pos=True)
    whole_img.write_image(os.path.join(root_path, "whole.pdf"))
    fig_bytes = whole_img.to_image(format="png")
    whole_img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(root_path, "whole.png"), whole_img)
    if is_visualize_whole_image:
        cv2.namedWindow("1", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("1", 2000, 300)
        cv2.imshow("1", whole_img)
        cv2.waitKey(0)

    if is_visualize_whole_image_details:
        visualize_whole_image_details(tree, tree_states, whole_tree)

    # Sub-tree
    pool = Pool(18)
    os.makedirs(os.path.join(root_path, "inter_imgs"), exist_ok=True)
    if is_generate_subtree:
        tree, tree_states = read_tree(os.path.join(root_path, "order_state.txt"), whole_tree)
        with open(os.path.join(root_path, "intermediate1.bin"), "wb") as f:
            f.write(dumps(tree))
        with open(os.path.join(root_path, "intermediate2.bin"), "wb") as f:
            f.write(dumps(tree_states))
    else:
        with open(os.path.join(root_path, "intermediate1.bin"), "rb") as f:
            line = f.read()
            tree = loads(line)
        with open(os.path.join(root_path, "intermediate2.bin"), "rb") as f:
            line = f.read()
            tree_states = loads(line)

    if is_generate_sub_imgs:
        list(tqdm(pool.imap_unordered(
            partial(viz_item, v_graph=whole_tree, v_whole_img=whole_img, v_boundary=boundary, v_is_fix_pos=is_fix_pos,
                    v_states_range=np.where(np.asarray(tree_states)[:, 1] == 1)[0]),
            list(enumerate(zip(tree, tree_states))), chunksize=500), total=len(tree)))

    pool.close()

    if is_generate_video:
        write_video()

    if is_generate_bsp:
        id_specific_cut = np.where(np.asarray(tree_states)[:, 1] == 2)[0][-1]
        bsp_img = generate_small_tree(os.path.join(root_path, "small_tree.txt"))

        img = visualize_indicate(whole_tree, tree_states[id_specific_cut][0])
        fig_bytes = img.to_image(format="png")
        img = cv2.imdecode(np.frombuffer(fig_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(root_path, "bsp_indicate.png"), img)

        cv2.imwrite(os.path.join(root_path, "bsp.png"), bsp_img)
        cv2.namedWindow("1", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("1", 1000, 1000)
        cv2.imshow("1", bsp_img)
        cv2.waitKey(0)

    if is_generate_specific_cut:
        id_specific_cut = 80
        generate_specific_cut(tree[id_specific_cut], tree_states[id_specific_cut][0], whole_tree)