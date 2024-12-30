import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from networkx.algorithms.core import core_number
from scipy.cluster.hierarchy import single
from sympy import sequence
from sympy.physics.units import frequency
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from collections import deque
from pympler import asizeof
from scipy.sparse import lil_matrix
import time
import cProfile
import tracemalloc
from torch_geometric.data import TemporalData
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.dimenet import triplets

from model import TemporalGNN
from torch.nn import init
from torch_geometric.utils import k_hop_subgraph, subgraph
from tqdm import tqdm
import heapq

# 全局常量和变量

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = "cpu"

print(f"Using device: {device}")

# Tree
dataset_name = 'mathoverflow'
# dataset_name = 'wikitalk'
num_vertex = 0
num_edge = 0
num_timestamp = 0
time_edge = {}
num_core_number = 0
vertex_core_numbers = []
time_range_core_number = defaultdict(dict)
temporal_graph = []
time_range_layers = []
time_range_set = set()
max_time_range_layers = []
min_time_range_layers = []
sequence_features1_matrix = torch.empty(0, 0, 0)
model_time_range_layers = []
partition = 4
max_range = 40
max_degree = 0
max_layer_id = 0
root = None

inter_time = 0.0

# GNN
temporal_graph_pyg = TemporalData()
temporal_graph_pyg_dense = TemporalData()
subgraph_k_hop_cache = {}
subgraph_pyg_cache = {}
subgraph_vertex_map_cache = {}
filtered_temporal_graph_pyg = TemporalData()
time_range_link_samples_cache = defaultdict(dict)  # {(t_start, t_end): {anchor: {vertex: [(pos, neg), ...]}}}
# 超参数
node_in_channels = 8
node_out_channels = 16
edge_dim = 8
learning_rate = 0.001
epochs = 200
batch_size = 8
k_hop = 5
positive_hop = 3
alpha = 0.1
num_time_range_samples = 10
num_anchor_samples = 100
test_result_list = []

# 读取时间序列图数据-优化内存
def read_temporal_graph():
    print("Loading the graph...")
    global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree, temporal_graph_pyg
    time_edge = defaultdict(set)
    temporal_graph = defaultdict(lambda: defaultdict(list))

    filename = f'../datasets/{dataset_name}.txt'
    with open(filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
                first_line = False
                continue
            v1, v2, t = map(int, line.strip().split())
            if v1 == v2:
                continue
            if v1 > v2:
                v1, v2 = v2, v1
            time_edge[t].add((v1, v2))
            # 构建temporal_graph
            temporal_graph[v1][t].append(v2)
            temporal_graph[v2][t].append(v1)

    total_size = asizeof.asizeof(temporal_graph)
    print(f"temporal_graph 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")

    # 转为pyG Data格式
    edge_to_timestamps = {}
    for t, edges in time_edge.items():
        for src, dst in edges:
            edge_to_timestamps.setdefault((src, dst), set()).add(t)
            edge_to_timestamps.setdefault((dst, src), set()).add(t)

    # 构造edge_index和稀疏edge_attr
    edge_index = []
    edge_attr_indices = []  # [2, num_timestamps] - [edge_idx, timestamp_idx]
    edge_attr_values = []  # [num_timestamps]

    for (src, dst), timestamps in edge_to_timestamps.items():
        curr_edge_idx = len(edge_index)
        edge_index.append([src, dst])

        # 为每个时间戳创建一个位置索引
        for t_idx, t in enumerate(sorted(timestamps)):
            edge_attr_indices.append([curr_edge_idx, t_idx])  # [边的索引, 时间戳的序号]
            edge_attr_values.append(float(t))  # 实际的时间戳值

    # 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr_indices = torch.tensor(edge_attr_indices, dtype=torch.long).t()
    edge_attr_values = torch.tensor(edge_attr_values, dtype=torch.float32)

    # 获取每条边最大的时间戳数量
    max_timestamps_per_edge = max(len(timestamps) for timestamps in edge_to_timestamps.values())

    # 创建稀疏张量
    edge_attr = torch.sparse_coo_tensor(
        edge_attr_indices,
        edge_attr_values,
        size=(len(edge_index.t()), max_timestamps_per_edge),  # [num_edges, max_timestamps_per_edge]
        device=device
    ).coalesce()

    # 构造Data对象
    temporal_graph_pyg = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,  # 现在是包含实际时间戳的稀疏张量
        num_nodes=num_vertex,
    )

    temporal_graph_pyg = temporal_graph_pyg.to(device)

# def read_temporal_graph():
#     print("Loading the graph...")
#     filename = f'../datasets/{dataset_name}.txt'
#     global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree, temporal_graph_pyg
#     time_edge = defaultdict(set)
#     # temporal_graph = defaultdict(default_dict_factory)
#     temporal_graph = defaultdict(lambda: defaultdict(list))
#
#     with open(filename, 'r') as f:
#         first_line = True
#         for line in f:
#             if first_line:
#                 num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
#                 first_line = False
#                 continue
#             v1, v2, t = map(int, line.strip().split())
#             if v1 == v2:
#                 continue
#             if v1 > v2:
#                 v1, v2 = v2, v1
#             time_edge[t].add((v1, v2))
#             # 构建temporal_graph
#             temporal_graph[v1][t].append(v2)
#             temporal_graph[v2][t].append(v1)
#
#     total_size = asizeof.asizeof(temporal_graph)
#     print(f"temporal_graph 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")
#
#     # 转为pyG Data格式
#     edge_to_timestamps = {}
#     for t, edges in time_edge.items():
#         for src, dst in edges:
#             # 确保无向图 (src, dst) 和 (dst, src) 都添加时间戳
#             edge_to_timestamps.setdefault((src, dst), []).append(t)
#             edge_to_timestamps.setdefault((dst, src), []).append(t)
#     # 构造 edge_index 和 edge_attr
#     edge_index = []
#     edge_attr = []
#
#     for (src, dst), timestamps in edge_to_timestamps.items():
#         edge_index.append([src, dst])  # 添加边
#         edge_attr.append(timestamps)  # 添加时间戳
#
#     # 将 edge_index 转换为张量
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 [2, num_edges]
#
#     # 将 edge_attr 转换为张量 (需要统一长度)
#     max_timestamps = max(len(ts) for ts in edge_attr)  # 找到时间戳的最大数量
#     edge_attr = [
#         ts + [-1] * (max_timestamps - len(ts))  # 用 0 填充到相同长度
#         for ts in edge_attr
#     ]
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 转为张量
#
#     # 构造 Data 对象
#     temporal_graph_pyg = Data(
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         num_nodes=num_vertex,
#     )
#
#     temporal_graph_pyg = temporal_graph_pyg.to(device)

def read_core_number():
    print("Loading the core number...")
    global num_core_number, vertex_core_numbers
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    core_number_filename = f'../datasets/{dataset_name}-core_number.txt'
    with open(core_number_filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                num_core_number = int(line.strip())
                first_line = False
                continue
            range_part, core_numbers_part = line.split(' ', 1)
            range_start, range_end = map(int, range_part.strip('[]').split(','))
            is_node_range = False
            if (range_start, range_end) in time_range_set:
                is_node_range = True
            for pair in core_numbers_part.split():
                vertex, core_number = map(int, pair.split(':'))
                if range_start == range_end:
                    vertex_core_numbers[vertex][range_start] = core_number
                if is_node_range:
                    time_range_core_number[(range_start, range_end)][vertex] = core_number


# 获取时间范围层次
def get_timerange_layers():
    global time_range_layers, min_time_range_layers, max_time_range_layers
    temp_range = num_timestamp
    while temp_range > max_range:
        temp_range = temp_range // partition
    layer_id = 0
    while temp_range < num_timestamp:
        range_start = 0
        time_range_layers.append([])
        temp_max_range = 0
        temp_min_range = num_timestamp
        while range_start < num_timestamp:
            range_end = range_start + temp_range - 1
            range_end = min(range_end, num_timestamp - 1)
            if range_end + temp_range > num_timestamp - 1:
                range_end = num_timestamp - 1
            time_range_layers[layer_id].append((range_start, range_end))
            time_range_set.add((range_start, range_end))
            if range_end - range_start + 1 > temp_max_range:
                temp_max_range = range_end - range_start + 1
            if range_end - range_start + 1 < temp_min_range:
                temp_min_range = range_end - range_start + 1
            range_start = range_end + 1
        max_time_range_layers.append(temp_max_range)
        min_time_range_layers.append(temp_min_range)
        temp_range = temp_range * partition
        layer_id = layer_id + 1
    time_range_layers.reverse()
    max_time_range_layers.reverse()
    min_time_range_layers.reverse()


def construct_feature_matrix():
    print("Constructing the feature matrix...")
    global sequence_features1_matrix, indices_vertex_of_matrix
    indices = []
    values = []

    for v in range(num_vertex):
        # print(v)
        for t, neighbors in temporal_graph[v].items():
            core_number = vertex_core_numbers[v].get(t, 0)  # 获取核心数，默认为0
            neighbor_count = len(neighbors)  # 邻居数量
            if core_number > 0:
                # 添加索引 [vertex_index, timestamp, feature_index]
                indices.append([v, t, 0])
                values.append(core_number)
            if neighbor_count > 0:
                indices.append([v, t, 1])
                values.append(neighbor_count)

    # 将索引和数值转换为张量
    indices = torch.tensor(indices).T  # 转置为形状 (3, N)
    values = torch.tensor(values, dtype=torch.float32)

    # 对索引排序 方便后续的二分查找
    sorted_order = torch.argsort(indices[0])
    sorted_indices = indices[:, sorted_order]  # 对 indices 排序
    sorted_values = values[sorted_order]  # 对 values 按相同顺序排序

    # 创建稀疏张量
    sequence_features1_matrix = torch.sparse_coo_tensor(
        sorted_indices,
        sorted_values,
        size=(num_vertex, num_timestamp, 2),
        device=device
    )

    sequence_features1_matrix = sequence_features1_matrix.coalesce()

    # 计算稀疏张量实际占用的内存大小
    indices_size = indices.element_size() * indices.numel()
    values_size = values.element_size() * values.numel()
    total_size = indices_size + values_size
    print(f"feature matrix 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")


# @profile
def init_vertex_features(t_start, t_end, vertex_set, feature_dim, anchor):
    # vertex_indices = torch.tensor(list(vertex_set), device=device).sort().values
    vertex_indices = vertex_set

    # 获取 indices 和 values
    indices = sequence_features1_matrix.indices()  # (n, nnz)
    values = sequence_features1_matrix.values()  # (nnz,)

    start_idx = torch.searchsorted(indices[0], vertex_indices, side='left')
    end_idx = torch.searchsorted(indices[0], vertex_indices, side='right')

    # 利用 start_idx 和 end_idx 构建 mask_indices，而不是逐个遍历生成
    range_lengths = end_idx - start_idx  # 每个范围的长度
    total_indices = range_lengths.sum()  # 总共的索引数

    # 创建一个平铺的范围张量，直接表示所有 mask_indices
    range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
    flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

    # 映射 flat_indices 到实际的 mask_indices 范围
    mask_indices = start_idx.repeat_interleave(range_lengths) + flat_indices

    vertex_mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=device)
    vertex_mask[mask_indices] = True

    filtered_indices = indices[:, vertex_mask]
    filtered_values = values[vertex_mask]

    # 筛选时间范围
    time_mask = (
            (filtered_indices[1] >= t_start) &
            (filtered_indices[1] <= t_end)
    )
    final_indices = filtered_indices[:, time_mask]
    final_values = filtered_values[time_mask]

    # 构建 vertex_map 的张量版本
    vertex_map = torch.zeros(vertex_indices.max() + 1, dtype=torch.long, device=device)
    vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

    # 通过张量索引映射 final_indices[0]
    final_indices[0] = vertex_map[final_indices[0]]

    final_indices[1] -= t_start

    # 构造筛选后的稀疏张量
    result_size = (
        len(vertex_indices), t_end - t_start + 1, sequence_features1_matrix.size(2)
    )
    result_sparse_tensor = torch.sparse_coo_tensor(
        final_indices, final_values, size=result_size
    )
    degree_tensor = result_sparse_tensor.to_dense()[:, :, 1]
    degree_tensor.to(device)

    # 矢量化缩放 degree_tensor
    degree_tensor = degree_tensor.unsqueeze(1)
    if degree_tensor.shape[2] < feature_dim - 1:
        degree_tensor = F.interpolate(degree_tensor, size=feature_dim - 2, mode='linear', align_corners=True)
    else:
        degree_tensor = F.adaptive_avg_pool1d(degree_tensor, output_size=feature_dim - 2)
    degree_tensor = degree_tensor.squeeze(1)
    core_number_values = torch.tensor([time_range_core_number[(t_start, t_end)].get(v.item(), 0) for v in vertex_set],
                                      dtype=torch.float32, device=device)
    core_number_values = (core_number_values - core_number_values.min()) / (
                core_number_values.max() - core_number_values.min() + 1e-6)


    # 全矩阵归一化
    vertex_features_matrix = degree_tensor
    matrix_max = torch.max(vertex_features_matrix)
    matrix_min = torch.min(vertex_features_matrix)
    vertex_features_matrix = (vertex_features_matrix - matrix_min) / (matrix_max - matrix_min + 1e-6)
    vertex_features_matrix = torch.cat([core_number_values.unsqueeze(1), vertex_features_matrix], dim=1)

    query_feature = torch.zeros(len(vertex_set), 1, device=device)
    if anchor != -1:
        query_feature[vertex_map[anchor]][0] = 1
    vertex_features_matrix = torch.cat([query_feature, vertex_features_matrix], dim=1)
    return vertex_features_matrix

# 无归一化loss
def margin_triplet_loss(anchor, positives, negatives, margin=1):
    # 计算锚点与正样本的欧氏距离平方
    D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)  # 形状：(num_positives,)

    # 计算锚点与负样本的欧氏距离平方
    D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)  # 形状：(num_negatives,)

    # 扩展维度以便进行广播
    D_ap_expanded = D_ap.unsqueeze(1)  # 形状：(num_positives, 1)
    D_an_expanded = D_an.unsqueeze(0)  # 形状：(1, num_negatives)

    # 计算所有正负样本对的损失矩阵
    losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)  # 形状：(num_positives, num_negatives)

    # 计算平均损失
    loss = losses.mean()
    return loss


def compute_link_loss(embeddings, vertex_map, node_indices, t_start, t_end, anchor, margin=0.2):
    """
    计算link loss的向量化实现
    """
    # 获取当前anchor的所有link samples
    link_samples_dict = time_range_link_samples_cache[(t_start, t_end)].get(anchor, {})
    if not link_samples_dict:
        return torch.tensor(0.0, device=embeddings.device)

    # 收集所有有效的样本
    vertices = []
    positives = []
    negatives = []

    # 一次性收集所有有效样本
    for vertex, samples in link_samples_dict.items():
        if vertex_map[vertex].item() != -1:  # 顶点在子图中
            for pos, neg in samples:
                if vertex_map[pos].item() != -1 and vertex_map[neg].item() != -1:
                    vertices.append(vertex)
                    positives.append(pos)
                    negatives.append(neg)

    if not vertices:  # 如果没有有效样本
        return torch.tensor(0.0, device=embeddings.device)

    # 转换为张量并一次性获取所有嵌入
    vertex_indices = node_indices[vertex_map[vertices]]
    pos_indices = node_indices[vertex_map[positives]]
    neg_indices = node_indices[vertex_map[negatives]]

    vertex_embs = embeddings[vertex_indices]
    pos_embs = embeddings[pos_indices]
    neg_embs = embeddings[neg_indices]

    # 批量计算距离
    d_pos = torch.sum((vertex_embs - pos_embs) ** 2, dim=1)
    d_neg = torch.sum((vertex_embs - neg_embs) ** 2, dim=1)

    # 批量计算损失
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)

    return losses.mean()

# 自适应归一化loss
# def margin_triplet_loss(anchor, positives, negatives, margin=1, alpha=0.1):
#     # 计算原始距离
#     D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)
#     D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)
#
#     # 计算当前batch的特征尺度
#     scale = torch.cat([D_ap, D_an]).mean().detach()
#
#     # 自适应归一化
#     D_ap = D_ap / (scale + alpha)
#     D_an = D_an / (scale + alpha)
#
#     # 计算损失
#     D_ap_expanded = D_ap.unsqueeze(1)
#     D_an_expanded = D_an.unsqueeze(0)
#     losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)
#
#     return losses.mean()

# # sigmod归一化loss
# def margin_triplet_loss(anchor, positives, negatives, margin=1, temperature=1.0, scale=1.0):
#     # 计算距离
#     D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)
#     D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)
#
#     # 先缩放再Sigmoid
#     D_ap = torch.sigmoid(scale * D_ap / temperature)
#     D_an = torch.sigmoid(scale * D_an / temperature)
#
#     # 计算损失
#     D_ap_expanded = D_ap.unsqueeze(1)
#     D_an_expanded = D_an.unsqueeze(0)
#     losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)
#
#     return losses.mean()

# # 提取coreness大于当前顶点的子图
# def get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight):
#     global subgraph_k_hop_cache
#     visited = set()
#     visited.add(center_vertex)
#     query_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
#     queue = deque([(center_vertex, query_core_number, 0)])  # 队列初始化 (节点, core number, 距离)
#     subgraph_result = set()
#     core_number_condition = query_core_number * 0.5
#     current_hop = 0
#     while queue:
#         top_vertex, _, hop = queue.popleft()
#         if hop > k:
#             continue
#         subgraph_result.add(top_vertex)
#         for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
#             if neighbor not in visited and neighbor_core_number >= core_number_condition:
#             # if neighbor not in visited:
#                 queue.append((neighbor, neighbor_core_number, hop + 1))
#                 visited.add(neighbor)
#     return subgraph_result

# 根据平均coreness提取子图
def get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight):
    global subgraph_k_hop_cache
    visited = set()
    visited.add(center_vertex)
    query_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
    queue = deque([(center_vertex, query_core_number, 0)])  # 队列初始化 (节点, core number, 距离)
    subgraph_result = set()
    core_number_condition = query_core_number * 0.2
    current_hop = 0
    total_core_number = 0
    tau = 0.5
    best_avg_core_number = 0
    while queue:
        top_vertex, neighbor_core_number, hop = queue.popleft()
        if hop > k:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            print(f"Average core number: {average_core_number}")
            break
        if hop > current_hop:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            print(f"Average core number: {average_core_number}")
            current_hop = hop
            if average_core_number > best_avg_core_number:
                best_avg_core_number = average_core_number
            else:
                break
        subgraph_result.add(top_vertex)
        if len(subgraph_result) > 8000:
            break
        total_core_number += neighbor_core_number
        for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
            # if neighbor not in visited and neighbor_core_number >= core_number_condition:
            if neighbor not in visited:
                queue.append((neighbor, neighbor_core_number, hop + 1))
                visited.add(neighbor)
    return subgraph_result

def compute_modularity(subgraph, filtered_temporal_graph, total_edge_weight):
    subgraph = set(subgraph)
    internal_weights = 0  # 子图内部边的权重和
    total_weights = 0  # 所有边的权重和

    # 1. 统计边权重和节点强度
    for vertex in subgraph:
        for (neighbor, edge_count, _) in filtered_temporal_graph[vertex]:
            # 更新节点强度和总权重
            total_weights += edge_count

            # 统计内部边权重（只统计一个方向）
            if neighbor in subgraph and vertex < neighbor:
                internal_weights += edge_count

    # 如果没有边，返回0
    if total_weights == 0:
        return 0.0
    # 3. 计算带权重的modularity
    # density modularity
    modularity = (1.0 / len(subgraph)) * (internal_weights - (total_weights ** 2) / (4 * total_edge_weight))
    # classic modularity
    # modularity = (1.0 / (2 * total_edge_weight)) * (internal_weights - (total_weights ** 2) / (2 * total_edge_weight))


    return modularity


# 不断根据modularity，扩展子图
# def get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight):
#     global subgraph_k_hop_cache
#
#     def bfs_within_range(start_vertices, c_lower, c_upper):
#         """在给定core number范围内进行BFS搜索"""
#         subgraph = set(start_vertices)
#         visited = set(start_vertices)
#         queue = deque(start_vertices)
#
#         while queue:
#             vertex = queue.popleft()
#             for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[vertex]:
#                 if (neighbor not in visited and
#                         c_lower <= neighbor_core_number <= c_upper):
#                     visited.add(neighbor)
#                     queue.append(neighbor)
#                     subgraph.add(neighbor)
#         return subgraph
#
#     def get_boundary_core_numbers(subgraph):
#         """获取子图邻居的边界core numbers"""
#         lower_cores = set()
#         upper_cores = set()
#         current_range = [float('inf'), float('-inf')]  # [min, max] of subgraph
#
#         # 获取当前子图的core number范围
#         for v in subgraph:
#             v_core = time_range_core_number[(t_start, t_end)].get(v, 0)
#             current_range[0] = min(current_range[0], v_core)
#             current_range[1] = max(current_range[1], v_core)
#
#         c_l, c_u = current_range[0], current_range[1]
#
#         # 检查邻居的core numbers
#         for v in subgraph:
#             for (neighbor, _, neighbor_core_number) in filtered_temporal_graph[v]:
#                 if neighbor not in subgraph:
#                     if neighbor_core_number < c_l:
#                         lower_cores.add(neighbor_core_number)
#                     elif neighbor_core_number > c_u:
#                         upper_cores.add(neighbor_core_number)
#
#         # 如果没有找到满足条件的邻居，保持原来的边界值
#         c_l_prime = max(lower_cores) if lower_cores else c_l
#         c_u_prime = min(upper_cores) if upper_cores else c_u
#
#         return c_l_prime, c_u_prime, c_l, c_u
#
#     # 1. 初始化
#     query_core = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
#     current_subgraph = bfs_within_range([center_vertex], query_core, query_core)
#     best_subgraph = current_subgraph
#     best_modularity = compute_modularity(current_subgraph, filtered_temporal_graph, total_edge_weight)
#
#     # 2. 迭代扩展
#     while True:
#         # 获取边界core numbers
#         c_l_prime, c_u_prime, c_l, c_u = get_boundary_core_numbers(current_subgraph)
#         improved = False
#
#         up_modularity = float('-inf')
#         low_modularity = float('-inf')
#         up_subgraph = set()
#         low_subgraph = set()
#         # 尝试向上扩展
#         if c_u_prime > c_u:
#             expanded_subgraph = bfs_within_range(current_subgraph, c_l, c_u_prime)
#             expanded_modularity = compute_modularity(expanded_subgraph, filtered_temporal_graph, total_edge_weight)
#             up_modularity = expanded_modularity
#             up_subgraph = expanded_subgraph
#
#             if expanded_modularity > best_modularity:
#                 best_subgraph = expanded_subgraph
#                 best_modularity = expanded_modularity
#                 improved = True
#                 # print("up")
#
#         # 尝试向下扩展
#         if c_l_prime < c_l:
#             expanded_subgraph = bfs_within_range(current_subgraph, c_l_prime, c_u)
#             expanded_modularity = compute_modularity(expanded_subgraph, filtered_temporal_graph, total_edge_weight)
#             low_modularity = expanded_modularity
#             low_subgraph = expanded_subgraph
#
#             if expanded_modularity > best_modularity:
#                 best_subgraph = expanded_subgraph
#                 best_modularity = expanded_modularity
#                 improved = True
#                 # print("down")
#
#         # 如果没有改进，则停止迭代
#         if not improved:
#             if len(best_subgraph) > 2000:
#                 break
#             else:
#                 if up_modularity > low_modularity:
#                     current_subgraph = up_subgraph
#                     best_modularity = up_modularity
#                     best_subgraph = current_subgraph
#                     # print("+")
#                 elif low_modularity > up_modularity:
#                     current_subgraph = low_subgraph
#                     best_modularity = low_modularity
#                     best_subgraph = current_subgraph
#                     # print("-")
#                 else:
#                     break
#         else:
#             current_subgraph = best_subgraph
#
#     # print(len(best_subgraph))
#     return best_subgraph


# 根据平均聚类稀疏，扩展子图
# def get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph):
#     """使用平均聚类系数控制子图扩展"""
#
#     def compute_clustering_coefficient(vertex, subgraph):
#         """计算单个节点的聚类系数"""
#         neighbors = set()
#         for (neighbor, _, _) in filtered_temporal_graph[vertex]:
#             if neighbor in subgraph:  # 只考虑子图内的邻居
#                 neighbors.add(neighbor)
#
#         if len(neighbors) < 2:
#             return 0.0
#
#         # 计算邻居之间的实际边数
#         actual_edges = 0
#         for u in neighbors:
#             u_neighbors = set(n for (n, _, _) in filtered_temporal_graph[u])
#             actual_edges += len(u_neighbors & neighbors)
#
#         actual_edges /= 2  # 每条边被计算了两次
#         possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
#
#         return actual_edges / possible_edges if possible_edges > 0 else 0
#
#     def compute_avg_clustering(subgraph):
#         """计算子图的平均聚类系数"""
#         if len(subgraph) < 3:
#             return 0.0
#
#         total_coef = sum(compute_clustering_coefficient(v, subgraph)
#                          for v in subgraph)
#         return total_coef / len(subgraph)
#
#     # 初始化
#     current_subgraph = {center_vertex}
#     best_subgraph = current_subgraph
#     current_avg_clustering = compute_avg_clustering(current_subgraph)
#
#     while True:
#         improved = False
#         candidates = set()
#
#         # 获取候选节点
#         for v in current_subgraph:
#             for (neighbor, _, _) in filtered_temporal_graph[v]:
#                 if neighbor not in current_subgraph:
#                     candidates.add(neighbor)
#
#         # 评估每个候选节点
#         best_candidate = None
#         best_clustering = current_avg_clustering
#
#         for candidate in candidates:
#             temp_subgraph = current_subgraph | {candidate}
#             new_clustering = compute_avg_clustering(temp_subgraph)
#
#             # 如果新的聚类系数更高，更新最佳候选
#             if new_clustering >= best_clustering:
#                 best_clustering = new_clustering
#                 best_candidate = candidate
#                 improved = True
#
#         # 如果找到更好的候选节点，扩展子图
#         if improved and best_candidate is not None:
#             current_subgraph.add(best_candidate)
#             best_subgraph = current_subgraph
#             current_avg_clustering = best_clustering
#         else:
#             break
#
#         # 控制子图大小
#         if len(best_subgraph) > 1000:
#             break
#
#     return best_subgraph

def get_samples(center_vertex, k, t_start, t_end, filtered_temporal_graph, vertex_connect_scores, total_edge_weight):
    global subgraph_k_hop_cache
    candidates_neighbors = get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight)
    print(f"K-hop Neighbors: {len(candidates_neighbors)}")

    # 同时考虑时态边和coreness
    positive_neighbors_list = []
    positive_neighbors = set()
    hard_negative_neighbors = set()
    visited = set()
    visited.add(center_vertex)
    queue = []  # 优先队列
    heapq.heappush(queue, (0, center_vertex))
    query_vertex_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
    while queue:
        _, top_vertex = heapq.heappop(queue)
        v_core_number = time_range_core_number[(t_start, t_end)].get(top_vertex, 0)
        if v_core_number >= query_vertex_core_number:
            hard_negative_neighbors.add(top_vertex)
        positive_neighbors_list.append(top_vertex)
        for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
            if neighbor not in visited and neighbor in candidates_neighbors:
                heapq.heappush(queue, (-vertex_connect_scores[neighbor], neighbor))
                visited.add(neighbor)

    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.3:
    # while len(positive_neighbors) < len(candidates_neighbors) * 0.3:
        left_vertex = positive_neighbors_list.pop(0)
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)
    hard_negative_neighbors = hard_negative_neighbors - positive_neighbors - {center_vertex}
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(candidates_neighbors)

    return positive_neighbors, hard_negative_neighbors, candidates_neighbors


def generate_time_range_link_samples(k_hop_samples, filtered_temporal_graph):
    link_samples_dict = defaultdict(list)
    min_neighbors = 3
    for vertex in k_hop_samples:

        # 如果邻居数量不够，跳过
        if len(filtered_temporal_graph[vertex]) < min_neighbors:
            continue

        # 统计与每个邻居的连边数
        neighbor_edges = {}
        for neighbor, edge_count, _ in filtered_temporal_graph[vertex]:
            neighbor_edges[neighbor] = edge_count

        # 如果没有邻居，跳过
        if not neighbor_edges:
            continue

        # 按连边数排序邻居
        sorted_neighbors = sorted(neighbor_edges.items(), key=lambda x: x[1])

        # 确保至少有两个不同连边数的邻居
        if len(sorted_neighbors) >= 2 and sorted_neighbors[-1][1] > sorted_neighbors[0][1]:
            neg_neighbor = sorted_neighbors[0][0]  # 连边最少的邻居
            pos_neighbor = sorted_neighbors[-1][0]  # 连边最多的邻居
            link_samples_dict[vertex].append((pos_neighbor, neg_neighbor))

    # 随机选择100个样本
    if len(link_samples_dict) > 100:
        selected_vertices = random.sample(list(link_samples_dict.keys()), 100)
        selected_samples = defaultdict(list)
        for vertex in selected_vertices:
            selected_samples[vertex] = link_samples_dict[vertex]
        return selected_samples

    return link_samples_dict


# 同时考虑coreness和时态边
def generate_triplets(center_vertices, k_hop, t_start, t_end):
    triplets = []
    idx = 0

    # generate filtered subgraph
    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0
    for vertex in range(num_vertex):
        neighbor_time_edge_count = defaultdict(int)
        total_time_edge_count = 0
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    neighbor_time_edge_count[neighbor] += 1
                    total_time_edge_count += 1
        neighbors_list = []
        for neighbor, count in neighbor_time_edge_count.items():
            core_number = time_range_core_number[(t_start, t_end)].get(neighbor, 0)
            neighbors_list.append((neighbor, count, core_number))
            if vertex < neighbor:
                total_edge_weight += count
        filtered_subgraph[vertex] = neighbors_list
        vertex_core_number = time_range_core_number[(t_start, t_end)].get(vertex, 0)
        vertex_connect_scores[vertex] = vertex_core_number * total_time_edge_count / len(neighbors_list) if len(
            neighbors_list) != 0 else 0


    # 原图编号
    for anchor in center_vertices:
        if idx % 100 == 0:
            print(f"{idx}/{len(center_vertices)}")
        idx = idx + 1
        # 找到 k 跳邻居作为正样本
        positive_samples, hard_negative_samples, k_hop_samples = get_samples(anchor, k_hop, t_start, t_end,
                                                                             filtered_subgraph, vertex_connect_scores, total_edge_weight)
        if len(positive_samples) == 0:
            continue
        # 提取负样本
        easy_negative_samples = random.choices(
            list(k_hop_samples - positive_samples - hard_negative_samples - {anchor}),
            k=min(int(len(positive_samples) * 0.8), len(k_hop_samples - positive_samples - hard_negative_samples - {anchor})))
        hard_negative_samples = random.choices(list(hard_negative_samples),
                                               k=min(int(len(positive_samples) - len(easy_negative_samples)),
                                                     len(hard_negative_samples)))
        negative_samples = hard_negative_samples + easy_negative_samples

        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue

        # 生成三元组
        triplets.append((anchor, list(positive_samples), list(negative_samples)))

        # 生成link loss三元组
        link_samples = generate_time_range_link_samples(k_hop_samples, filtered_subgraph)
        time_range_link_samples_cache[(t_start, t_end)][anchor] = link_samples



    return triplets


class MultiSampleQuadrupletDataset(Dataset):
    def __init__(self, quadruplets):
        self.quadruplets = quadruplets

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor, positives, negatives, time_range = self.quadruplets[idx]
        return anchor, list(positives), list(negatives), time_range


def quadruplet_collate_fn(batch):
    anchors = torch.tensor([item[0] for item in batch], dtype=torch.long)  # 锚点
    positives = [item[1] for item in batch]  # 正样本列表
    negatives = [item[2] for item in batch]  # 负样本列表
    time_ranges = [item[3] for item in batch]  # 时间范围
    return anchors, positives, negatives, time_ranges

# @profile
def extract_subgraph_for_anchor(anchor, t_start, t_end):
    subgraph_pyg = subgraph_pyg_cache[(t_start, t_end)].to(device)
    neighbors_k_hop = subgraph_k_hop_cache[(int(anchor), (t_start, t_end))]
    neighbors_k_hop = torch.tensor(neighbors_k_hop, device=device)
    sub_edge_index, sub_edge_attr = subgraph(subset=neighbors_k_hop, edge_index=subgraph_pyg.edge_index, edge_attr=subgraph_pyg.edge_attr, relabel_nodes=True)
    # 构建 vertex_map
    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
    vertex_map[neighbors_k_hop] = torch.arange(len(neighbors_k_hop), device=device)

    # 提取特征矩阵
    old_vertex_map = subgraph_vertex_map_cache[(t_start, t_end)].to(device)
    feature_matrix = subgraph_pyg.x[old_vertex_map[neighbors_k_hop],:]
    feature_matrix[vertex_map[int(anchor)]][0] = 1
    # 构建子图 Data 对象
    subgraph_pyg = Data(
        x=feature_matrix,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
    )
    vertex_map = vertex_map.to(device)

    return subgraph_pyg, vertex_map

# sparse
def extract_subgraph_for_time_range(anchors, t_start, t_end, feature_dim):
    global subgraph_k_hop_cache
    neighbors = set()
    for anchor in anchors:
        neighbors |= set(subgraph_k_hop_cache[(anchor, (t_start, t_end))])
    neighbors = torch.tensor(sorted(neighbors), device=device)

    # 构建节点映射
    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
    vertex_map[neighbors] = torch.arange(len(neighbors), device=device)

    # 获取子图的边索引
    sub_edge_index, _, edge_mask = subgraph(
        subset=neighbors,
        edge_index=temporal_graph_pyg.edge_index,
        return_edge_mask=True,
        relabel_nodes=False
    )

    # 获取原始边属性（稀疏时间戳张量）
    edge_attr = temporal_graph_pyg.edge_attr
    edge_indices = edge_attr.indices()
    edge_values = edge_attr.values()

    # 1. 首先筛选时间范围
    time_mask = (edge_values >= t_start) & (edge_values <= t_end)
    valid_time_indices = edge_indices[:, time_mask]
    valid_time_values = edge_values[time_mask]

    # 2. 先筛选在子图中的边
    edge_in_subgraph_mask = edge_mask[valid_time_indices[0]]
    valid_time_indices = valid_time_indices[:, edge_in_subgraph_mask]
    valid_time_values = valid_time_values[edge_in_subgraph_mask]

    # 3. 创建边索引映射
    unique_edges = torch.unique(valid_time_indices[0])
    edge_id_map = torch.full((edge_mask.size(0),), -1, dtype=torch.long, device=device)
    edge_id_map[unique_edges] = torch.arange(len(unique_edges), device=device)
    sub_edge_index = temporal_graph_pyg.edge_index[:, unique_edges]

    # 4. 构建稠密边属性矩阵
    dense_edge_attr = torch.zeros(len(unique_edges), t_end - t_start + 1, device=device)
    new_edge_indices = edge_id_map[valid_time_indices[0]]
    adjusted_times = (valid_time_values - t_start).long()
    dense_edge_attr[new_edge_indices, adjusted_times] = 1

    # 应用傅里叶变换
    target_dim = edge_dim * 2 - 2
    if dense_edge_attr.shape[1] < target_dim:
        dense_edge_attr = F.interpolate(dense_edge_attr.unsqueeze(1), size=target_dim, mode='linear', align_corners=True)
    else:
        dense_edge_attr = F.adaptive_avg_pool1d(dense_edge_attr.unsqueeze(1), target_dim)
    sub_edge_attr = dense_edge_attr.squeeze(1)
    fft_result = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))
    sub_edge_attr = fft_result[:, :edge_dim]
    # dense_edge_attr = F.adaptive_avg_pool1d(dense_edge_attr.unsqueeze(1), edge_dim)
    # sub_edge_attr = dense_edge_attr.squeeze(1)
    # sub_edge_attr = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))

    # 构建子图
    subgraph_pyg = Data(
        x=init_vertex_features(t_start, t_end, neighbors, feature_dim, -1),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device
    )

    return subgraph_pyg, vertex_map

# dense
# def extract_subgraph_for_time_range(anchors, t_start, t_end, feature_dim):
#     global subgraph_k_hop_cache
#     neighbors = set()
#     for anchor in anchors:
#         neighbors |= set(subgraph_k_hop_cache[(anchor, (t_start, t_end))])
#     neighbors = torch.tensor(sorted(neighbors), device=device)
#     sub_edge_index, sub_edge_attr = subgraph(subset=neighbors, edge_index=temporal_graph_pyg.edge_index, edge_attr=temporal_graph_pyg.edge_attr, relabel_nodes=False)
#     time_mask = (sub_edge_attr >= t_start) & (sub_edge_attr <= t_end)
#     valid_edges = time_mask.any(dim=1)
#     sub_edge_attr[~time_mask] = -1
#     sub_edge_attr = sub_edge_attr[valid_edges]
#     sub_edge_index = sub_edge_index[:, valid_edges]
#
#     # 构建 vertex_map
#     vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
#     vertex_map[neighbors] = torch.arange(len(neighbors), device=device)
#
#     # 生成边属性-傅立叶变换
#     mask = (sub_edge_attr != -1)
#     indices = mask.nonzero(as_tuple=True)
#     origin_edge_attr = torch.zeros(sub_edge_attr.shape[0], t_end - t_start + 1, device=device)
#     # show the memory usage of origin_edge_attr
#     print(f"origin_edge_attr 占用的内存大小为 {origin_edge_attr.element_size() * origin_edge_attr.numel() / (1024 ** 2):.2f} MB")
#     origin_edge_attr[indices[0], (sub_edge_attr[indices] - t_start).long()] = 1
#     target_dim = edge_dim
#     origin_edge_attr = F.adaptive_avg_pool1d(origin_edge_attr.unsqueeze(1), target_dim)
#     sub_edge_attr = origin_edge_attr.squeeze(1)
#     sub_edge_attr = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))
#
#
#     # 构建子图 Data 对象
#     subgraph_pyg = Data(
#         x=init_vertex_features(t_start, t_end, neighbors, feature_dim, -1),
#         edge_index=sub_edge_index,
#         edge_attr=sub_edge_attr,
#         device=device
#     )
#     return subgraph_pyg, vertex_map


# 验证逻辑
def validate_model(model, val_loader, k_hop):
    model.eval()
    val_loss = 0.0
    top_num = 0.0
    feature_dim = node_in_channels

    with torch.no_grad():
        for batch in val_loader:
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device).clone()
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            subgraphs = []
            vertex_maps = []

            # 提取每个子图并合并
            for anchor, pos_samples, neg_samples, time_range in zip(anchors, positives, negatives, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1])
                if len(vertex_map) == 0:
                    continue
                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            if len(subgraphs) == 0:  # 跳过空批次
                continue

            # 合并子图为一个批次
            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

            # 模型前向传播
            embeddings = model(batched_subgraphs)

            # 使用 Batch.batch 获取每个子图的全局编号范围
            batch_indices = batched_subgraphs.batch

            del batched_subgraphs

            # 计算损失
            batch_loss = 0.0
            # batch_top_num = 0.0
            for i, (anchor, pos_samples, neg_samples, vertex_map, subgraph_pyg, time_range) in enumerate(
                    zip(anchors, positives, negatives, vertex_maps, subgraphs, time_ranges)):
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

                # 调整编号
                anchor_idx = node_indices[vertex_map[anchor.item()]]
                pos_indices = [node_indices[vertex_map[p.item()]].item() for p in pos_samples if
                               vertex_map[p.item()].item() != -1]
                neg_indices = [node_indices[vertex_map[n.item()]].item() for n in neg_samples if
                               vertex_map[n.item()].item() != -1]

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]
                subgraph_emb = embeddings[node_indices]

                loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                link_loss_value = compute_link_loss(embeddings, vertex_map, node_indices,
                                                    time_range[0], time_range[1], anchor.item())
                loss += alpha * link_loss_value
                batch_loss += loss

                torch.cuda.empty_cache()


            # 平均批次损失
            if len(vertex_maps) > 0:  # 避免除以零
                batch_loss = batch_loss / len(vertex_maps)
                # batch_top_num = batch_top_num / len(vertex_maps)

            val_loss += batch_loss.item()
            # top_num += batch_top_num

    val_loss /= len(val_loader)
    top_num /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    # print(f"Top Positive Samples in all Top tensors: {top_num:.4f}")
    return val_loss


# 测试逻辑
def test_model(model, test_loader, k_hop):
    """
    测试逻辑：统计正样本和负样本的平均相似度。

    参数：
    - model: 模型实例
    - test_loader: 测试数据加载器
    - k_hop: 提取子图的跳数

    返回：
    - avg_sim_pos: 平均正样本相似度
    - avg_sim_neg: 平均负样本相似度
    """
    model.eval()
    avg_sim_pos, avg_sim_neg = 0.0, 0.0
    total_pos_samples, total_neg_samples = 0, 0
    feature_dim = node_in_channels

    with torch.no_grad():
        for batch in test_loader:
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device)
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            subgraphs = []
            vertex_maps = []

            # 提取每个子图并合并
            for anchor, pos_samples, neg_samples, time_range in zip(anchors, positives, negatives, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1])
                if len(vertex_map) == 0:
                    continue
                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            if len(subgraphs) == 0:  # 跳过空批次
                continue

            # 合并子图为一个批次
            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

            # 模型前向传播
            embeddings = model(batched_subgraphs)

            # 使用 Batch.batch 获取每个子图的全局编号范围
            batch_indices = batched_subgraphs.batch

            # 统计正负样本相似度
            for i, (anchor, pos_samples, neg_samples, vertex_map) in enumerate(
                    zip(anchors, positives, negatives, vertex_maps)):
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

                # 调整编号
                anchor_idx = node_indices[vertex_map[anchor.item()]]
                pos_indices = [node_indices[vertex_map[p.item()]].item() for p in pos_samples if
                               vertex_map[p.item()] != -1]
                neg_indices = [node_indices[vertex_map[n.item()]].item() for n in neg_samples if
                               vertex_map[n.item()] != -1]

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]

                # 计算相似度
                sim_pos = F.cosine_similarity(anchor_emb.unsqueeze(0), positive_emb).mean().item()
                sim_neg = F.cosine_similarity(anchor_emb.unsqueeze(0), negative_emb).mean().item()

                avg_sim_pos += sim_pos * len(pos_indices)
                avg_sim_neg += sim_neg * len(neg_indices)
                total_pos_samples += len(pos_indices)
                total_neg_samples += len(neg_indices)

    avg_sim_pos /= total_pos_samples
    avg_sim_neg /= total_neg_samples

    print(f"Test Avg Positive Similarity: {avg_sim_pos:.4f}")
    print(f"Test Avg Negative Similarity: {avg_sim_neg:.4f}")
    return avg_sim_pos, avg_sim_neg


# 模型训练和验证
# @profile
def train_and_evaluation():
    global filtered_temporal_graph_pyg

    # 初始化模型
    model = TemporalGNN(node_in_channels, node_out_channels, edge_dim=edge_dim).to(device)
    # model = TemporalGNN(node_in_channels, node_out_channels, edge_dim=edge_dim).to(device)

    # 损失函数和数据集
    # criterion = margin_triplet_loss
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    train_time_range_list = []
    while len(train_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in train_time_range_list:
            train_time_range_list.append((t_start, t_end))

    # get training data
    # test
    quadruplet = []  # 训练数据用四元组形式存储
    # center_vertices = {198, 311, 441, 458, 506, 604, 612, 944, 1118, 1208, 1354, 1585, 1885, 1989, 2076, 2095, 2131, 2211, 2247, 2316, 2899, 3141, 3288, 3599, 3798, 4016, 4066, 4253, 4444, 4448, 4633, 4681, 4912, 4940, 4962, 5443, 5679, 5705, 6095, 6174, 6314, 7031, 7054, 7804, 8114, 8421, 8978, 9443, 9588, 10283, 10744, 11228, 11853, 12177, 12285, 12788, 13253, 13433, 13934, 14107, 14253, 14288, 14384, 14411, 14421, 14871, 14898, 15020, 15284, 15315, 15814, 16043, 16275, 16307, 16343, 16461, 16857, 16960, 17006, 17054, 17151, 17159, 17201, 17214, 17224, 17231, 17310, 17342, 17424, 17538, 17546, 17685, 17714, 17762, 17923, 18008, 18021, 18047, 21609, 23450}
    # t_start = 1728
    # t_end = 1871
    # triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
    # for triplet in triplets:
    #     quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))
    # temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels)

    for i, time_range in enumerate(train_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        print(f"Generating training data {i + 1}/{len(train_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        select_limit = 50
        select_cnt = 0
        while len(center_vertices) < num_anchor_samples:
            if select_cnt >= select_limit:
                break
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            min_core_numer = 5
            if t_end - t_start > 300:
                min_core_numer = 10
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
        for triplet in triplets:
            quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))
        # generate pyg data for the subgraph of the current time range
        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels)
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()

    test_time_range_list = []
    while len(test_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in test_time_range_list and (t_start, t_end) not in train_time_range_list:
            test_time_range_list.append((t_start, t_end))
    test_quadruplet = []
    for i, time_range in enumerate(test_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        print(f"Generating testing data {i + 1}/{len(test_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        select_limit = 50
        select_cnt = 0
        min_core_numer = 5
        if t_end - t_start > 300:
            min_core_numer = 10
        while len(center_vertices) < num_anchor_samples:
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            if select_cnt >= select_limit:
                break
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
        for triplet in triplets:
            test_quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))
        # generate pyg data for the subgraph of the current time range
        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end,
                                                                        node_in_channels)
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()

    test_quadruplet = random.choices(test_quadruplet, k=200)
    # 切割数据集
    train_quadruplet = quadruplet
    val_quadruplet, test_quadruplet = train_test_split(test_quadruplet, test_size=0.5, random_state=42)  # 20% 的一半用于验证集

    # 创建数据集
    train_dataset = MultiSampleQuadrupletDataset(train_quadruplet)
    val_dataset = MultiSampleQuadrupletDataset(val_quadruplet)
    test_dataset = MultiSampleQuadrupletDataset(test_quadruplet)

    # 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=quadruplet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # 用于混合精度训练
    feature_dim = node_in_channels

    # 初始化早停相关变量
    best_val_loss = float('inf')
    patience = 15  # 可以根据需要调整耐心值
    patience_counter = 0
    sample_cache = {}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device).clone()
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            optimizer.zero_grad()

            subgraphs = []
            vertex_maps = []
            for anchor, time_range in zip(anchors, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1])
                if len(vertex_map) == 0:
                    continue

                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            # for anchor, pos_samples, neg_sample, time_range in zip(anchors, positives, negatives, time_ranges):
            #     subgraph_pyg1, vertex_map1 = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1])
            #     subgraph_pyg, vertex_map = extract_subgraph(anchor.item(), time_range[0], time_range[1], k_hop,
            #                                                 feature_dim)
            #     compare1 = torch.equal(vertex_map1, vertex_map)
            #     compare2 = torch.equal(subgraph_pyg1.x, subgraph_pyg.x)
            #     if len(vertex_map) == 0:
            #         continue
            #
            #     subgraphs.append(subgraph_pyg)
            #     vertex_maps.append(vertex_map)
            #     del subgraph_pyg

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

            # 模型前向传播
            embeddings = model(batched_subgraphs)

            # 使用 Batch.batch 获取每个子图的全局编号范围
            batch_indices = batched_subgraphs.batch

            # release the memory of subgraphs
            del batched_subgraphs

            # 计算损失
            batch_loss = 0.0
            for i, (anchor, pos_samples, neg_samples, vertex_map, time_range) in enumerate(
                    zip(anchors, positives, negatives, vertex_maps, time_ranges)):
                # 获取当前子图的节点范围
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

                # 调整锚点、正样本和负样本的编号
                # anchor_idx = node_indices[vertex_map[anchor.item()]]
                # pos_indices = [node_indices[vertex_map[p.item()]].item() for p in pos_samples]
                # neg_indices = [node_indices[vertex_map[n.item()]].item() for n in neg_samples]

                anchor_idx = node_indices[vertex_map[anchor.long()]]
                pos_samples = pos_samples.to(device=vertex_map.device).long()
                neg_samples = neg_samples.to(device=vertex_map.device).long()
                pos_mapped_indices = vertex_map[pos_samples]  # 批量映射
                neg_mapped_indices = vertex_map[neg_samples]  # 批量映射
                pos_indices = node_indices[pos_mapped_indices]
                neg_indices = node_indices[neg_mapped_indices]

                    # 跳过无效样本
                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                # 提取嵌入
                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]
                subgraph_emb = embeddings[node_indices]

                # 计算三元组损失
                with autocast():
                    loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    link_loss_value = compute_link_loss(embeddings, vertex_map, node_indices,
                                                        time_range[0], time_range[1], anchor.item())
                    loss += alpha * link_loss_value
                    # loss += alpha * link_loss(subgraph_emb, subgraph_pyg)
                    # loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                batch_loss += loss
                torch.cuda.empty_cache()

            # 平均批次损失
            if len(vertex_maps) > 0:  # 避免除以零
                batch_loss = batch_loss / len(vertex_maps)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=epoch_loss / (batch_idx + 1))
            torch.cuda.empty_cache()
            # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
            # print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", end=' ')

        # 验证模型
        val_loss = validate_model(model, val_loader, k_hop)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最佳模型
            # torch.save(model.state_dict(), 'best_model.pth')
            # torch.save(model.state_dict(), 'model_L1_L2.pth')
            torch.save(model.state_dict(), f'model_L1_{dataset_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                # 加载最佳模型参数
                # model.load_state_dict(torch.load('best_model.pth'))
                # model.load_state_dict(torch.load('model_L1_L2.pth'))
                model.load_state_dict(torch.load(f'model_L1_{dataset_name}.pth'))
                break

    # 测试模型
    avg_sim_pos, avg_sim_neg = test_model(model, test_loader, k_hop)
    test_result_list.append((avg_sim_pos, avg_sim_neg))


# 主函数
def main():
    # # trace memory usage
    # tracemalloc.start()

    global root
    read_temporal_graph()
    # read_temporal_graph_dense()
    get_timerange_layers()
    read_core_number()
    construct_feature_matrix()
    train_and_evaluation()


if __name__ == "__main__":
    # cProfile.run('main()')
    # torch.cuda.empty_cache()
    main()