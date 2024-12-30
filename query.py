import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL.features import features
from networkx.algorithms.core import core_number
from scipy.cluster.hierarchy import single
from sympy import sequence
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
from torch_geometric.utils import k_hop_subgraph,subgraph
from tqdm import tqdm
import heapq
from sklearn.cluster import KMeans, DBSCAN
import math
import matplotlib.pyplot as plt
import time
from model import *

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
k_core_conductance = 0.0
k_core_density = 0.0
k_core_num = 0
root = None

inter_time = 0.0

# GNN
temporal_graph_pyg = TemporalData()
subgraph_k_hop_cache = {}
subgraph_pyg_cache = {}
subgraph_vertex_map_cache = {}
filtered_temporal_graph_pyg = TemporalData()
# 超参数
node_in_channels = 8
node_out_channels = 16
learning_rate = 0.001
epochs = 200
batch_size = 8
k_hop = 5
positive_hop = 3
edge_dim = 8
test_result_list = []

def generate_triplets(center_vertices, k_hop, t_start, t_end):
    triplets = []
    idx = 0

    # generate filtered subgraph
    filtered_subgraph = {}
    vertex_connect_scores = {}
    filtered_subgraph_tensor = {}
    valid_vertices = set()
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
                                                                             filtered_subgraph, vertex_connect_scores)
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
    return triplets

def get_k_hop_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph):
    global subgraph_k_hop_cache
    neighbors_k_hop = set()
    visited = set()
    visited.add(center_vertex)
    queue = deque([(center_vertex, 0)])  # 队列初始化 (节点, 当前跳数)
    while queue:
        top_vertex, depth = queue.popleft()
        if depth > k:
            continue
        neighbors_k_hop.add(top_vertex)
        for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
                visited.add(neighbor)
    return neighbors_k_hop

def get_samples(center_vertex, k, t_start, t_end, filtered_temporal_graph, vertex_connect_scores):
    global subgraph_k_hop_cache
    k_hop_neighbors = get_k_hop_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph)
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(k_hop_neighbors)

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
            if neighbor not in visited and neighbor in k_hop_neighbors:
                heapq.heappush(queue, (-vertex_connect_scores[neighbor], neighbor))
                visited.add(neighbor)

    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.3:
        left_vertex = positive_neighbors_list.pop(0)
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)
    hard_negative_neighbors = hard_negative_neighbors - positive_neighbors - {center_vertex}

    return positive_neighbors, hard_negative_neighbors, k_hop_neighbors




# 读取时间序列图数据-优化内存
def read_temporal_graph():
    print("Loading the graph...")
    filename = f'../datasets/{dataset_name}.txt'
    global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree, temporal_graph_pyg
    time_edge = defaultdict(set)
    # temporal_graph = defaultdict(default_dict_factory)
    temporal_graph = defaultdict(lambda: defaultdict(list))

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
            # 确保无向图 (src, dst) 和 (dst, src) 都添加时间戳
            edge_to_timestamps.setdefault((src, dst), []).append(t)
            edge_to_timestamps.setdefault((dst, src), []).append(t)
    # 构造 edge_index 和 edge_attr
    edge_index = []
    edge_attr = []

    for (src, dst), timestamps in edge_to_timestamps.items():
        edge_index.append([src, dst])  # 添加边
        edge_attr.append(timestamps)  # 添加时间戳

    # 将 edge_index 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 [2, num_edges]

    # 将 edge_attr 转换为张量 (需要统一长度)
    max_timestamps = max(len(ts) for ts in edge_attr)  # 找到时间戳的最大数量
    edge_attr = [
        ts + [0] * (max_timestamps - len(ts))  # 用 0 填充到相同长度
        for ts in edge_attr
    ]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 转为张量

    # 构造 Data 对象
    temporal_graph_pyg = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_vertex,
    )

    temporal_graph_pyg = temporal_graph_pyg.to(device)


# 读取节点的核心数
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


    vertex_features_matrix = torch.cat([core_number_values.unsqueeze(1), degree_tensor], dim=1)

    # 全矩阵归一化
    matrix_max = torch.max(vertex_features_matrix)
    matrix_min = torch.min(vertex_features_matrix)
    vertex_features_matrix = (vertex_features_matrix - matrix_min) / (matrix_max - matrix_min + 1e-6)
    query_feature = torch.zeros(len(vertex_set), 1, device=device)
    query_feature[vertex_map[anchor]][0] = 1
    vertex_features_matrix = torch.cat([query_feature, vertex_features_matrix], dim=1)


    return vertex_features_matrix


def margin_triplet_loss(anchor, positives, negatives, margin=1):
    """
        计算三元组损失。

        参数：
        - anchor: 张量，形状为 (embedding_dim,)
        - positives: 列表，包含一个或多个张量，每个形状为 (embedding_dim,)
        - negatives: 列表，包含多个张量，每个形状为 (embedding_dim,)
        - margin: float，边距超参数

        返回：
        - loss: 标量，损失值
        """

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


# @profile
def extract_subgraph(anchor, t_start, t_end, k, feature_dim):
    visited = set()
    visited.add(anchor)
    queue = deque([(anchor, 0)])  # 队列初始化 (节点, 当前跳数)
    neighbors_k_hop = set()
    while queue:
        top_vertex, depth = queue.popleft()
        if depth > k:
            continue
        neighbors_k_hop.add(top_vertex)
        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                        visited.add(neighbor)

    neighbors_k_hop = torch.tensor(sorted(neighbors_k_hop), device=device)

    # 构建 vertex_map
    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long)
    vertex_map[neighbors_k_hop] = torch.arange(len(neighbors_k_hop))
    vertex_map = vertex_map.to(device)

    # 过滤条件
    mask = (
            (vertex_map[temporal_graph_pyg.edge_index[0]] != -1)
            & (vertex_map[temporal_graph_pyg.edge_index[1]] != -1)
    )
    mask = mask.to(device)

    # print(f"edge_index device: {temporal_graph_pyg.edge_index.device}")
    # print(f"mask device: {mask.device}")

    sub_edge_index = temporal_graph_pyg.edge_index[:, mask]

    # 将边索引映射到子图编号
    sub_edge_index = vertex_map[sub_edge_index]

    # 提取边属性
    sub_edge_attr = temporal_graph_pyg.edge_attr[mask]
    time_mask = (sub_edge_attr >= t_start) & (sub_edge_attr <= t_end)
    valid_edges = time_mask.any(dim=1)
    sub_edge_attr[~time_mask] = -1
    sub_edge_attr = sub_edge_attr[valid_edges]
    sub_edge_index = sub_edge_index[:, valid_edges]

    # 生成边属性-傅立叶变换
    mask = (sub_edge_attr != -1)
    indices = mask.nonzero(as_tuple=True)
    origin_edge_attr = torch.zeros(sub_edge_attr.shape[0], t_end - t_start + 1, device=device)
    origin_edge_attr[indices[0], (sub_edge_attr[indices] - t_start).long()] = 1
    target_dim = edge_dim
    origin_edge_attr = F.adaptive_avg_pool1d(origin_edge_attr.unsqueeze(1), target_dim)
    sub_edge_attr = origin_edge_attr.squeeze(1)
    sub_edge_attr = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))

    # 构建子图 Data 对象
    subgraph_pyg = Data(
        x=init_vertex_features(t_start, t_end, neighbors_k_hop, feature_dim, anchor),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        num_nodes=len(neighbors_k_hop),
        device=device
    )
    return subgraph_pyg, vertex_map, neighbors_k_hop

def compute_density(vertex_set, t_start, t_end):
    time_edges_count = 0
    time_stamps = set()
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor in vertex_set:
                        time_edges_count += 1
                        time_stamps.add(t)
    temporal_density = time_edges_count / ((len(vertex_set) * (len(vertex_set) - 1)) * len(time_stamps))
    return temporal_density

def compute_conductance(vertex_set, t_start, t_end):
    time_edges_count = 0
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in vertex_set:
                        time_edges_count += 1
    degree_community = 0
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                degree_community += len(neighbors)
    degree_not_in_community = 0
    for vertex in range(num_vertex):
        if vertex not in vertex_set:
            for t, neighbors in temporal_graph[vertex].items():
                if t_start <= t <= t_end:
                    degree_not_in_community += len(neighbors)
    temporal_conductance = time_edges_count / min(degree_community, degree_not_in_community)
    return temporal_conductance

def temporal_test_k_core(query_vertex, t_start, t_end):
    global k_core_num
    visited = set()
    visited.add(query_vertex)
    queue = deque([query_vertex])  # 队列初始化 (节点, 当前跳数)
    result_k_core = set()
    query_core_number = time_range_core_number[(t_start, t_end)].get(query_vertex, 0)
    while queue:
        # if len(result_k_core) > 50:
        #     break
        top_vertex = queue.popleft()
        result_k_core.add(top_vertex)
        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    neighbor_core_number = time_range_core_number[(t_start, t_end)].get(neighbor, 0)
                    if neighbor_core_number >= query_core_number:
                        if neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)

    temporal_density = compute_density(result_k_core, t_start, t_end)

    # count the conductance
    temporal_conductance = compute_conductance(result_k_core, t_start, t_end)
    k_core_num = len(result_k_core)
    print(f"k-core Result Number: {len(result_k_core)}")
    return temporal_density, temporal_conductance

def temporal_test_GNN(distances, vertex_map, query_vertex, t_start, t_end):
    # test with GNN using threshold
    visited = set()
    visited.add(query_vertex)
    result = set()
    queue = []
    heapq.heappush(queue, (0, query_vertex))
    # # 归一化距离
    mask = (distances != 0)
    temp_distances = distances[mask]
    distances = (distances - temp_distances.min()) / (distances.max() - temp_distances.min() + 1e-6)
    result_distance = []
    threshold = distances.mean().item()
    less_num = torch.sum(distances < threshold).item()

    while queue:
        distance, top_vertex = heapq.heappop(queue)
        result.add(top_vertex)
        result_distance.append(distance)
        if distance > threshold:
            break
        if len(result) > 1:
            tau = 0.9
            alpha = np.cos((np.pi / 2) * (len(result) / (less_num ** tau)))
            threshold = alpha * threshold + (1-alpha) * (sum(result_distance) / (len(result_distance) - 1))

        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if vertex_map[neighbor] != -1:
                            heapq.heappush(queue, (distances[vertex_map[neighbor]].item(), neighbor))
    if len(result) == 0:
        return 0, 0
    temporal_density = compute_density(result, t_start, t_end)
    # print(len(result))

    temporal_conductance = compute_conductance(result, t_start, t_end)
    print(f"Result Number: {len(result)}")
    return temporal_density, temporal_conductance


def temporal_test_GNN_query_time(distances, vertex_map, query_vertex, t_start, t_end):
    # test with GNN using threshold
    visited = set()
    visited.add(query_vertex)
    result = set()
    queue = []
    heapq.heappush(queue, (0, query_vertex))
    # # 归一化距离
    mask = (distances != 0)
    temp_distances = distances[mask]
    distances = (distances - temp_distances.min()) / (distances.max() - temp_distances.min() + 1e-6)
    result_distance = []
    threshold = distances.mean().item()
    less_num = torch.sum(distances < threshold).item()

    while queue:
        distance, top_vertex = heapq.heappop(queue)
        result.add(top_vertex)
        result_distance.append(distance)
        if distance > threshold:
            break
        if len(result) > 1:
            tau = 0.9
            alpha = np.cos((np.pi / 2) * (len(result) / (less_num ** tau)))
            threshold = alpha * threshold + (1-alpha) * (sum(result_distance) / (len(result_distance) - 1))

        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if vertex_map[neighbor] != -1:
                            heapq.heappush(queue, (distances[vertex_map[neighbor]].item(), neighbor))
    if len(result) == 0:
        return 0, 0
    temporal_density = compute_density(result, t_start, t_end)
    # print(len(result))
    temporal_conductance = compute_conductance(result, t_start, t_end)
    print(f"Result Number: {len(result)}")
    return temporal_density, temporal_conductance,result

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

def extract_subgraph_for_anchor(anchor, t_start, t_end,feature_dim):
    """
    为单个 anchor 提取子图。

    Args:
        anchor: 单个 anchor 节点
        t_start: 时间窗口起始时间
        t_end: 时间窗口结束时间
        feature_dim: 节点特征维度
        temporal_graph_pyg: 原始时间图
        subgraph_k_hop_cache: k-hop 邻居缓存
        num_vertex: 总节点数
        init_vertex_features: 初始化节点特征的函数
        edge_dim: 边的维度
        device: 设备

    Returns:
        subgraph_pyg: 子图 Data 对象
        vertex_map: 顶点映射
    """
    neighbors = subgraph_k_hop_cache[(anchor, (t_start, t_end))]
    neighbors = torch.tensor(sorted(neighbors), device=device)
    sub_edge_index, sub_edge_attr = subgraph(subset=neighbors, edge_index=temporal_graph_pyg.edge_index, edge_attr=temporal_graph_pyg.edge_attr, relabel_nodes=False)
    time_mask = (sub_edge_attr >= t_start) & (sub_edge_attr <= t_end)
    valid_edges = time_mask.any(dim=1)
    sub_edge_attr[~time_mask] = -1
    sub_edge_attr = sub_edge_attr[valid_edges]
    sub_edge_index = sub_edge_index[:, valid_edges]


    # 构建 vertex_map
    vertex_map = torch.full((num_vertex,), -1, dtype=torch.long, device=device)
    vertex_map[neighbors] = torch.arange(len(neighbors), device=device)
    
    # 生成边属性-傅立叶变换
    mask = (sub_edge_attr != -1)
    indices = mask.nonzero(as_tuple=True)
    origin_edge_attr = torch.zeros(sub_edge_attr.shape[0], t_end - t_start + 1, device=device)
    origin_edge_attr[indices[0], (sub_edge_attr[indices] - t_start).long()] = 1
    target_dim = edge_dim
    origin_edge_attr = F.adaptive_avg_pool1d(origin_edge_attr.unsqueeze(1), target_dim)
    sub_edge_attr = origin_edge_attr.squeeze(1)
    sub_edge_attr = torch.abs(torch.fft.fft(sub_edge_attr, dim=1))
    
    # 构建子图 Data 对象
    subgraph_pyg = Data(
        x=init_vertex_features(t_start, t_end, neighbors, feature_dim, -1),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device
    )

    return subgraph_pyg, vertex_map

def query_test():
    global k_core_conductance, k_core_density
    
    # 初始化模型并加载预训练权重
    model = AdapterTemporalGNN(
        node_in_channels=node_in_channels,
        node_out_channels=node_out_channels,
        edge_dim=edge_dim,
        adapter_dim=16,
        use_adapter3=False
    ).to(device)
    model.load_state_dict(torch.load("model_L1.pth"), strict=False)
    
    # 冻结非adapter参数
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'gating_params' not in name:
            param.requires_grad = False
    
    # 只优化adapter参数
    adapter_params = [p for n, p in model.named_parameters() if 'adapter' in n or 'gating_params' in n]
    optimizer = torch.optim.Adam(adapter_params, lr=0.001)
    scaler = GradScaler()  # 混合精度训练
    
    test_time_range_list = []
    while len(test_time_range_list) < 10:
        t_layer = random.randint(1, len(time_range_layers) - 2)
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in test_time_range_list:
            test_time_range_list.append((t_start, t_end))
    
    temporal_density_ratio = 0
    temporal_conductance_ratio = 0
    valid_cnt = 0
    total_time = 0
    result_len = 0
    
    for t_start, t_end in test_time_range_list:
        print(f"Test time range: [{t_start}, {t_end}]")
        query_vertex_list = set()
        while len(query_vertex_list) < 10:
            query_vertex = random.choice(range(num_vertex))
            core_number = time_range_core_number[(t_start, t_end)].get(query_vertex, 0)
            while core_number < 5:
                query_vertex = random.choice(range(num_vertex))
                core_number = time_range_core_number[(t_start, t_end)].get(query_vertex, 0)
            query_vertex_list.add(query_vertex)
            
        for query_vertex in query_vertex_list:
            print(valid_cnt)
            start_time = time.time()
            
            # 为查询节点生成训练数据
            center_vertices = {query_vertex}  # 只使用查询节点作为中心
            triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
            quadruplet = [(t[0], t[1], t[2], (t_start, t_end)) for t in triplets]
            
            # 创建数据加载器
            query_dataset = MultiSampleQuadrupletDataset(quadruplet)
            query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=quadruplet_collate_fn)
            # 查询特定的微调
            model.train()
            for _ in range(5):  # 微调步数
                for batch in query_loader:
                    anchors, positives, negatives, time_ranges = batch
                    anchors = torch.tensor(anchors, device=device).clone()
                    positives = [torch.tensor(pos, device=device) for pos in positives]
                    negatives = [torch.tensor(neg, device=device) for neg in negatives]

                    optimizer.zero_grad()
                    
                    # 提取子图并获取嵌入
                    subgraphs = []
                    vertex_maps = []
                    for anchor, time_range in zip(anchors, time_ranges):
                        subgraph_pyg, vertex_map = extract_subgraph_for_anchor(
                            anchor.item(), time_range[0], time_range[1],8)
                        if subgraph_pyg is None:
                            continue # 跳过没有子图的样本
                        subgraphs.append(subgraph_pyg)
                        vertex_maps.append(vertex_map)
                        
                    batched_subgraphs = Batch.from_data_list(subgraphs).to(device)
                    with autocast():
                        embeddings = model(batched_subgraphs)
                    batch_indices = batched_subgraphs.batch
                    
                    # 计算损失
                    batch_loss = 0.0
                    for i, (anchor, pos_samples, neg_samples, vertex_map, time_range) in enumerate(
                            zip(anchors, positives, negatives, vertex_maps, time_ranges)):
                        
                        node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]
                        anchor_idx = node_indices[vertex_map[anchor.long()]]
                        pos_samples = pos_samples.to(device=vertex_map.device).long()
                        neg_samples = neg_samples.to(device=vertex_map.device).long()
                        pos_mapped_indices = vertex_map[pos_samples]
                        neg_mapped_indices = vertex_map[neg_samples]
                        pos_indices = node_indices[pos_mapped_indices]
                        neg_indices = node_indices[neg_mapped_indices]
                        
                        if len(pos_indices) == 0 or len(neg_indices) == 0:
                            continue
                            
                        anchor_emb = embeddings[anchor_idx]
                        positive_emb = embeddings[pos_indices]
                        negative_emb = embeddings[neg_indices]
                        
                        # 组合损失：三元组损失 + 连接损失
                        loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                        # loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                        batch_loss += loss
                    
                    if len(vertex_maps) > 0:
                        batch_loss = batch_loss / len(vertex_maps)
                        
                    # 更新参数
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            
            # 评估阶段
            model.eval()
            with torch.no_grad():
                feature_dim = node_in_channels
                subgraph, vertex_map, neighbors_k_hop = extract_subgraph(
                    query_vertex, t_start, t_end, k_hop, feature_dim)
                
                embeddings = model(subgraph)
                query_vertex_embedding = embeddings[vertex_map[query_vertex]].unsqueeze(0)
                neighbors_embeddings = embeddings[vertex_map[neighbors_k_hop]]
                distances = F.pairwise_distance(query_vertex_embedding, neighbors_embeddings)
                
                GNN_temporal_density, GNN_temporal_conductance, result = temporal_test_GNN_query_time(
                    distances, vertex_map, query_vertex, t_start, t_end)
                
                temporal_density_ratio += GNN_temporal_density
                temporal_conductance_ratio += GNN_temporal_conductance
                result_len += len(result)
            
            end_time = time.time()
            total_time += end_time - start_time
            valid_cnt += 1
            
            # 清理缓存
            torch.cuda.empty_cache()
    
    # 打印结果
    print(f"Valid test number: {valid_cnt}")
    print(f"Average temporal density ratio: {temporal_density_ratio / valid_cnt}")
    print(f"Average temporal conductance ratio: {temporal_conductance_ratio / valid_cnt}")
    print(f"Average time: {total_time / valid_cnt}s")
    print(f"Average length: {result_len / valid_cnt}")

# 主函数
def main():
    # # trace memory usage
    # tracemalloc.start()

    global root
    read_temporal_graph()
    get_timerange_layers()
    read_core_number()
    construct_feature_matrix()
    query_test()


if __name__ == "__main__":
    # cProfile.run('main()')
    # torch.cuda.empty_cache()
    main()