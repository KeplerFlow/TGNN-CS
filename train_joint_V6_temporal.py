import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from collections import deque
from pympler import asizeof
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
edge_dim = 8
learning_rate = 0.001
epochs = 200
batch_size = 8
k_hop = 5
positive_hop = 3
alpha = 0.5
num_time_range_samples = 10
num_anchor_samples = 10
test_result_list = []

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
        ts + [-1] * (max_timestamps - len(ts))  # 用 0 填充到相同长度
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

def margin_triplet_loss(anchor, positives, negatives, margin=1.5):
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

def link_loss(vertex_map, subgraph_emb, t_start, t_end):
    return 0.0
    num_samples = 100
    # 从缓存中获取有效顶点和子图数据
    valid_vertices = filtered_subgraph_tensor_cache[(t_start, t_end)][0].to(device)
    filter_subgraph = filtered_subgraph_tensor_cache[(t_start, t_end)][1]  # 假设里面是某种可索引结构

    # 随机采样顶点
    sampled_vertices = valid_vertices[torch.randint(0, len(valid_vertices), (num_samples,), device=device)]

    # 获取邻居和频率 (尽量将这个步骤也向张量化或至少批量化方向努力)
    # 假定 filter_subgraph[v] = (neighbor_tensor, frequency_tensor) 都在 GPU 上
    # 如果不是，需要提前进行数据准备，将整个 filter_subgraph 转换成支持索引的 GPU 张量结构。
    neighbors_list = []
    frequencies_list = []
    for v in sampled_vertices:
        n, f = filter_subgraph[v.item()]  # 假设这已经是tensor
        neighbors_list.append(n)
        frequencies_list.append(f)

    # 找出最大邻居数，做padding
    max_num_neighbors = max(n.shape[0] for n in neighbors_list)

    # 初始化邻居和频率张量
    # neighbor_embs_tensor: shape (num_samples, max_num_neighbors, emb_dim)
    # frequencies_tensor: shape (num_samples, max_num_neighbors)
    emb_dim = subgraph_emb.shape[1]
    neighbor_indices_padded = torch.zeros(num_samples, max_num_neighbors, dtype=sampled_vertices.dtype, device=device)
    frequencies_padded = torch.zeros(num_samples, max_num_neighbors, dtype=subgraph_emb.dtype, device=device)
    mask = torch.zeros(num_samples, max_num_neighbors, dtype=torch.bool, device=device)

    for i, (n, f) in enumerate(zip(neighbors_list, frequencies_list)):
        length = n.shape[0]
        neighbor_indices_padded[i, :length] = n
        frequencies_padded[i, :length] = f
        mask[i, :length] = True

    # 获取 vertex embeddings
    vertex_embs = subgraph_emb[vertex_map[sampled_vertices]]  # shape: (num_samples, emb_dim)

    # 获取 neighbor embeddings
    neighbor_embs_tensor = subgraph_emb[
        vertex_map[neighbor_indices_padded]]  # (num_samples, max_num_neighbors, emb_dim)

    # 计算距离 (广播计算)
    # vertex_embs.unsqueeze(1): (num_samples, 1, emb_dim)
    # neighbor_embs_tensor: (num_samples, max_num_neighbors, emb_dim)
    # distances: (num_samples, max_num_neighbors)
    distances = torch.sum((vertex_embs.unsqueeze(1) - neighbor_embs_tensor) ** 2, dim=2)

    # 只对有邻居的地方做加权求和并平均
    # frequencies_padded: (num_samples, max_num_neighbors)
    weighted_sum = torch.sum(distances * frequencies_padded, dim=1)
    neighbor_count = mask.sum(dim=1).float()
    weighted_losses = weighted_sum / neighbor_count

    # 最终 loss
    total_loss = weighted_losses.mean()
    return total_loss


# def find_k_hop_neighbors_old(center_vertex, k, t_start, t_end, filtered_temporal_graph):
#     # 同时考虑时态边和coreness V2
#     global subgraph_k_hop_cache
#     positive_neighbors_list = []
#     positive_neighbors = set()
#     hard_negative_neighbors = set()
#     neighbors_k_hop = set()
#     visited = set()
#     visited.add(center_vertex)
#     queue = deque([(center_vertex, 0)])  # 队列初始化 (节点, 当前跳数)
#     anchor_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
#     while queue:
#         top_vertex, depth = queue.popleft()
#         if depth > k:
#             continue
#         neighbors_k_hop.add(top_vertex)
#         for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
#             if neighbor not in visited:
#                 queue.append((neighbor, depth + 1))
#                 visited.add(neighbor)
#             if (
#                     top_vertex in positive_neighbors or depth == 0) and neighbor not in positive_neighbors and depth + 1 <= k:
#                 if edge_count + neighbor_core_number > anchor_core_number + 1:
#                     if neighbor not in positive_neighbors:
#                         positive_neighbors.add(neighbor)
#                         positive_neighbors_list.append(neighbor)
#                         if neighbor in hard_negative_neighbors:
#                             hard_negative_neighbors.remove(neighbor)
#                 else:
#                     hard_negative_neighbors.add(neighbor)
#     hard_negative_neighbors = hard_negative_neighbors - positive_neighbors - {center_vertex}
#     positive_neighbors = positive_neighbors - {center_vertex}
#
#     # if len(positive_neighbors) != 0:
#     subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(neighbors_k_hop)
#     num_positive_limit = len(positive_neighbors) * 0.3
#     positive_neighbors.clear()
#     while len(positive_neighbors) < num_positive_limit:
#         positive_neighbors.add(positive_neighbors_list.pop(0))
#
#     return positive_neighbors, hard_negative_neighbors, neighbors_k_hop

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


# 同时考虑coreness和时态边
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

def extract_subgraph_for_time_range(anchors, t_start, t_end, feature_dim):
    global subgraph_k_hop_cache
    neighbors = set()
    for anchor in anchors:
        neighbors |= set(subgraph_k_hop_cache[(anchor, (t_start, t_end))])
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
                # loss += alpha * link_loss(subgraph_emb, subgraph_pyg)
                loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                # loss += alpha * link_loss(subgraph_emb, t_start, t_end)
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

    ######################
    # 1) 初始化带 Adapter 的模型
    ######################
    model = AdapterTemporalGNN(
        node_in_channels=node_in_channels,
        node_out_channels=node_out_channels,
        edge_dim=edge_dim,
        adapter_dim=16,    # 根据需要调整 Adapter 的隐藏维度
        use_adapter3=False # 是否在第三层后也插入一个 Adapter
    ).to(device)

    ######################
    # 2) 加载预训练好的权重 (strict=False 允许新模型存在额外参数)
    ######################
    # pretrained_path = "/home/asc23/lcy/gnn/TGNN/GNN/model_L1.pth"  # 这里替换为您预训练模型的路径
    # pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    # model_dict = model.state_dict()

    # # 只加载能对得上的部分参数, Adapter 模块不在预训练模型中, 因此用 strict=False
    # for k, v in pretrained_state_dict.items():
    #     if k in model_dict:
    #         model_dict[k] = v
    # model.load_state_dict(model_dict, strict=False)

    # # 3) 冻结原有卷积层参数, 只训练 Adapter 相关参数
    # for name, param in model.named_parameters():
    #     # 如果参数名里不包含 "adapter" 和 "gating_params"，则冻结
    #     if ("adapter" not in name) and ("gating_params" not in name):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    # adapter_flag = True

    ######################
    # 4) 生成训练/验证/测试数据
    ######################
    train_time_range_list = []
    while len(train_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in train_time_range_list:
            train_time_range_list.append((t_start, t_end))

    quadruplet = []
    for i, time_range in enumerate(train_time_range_list):
        t_start, t_end = time_range
        print(f"Generating training data {i + 1}/{len(train_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        while len(center_vertices) < num_anchor_samples:
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            if temp_core_number >= 3:
                center_vertices.add(temp_vertex)
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
        for triplet in triplets:
            quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))

        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels)
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map

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
        t_start, t_end = time_range
        print(f"Generating testing data {i + 1}/{len(test_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        while len(center_vertices) < num_anchor_samples:
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            if temp_core_number >= 3:
                center_vertices.add(temp_vertex)
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end)
        for triplet in triplets:
            test_quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))

        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels)
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map

    test_quadruplet = random.choices(test_quadruplet, k=200)
    train_quadruplet = quadruplet
    val_quadruplet, test_quadruplet = train_test_split(test_quadruplet, test_size=0.5, random_state=42)

    train_dataset = MultiSampleQuadrupletDataset(train_quadruplet)
    val_dataset = MultiSampleQuadrupletDataset(val_quadruplet)
    test_dataset = MultiSampleQuadrupletDataset(test_quadruplet)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, collate_fn=quadruplet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)

    ######################
    # 5) 优化器：只更新可训练参数 (Adapter + gating)
    ######################
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scaler = GradScaler()  # 混合精度

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    ######################
    # 6) 训练循环
    ######################
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

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)
            embeddings = model(batched_subgraphs)
            batch_indices = batched_subgraphs.batch

            del batched_subgraphs

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
                subgraph_emb = embeddings[node_indices]

                # 三元组损失 + link_loss
                with autocast():
                    loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                batch_loss += loss

            if len(vertex_maps) > 0:
                batch_loss = batch_loss / len(vertex_maps)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=epoch_loss / (batch_idx + 1))
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", end=' ')

        # 验证模型
        val_loss = validate_model(model, val_loader, k_hop)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model_L1_finetune.pth')  # 保存fine-tune后的最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                model.load_state_dict(torch.load('model_L1_finetune.pth'))
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
    get_timerange_layers()
    read_core_number()
    construct_feature_matrix()
    train_and_evaluation()


if __name__ == "__main__":
    # cProfile.run('main()')
    # torch.cuda.empty_cache()
    main()