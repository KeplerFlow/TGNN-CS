from collections import deque
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
import random
from Tree import *
from MLP_models import *


def get_timerange_layers(num_timestamp, max_range, partition):
    print("Calculating time range layers...")
    time_range_set = set()
    time_range_layers = []
    max_time_range_layers = []
    min_time_range_layers = []
    
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
    
    max_layer_id = layer_id
    print(f"Number of layers: {max_layer_id}")
    
    return time_range_layers, min_time_range_layers, max_time_range_layers, time_range_set, max_layer_id

def read_core_number(dataset_name, num_vertex, time_range_set):

    print("Loading the core number...")
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    time_range_core_number = {time_range: {} for time_range in time_range_set}  # Initialize dictionary for time ranges
    core_number_filename = f'../datasets/{dataset_name}-core_number.txt'

    with open(core_number_filename, 'r') as f:
        num_core_number = int(f.readline().strip())  # Read the first line as num_core_number directly

        for line in f:
            range_part, core_numbers_part = line.split(' ', 1)
            range_start, range_end = map(int, range_part.strip('[]').split(','))
            is_node_range = (range_start, range_end) in time_range_set

            for pair in core_numbers_part.split():
                vertex, core_number = map(int, pair.split(':'))
                if range_start == range_end:
                    vertex_core_numbers[vertex][range_start] = core_number
                if is_node_range:
                    time_range_core_number[(range_start, range_end)][vertex] = core_number

    return num_core_number, vertex_core_numbers, time_range_core_number

def read_core_number_tree(dataset_name, num_vertex, time_range_set):
    print("Loading the core number...")
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    time_range_core_number = defaultdict(dict)
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
    return num_core_number, vertex_core_numbers, time_range_core_number


def construct_feature_matrix(num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device):
    print("Constructing the feature matrix...")
    sequence_features1_matrix = torch.empty(0, 0, 0)
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

    return sequence_features1_matrix.to(device)

def construct_feature_matrix_tree(num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device):
    indices_vertex_of_matrix = torch.zeros(num_vertex, 2, dtype=torch.int64, device=device)
    indices = []
    values = []

    idx = 0
    for v in range(num_vertex):
        start_idx = idx
        for t, neighbors in temporal_graph[v].items():
            core_number = vertex_core_numbers[v].get(t, 0)  # 获取核心数，默认为0
            neighbor_count = len(neighbors)  # 邻居数量
            if core_number > 0:
                # 添加索引 [vertex_index, timestamp, feature_index]
                indices.append([v, t, 0])
                values.append(core_number)
                idx = idx + 1
            if neighbor_count > 0:
                indices.append([v, t, 1])
                values.append(neighbor_count)
                idx = idx + 1
        if start_idx != idx:
            indices_vertex_of_matrix[v][0] = start_idx
            indices_vertex_of_matrix[v][1] = idx - 1
        else:
            indices_vertex_of_matrix[v][0] = -1
            indices_vertex_of_matrix[v][1] = -1

    # 将索引和数值转换为张量
    indices = torch.tensor(indices).T  # 转置为形状 (3, N)
    values = torch.tensor(values, dtype=torch.float32)

    # 对索引排序 方便后续的二分查找
    sorted_order = torch.argsort(indices[0])
    sorted_indices = indices[:, sorted_order]       # 对 indices 排序
    sorted_values = values[sorted_order]            # 对 values 按相同顺序排序

    # 创建稀疏张量
    sequence_features1_matrix = torch.sparse_coo_tensor(
        sorted_indices,
        sorted_values,
        size=(num_vertex, num_timestamp, 2),
        device=device
    )

    sequence_features1_matrix = sequence_features1_matrix.coalesce()

    # 计算稀疏张量实际占用的内存大小
    indices_size = sorted_indices.element_size() * sorted_indices.numel()
    values_size = sorted_values.element_size() * sorted_values.numel()
    total_size = indices_size + values_size
    print(f"feature matrix 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")

    return sequence_features1_matrix

def init_vertex_features(t_start, t_end, vertex_set, feature_dim, anchor, sequence_features1_matrix, time_range_core_number, device):

    vertex_indices = vertex_set
    
    indices = sequence_features1_matrix.indices()
    values = sequence_features1_matrix.values()

    start_idx = torch.searchsorted(indices[0], vertex_indices, side='left')
    end_idx = torch.searchsorted(indices[0], vertex_indices, side='right')

    range_lengths = end_idx - start_idx
    total_indices = range_lengths.sum()

    range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
    flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

    mask_indices = start_idx.repeat_interleave(range_lengths) + flat_indices

    vertex_mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=device)
    vertex_mask[mask_indices] = True

    filtered_indices = indices[:, vertex_mask]
    filtered_values = values[vertex_mask]

    time_mask = (
            (filtered_indices[1] >= t_start) &
            (filtered_indices[1] <= t_end)
    )
    final_indices = filtered_indices[:, time_mask]
    final_values = filtered_values[time_mask]

    vertex_map = torch.zeros(vertex_indices.max() + 1, dtype=torch.long, device=device)
    vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

    final_indices[0] = vertex_map[final_indices[0]]
    final_indices[1] -= t_start

    result_size = (
        len(vertex_indices), t_end - t_start + 1, sequence_features1_matrix.size(2)
    )
    result_sparse_tensor = torch.sparse_coo_tensor(
        final_indices, final_values, size=result_size
    )
    degree_tensor = result_sparse_tensor.to_dense()[:, :, 1]
    degree_tensor.to(device)

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

def get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight, time_range_core_number, subgraph_k_hop_cache):

    cache_key = (center_vertex, k, t_start, t_end)
    if cache_key in subgraph_k_hop_cache:
      return subgraph_k_hop_cache[cache_key]
    
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
            # print(f"Average core number: {average_core_number}")
            break
        if hop > current_hop:
            average_core_number = total_core_number / (len(subgraph_result) ** tau)
            # print(f"Average core number: {average_core_number}")
            current_hop = hop
            if average_core_number > best_avg_core_number:
                best_avg_core_number = average_core_number
            else:
                break
        subgraph_result.add(top_vertex)
        if len(subgraph_result) > 8000:
            break
        total_core_number += neighbor_core_number
        if top_vertex in filtered_temporal_graph:
           for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
               # if neighbor not in visited and neighbor_core_number >= core_number_condition:
               if neighbor not in visited:
                 queue.append((neighbor, neighbor_core_number, hop + 1))
                 visited.add(neighbor)
    subgraph_k_hop_cache[cache_key] = subgraph_result
    return subgraph_result

def compute_modularity(subgraph, filtered_temporal_graph, total_edge_weight):
    
    subgraph = set(subgraph)
    internal_weights = 0  # 子图内部边的权重和
    total_weights = 0  # 所有边的权重和

    # 1. 统计边权重和节点强度
    for vertex in subgraph:
        if vertex in filtered_temporal_graph:
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

def build_tree(num_timestamp, time_range_core_number, num_vertex, temporal_graph, time_range_layers):
    node_stack = []
    root_node = TreeNode((0, num_timestamp-1), 0)
    max_layer_id = 0 # Initialize max_layer_id locally
    node_stack.append(root_node)
    while len(node_stack) > 0:
        current_node = node_stack.pop()
        current_node.vertex_core_number = time_range_core_number[(current_node.time_start, current_node.time_end)]
        current_node.vertex_degree = defaultdict(int)
        if current_node.layer_id != 0:
            for v in range(num_vertex):
                neighbors_set = set()
                for t, neighbors in temporal_graph[v].items():
                    if current_node.time_start <= t <= current_node.time_end:
                        neighbors_set.update(neighbors)
                if len(neighbors_set) > 0:
                    current_node.vertex_degree[v] = len(neighbors_set)

        if current_node.layer_id < len(time_range_layers) - 1:
            for i in range(len(time_range_layers[current_node.layer_id + 1])):
                temp_time_start = time_range_layers[current_node.layer_id +1][i][0]
                temp_time_end = time_range_layers[current_node.layer_id + 1][i][1]
                if temp_time_start >= current_node.time_start and temp_time_end <= current_node.time_end:
                    child_node = TreeNode((temp_time_start, temp_time_end), current_node.layer_id + 1)
                    current_node.add_child(child_node)
                    node_stack.append(child_node)
    max_layer_id = len(time_range_layers) - 1
    return root_node, max_layer_id

def tree_query(time_start, time_end, num_timestamp, root, max_layer_id):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None

    node = root
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                move_to_next = True
                break
        if not move_to_next:
            break
    return node

def get_node_path(time_start, time_end, num_timestamp, root, max_layer_id):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None
    path = [root]
    node = root
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                path.append(node)
                move_to_next = True
                break
        if not move_to_next:
            break
    return path

def model_output_for_path(time_start, time_end, vertex_set, sequence_features, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return torch.zeros(len(vertex_set), 1, device=device)
    sequence_features = sequence_features.to(device)
    node = root
    
    path = [node]

    print(path)

    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                path.append(node)
                move_to_next = True
                break
        if not move_to_next:
            break
    if len(path) == 1:
        return torch.zeros(len(vertex_set), 1, device=device)

    # 计算模型输出
    path.pop()
    output = torch.zeros(len(vertex_set), 1, dtype=torch.float32, device=device)
    sequence_input1 = torch.zeros(len(vertex_set), max_time_range_layers[0], 2, device=device)
    # sequence_features = sequence_features.to_dense()  # 添加这一行以确保是稠密张量
    sequence_input1[:, 0:sequence_features.shape[1], :] = sequence_features
    for node in path:
        max_length1 = max_time_range_layers[node.layer_id + 1] * 2
        max_length2 = partition
        # 构造模型输入
        sequence_input1 = sequence_input1[:, :max_length1, :]
        sequence_input2 = torch.zeros(len(vertex_set), max_length2, 2, device=device)

        single_value = output

        model = node.model
        with torch.no_grad():
            output = model(sequence_input1, sequence_input2, single_value)
            if output.dim() == 0:
                output = output.reshape(len(vertex_set), 1)
    return output

def load_models(depth_id, time_range_layers, max_layer_id, device, dataset_name, 
                max_time_range_layers, partition,root,num_timestamp):
    model_time_range_layers = [[] for _ in range(len(time_range_layers))]
    print("Loading models...")
    for layer_id in range(0, depth_id + 1):
        for i in range(len(time_range_layers[layer_id])):
            print(f"{i+1}/{len(time_range_layers[layer_id])}")
            time_start = time_range_layers[layer_id][i][0]
            time_end = time_range_layers[layer_id][i][1]
            model = None
            if layer_id != max_layer_id:
                model = MLPNonleaf(2, max_time_range_layers[layer_id + 1] * 2, partition, 64).to(device)
                model_path = f'models/{dataset_name}/model_{layer_id}_{i}.pth'
            else:
                model = MLP(2, max_time_range_layers[max_layer_id], 64).to(device)
                model_path = f'models/{dataset_name}/model_{layer_id}_{i}.pth'

            try:
                model.load_state_dict(torch.load(model_path))
                model.eval()
            except FileNotFoundError:
                print(f"Warning: Model file not found: {model_path}")
                continue  # 或者根据需要进行其他处理

            # 使用传入的 tree_query_func
            node = tree_query(time_start, time_end, num_timestamp, root, max_layer_id)
            if node is None:
                print("Error: node not found.")
            else:
                node.set_model(model)

class MultiSampleQuadrupletDataset():
    def __init__(self, quadruplets):
        self.quadruplets = quadruplets

    def __len__(self):
        return len(self.quadruplets)

    def __getitem__(self, idx):
        anchor, positives, negatives, time_range = self.quadruplets[idx]
        return anchor, list(positives), list(negatives), time_range

