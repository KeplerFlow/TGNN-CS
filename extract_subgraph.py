import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from collections import defaultdict, deque
import random
import heapq
import numpy as np

from utils import *
from index import *

def get_samples(center_vertex, k, t_start, t_end, filtered_temporal_graph, vertex_connect_scores, total_edge_weight, time_range_core_number, subgraph_k_hop_cache):

    candidates_neighbors = get_candidate_neighbors(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight, time_range_core_number, subgraph_k_hop_cache)
    # print(f"K-hop Neighbors: {len(candidates_neighbors)}")
    
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
        if top_vertex in filtered_temporal_graph:
            for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
                if neighbor not in visited and neighbor in candidates_neighbors:
                    heapq.heappush(queue, (-vertex_connect_scores[neighbor], neighbor))
                    visited.add(neighbor)

    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.3:
        left_vertex = positive_neighbors_list.pop(0)
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)
    hard_negative_neighbors = hard_negative_neighbors - positive_neighbors - {center_vertex}
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(candidates_neighbors)
    
    return positive_neighbors, hard_negative_neighbors, candidates_neighbors 

def generate_time_range_link_samples(k_hop_samples, filtered_temporal_graph, num_vertex):
    link_samples_dict = defaultdict(list)
    min_neighbors = 3
    for vertex in k_hop_samples:
         if vertex in filtered_temporal_graph:
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

def get_samples_index(center_vertex, k, t_start, t_end, filtered_temporal_graph, vertex_connect_scores, total_edge_weight, time_range_core_number, subgraph_k_hop_cache,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root):

    candidates_neighbors = get_candidate_neighbors_index(center_vertex, k, t_start, t_end, filtered_temporal_graph, total_edge_weight, time_range_core_number, subgraph_k_hop_cache,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)
    # print(f"K-hop Neighbors: {len(candidates_neighbors)}")
    
    positive_neighbors_list = []
    positive_neighbors = set()
    hard_negative_neighbors = set()
    visited = set()
    visited.add(center_vertex)
    queue = []  # 优先队列
    heapq.heappush(queue, (0, center_vertex))
    query_vertex_core_number = time_range_core_number[(t_start, t_end)].get(center_vertex, 0)
    # query_vertex_core_number = model_out_put_for_any_range_vertex_set([center_vertex],t_start,t_end,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)

    while queue:
        _, top_vertex = heapq.heappop(queue)
        v_core_number = time_range_core_number[(t_start, t_end)].get(top_vertex, 0)
        # v_core_number = model_out_put_for_any_range_vertex_set([top_vertex],t_start,t_end,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)

        if v_core_number >= query_vertex_core_number:
            hard_negative_neighbors.add(top_vertex)
        positive_neighbors_list.append(top_vertex)
        if top_vertex in filtered_temporal_graph:
            for (neighbor, edge_count, neighbor_core_number) in filtered_temporal_graph[top_vertex]:
                if neighbor not in visited and neighbor in candidates_neighbors:
                    heapq.heappush(queue, (-vertex_connect_scores[neighbor], neighbor))
                    visited.add(neighbor)

    while len(positive_neighbors) < len(hard_negative_neighbors) * 0.3:
        left_vertex = positive_neighbors_list.pop(0)
        if left_vertex != center_vertex:
            positive_neighbors.add(left_vertex)
    hard_negative_neighbors = hard_negative_neighbors - positive_neighbors - {center_vertex}
    subgraph_k_hop_cache[(center_vertex, (t_start, t_end))] = sorted(candidates_neighbors)
    
    return positive_neighbors, hard_negative_neighbors, candidates_neighbors 

def generate_triplets(center_vertices, k_hop, t_start, t_end, num_vertex, temporal_graph, time_range_core_number, time_range_link_samples_cache, subgraph_k_hop_cache):
    triplets = []
    idx = 0

    # generate filtered subgraph
    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0
    for vertex in range(num_vertex):
        neighbor_time_edge_count = defaultdict(int)
        total_time_edge_count = 0
        if vertex in temporal_graph:
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
                                                                             filtered_subgraph, vertex_connect_scores, total_edge_weight, time_range_core_number, subgraph_k_hop_cache)
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
        link_samples = generate_time_range_link_samples(k_hop_samples, filtered_subgraph, num_vertex)
        time_range_link_samples_cache[(t_start, t_end)][anchor] = link_samples

    return triplets

def generate_triplets_index(center_vertices, k_hop, t_start, t_end, num_vertex, temporal_graph, time_range_core_number, time_range_link_samples_cache, subgraph_k_hop_cache,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root):
    triplets = []
    idx = 0

    # generate filtered subgraph
    filtered_subgraph = {}
    vertex_connect_scores = {}
    total_edge_weight = 0
    for vertex in range(num_vertex):
        neighbor_time_edge_count = defaultdict(int)
        total_time_edge_count = 0
        if vertex in temporal_graph:
           for t, neighbors in temporal_graph[vertex].items():
              if t_start <= t <= t_end:
                for neighbor in neighbors:
                    neighbor_time_edge_count[neighbor] += 1
                    total_time_edge_count += 1
        neighbors_list = []
        for neighbor, count in neighbor_time_edge_count.items():
            core_number = time_range_core_number[(t_start, t_end)].get(neighbor, 0)
            # core_number = model_out_put_for_any_range_vertex_set([neighbor],t_start,t_end,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)

            neighbors_list.append((neighbor, count, core_number))
            if vertex < neighbor:
                total_edge_weight += count
        filtered_subgraph[vertex] = neighbors_list
        vertex_core_number = time_range_core_number[(t_start, t_end)].get(vertex, 0)
        # vertex_core_number = model_out_put_for_any_range_vertex_set([vertex],t_start,t_end,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)

        vertex_connect_scores[vertex] = vertex_core_number * total_time_edge_count / len(neighbors_list) if len(
            neighbors_list) != 0 else 0

    # 原图编号
    for anchor in center_vertices:
        if idx % 100 == 0:
            print(f"{idx}/{len(center_vertices)}")
        idx = idx + 1
        # 找到 k 跳邻居作为正样本
        positive_samples, hard_negative_samples, k_hop_samples = get_samples_index(anchor, k_hop, t_start, t_end,
                                                                             filtered_subgraph, vertex_connect_scores, total_edge_weight, time_range_core_number, subgraph_k_hop_cache,max_layer_id,max_time_range_layers,device,sequence_features1_matrix,partition,num_timestamp, root)
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
        link_samples = generate_time_range_link_samples(k_hop_samples, filtered_subgraph, num_vertex)
        time_range_link_samples_cache[(t_start, t_end)][anchor] = link_samples

    return triplets

def extract_subgraph_for_time_range(anchors, t_start, t_end, feature_dim, temporal_graph_pyg, num_vertex, edge_dim, device, vertex_features, time_range_core_number,subgraph_k_hop_cache):

    neighbors = set()
    
    for anchor in anchors:
        neighbors |= set(subgraph_k_hop_cache[(anchor, (t_start, t_end))])
    neighbors = torch.tensor(sorted(neighbors), device=device)

    print("don1.5")
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
        x=init_vertex_features(t_start, t_end, neighbors, feature_dim, -1, vertex_features, time_range_core_number, device),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device
    )

    return subgraph_pyg, vertex_map

def extract_subgraph_for_anchor(anchor, t_start, t_end, subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex, device):
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

def extract_subgraph(anchor, t_start, t_end, k, feature_dim, temporal_graph, temporal_graph_pyg, num_vertex, edge_dim, vertex_features, time_range_core_number,device):
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

    neighbors_k_hop = torch.tensor(sorted(list(neighbors_k_hop)), device=device, dtype=torch.long)
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
    sub_edge_attr = temporal_graph_pyg.edge_attr.coalesce().to_dense()[mask]
    # sub_edge_attr = temporal_graph_pyg.edge_attr[mask]
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
        x=init_vertex_features(t_start, t_end, neighbors_k_hop, feature_dim, -1, vertex_features, time_range_core_number, device),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device
    )
    return subgraph_pyg, vertex_map, neighbors_k_hop

def extract_subgraph_multiple_query(query_vertex, t_start, t_end, k, feature_dim, temporal_graph,
                                                                        temporal_graph_pyg, num_vertex, edge_dim,
                                                                        sequence_features1_matrix,
                                                                        time_range_core_number, max_layer_id,max_time_range_layers,device,partition,num_timestamp, root):
    # 确保 query_vertex 是张量类型
    if not isinstance(query_vertex, torch.Tensor):
        query_vertex = torch.tensor(query_vertex, device=device)
    
    # 初始化访问集合，将所有查询节点加入
    visited = set(query_vertex.cpu().numpy())
    
    # 初始化队列，将所有查询节点加入
    queue = deque([(v.item(), 0) for v in query_vertex])  # (节点, 当前跳数)
    neighbors_k_hop = set(query_vertex.cpu().numpy())
    
    while queue:
        top_vertex, depth = queue.popleft()
        if depth > k:
            continue
            
        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                        visited.add(neighbor)
                        neighbors_k_hop.add(neighbor)
    
    neighbors_k_hop = torch.tensor(sorted(list(neighbors_k_hop)), device=device, dtype=torch.long)
    
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
    
    sub_edge_index = temporal_graph_pyg.edge_index[:, mask]
    
    # 将边索引映射到子图编号
    sub_edge_index = vertex_map[sub_edge_index]
    
    # 提取边属性
    sub_edge_attr = temporal_graph_pyg.edge_attr.coalesce().to_dense()[mask]
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
        x=init_vertex_features_index(t_start, t_end, neighbors_k_hop, feature_dim, -1, sequence_features1_matrix, time_range_core_number,max_layer_id,max_time_range_layers,device,partition,num_timestamp, root),
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        device=device
    )
    
    return subgraph_pyg, vertex_map, neighbors_k_hop