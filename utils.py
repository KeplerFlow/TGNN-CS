import torch
import networkx as nx
import numpy as np
import random
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def initialize_features(data, dimensions=128):
    # 找到最大的k-core值，用于归一化
    max_coreness = data.k_core.max().item() if data.k_core.numel() > 0 else 1

    # 初始化节点特征
    node_features = torch.zeros((data.num_nodes, dimensions))
    
    # 为每个节点生成归一化的coreness特征并广播到128维
    for node in range(data.num_nodes):
        normalized_coreness = data.k_core[node].item() / max_coreness
        coreness_vector = torch.full((dimensions,), normalized_coreness, dtype=torch.float32)
        node_features[node] = coreness_vector
    
    return node_features

def sample_graph_nodes(sub_graph, proportions=(0.5, 0.3, 0.2), total_proportion=0.1):
    # 获取节点的k-core值
    k_cores = sub_graph.k_core.numpy()  # 假设k_core是存储在子图中的Tensor

    # 分级节点
    thresholds = np.percentile(k_cores, [50, 80])
    high_core_nodes = np.where(k_cores >= thresholds[1])[0]
    medium_core_nodes = np.where((k_cores < thresholds[1]) & (k_cores >= thresholds[0]))[0]
    low_core_nodes = np.where(k_cores < thresholds[0])[0]

    # 按比例采样
    num_high = int(len(high_core_nodes) * proportions[0])
    num_medium = int(len(medium_core_nodes) * proportions[1])
    num_low = int(len(low_core_nodes) * proportions[2])

    sampled_nodes = np.concatenate([
        np.random.choice(high_core_nodes, num_high, replace=False),
        np.random.choice(medium_core_nodes, num_medium, replace=False),
        np.random.choice(low_core_nodes, num_low, replace=False)
    ])

    # 进一步采样以确保总数为子图节点的10%
    total_sample_size = int(len(sub_graph.k_core) * total_proportion)
    if len(sampled_nodes) > total_sample_size:
        sampled_nodes = np.random.choice(sampled_nodes, total_sample_size, replace=False)

    return sampled_nodes

def generate_negative_samples(subgraph, positive_indices, num_negatives):
    all_indices = set(range(subgraph.num_nodes))
    positive_indices_set = set(positive_indices.tolist())
    negative_candidates = list(all_indices - positive_indices_set)
    
    if not negative_candidates:
        raise ValueError("No negative candidates available for sampling.")
    
    if len(negative_candidates) < num_negatives:
        # 如果可用的负样本不足，则重复一些选择
        return torch.tensor(random.choices(negative_candidates, k=num_negatives))
    else:
        return torch.tensor(random.sample(negative_candidates, k=num_negatives))

def calculate_temporal_modularity_for_jumps(data, sampled_nodes, max_jumps):
    if not torch.all((data.timestamp.squeeze() >= data.start_time) & (data.timestamp.squeeze() < data.end_time)):
        print("Error: Some timestamps are not within the specified time window.")
    
    # Convert PyG Data object to an undirected NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Add timestamp information to edges in the NetworkX graph
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        timestamp = data.timestamp[i].item()
        G[src][dst]['timestamp'] = timestamp
        G[dst][src]['timestamp'] = timestamp  # This is redundant in an undirected graph
    
    # Create a mapping from edges to their timestamps
    edge_timestamps = {(u, v): G[u][v]['timestamp'] for u, v in G.edges()}
    
    # Initialize a dictionary to store the optimal number of jumps and modularity for each node
    node_optimal_jumps = {node: (0, -np.inf) for node in sampled_nodes}

    # Iterate over all possible numbers of jumps
    for jumps in range(1, max_jumps + 1):
        for node in sampled_nodes:
            if node not in G:
                continue
            
            # 使用 single_source_shortest_path_length 获取指定跳数内的所有节点
            hop_dict = nx.single_source_shortest_path_length(G, node, cutoff=jumps)
            visited = set(hop_dict.keys())
            
            # Get the subgraph for current node within the specified jumps
            subgraph = G.subgraph(visited)
            nodes = list(visited)
            sub_edge_timestamps = {
                (u, v): edge_timestamps.get((u, v), edge_timestamps.get((v, u), None))
                for u, v in subgraph.edges()
                if (u, v) in edge_timestamps or (v, u) in edge_timestamps
            }

            # Calculate temporal modularity for the subgraph
            modularity = temporal_modularity(G, nodes, sub_edge_timestamps, data.start_time)
            
            # Update the maximum modularity and its corresponding number of jumps
            if modularity > node_optimal_jumps[node][1]:
                node_optimal_jumps[node] = (jumps, modularity)

    return node_optimal_jumps

def temporal_modularity(G, C, edge_timestamps, t_s):
    # 创建一个新的图并确保没有自环
    G_copy = G.copy()
    G_copy.remove_edges_from(nx.selfloop_edges(G_copy))
    
    # 确保所有节点都在图中
    C = [v for v in C if v in G_copy]
    if not C:
        return 0.0
        
    # 获取子图中的边
    E_C = []
    for u in C:
        for v in C:
            if u < v and G_copy.has_edge(u, v):  # 只检查一个方向避免重复
                if (u, v) in edge_timestamps or (v, u) in edge_timestamps:
                    E_C.append((u, v))
    
    if not E_C:  # 如果没有边，返回0
        return 0.0
    
    try:
        core_sum = sum(nx.core_number(G_copy)[v] for v in C)
        d_C = sum(G_copy.degree(v) for v in C)
        total_edges = G_copy.number_of_edges()
        if total_edges == 0:  # 避免除以0
            return 0.0
            
        modularity = (core_sum / 2*len(C)) * (2 * len(E_C) - (d_C**2 / (2 * total_edges)))

        # 计算时间因子
        time_diffs = []
        for e in E_C:
            if e in edge_timestamps:
                time_diff = edge_timestamps[e] - t_s
                time_diffs.append(max(time_diff, 0))  # 确保时间差非负
        
        if not time_diffs:  # 如果没有有效的时间差
            return 0.0
            
        temporal_factor = len(E_C) / (sum(time_diffs) + 1e-10)  # 添加小量避免除以0
        
        return modularity * temporal_factor
        
    except Exception as e:
        print(f"Error in temporal_modularity: {e}")
        return 0.0

def get_optimal_subgraphs(data, node_optimal_jumps):
    # 转换为NetworkX图以便使用其API
    G = to_networkx(data, to_undirected=True)
    
    # 移除自环
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 存储每个节点的最优子图
    optimal_subgraphs = {}
    
    # 对每个节点获取其最优跳数对应的子图
    for node, (optimal_jumps, _) in node_optimal_jumps.items():
        if node not in G:
            optimal_subgraphs[node] = {node}  # 如果节点不在图中，只包含自身
            continue
            
        # 使用最短路径获取指定跳数内的所有节点
        hop_dict = nx.single_source_shortest_path_length(G, node, cutoff=optimal_jumps)
        
        # 将字典的键（即可达节点）转换为集合
        reachable_nodes = set(hop_dict.keys())
        
        # 存储结果
        optimal_subgraphs[node] = reachable_nodes
    
    return optimal_subgraphs

def compute_core_numbers(G):
    # 将 PyG 图转换为 NetworkX 图
    edge_index = G.edge_index.cpu().numpy()
    nx_graph = nx.Graph()
    
    # 添加边到 NetworkX 图中，同时跳过自环
    edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src != dst:  # 跳过自环
            edges.append((src, dst))
    
    nx_graph.add_edges_from(edges)
    
    # 确保所有节点都在图中（包括孤立节点）
    max_node_idx = max(max(edge_index[0]), max(edge_index[1]))
    nx_graph.add_nodes_from(range(max_node_idx + 1))
    
    # 计算 core numbers
    core_numbers = nx.core_number(nx_graph)
    
    # 转换为张量
    core_tensor = torch.zeros(max_node_idx + 1, dtype=torch.float)
    
    for node, core in core_numbers.items():
        core_tensor[node] = float(core)
    
    # 如果使用 GPU，将张量移到相应设备
    if G.edge_index.is_cuda:
        core_tensor = core_tensor.cuda()
    
    return core_tensor