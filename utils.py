import torch
import networkx as nx
import numpy as np
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

global_node_features = {}  # {original_node_id: feature_vector}

def initialize_features(data, dimensions=128, use_cached=True):
    # 获取原始节点ID到重映射ID的映射关系
    remapped_idx = data.node_remapped_idx
    
    # 初始化特征矩阵
    node_features = torch.zeros((data.num_nodes, dimensions))
    
    # 转换为NetworkX图用于k-core计算
    G_nx = to_networkx(data, to_undirected=True)
    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))
    coreness_dict = nx.core_number(G_nx)
    max_coreness = max(coreness_dict.values()) if coreness_dict else 1
    
    # 遍历每个节点
    for new_idx in range(data.num_nodes):
        # 获取原始节点ID
        orig_idx = torch.where(remapped_idx == new_idx)[0]
        if len(orig_idx) > 0:
            orig_idx = orig_idx[0].item()
            
            # 检查是否有缓存的特征
            if use_cached and orig_idx in global_node_features:
                node_features[new_idx] = global_node_features[orig_idx]
            else:
                # 生成新的特征
                normalized_coreness = coreness_dict[new_idx] / max_coreness
                feature_vector = torch.full((dimensions,), normalized_coreness, dtype=torch.float32)
                node_features[new_idx] = feature_vector
                # 缓存特征
                global_node_features[orig_idx] = feature_vector
    
    return node_features

def update_global_features(data, features):
    remapped_idx = data.node_remapped_idx
    for new_idx in range(data.num_nodes):
        orig_idx = torch.where(remapped_idx == new_idx)[0]
        if len(orig_idx) > 0:
            orig_idx = orig_idx[0].item()
            global_node_features[orig_idx] = features[new_idx].clone()

def sample_graph_nodes(sub_graph, proportions=(0.5, 0.3, 0.2)):
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

    return sampled_nodes


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
            
        modularity = (core_sum / 1) * (2 * len(E_C) - (d_C**2 / (2 * total_edges)))

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

def encode_temporal_features(timestamps, num_features=16):
    # 生成随机频率和相位偏移
    omega = torch.randn(num_features)  # [num_features]
    phi = torch.randn(num_features)    # [num_features]
    
    # 广播时间戳
    t = timestamps.unsqueeze(1)  # [num_edges, 1]
    
    # 计算随机傅里叶特征
    linear_term = omega[0] * t + phi[0]
    sin_terms = torch.sin(omega[1:] * t + phi[1:])
    
    # 拼接线性项和正弦项
    temporal_features = torch.cat([linear_term, sin_terms], dim=1)  # [num_edges, num_features]
    
    return temporal_features

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
    """
    计算图中所有节点的 core number，处理自环问题
    
    参数:
    G: PyG Data对象
    
    返回:
    core_numbers: tensor, 每个节点的 core number
    """
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