import torch
import networkx as nx
import numpy as np
import random
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch.nn.functional as F 

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

def community_search(z, query_idx, subgraph, t_s, t_e, temporal_encoder, similarity_threshold=0.1, time_similarity_threshold=0.1):
    # 获取子图中的节点数
    num_nodes = subgraph.num_nodes

    # 获取查询节点的嵌入
    z_query = z[query_idx]  # [embedding_dim]

    # 计算所有节点与查询节点的相似度（余弦相似度）
    cos_sim = F.cosine_similarity(z_query.unsqueeze(0), z, dim=1).squeeze()  # [num_nodes]

    # 一.筛选相似度超过阈值的节点
    similar_nodes = set(torch.where(cos_sim >= similarity_threshold)[0].tolist())

    # 二.选择相似度较大的前20%的节点
    # num_top_nodes = int(len(cos_sim) * 0.2)
    # top_indices = torch.topk(cos_sim, num_top_nodes).indices
    # similar_nodes = set(top_indices.tolist())

    # 获取在时间窗口内的边
    edge_index = subgraph.edge_index  # [2, num_edges]
    timestamps = subgraph.timestamp.squeeze()  # [num_edges]

    # 筛选时间窗口内的边
    time_mask = (timestamps >= t_s) & (timestamps <= t_e)
    edge_index_time = edge_index[:, time_mask]
    edge_times = timestamps[time_mask]

    # 构建时间窗口内的邻接表
    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in edge_index_time.t().tolist():
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # 如果是无向图

    # 通过 BFS 扩展，找到与查询节点连接的节点
    visited = set()
    queue = [query_idx.item()]
    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            # 只添加相似节点
            neighbors = [n for n in adj_list[current] if n in similar_nodes]
            queue.extend(neighbors)

    community_nodes = list(visited)
    time_similar_nodes = set()

    # 获取社区节点的时间嵌入
    node_time_encodings = z[community_nodes][:, -64:]  # [num_community_nodes, 64]

    # 计算时间窗口内的时间编码
    phi_t_window = time_encoding(temporal_encoder.omega, temporal_encoder.phi, edge_times)  # [num_edges, 64]

    # 计算相似度矩阵
    time_sims = F.cosine_similarity(node_time_encodings.unsqueeze(1), phi_t_window.unsqueeze(0), dim=2)  # [num_community_nodes, num_edges]

    # 一.筛选超过阈值的节点对
    time_similar_pairs = (time_sims >= time_similarity_threshold).nonzero(as_tuple=False)  # [num_pairs, 2]

    # 二.筛选相似度较大的前50%的节点对
    # num_top_pairs = int(time_sims.numel() * 0.5)
    # top_values, top_indices = torch.topk(time_sims.view(-1), num_top_pairs)
    # time_similar_pairs = torch.stack([top_indices // time_sims.size(1), top_indices % time_sims.size(1)], dim=1)

    # 获取对应的节点索引
    for i, j in time_similar_pairs.tolist():
        src, dst = edge_index_time[:, j].tolist()
        time_similar_nodes.add(src)
        time_similar_nodes.add(dst)

    return time_similar_nodes

def time_encoding(omega, phi, timestamps):
    t = timestamps.unsqueeze(1)  # [num_timestamps, 1]
    linear_term = omega[0] * t + phi[0]  # [num_timestamps, 1]
    sin_terms = torch.sin(omega[1:] * t + phi[1:])  # [num_timestamps, num_features - 1]
    phi_t = torch.cat([linear_term, sin_terms], dim=1)  # [num_timestamps, num_features]
    phi_t = torch.tanh(phi_t)
    return phi_t

def evaluate_community(subgraph, community_nodes, t_s, t_e):

    # 将社区节点集合转换为集合类型，方便后续操作
    S = set(community_nodes)
    V = set(range(subgraph.num_nodes))

    # 获取边索引和时间戳
    edge_index = subgraph.edge_index  # [2, num_edges]
    timestamps = subgraph.timestamp.squeeze()  # [num_edges]

    # 初始化计数器
    internal_edges = 0       # 时间边数量
    cut_edges = 0            # 跨越社区边界的时间边数量
    T_S_set = set()          # 社区内部时间戳集合
    T_vol_S = 0              # 社区内部的时间边数量（考虑时间）
    T_vol_V_minus_S = 0      # 社区外部的时间边数量（考虑时间）
    seen_edges = set()        # 去重集合
    cut_seen_edges = set()    # 跨越社区边界的时间边集合

    for idx in range(edge_index.size(1)):
        u = edge_index[0, idx].item()
        v = edge_index[1, idx].item()
        t = timestamps[idx].item()
        internal_edges+=1
        if u in S and v in S:
            edge = (min(u, v), max(u, v))  # 确保边的顺序一致以便去重
            if edge not in seen_edges:
                seen_edges.add(edge)  # 添加到去重集合
            T_S_set.add(t)
            T_vol_S += 1

        if u in S and v not in S:
            cut_edges += 1
        if u in S :
            T_vol_V_minus_S+=1

    # 计算社区节点对的最大可能连接数
    num_S = len(S)
    edges_num = len(seen_edges)
    # 计算时间密度 TD(S) large better
    TD_S = (2 * T_vol_S ) / (num_S * (num_S - 1) * len(T_S_set)+0.00001)

    cut_edges_num = len(cut_seen_edges)

    # 计算时间割 TC(S) small better
    if T_vol_S > 0 and T_vol_V_minus_S > 0:
        denominator = min(internal_edges-T_vol_V_minus_S, T_vol_V_minus_S)
        TC_S = cut_edges / denominator
    else:
        TC_S = 0  # 避免除以零

    return TD_S, TC_S
