import torch
from collections import defaultdict
from pympler import asizeof
from torch_geometric.data import Data

def read_temporal_graph(dataset_name, device):
    print("Loading the graph...")
    time_edge = defaultdict(set)
    temporal_graph = defaultdict(lambda: defaultdict(list))
    num_vertex = 0
    num_edge = 0
    num_timestamp = 0
    
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

    return num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, temporal_graph_pyg