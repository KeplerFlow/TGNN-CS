import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_graph_from_txt_pyg(file_path):
    # 读取文件
    data = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target', 'timestamp'])

    # 将源节点和目标节点转换为PyTorch tensor
    edge_index = torch.tensor(data[['source', 'target']].values.T, dtype=torch.long)

    # 添加反向边以使图无向
    edge_index_reversed = edge_index[[1, 0], :]  # 交换行以获得反向边
    edge_index = torch.cat([edge_index, edge_index_reversed], dim=1)  # 合并原始边和反向边
    
    # 创建边的特征，例如时间戳
    edge_attr = torch.tensor(data['timestamp'].values, dtype=torch.long).unsqueeze(1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_attr shape: {edge_attr.shape}")
    # 创建PyTorch Geometric图
    G_pyg = Data(edge_index=edge_index, edge_attr=edge_attr)
    
    return G_pyg

def split_graph_by_time_pyg(graph, days):
    seconds_in_day = 86400
    time_window = days * seconds_in_day

    subgraphs = []
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr.squeeze()
    
    min_time = torch.min(edge_attr).item()
    max_time = torch.max(edge_attr).item()

    start_time = min_time
    while start_time <= max_time:
        end_time = start_time + time_window
        mask = (edge_attr >= start_time) & (edge_attr < end_time)

        if mask.any():
            sub_edge_index = edge_index[:, mask]
            sub_edge_attr = edge_attr[mask].unsqueeze(1)
            
            # 重新映射节点ID
            unique_nodes = torch.unique(sub_edge_index)
            node_idx = torch.zeros(torch.max(sub_edge_index).item() + 1, 
                                 dtype=torch.long)
            node_idx[unique_nodes] = torch.arange(len(unique_nodes))
            
            # 更新边索引使用新的节点ID
            remapped_edge_index = node_idx[sub_edge_index]

            # 创建字典来存储每条边的时间戳列表
            edge_dict = {}
            for i in range(remapped_edge_index.shape[1]):
                edge = tuple(remapped_edge_index[:, i].tolist())
                timestamp = sub_edge_attr[i].item()
                if edge in edge_dict:
                    edge_dict[edge].append(timestamp)
                else:
                    edge_dict[edge] = [timestamp]
            
            # 生成不重复的边和对应的时间戳列表
            # 将字典中的边转换为tensor格式
            unique_edges = torch.tensor([edge for edge in edge_dict.keys()], dtype=torch.long).t()
            timestamp_lists = []
            for edge,times in edge_dict.items():
                timestamp_lists.append(times)

            # 对每个边的时间戳列表进行排序
            timestamp_lists = [sorted(times) for times in timestamp_lists]           

            # 已经是无向图,邻接表已经对称            

            # 创建新的子图，确保边属性被命名为timestamp,并保存节点映射
            sub_graph = Data(
                edge_index=remapped_edge_index,
                timestamp=sub_edge_attr,  # 明确命名为timestamp
                num_nodes=node_idx.max().item() + 1,
                node_remapped_idx=node_idx,
                start_time=start_time,
                end_time=end_time,
                unique_edges=unique_edges,
                timestamp_lists=timestamp_lists
            )

            # 转换为NetworkX图计算k-core
            G_nx = to_networkx(sub_graph, to_undirected=True)
            
            #remove self-loops
            G_nx.remove_edges_from(nx.selfloop_edges(G_nx))
            
            core_dict = nx.core_number(G_nx)

            # 将k-core结果保存到子图中
            sub_graph.k_core = torch.tensor([core_dict[node] for node in range(len(unique_nodes))], dtype=torch.long)

            subgraphs.append(sub_graph)
        
        start_time = end_time
    
    return subgraphs

def validate_and_summarize_subgraphs(subgraphs, original_graph):
    total_edges_in_subgraphs = 0
    print("Validation and Summary of Each Subgraph:")
    
    # 遍历每个子图
    for idx, sg in enumerate(subgraphs):
        num_edges = sg.edge_index.size(1)  # 子图中的边数
        num_nodes = torch.unique(sg.edge_index).numel()  # 子图中的节点数（去重）

        # 计算每个子图中除去多边后的唯一边数
        unique_edges = torch.unique(sg.edge_index, dim=1)  # 根据每列去重，得到唯一的边
        num_unique_edges = unique_edges.size(1)
        
        # 转换为NetworkX图
        G_nx = to_networkx(sg, to_undirected=True)
        
        # 移除自环
        G_nx.remove_edges_from(nx.selfloop_edges(G_nx))
        
        # 计算k-core
        core_dict = nx.core_number(G_nx)
        max_k = max(core_dict.values()) if core_dict else 0
        avg_k = sum(core_dict.values()) / len(core_dict) if core_dict else 0
        
        # 检查节点ID是否连续
        node_ids = list(G_nx.nodes())
        min_node_id = min(node_ids)
        max_node_id = max(node_ids)
        expected_ids = set(range(min_node_id, max_node_id + 1))
        actual_ids = set(node_ids)
        is_continuous = (expected_ids == actual_ids)
        
        print(f"Subgraph {idx + 1}: Nodes = {num_nodes}, Edges = {num_edges}, Unique Edges = {num_unique_edges}, Avg k-core = {avg_k:.2f}, Max k-core = {max_k}, ID Continuous = {is_continuous}")
        total_edges_in_subgraphs += num_edges
    
    # 检查边的总数是否与原图相等
    original_total_edges = original_graph.edge_index.size(1)
    print("\nTotal edges in all subgraphs:", total_edges_in_subgraphs)
    print("Total edges in original graph:", original_total_edges)

    print("Number of subgraphs:", len(subgraphs))
    for idx, sg in enumerate(subgraphs):
        print(f"Subgraph {idx} time range: {sg.start_time} to {sg.end_time}")
        print("Edge timestamps:", sg.timestamp)
        # 验证是否所有的时间戳都在start_time和end_time之间
        if not all(sg.start_time <= ts.item() < sg.end_time for ts in sg.timestamp):
            print(f"Error in timestamps for subgraph {idx}")

    # 输出验证结果
    if total_edges_in_subgraphs == original_total_edges:
        print("Validation Passed: The sum of edges in all subgraphs equals the total edges in the original graph.")
    else:
        print("Validation Failed: The sum of edges in all subgraphs does not equal the total edges in the original graph.")
