from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_loader import *
from utils import *
from model import *
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import os
import time
import networkx as nx
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd 
import random

file_path = "../TCS/data/CollegeMsg.txt"
graph =  read_graph_from_txt_pyg(file_path)
print(f"successfully read graph")

window_size = 5

subgraphs = split_graph_by_time_pyg(graph,window_size)  # x days window
print(f"successfully split graph")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalGNN(
    in_channels=128,
    hidden_channels=64,
    temporal_features_dim=64,  # 你可以根据需要调整
    num_layers=3
).to(device)
# model.load_state_dict(torch.load("temporal_gnn_model.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = TemporalContrastiveLoss(temporal_encoder=model.temporal_encoder)
num_epochs = 50

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model = model.to(device)
model.train()
print(f"begin to train")
for subgraph in subgraphs:
    subgraph.x = initialize_features(subgraph)
    subgraph.x = subgraph.x.to(device)
    subgraph.edge_index = subgraph.edge_index.to(device)
    subgraph.timestamp = subgraph.timestamp.to(device)
    subgraph.unique_edges = subgraph.unique_edges.to(device)
    subgraph.timestamp_lists = [torch.tensor(times, dtype=torch.float32).to(device) for times in subgraph.timestamp_lists]

    sampled_nodes = sample_graph_nodes(subgraph)
    if len(sampled_nodes) == 0:
        continue 
    node_optimal_jumps = calculate_temporal_modularity_for_jumps(subgraph, sampled_nodes, 3)
    optimal_subgraphs = get_optimal_subgraphs(subgraph, node_optimal_jumps)

    best_community = None
    best_TD = -1

    for epoch in range(num_epochs):
        total_loss = 0

        query_idx = torch.tensor([random.choice(sampled_nodes)])
        t_s = subgraph.start_time
        t_e = subgraph.end_time 
        time_window = t_e - t_s
        mid_point = t_s + time_window // 2
        
        t_s = mid_point - time_window // 4
        t_e = mid_point + time_window // 4
        t_s = subgraph.start_time
        t_e = subgraph.end_time

        z = model(subgraph.x, subgraph.edge_index, subgraph.timestamp.squeeze(), subgraph.timestamp.squeeze() - subgraph.start_time, subgraph.unique_edges, subgraph.timestamp_lists)

        communities = community_search(z, query_idx, subgraph, t_s, t_e, model.temporal_encoder)

        if len(communities) > 0:
            neighbor_idx = torch.tensor(np.array(list(communities)), dtype=torch.long)
        else:
            neighbor_idx = torch.tensor(list(optimal_subgraphs[query_idx.item()]))

        TD_S, TC_S = evaluate_community(subgraph, communities, t_s, t_e)
        
        if len(communities) > 10 and TD_S > best_TD:
            best_community = communities
            best_TD = TD_S

        query_idx = query_idx.to(device)
        neighbor_idx = neighbor_idx.to(device)
        edge_times = subgraph.timestamp.squeeze().to(device)

        loss = criterion(
            z=z,
            query_idx=query_idx,
            neighbor_idx=torch.tensor(list(optimal_subgraphs[query_idx.item()])).to(device),
            edge_times=subgraph.timestamp.squeeze(),
            current_time=subgraph.end_time,
            t_s=t_s,
            t_e=t_e,
            G=subgraph
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    if best_community is not None:
        print(f"最佳社区节点数: {len(best_community)}, 时间密度: {best_TD:.4f}")
        print(f"子图节点数为:{subgraph.x.shape[0]}")
        print(f"时间窗口: {t_s} - {t_e}")

# 在训练结束后保存模型
torch.save(model.state_dict(), "temporal_gnn_model.pth")

test_path = "../TCS/data/sx-mathoverflow.txt"
test_graph = read_graph_from_txt_pyg(test_path)
print(f"successfully read test graph")

test_window_size = 21

test_subgraphs = split_graph_by_time_pyg(test_graph,test_window_size)

model.eval()
total_loss = 0
with torch.no_grad():  # 禁用梯度计算
    for subgraph in test_subgraphs:
        subgraph.x = initialize_features(subgraph)
        sampled_nodes = sample_graph_nodes(subgraph)
        node_optimal_jumps = calculate_temporal_modularity_for_jumps(subgraph, sampled_nodes, 3)
        optimal_subgraphs = get_optimal_subgraphs(subgraph, node_optimal_jumps)
        
        for epoch in range(num_epochs):
            
            # 随机选择一个查询节点
            query_idx = torch.tensor([random.choice(sampled_nodes)])
            # 计算时间窗口的中点
            t_s = subgraph.start_time
            t_e = subgraph.end_time 
            time_window = t_e - t_s
            mid_point = t_s + time_window // 2
            
            # 更新时间窗口为一半大小
            t_s = mid_point - time_window // 4  # 从中点往前1/4窗口
            t_e = mid_point + time_window // 4  # 从中点往后1/4窗口
            t_s = subgraph.start_time
            t_e = subgraph.end_time

            # 前向传播
            z = model(subgraph.x, subgraph.edge_index, subgraph.timestamp.squeeze(), subgraph.timestamp.squeeze() - subgraph.start_time, subgraph.unique_edges, subgraph.timestamp_lists)

            communities = community_search(z, query_idx, subgraph, t_s, t_e,model.temporal_encoder)

            if(len(communities)>0):
                neighbor_idx = torch.tensor(np.array(list(communities)), dtype=torch.long)
            else:
                neighbor_idx = torch.tensor(list(optimal_subgraphs[query_idx.item()])) 

            # 计算社区和子图节点占比
            community_size = len(communities)
            subgraph_size = subgraph.num_nodes
            node_ratio = community_size / subgraph_size
            print(f"社区节点数: {community_size}, 子图节点总数: {subgraph_size}，邻居节点数: {len(neighbor_idx)}")
            print(f"社区节点占比: {node_ratio:.2%}")

            # 计算损失
            loss = criterion(
                z=z,
                query_idx=query_idx,
                neighbor_idx=neighbor_idx,
                edge_times=subgraph.timestamp.squeeze(),
                current_time=subgraph.end_time,
                t_s=t_s,
                t_e=t_e,
                G=subgraph
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(test_subgraphs)
    print(f"Validation Loss: {avg_loss:.4f}")
