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
from signal import signal, SIGPIPE, SIG_DFL
import pandas as pd 
import random
signal(SIGPIPE,SIG_DFL)

file_path = "../TCS/data/sx-mathoverflow.txt"
graph =  read_graph_from_txt_pyg(file_path)
print(f"successfully read graph")

window_size = 21

subgraphs = split_graph_by_time_pyg(graph,window_size)  # x days window
print(f"successfully split graph")
#validate the subgraphs
# validate_and_summarize_subgraphs(subgraphs, graph)

model = TemporalGNN(
    in_channels=128,
    hidden_channels=64,
    temporal_features_dim=16,  # 你可以根据需要调整
    num_layers=3
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = TemporalContrastiveLoss(temporal_encoder=model.temporal_encoder)
num_epochs = 50

print(f"begin to train")
for subgraph in subgraphs:
    subgraph.x = initialize_features(subgraph)
    sampled_nodes = sample_graph_nodes(subgraph)  # 采样一些节点作为查询节点
    node_optimal_jumps = calculate_temporal_modularity_for_jumps(subgraph, sampled_nodes, 3)
    print(f"the optimal jumps: {node_optimal_jumps}")
    optimal_subgraphs = get_optimal_subgraphs(subgraph, node_optimal_jumps)
    # print(f"the optimal subgraphs: {optimal_subgraphs}")
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        # 随机选择一个查询节点
        query_idx = torch.tensor([random.choice(sampled_nodes)])

        # 准备数据
        x = subgraph.x
        edge_index = subgraph.edge_index
        timestamps = subgraph.timestamp.squeeze()
        time_diffs = timestamps - subgraph.start_time
        unique_edges = subgraph.unique_edges
        timestamp_lists = subgraph.timestamp_lists

        current_time = subgraph.start_time
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
        z = model(x, edge_index, timestamps, time_diffs, unique_edges, timestamp_lists)

        # 社区搜索
        communities = community_search(z, query_idx, subgraph, t_s, t_e)
        communities_tensor = torch.tensor(list(communities))

        # 将社区中的节点作为邻居节点
        query_idx_value = query_idx.item()

        neighbor_idx = torch.tensor(list(optimal_subgraphs[query_idx_value]))

        # 计算社区和子图节点占比
        community_size = len(communities)
        subgraph_size = subgraph.num_nodes
        node_ratio = community_size / subgraph_size
        print(f"社区节点数: {community_size}, 子图节点总数: {subgraph_size}，邻居节点数: {len(neighbor_idx)}")
        print(f"社区节点占比: {node_ratio:.2%}")

        # 生成负样本
        # neg_idxs = generate_negative_samples(subgraph, neighbor_idx, num_negatives=10)

        if (epoch==45):
            TD_S, TC_S = evaluate_community(subgraph, communities, t_s, t_e)
            print(f"社区的时间密度 TD(S): {TD_S:.4f}")
            print(f"社区的时间割 TC(S): {TC_S:.4f}")

        # 计算损失
        loss = criterion(
            z=z,
            query_idx=query_idx,
            neighbor_idx=communities_tensor,
            edge_times=timestamps,
            current_time=subgraph.start_time,
            t_s=t_s,
            t_e=t_e,
            G=subgraph
        )

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 使用训练好的模型生成节点嵌入
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    for unseen_subgraph in unseen_subgraphs:
        unseen_subgraph.x = initialize_features(unseen_subgraph)
        
        # 准备数据
        x = unseen_subgraph.x
        edge_index = unseen_subgraph.edge_index
        timestamps = unseen_subgraph.timestamp.squeeze()
        time_diffs = timestamps - unseen_subgraph.start_time
        
        # 前向传播生成节点嵌入
        z = model(x, edge_index, timestamps, time_diffs)
        
        # 在给定时间窗口内进行社区搜索
        # 这里假设您有一个函数 `community_search` 来执行社区搜索
        communities = community_search(z, unseen_subgraph, time_window=window_size)
        
        # 打印或保存社区搜索结果
        print(f"Communities found in unseen subgraph: {communities}")
