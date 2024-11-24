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

window_size = 7

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
criterion = TemporalContrastiveLoss()
num_epochs = 50
print(f"begin to train")
# 训练循环
for subgraph in subgraphs:
    subgraph.x = initialize_features(subgraph)
    sampled_nodes = sample_graph_nodes(subgraph)  # 采样一些节点作为查询节点
    node_optimal_jumps = calculate_temporal_modularity_for_jumps(subgraph, sampled_nodes, 3)
    optimal_subgraphs = get_optimal_subgraphs(subgraph, node_optimal_jumps)
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        
        # 随机选择一个查询节点
        query_idx = torch.tensor([random.choice(sampled_nodes)])
        
        # 获取查询节点的最优子图中的节点作为邻居节点
        neighbor_idx = torch.tensor(list(optimal_subgraphs[query_idx.item()]))
        
        # 准备数据
        x = subgraph.x
        edge_index = subgraph.edge_index
        timestamps = subgraph.timestamp.squeeze()
        time_diffs = timestamps - subgraph.start_time
        
        # 前向传播
        z = model(x, edge_index, timestamps, time_diffs)
        
        # 计算损失
        loss = criterion(
            z=z,
            query_idx=query_idx,
            neighbor_idx=neighbor_idx,
            edge_times=timestamps,
            current_time=subgraph.end_time,
            G=subgraph
        )
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# evaluate the model
unseen_file_path = "../TCS/data/unseen_graph.txt"
unseen_graph = read_graph_from_txt_pyg(unseen_file_path)
print(f"successfully read unseen graph")

# 将未见过的图分割为子图
unseen_subgraphs = split_graph_by_time_pyg(unseen_graph, window_size)  # 使用相同的时间窗口

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
