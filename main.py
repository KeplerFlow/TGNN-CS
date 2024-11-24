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

subgraphs = split_graph_by_time_pyg(graph, 21)  # x days window
print(f"successfully split graph")
#validate the subgraphs
# validate_and_summarize_subgraphs(subgraphs, graph)

features = initialize_features(subgraphs[0], dimensions=128, use_cached=True)
print(f"successfully initialize features")

model = TemporalGNN(
    in_channels=features.size(1),
    hidden_channels=64,
    temporal_features_dim=16,  # 你可以根据需要调整
    num_layers=3
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = TemporalContrastiveLoss()

num_epochs = 50
# 只使用第一个子图进行训练
subgraph = subgraphs[0]  # 获取第一个子图

# 在训练循环之前，获取采样节点和最优跳数
sampled_nodes = sample_graph_nodes(subgraph)  # 采样一些节点作为查询节点
node_optimal_jumps = calculate_temporal_modularity_for_jumps(subgraph, sampled_nodes, 3)

# 获取每个节点的最优子图
optimal_subgraphs = get_optimal_subgraphs(subgraph, node_optimal_jumps)
print(f"successfully calculate temporal modularity")

print(f"begin to train")
# 训练循环

for epoch in range(num_epochs):
    for subgraph in subgraphs:
        
        # 初始化特征，使用缓存的特征
        features = initialize_features(subgraph, dimensions=128, use_cached=True)
        subgraph.x = features

        model.train()
        optimizer.zero_grad()
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
        out = model(x, edge_index, timestamps, time_diffs)
        
        # 计算损失
        loss = criterion(
            z=out,
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

        update_global_features(subgraph, out.detach())
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")