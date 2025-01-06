import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from networkx.algorithms.core import core_number
from scipy.cluster.hierarchy import single
from sympy import sequence
from sympy.physics.units import frequency
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from collections import deque
from pympler import asizeof
from scipy.sparse import lil_matrix
import time
import cProfile
import tracemalloc
from torch_geometric.data import TemporalData
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.dimenet import triplets

from model import *
from torch.nn import init
from torch_geometric.utils import k_hop_subgraph, subgraph
from tqdm import tqdm
import heapq

from data_loader import *
from utils import *
from loss import *
from extract_subgraph import *
from train import *
from query import *
from index import *

import cProfile


# 主函数
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = "cpu"
    
    dataset_name = 'mathoverflow'
    # dataset_name = 'wikitalk'
    num_vertex = 0
    num_edge = 0
    num_timestamp = 0
    time_edge = {}
    num_core_number = 0
    vertex_core_numbers = []
    time_range_core_number = defaultdict(dict)
    temporal_graph = []
    time_range_layers = []
    time_range_set = set()
    max_time_range_layers = []
    min_time_range_layers = []
    sequence_features1_matrix = torch.empty(0, 0, 0)
    model_time_range_layers = []
    partition = 4
    max_range = 40
    max_degree = 0
    max_layer_id = 0
    
    inter_time = 0.0

    # GNN
    temporal_graph_pyg = TemporalData()
    temporal_graph_pyg_dense = TemporalData()
    subgraph_k_hop_cache = {}
    subgraph_pyg_cache = {}
    subgraph_vertex_map_cache = {}
    filtered_temporal_graph_pyg = TemporalData()
    time_range_link_samples_cache = defaultdict(dict)  # {(t_start, t_end): {anchor: {vertex: [(pos, neg), ...]}}}
    # 超参数
    node_in_channels = 8
    node_out_channels = 16
    edge_dim = 8
    learning_rate = 0.001
    epochs = 3
    batch_size = 4
    k_hop = 5
    positive_hop = 3
    alpha = 0.1
    num_time_range_samples = 10
    num_anchor_samples = 10
    test_result_list = []

    num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, temporal_graph_pyg = read_temporal_graph(dataset_name,device)
    
    time_range_layers, min_time_range_layers, max_time_range_layers, time_range_set, max_layer_id = get_timerange_layers(num_timestamp, max_range, partition)

    print(time_range_layers)
    
    num_core_number, vertex_core_numbers, time_range_core_number = read_core_number(dataset_name, num_vertex, time_range_set)
    
    sequence_features1_matrix = construct_feature_matrix(num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device)
    
    #####################

    root, max_layer_id = build_tree(num_timestamp, time_range_core_number, num_vertex, temporal_graph, time_range_layers)

    load_models(0, time_range_layers, max_layer_id, device, dataset_name, 
                max_time_range_layers, partition,root,num_timestamp)

    sequence_features_matrix_tree, indices_vertex_of_matrix = construct_feature_matrix_tree(num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device)

    query_time_range_start = random.randint(0, num_timestamp-1)
    query_time_range_end = random.randint(query_time_range_start, min(num_timestamp-1, query_time_range_start + 100))
    query_vertex = random.randint(0, num_vertex - 1)

    query_vertex_core_number = model_out_put_for_any_range_vertex_set(
        [query_vertex], query_time_range_start, query_time_range_end,
        max_layer_id,
        max_time_range_layers,
        device,
        sequence_features_matrix_tree,
        partition,
        num_timestamp, 
        root
    )
    print(f"query_vertex: {query_vertex}")

    print(f"query_vertex_core_number:{query_vertex_core_number}")

    ######################

    model = AdapterTemporalGNN(node_in_channels, node_out_channels, edge_dim=edge_dim).to(device)
    
    train_time_range_list = []
    while len(train_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in train_time_range_list:
            train_time_range_list.append((t_start, t_end))
            
    quadruplet = []  # 训练数据用四元组形式存储

    print(subgraph_k_hop_cache)
    
    for i, time_range in enumerate(train_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        print(f"Generating training data {i + 1}/{len(train_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        select_limit = 50
        select_cnt = 0
        while len(center_vertices) < num_anchor_samples:
            if select_cnt >= select_limit:
                break
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            min_core_numer = 5
            if t_end - t_start > 300:
                min_core_numer = 10
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue
        
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end, num_vertex, temporal_graph, time_range_core_number, time_range_link_samples_cache, subgraph_k_hop_cache)
        for triplet in triplets:
            quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))
        
        # generate pyg data for the subgraph of the current time range
        print("done1")
        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels, temporal_graph_pyg, num_vertex, edge_dim, device,sequence_features1_matrix, time_range_core_number,subgraph_k_hop_cache)
        print("done2")
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()
        print("done3")

    test_time_range_list = []
    while len(test_time_range_list) < num_time_range_samples:
        t_layer = random.randint(0, len(time_range_layers) - 1)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in test_time_range_list and (t_start, t_end) not in train_time_range_list:
            test_time_range_list.append((t_start, t_end))

    test_quadruplet = []
    for i, time_range in enumerate(test_time_range_list):
        t_start = time_range[0]
        t_end = time_range[1]
        print(f"Generating testing data {i + 1}/{len(test_time_range_list)}...")
        print(f"Time range: {t_start} - {t_end}")
        center_vertices = set()
        select_limit = 50
        select_cnt = 0
        min_core_numer = 5
        if t_end - t_start > 300:
            min_core_numer = 10
        while len(center_vertices) < num_anchor_samples:
            temp_vertex = random.choice(list(time_range_core_number[(t_start, t_end)].keys()))
            temp_core_number = time_range_core_number[(t_start, t_end)][temp_vertex]
            if select_cnt >= select_limit:
                break
            if temp_core_number >= min_core_numer:
                if temp_vertex not in center_vertices:
                    center_vertices.add(temp_vertex)
                    select_cnt = 0
                else:
                    select_cnt += 1
            else:
                select_cnt += 1
        if len(center_vertices) == 0:
            continue
        triplets = generate_triplets(center_vertices, k_hop, t_start, t_end, num_vertex, temporal_graph, time_range_core_number, time_range_link_samples_cache, subgraph_k_hop_cache)
        for triplet in triplets:
            test_quadruplet.append((triplet[0], triplet[1], triplet[2], (t_start, t_end)))
        # generate pyg data for the subgraph of the current time range
        temp_subgraph_pyg, vertex_map = extract_subgraph_for_time_range(center_vertices, t_start, t_end, node_in_channels, temporal_graph_pyg, num_vertex, edge_dim, device,sequence_features1_matrix, time_range_core_number,subgraph_k_hop_cache)
        
        temp_subgraph_pyg = temp_subgraph_pyg.to('cpu')
        vertex_map = vertex_map.to('cpu')
        subgraph_pyg_cache[(t_start, t_end)] = temp_subgraph_pyg
        subgraph_vertex_map_cache[(t_start, t_end)] = vertex_map
        torch.cuda.empty_cache()

    test_quadruplet = random.choices(test_quadruplet, k=200)

    # 切割数据集
    train_quadruplet = quadruplet
    val_quadruplet, test_quadruplet = train_test_split(test_quadruplet, test_size=0.5, random_state=42)  # 20% 的一半用于验证集

    # 创建数据集
    train_dataset = MultiSampleQuadrupletDataset(train_quadruplet)
    val_dataset = MultiSampleQuadrupletDataset(val_quadruplet)
    test_dataset = MultiSampleQuadrupletDataset(test_quadruplet)

    # 创建加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=quadruplet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=quadruplet_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # 用于混合精度训练
    feature_dim = node_in_channels

    # 初始化早停相关变量
    best_val_loss = float('inf')
    patience = 15  # 可以根据需要调整耐心值
    patience_counter = 0
    sample_cache = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device).clone()
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            optimizer.zero_grad()

            subgraphs = []
            vertex_maps = []
            for anchor, time_range in zip(anchors, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1],subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex, device)
                if len(vertex_map) == 0:
                    continue

                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)

            # 模型前向传播
            embeddings = model(batched_subgraphs)

            # 使用 Batch.batch 获取每个子图的全局编号范围
            batch_indices = batched_subgraphs.batch

            # release the memory of subgraphs
            del batched_subgraphs

            # 计算损失
            batch_loss = 0.0
            for i, (anchor, pos_samples, neg_samples, vertex_map, time_range) in enumerate(
                    zip(anchors, positives, negatives, vertex_maps, time_ranges)):
                # 获取当前子图的节点范围
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

                anchor_idx = node_indices[vertex_map[anchor.long()]]
                pos_samples = pos_samples.to(device=vertex_map.device).long()
                neg_samples = neg_samples.to(device=vertex_map.device).long()
                pos_mapped_indices = vertex_map[pos_samples]  # 批量映射
                neg_mapped_indices = vertex_map[neg_samples]  # 批量映射
                pos_indices = node_indices[pos_mapped_indices]
                neg_indices = node_indices[neg_mapped_indices]

                    # 跳过无效样本
                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                # 提取嵌入
                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]
                subgraph_emb = embeddings[node_indices]

                # 计算三元组损失
                with autocast():
                    loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    link_loss_value = compute_link_loss(embeddings, vertex_map, node_indices,
                                                        time_range[0], time_range[1], anchor.item(),time_range_link_samples_cache, margin=0.2)
                    loss += alpha * link_loss_value
                    # loss += alpha * link_loss(subgraph_emb, subgraph_pyg)
                    # loss += alpha * link_loss(vertex_map, subgraph_emb, time_range[0], time_range[1])
                batch_loss += loss
                torch.cuda.empty_cache()

            # 平均批次损失
            if len(vertex_maps) > 0:  # 避免除以零
                batch_loss = batch_loss / len(vertex_maps)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=epoch_loss / (batch_idx + 1))
            torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", end=' ')

        # 验证模型
        val_loss = validate_model(model, val_loader, device, alpha, subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex,time_range_link_samples_cache)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最佳模型

            # torch.save(model.state_dict(), f'model_L1_{dataset_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                # 加载最佳模型参数

                model.load_state_dict(torch.load(f'model_L1_{dataset_name}.pth'))
                break
        
    # 测试模型
    avg_sim_pos, avg_sim_neg = test_model(model, test_loader, device, subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex)
    test_result_list.append((avg_sim_pos, avg_sim_neg))

    # 1. 首先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 2. 只解冻Adapter和gate参数
    for name, param in model.named_parameters():
        if 'adapter' in name or 'gating_params' in name:
            param.requires_grad = True

    optimizer = optim.Adam(
        [p for n, p in model.named_parameters() if ('adapter' in n or 'gating_params' in n) and p.requires_grad],
        lr=learning_rate
    )
    scaler = GradScaler()  # 混合精度训练

    sequence_features2_matrix = construct_feature_matrix(num_vertex, num_timestamp, temporal_graph, vertex_core_numbers, device)

    test_time_range_list = []
    while len(test_time_range_list) < 10:
        t_layer = random.randint(1, len(time_range_layers) - 2)
        if t_layer == 0:
            continue
        t_idx = random.randint(0, len(time_range_layers[t_layer]) - 1)
        t_start, t_end = time_range_layers[t_layer][t_idx][0], time_range_layers[t_layer][t_idx][1]
        if (t_start, t_end) not in test_time_range_list:
            test_time_range_list.append((t_start, t_end))
    temporal_density_ratio = 0
    temporal_conductance_ratio = 0
    valid_cnt = 0
    total_time = 0
    result_len = 0
    epochs=5

    for t_start, t_end in test_time_range_list:
        print(f"Test time range: [{t_start}, {t_end}]")
        query_vertex_list = set()
        while len(query_vertex_list) < 10:
            query_vertex = random.choice(range(num_vertex))
            core_number = time_range_core_number[(t_start, t_end)].get(query_vertex, 0)
            while core_number < 5:
                query_vertex = random.choice(range(num_vertex))
                core_number = time_range_core_number[(t_start, t_end)].get(query_vertex, 0)
            query_vertex_list.add(query_vertex)

        for query_vertex in query_vertex_list:
            print(valid_cnt)
            start_time = time.time()

            # 为查询节点生成训练数据
            center_vertices = {query_vertex}  # 只使用查询节点作为中心
            triplets = generate_triplets(center_vertices, k_hop, t_start, t_end, num_vertex, temporal_graph,
                                        time_range_core_number, time_range_link_samples_cache, subgraph_k_hop_cache)
            quadruplet = [(t[0], t[1], t[2], (t_start, t_end)) for t in triplets]

            # 创建数据加载器
            query_dataset = MultiSampleQuadrupletDataset(quadruplet)
            query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True,
                                    collate_fn=quadruplet_collate_fn)
            # before finetune
            # 评估阶段
            model.eval()
            with torch.no_grad():
                feature_dim = node_in_channels
                subgraph, vertex_map, neighbors_k_hop = extract_subgraph(query_vertex, t_start, t_end, k_hop,
                                                                        feature_dim, temporal_graph,
                                                                        temporal_graph_pyg, num_vertex, edge_dim,
                                                                        sequence_features2_matrix,
                                                                        time_range_core_number, device)
                if subgraph is not None and vertex_map is not None and query_vertex in vertex_map:
                    embeddings = model(subgraph.to(device))
                    query_vertex_embedding = embeddings[vertex_map[query_vertex]].unsqueeze(0)

                    # 确保 neighbors_k_hop 中的邻居在 vertex_map 中
                    valid_neighbors_k_hop = [n for n in neighbors_k_hop if n in vertex_map]
                    if valid_neighbors_k_hop:
                        neighbors_embeddings = embeddings[torch.tensor([vertex_map[i] for i in valid_neighbors_k_hop], device=device)]
                        distances = F.pairwise_distance(query_vertex_embedding, neighbors_embeddings)
                        GNN_temporal_density, GNN_temporal_conductance, result = temporal_test_GNN_query_time(
                            distances, vertex_map, query_vertex, t_start, t_end, temporal_graph, device, num_vertex
                        )
                        temporal_density_ratio += GNN_temporal_density
                        temporal_conductance_ratio += GNN_temporal_conductance
                        result_len += len(result)
                        print(f"before test number: {len(result)}")
                        print(f"before temporal density ratio: {GNN_temporal_density}")
                        print(f"before temporal conductance ratio: {GNN_temporal_conductance}")
                else:
                    continue
            
            # 查询特定的微调
            model.train()
            subgraph_pyg, vertex_map,neighbors_k_hop = extract_subgraph(query_vertex, t_start, t_end, k_hop,
                                                                        feature_dim, temporal_graph,
                                                                        temporal_graph_pyg, num_vertex, edge_dim,
                                                                        sequence_features2_matrix,
                                                                        time_range_core_number, device)
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                epoch_start_time = time.time()
                timing_info = {
                    "data_loading": 0.0,
                    "data_preparation": 0.0,
                    "forward_pass": 0.0,
                    "loss_calculation": 0.0,
                    "backward_pass": 0.0,
                    "optimizer_step": 0.0,
                }
                for batch_idx, batch in enumerate(train_loader):
                    batch_start_time = time.time()
                    
                    data_loading_start_time = time.time()
                    anchors, positives, negatives, time_ranges = batch
                    anchors = torch.tensor(anchors, device=device)
                    timing_info["data_loading"] += time.time() - data_loading_start_time

                    data_preparation_start_time = time.time()
                    positives = [torch.tensor(pos, device=device) for pos in positives]
                    negatives = [torch.tensor(neg, device=device) for neg in negatives]

                    optimizer.zero_grad()

                    subgraphs = []
                    vertex_maps = []
                    for anchor, time_range in zip(anchors, time_ranges):
                        # Add subgraph creation logic here
                        subgraphs.append(subgraph_pyg)
                        vertex_maps.append(vertex_map)
                        
                    batched_subgraphs = Batch.from_data_list(subgraphs).to(device)
                    timing_info["data_preparation"] += time.time() - data_preparation_start_time

                    forward_start_time = time.time()
                    embeddings = model(batched_subgraphs)
                    timing_info["forward_pass"] += time.time() - forward_start_time

                    loss_calculation_start_time = time.time()
                    batch_indices = batched_subgraphs.batch
                    del batched_subgraphs

                    batch_loss = 0.0
                    for i, (anchor, pos_samples, neg_samples, vertex_map, time_range) in enumerate(
                            zip(anchors, positives, negatives, vertex_maps, time_ranges)):
                        node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]
                        anchor_idx = node_indices[vertex_map[anchor.long()]]
                        
                        pos_samples = pos_samples.to(device)
                        neg_samples = neg_samples.to(device)
                        pos_indices = node_indices[vertex_map[pos_samples.long()]]
                        neg_indices = node_indices[vertex_map[neg_samples.long()]]

                        if len(pos_indices) == 0 or len(neg_indices) == 0:
                            continue

                        with autocast():
                            loss = margin_triplet_loss(
                                embeddings[anchor_idx],
                                embeddings[pos_indices],
                                embeddings[neg_indices]
                            )
                            batch_loss += loss

                    if len(vertex_maps) > 0:
                        batch_loss = batch_loss / len(vertex_maps)
                    timing_info["loss_calculation"] += time.time() - loss_calculation_start_time

                    backward_start_time = time.time()
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(batch_loss).backward()
                    else:
                        batch_loss.backward()
                    timing_info["backward_pass"] += time.time() - backward_start_time

                    # 优化器步骤
                    optimizer_step_start_time = time.time()
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    timing_info["optimizer_step"] += time.time() - optimizer_step_start_time

                    epoch_loss += batch_loss.item()

                torch.cuda.empty_cache()
                epoch_end_time = time.time()
                epoch_total_time = epoch_end_time - epoch_start_time

                timing_percentages = {
                    k: (v / epoch_total_time) * 100 for k, v in timing_info.items()
                }
                if epoch == 0:
                    print(f"epoch_total_time: {epoch_total_time}s")
                    print("耗时占比:")
                    for name, percentage in timing_percentages.items():
                        print(f"  {name}: {percentage:.2f}%")
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}", end=' ')
                

            # 评估阶段
            model.eval()
            with torch.no_grad():
                feature_dim = node_in_channels
                subgraph, vertex_map, neighbors_k_hop = extract_subgraph(query_vertex, t_start, t_end, k_hop,
                                                                        feature_dim, temporal_graph,
                                                                        temporal_graph_pyg, num_vertex, edge_dim,
                                                                        sequence_features2_matrix,
                                                                        time_range_core_number, device)
                if subgraph is not None and vertex_map is not None and query_vertex in vertex_map:
                    embeddings = model(subgraph.to(device))
                    query_vertex_embedding = embeddings[vertex_map[query_vertex]].unsqueeze(0)

                    # 确保 neighbors_k_hop 中的邻居在 vertex_map 中
                    valid_neighbors_k_hop = [n for n in neighbors_k_hop if n in vertex_map]
                    if valid_neighbors_k_hop:
                        neighbors_embeddings = embeddings[torch.tensor([vertex_map[i] for i in valid_neighbors_k_hop], device=device)]
                        distances = F.pairwise_distance(query_vertex_embedding, neighbors_embeddings)
                        GNN_temporal_density, GNN_temporal_conductance, result = temporal_test_GNN_query_time(
                            distances, vertex_map, query_vertex, t_start, t_end, temporal_graph, device, num_vertex
                        )
                        temporal_density_ratio += GNN_temporal_density
                        temporal_conductance_ratio += GNN_temporal_conductance
                        result_len += len(result)
                        print(f"after test number: {len(result)}")
                        print(f"after temporal density ratio: {GNN_temporal_density}")
                        print(f"after temporal conductance ratio: {GNN_temporal_conductance}")

            end_time = time.time()
            total_time += end_time - start_time
            valid_cnt += 1

            # 清理缓存
            torch.cuda.empty_cache()


        # for query_vertex in query_vertex_list:
        #     print(valid_cnt)
        #     start_time = time.time()
        #     feature_dim = node_in_channels
        #     subgraph, vertex_map, neighbors_k_hop = extract_subgraph(query_vertex, t_start, t_end, k_hop, feature_dim, temporal_graph, temporal_graph_pyg, num_vertex, edge_dim, sequence_features2_matrix, time_range_core_number,device)
        #     embeddings = model(subgraph)
        #     query_vertex_embedding = embeddings[vertex_map[query_vertex]].unsqueeze(0)
        #     neighbors_embeddings = embeddings[vertex_map[neighbors_k_hop]]
        #     # 计算距离
        #     distances = F.pairwise_distance(query_vertex_embedding, neighbors_embeddings)

        #     # test query time
        #     GNN_temporal_density, GNN_temporal_conductance,result = temporal_test_GNN_query_time(distances, vertex_map, query_vertex, t_start, t_end, temporal_graph, device,num_vertex)
        #     temporal_density_ratio+=GNN_temporal_density
        #     temporal_conductance_ratio+=GNN_temporal_conductance
        #     result_len+=len(result)

        #     end_time = time.time()
        #     total_time += end_time - start_time
        #     valid_cnt += 1
    print(f"Valid test number: {valid_cnt}")
    print(f"Average temporal density ratio: {temporal_density_ratio / valid_cnt}")
    print(f"Average temporal conductance ratio: {temporal_conductance_ratio / valid_cnt}")
    print(f"Average time: {total_time / valid_cnt}s")
    print(f"Average length: {result_len / valid_cnt}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()