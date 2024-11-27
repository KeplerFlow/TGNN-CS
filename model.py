import torch
import torch.nn as nn
from layer import *
from utils import *
import torch.nn.functional as F

class TemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_features_dim, num_layers):
        super(TemporalGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.temporal_encoder = TemporalFeatureEncoder(num_features=temporal_features_dim)  # 时间特征编码器
        for _ in range(num_layers):
            layer_set = LayerSet(in_channels, hidden_channels, temporal_features_dim)
            self.layers.append(layer_set)
    
    def forward(self, x, edge_index, timestamps, time_diffs,unique_edges, timestamp_lists):
        z = x
        # 使用时间特征编码器生成时间特征
        temporal_features = self.temporal_encoder(timestamp_lists)
        for layer in self.layers:
            z = layer(z, edge_index, temporal_features, time_diffs,unique_edges)
        return z

class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temporal_encoder, weight_time=0.5, weight_core=0.5):
        super().__init__()
        self.weight_time = weight_time
        self.weight_core = weight_core
        self.omega = temporal_encoder.omega.clone()
        self.phi = temporal_encoder.phi.clone()

    def time_encoding(self, timestamps):
        """
        时间编码函数
        timestamps: 时间戳张量，形状为 [num_timestamps]
        返回值：编码后的时间特征，形状为 [num_timestamps, num_features]
        """
        t = timestamps.unsqueeze(1)  # [num_timestamps, 1]
        linear_term = self.omega[0] * t + self.phi[0]  # [num_timestamps, 1]
        sin_terms = torch.sin(self.omega[1:] * t + self.phi[1:])  # [num_timestamps, num_features - 1]
        phi_t = torch.cat([linear_term, sin_terms], dim=1)  # [num_timestamps, num_features]
        phi_t = torch.tanh(phi_t)
        return phi_t

    def forward(self, z, query_idx, neighbor_idx, edge_times, current_time, t_s, t_e, G):
        device = z.device

        # ==================== Loss 1: 时间编码差异损失 ====================
        # 筛选时间窗口内的所有时间戳
        time_mask = (edge_times >= t_s) & (edge_times <= t_e)
        timestamps_window = edge_times[time_mask]  # [num_timestamps_in_window]

        if timestamps_window.numel() == 0:
            # 如果时间窗口内没有边，损失为0
            loss_time = torch.tensor(0.0, device=device)
        else:
            # 对时间窗口内的时间戳进行编码
            phi_t_window = self.time_encoding(timestamps_window)  # [num_timestamps_in_window, num_features]

            # 获取时间窗口内的边索引
            edge_index_window = G.edge_index[:, time_mask]  # [2, num_edges_in_window]

            # 筛选出 neighbor_idx 之间的边
            neighbor_set = set(neighbor_idx.tolist())
            src_nodes = edge_index_window[0]
            dst_nodes = edge_index_window[1]
            mask_neighbor_edges = ((src_nodes.unsqueeze(1) == neighbor_idx.unsqueeze(0)).any(dim=1) &
                                   (dst_nodes.unsqueeze(1) == neighbor_idx.unsqueeze(0)).any(dim=1))

            neighbor_edge_times = edge_times[time_mask][mask_neighbor_edges]  # [num_neighbor_edges]

            if neighbor_edge_times.numel() == 0:
                loss_time = torch.tensor(0.0, device=device)
            else:
                # 对 neighbor_idx 之间的时间戳进行编码
                phi_t_neighbors = self.time_encoding(neighbor_edge_times)  # [num_neighbor_edges, num_features]

                # 对时间编码进行归一化
                mean_phi_t_window = F.normalize(phi_t_window.mean(dim=0), dim=0)  # [num_features]
                mean_phi_t_neighbors = F.normalize(phi_t_neighbors.mean(dim=0), dim=0)  # [num_features]

                # 计算损失（均方误差）
                loss_time = F.mse_loss(mean_phi_t_window, mean_phi_t_neighbors)

        # ==================== Loss 2: 核心度接近损失 ====================
        query_core = G.k_core[query_idx]  # 查询节点的核心度
        neighbor_cores = G.k_core[neighbor_idx]  # 邻居节点的核心度

        # 核心度归一化
        core_min = G.k_core.min().float()
        core_max = G.k_core.max().float()
        query_core_norm = (query_core.float() - core_min) / (core_max - core_min + 1e-8)
        neighbor_cores_norm = (neighbor_cores.float() - core_min) / (core_max - core_min + 1e-8)

        # 计算核心度的均方误差损失
        loss_core = F.mse_loss(neighbor_cores_norm, query_core_norm.expand_as(neighbor_cores_norm))

        # ==================== 总损失 ====================
        total_loss = self.weight_time * loss_time + self.weight_core * loss_core

        return total_loss

