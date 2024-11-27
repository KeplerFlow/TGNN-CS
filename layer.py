import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence

class TemporalFeatureEncoder(nn.Module):
    def __init__(self, num_features=16):
        super(TemporalFeatureEncoder, self).__init__()
        self.omega = nn.Parameter(torch.randn(num_features))
        self.phi = nn.Parameter(torch.randn(num_features))

    def forward(self, timestamps_list):
        # 将时间戳列表转换为张量列表
        timestamps_tensors = [torch.tensor(timestamps).float() for timestamps in timestamps_list]
        # 使用pad_sequence将序列填充到相同长度
        padded_timestamps = pad_sequence(timestamps_tensors, batch_first=True, padding_value=0.0)  # [num_edges, max_len]
        # 创建掩码，标记有效的时间戳位置
        mask = (padded_timestamps != 0)  # [num_edges, max_len]

        t = padded_timestamps.unsqueeze(-1)  # [num_edges, max_len, 1]
        linear_term = self.omega[0] * t + self.phi[0]  # [num_edges, max_len, 1]
        sin_terms = torch.sin(self.omega[1:] * t + self.phi[1:])  # [num_edges, max_len, num_features -1]
        temporal_features = torch.cat([linear_term, sin_terms], dim=-1)  # [num_edges, max_len, num_features]

        # 获取每个序列的有效长度
        lengths = mask.sum(dim=1)  # [num_edges]
        # 计算每个序列中最后一个有效时间戳的索引
        indices = (lengths - 1).unsqueeze(1)  # [num_edges, 1]
        # 获取每个序列的最后一个时间戳
        last_timestamps = padded_timestamps.gather(1, indices).squeeze(1)  # [num_edges]

        # 计算deltas
        deltas = last_timestamps.unsqueeze(1) - padded_timestamps  # [num_edges, max_len]
        # 将填充位置的deltas设为0
        deltas = deltas * mask.float()

        # 计算权重
        weights = torch.exp(-deltas)  # [num_edges, max_len]
        # 将填充位置的权重设为0
        weights = weights * mask.float()
        # 归一化权重
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # [num_edges, 1]
        weights = weights / weights_sum  # [num_edges, max_len]

        # 计算加权特征
        weighted_features = temporal_features * weights.unsqueeze(-1)  # [num_edges, max_len, num_features]
        # 在时间维度上求和，得到最终的特征表示
        phi_uv = weighted_features.sum(dim=1)  # [num_edges, num_features]

        phi_uv = torch.tanh(phi_uv)
        return phi_uv


class TemporalMessagePassingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_features_dim):
        super(TemporalMessagePassingLayer, self).__init__()
        self.W_S = nn.Linear(in_channels, out_channels)
        self.W_T = nn.Linear(in_channels, out_channels)
        self.temporal_mlp = nn.Linear(temporal_features_dim, out_channels)
    
    def forward(self, x, edge_index, temporal_features):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # temporal_features: [num_edges, temporal_features_dim]
        
        num_nodes = x.size(0)
        row, col = edge_index  # col -> row
        
        # 自身特征
        self_features = self.W_S(x)  # [num_nodes, out_channels]
        
        # 邻居特征
        neighbor_features = self.W_T(x[col])  # [num_edges, out_channels]
        
        # 时间特征编码
        temporal_encodings = self.temporal_mlp(temporal_features)  # [num_edges, out_channels]
        
        # 消息 = 邻居特征 + 时间特征
        messages = neighbor_features + temporal_encodings  # [num_edges, out_channels]
        
        # 聚合消息
        aggregated_messages = torch.zeros(num_nodes, messages.size(1)).to(x.device)  # [num_nodes, out_channels]
        aggregated_messages.index_add_(0, row, messages)
        
        # 最终更新
        r_u = F.relu(self_features + aggregated_messages)  # [num_nodes, out_channels]
        
        return r_u

class StructuralFeatureLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StructuralFeatureLayer, self).__init__()
        self.W_S = nn.Linear(in_channels, out_channels)
        self.W_N = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, time_diffs):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # time_diffs: [num_edges]  # t_e - t_s
        
        num_nodes = x.size(0)
        row, col = edge_index  # col -> row
        
        # 自身特征
        self_features = self.W_S(x)  # [num_nodes, out_channels]
        
        # 邻居特征
        neighbor_features = self.W_N(x[col])  # [num_edges, out_channels]
        
        # 时间窗口影响函数 k(t_e - t_s)
        time_diffs = time_diffs.float()

        time_diff_sum = torch.zeros(num_nodes).to(x.device)
        time_diff_sum.index_add_(0, row, time_diffs)
        time_diff_sum[time_diff_sum == 0] = 1.0  # 避免除以零
        
        relative_weights = time_diffs / time_diff_sum[row]
        
        # 加权邻居特征
        weighted_neighbor_features = neighbor_features * relative_weights.unsqueeze(1)  # [num_edges, out_channels]
        
        # 聚合加权邻居特征
        aggregated_features = torch.zeros(num_nodes, weighted_neighbor_features.size(1)).to(x.device)  # [num_nodes, out_channels]
        aggregated_features.index_add_(0, row, weighted_neighbor_features)
        
        # 最终更新
        gamma_u = F.relu(self_features + aggregated_features)  # [num_nodes, out_channels]
        
        return gamma_u

class FeatureFusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionLayer, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, r_u, gamma_u, z_prev):
        combined = torch.cat([r_u, gamma_u], dim=1)  # [num_nodes, 2 * in_channels]
        h_u = self.ffn(combined)  # [num_nodes, out_channels]
        z_u = z_prev + F.relu(h_u)  # 残差连接
        return z_u

class LayerSet(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_features_dim):
        super(LayerSet, self).__init__()
        self.message_passing = TemporalMessagePassingLayer(in_channels, hidden_channels, temporal_features_dim)
        self.structural_feature = StructuralFeatureLayer(in_channels, hidden_channels)
        self.fusion = FeatureFusionLayer(hidden_channels, in_channels)

    def forward(self, z, edge_index, temporal_features, time_diffs,unique_edges):
        r_u = self.message_passing(z, unique_edges, temporal_features)
        gamma_u = self.structural_feature(z, edge_index, time_diffs)
        z = self.fusion(r_u, gamma_u, z)
        return z
