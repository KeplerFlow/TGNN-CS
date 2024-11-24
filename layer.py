import torch.nn as nn
import torch.nn.functional as F
import torch
class TemporalFeatureEncoder(nn.Module):
    def __init__(self, num_features=16):
        super(TemporalFeatureEncoder, self).__init__()
        # 将频率和相位初始化为可训练参数
        self.omega = nn.Parameter(torch.randn(num_features))
        self.phi = nn.Parameter(torch.randn(num_features))

    def forward(self, timestamps):
        t = timestamps.unsqueeze(1)  # [num_edges, 1]
        linear_term = self.omega[0] * t + self.phi[0]
        sin_terms = torch.sin(self.omega[1:] * t + self.phi[1:])
        temporal_features = torch.cat([linear_term, sin_terms], dim=1)  # [num_edges, num_features]
        return temporal_features

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
        # 这里可以使用指数衰减函数，例如 exp(-time_diffs / tau)
        tau = 1.0  # 衰减常数，可以调节
        time_weights = torch.exp(-time_diffs / tau).unsqueeze(1)  # [num_edges, 1]
        
        # 加权邻居特征
        weighted_neighbor_features = neighbor_features * time_weights  # [num_edges, out_channels]
        
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
        # r_u: [num_nodes, in_channels]
        # gamma_u: [num_nodes, in_channels]
        # z_prev: [num_nodes, in_channels]
        
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

    def forward(self, z, edge_index, temporal_features, time_diffs):
        r_u = self.message_passing(z, edge_index, temporal_features)
        gamma_u = self.structural_feature(z, edge_index, time_diffs)
        z = self.fusion(r_u, gamma_u, z)
        return z
