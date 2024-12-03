import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_scatter

class TemporalFeatureEncoder(nn.Module):
    def __init__(self, num_features=64):
        super(TemporalFeatureEncoder, self).__init__()
        self.omega = nn.Parameter(torch.randn(num_features))
        self.phi = nn.Parameter(torch.randn(num_features))

    def forward(self, timestamps_list):
        # 将时间戳列表转换为张量列表
        timestamps_tensors = timestamps_list
        # 使用 pad_sequence 将序列填充到相同长度
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

        # 计算 deltas
        deltas_next = last_timestamps.unsqueeze(1) - padded_timestamps  # [num_edges, max_len]
        deltas_prev = padded_timestamps - padded_timestamps.roll(shifts=1, dims=1)  # [num_edges, max_len]
        deltas_prev[:, 0] = 0  # 第一个元素没有前一个元素，设为 0
        deltas = (deltas_next.abs() + deltas_prev.abs()) / 2  # 取绝对值平均

        # 将填充位置的 deltas 设为 0
        deltas = deltas * mask.float()

        # 计算权重
        weights = torch.exp(-deltas)  # [num_edges, max_len]
        # 将填充位置的权重设为 0
        weights = weights * mask.float()
        # 归一化权重
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # [num_edges, 1]
        weights = weights / weights_sum  # [num_edges, max_len]

        # 计算加权特征
        weighted_features = temporal_features * weights.unsqueeze(-1)  # [num_edges, max_len, num_features]
        # 在时间维度上求和，得到最终的特征表示
        phi_uv = weighted_features.sum(dim=1)  # [num_edges, num_features]

        return phi_uv

class TemporalMessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, temporal_features_dim):
        super(TemporalMessagePassingLayer, self).__init__(aggr='add')  # 使用 'add' 聚合
        self.W_S = nn.Linear(in_channels, out_channels)
        self.W_T = nn.Linear(in_channels, out_channels)
        self.temporal_mlp = nn.Linear(temporal_features_dim, out_channels)

    def forward(self, x, edge_index, temporal_features):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # temporal_features: [num_edges, temporal_features_dim]

        # 自身特征
        self_features = self.W_S(x)  # [num_nodes, out_channels]

        # 进行消息传递和聚合
        out = self.propagate(edge_index, x=x, temporal_encodings=temporal_features)

        # 最终更新
        r_u = F.relu(self_features + out)  # [num_nodes, out_channels]
        r_u_normalized = F.normalize(r_u, p=2, dim=1)  # L2 归一化

        return r_u_normalized

    def message(self, x_j, temporal_encodings):
        # x_j: [num_edges, in_channels]
        neighbor_features = self.W_T(x_j)  # [num_edges, out_channels]
        messages = neighbor_features + temporal_encodings  # [num_edges, out_channels]
        return messages

class StructuralFeatureLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(StructuralFeatureLayer, self).__init__(aggr='add')  # 使用 'add' 聚合
        self.W_S = nn.Linear(in_channels, out_channels)
        self.W_N = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, time_diffs):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # time_diffs: [num_edges]  # t_e - t_s

        # 自身特征
        self_features = self.W_S(x)  # [num_nodes, out_channels]

        # 计算相对权重
        row, col = edge_index
        time_diffs = time_diffs.float()
        time_diff_sum = torch_scatter.scatter_add(time_diffs, row, dim=0, dim_size=x.size(0))  # [num_nodes]
        time_diff_sum[time_diff_sum == 0] = 1.0  # 避免除以零
        relative_weights = time_diffs / time_diff_sum[row]  # [num_edges]

        # 进行消息传递和聚合
        out = self.propagate(edge_index, x=x, relative_weights=relative_weights)

        # 最终更新
        gamma_u = F.relu(self_features + out)  # [num_nodes, out_channels]
        gamma_u_normalized = F.normalize(gamma_u, p=2, dim=1)  # L2 归一化

        return gamma_u_normalized

    def message(self, x_j, relative_weights):
        neighbor_features = self.W_N(x_j)  # [num_edges, out_channels]
        weighted_neighbor_features = neighbor_features * relative_weights.unsqueeze(-1)  # [num_edges, out_channels]
        return weighted_neighbor_features

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

        # 对节点表示进行归一化
        z_u_normalized = F.normalize(z_u, p=2, dim=1)  # L2 归一化

        return z_u_normalized

class LayerSet(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_features_dim):
        super(LayerSet, self).__init__()
        self.message_passing = TemporalMessagePassingLayer(in_channels, hidden_channels, temporal_features_dim)
        self.structural_feature = StructuralFeatureLayer(in_channels, hidden_channels)
        self.fusion = FeatureFusionLayer(hidden_channels, in_channels)

    def forward(self, z, edge_index, temporal_features, time_diffs, unique_edges):
        # 后 64 维为时间特征
        r_u = self.message_passing(z, unique_edges, temporal_features)
        # 前 64 维为结构特征
        gamma_u = self.structural_feature(z, edge_index, time_diffs)
        z = self.fusion(r_u, gamma_u, z)
        return z
