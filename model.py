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
    
    def forward(self, x, edge_index, timestamps, time_diffs, unique_edges, timestamp_lists):
        z = x
        # 使用时间特征编码器生成时间特征
        temporal_features = self.temporal_encoder(timestamp_lists)
        for layer in self.layers:
            z = layer(z, edge_index, temporal_features, time_diffs,unique_edges)
        return z

class TemporalContrastiveLoss(nn.Module):
    def __init__(self, num_features=16, temperature=0.07, alpha_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        # 时间编码参数
        self.omega = nn.Parameter(torch.randn(num_features))
        self.phi = nn.Parameter(torch.randn(num_features))

    def time_encoding(self, delta_t):
        """
        时间编码函数
        delta_t: 时间差，形状为 [batch_size, 1]
        返回值：编码后的时间特征，形状为 [batch_size]
        """
        linear_term = self.omega[0] * delta_t + self.phi[0]
        sin_terms = torch.sin(self.omega[1:] * delta_t + self.phi[1:])
        phi_t = torch.cat([linear_term, sin_terms], dim=1).sum(dim=1)
        return phi_t  # [batch_size]

    def compute_mu(self, z_i, z_j):
        """
        计算μ(x, y) = -||z_x - z_y||^2
        z_i, z_j: 节点嵌入，形状为 [batch_size, embedding_dim]
        返回值：μ值，形状为 [batch_size]
        """
        return -torch.norm(z_i - z_j, dim=1).pow(2)  # [batch_size]

    def compute_alpha(self, z_i, z_neighbors, delta_t_neighbors):
        """
        计算注意力系数α(i, y)
        z_i: 查询节点的嵌入，形状为 [1, embedding_dim]
        z_neighbors: 邻居节点的嵌入，形状为 [num_neighbors, embedding_dim]
        delta_t_neighbors: 时间差，形状为 [num_neighbors, 1]
        返回值：注意力系数α，形状为 [num_neighbors]
        """
        # 计算μ(i, x)
        mu_ix = self.compute_mu(z_neighbors, z_i.expand_as(z_neighbors))  # [num_neighbors]
        # 时间编码
        phi_t = self.time_encoding(delta_t_neighbors)  # [num_neighbors]
        # 计算注意力得分
        exp_mu = torch.exp(mu_ix / self.temperature) * torch.exp(-phi_t)
        # 归一化
        alpha = exp_mu / (exp_mu.sum() + 1e-8)
        return alpha  # [num_neighbors]

    def compute_lambda_T(self, z, query_idx, neighbor_idxs, edge_times, current_time, G):
        """
        计算时间强度函数λ_{(x,y)}^T(t)
        """
        z_query = z[query_idx].unsqueeze(0)  # [1, embedding_dim]
        z_neighbors = z[neighbor_idxs]  # [num_neighbors, embedding_dim]

        # 计算μ(x, y)
        mu_xy = self.compute_mu(z_query, z_neighbors)  # [num_neighbors]

        # 获取查询节点和邻居节点的邻居
        neighbors_of_query = G.neighbors(query_idx)
        neighbors_of_neighbors = [G.neighbors(idx) for idx in neighbor_idxs]

        lambda_T_list = []
        for i, neighbor_idx in enumerate(neighbor_idxs):
            # 查询节点的邻居
            z_Nx = z[neighbors_of_query]  # [num_Nx, embedding_dim]
            delta_t_Nx = current_time - edge_times[(query_idx, neighbors_of_query)]  # [num_Nx, 1]
            alpha_i_y = self.compute_alpha(z_neighbors[i].unsqueeze(0), z_Nx, delta_t_Nx.unsqueeze(1))  # [num_Nx]

            mu_i_y = self.compute_mu(z_Nx, z_neighbors[i].expand_as(z_Nx))  # [num_Nx]

            # 邻居节点的邻居
            z_Ny = z[neighbors_of_neighbors[i]]  # [num_Ny, embedding_dim]
            delta_t_Ny = current_time - edge_times[(neighbor_idx, neighbors_of_neighbors[i])]  # [num_Ny, 1]
            alpha_i_x = self.compute_alpha(z_query, z_Ny, delta_t_Ny.unsqueeze(1))  # [num_Ny]

            mu_i_x = self.compute_mu(z_Ny, z_query.expand_as(z_Ny))  # [num_Ny]

            # 计算λ_{(x,y)}^T(t)
            lambda_T = mu_xy[i] + (alpha_i_y * mu_i_y).sum() + (alpha_i_x * mu_i_x).sum()
            lambda_T_list.append(lambda_T)

        lambda_T = torch.stack(lambda_T_list)  # [num_neighbors]
        return lambda_T

    def compute_lambda_S(self, z, query_idx, neighbor_idxs, G):
        """
        计算结构强度函数λ_{(x,y)}^S
        """
        z_query = z[query_idx].unsqueeze(0)  # [1, embedding_dim]
        z_neighbors = z[neighbor_idxs]  # [num_neighbors, embedding_dim]

        # 计算μ(x, y)
        mu_xy = self.compute_mu(z_query, z_neighbors)  # [num_neighbors]

        # 由于是结构强度，不考虑时间衰减和注意力系数，直接使用μ(x, y)
        lambda_S = mu_xy  # [num_neighbors]
        return lambda_S

    def compute_tightness_loss(self, lambda_S, neg_lambda_S):
        """
        计算紧密性损失L_tight
        lambda_S: 正样本的λ_{(x,y)}^S，形状为 [num_positive]
        neg_lambda_S: 负样本的λ_{(x,k)}^S，形状为 [num_negative]
        返回值：紧密性损失
        """
        pos_loss = -torch.log(torch.sigmoid(lambda_S) + 1e-8).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_lambda_S) + 1e-8).mean()
        tightness_loss = pos_loss + neg_loss
        return tightness_loss

    def compute_core_proximity_loss(self, cores, query_core):
        """
        计算核心接近损失
        cores: 社区中节点的核心值，形状为 [num_nodes]
        query_core: 查询节点的核心值，标量
        返回值：核心接近损失
        """
        core_loss = F.mse_loss(cores, query_core.expand_as(cores))
        return core_loss

    def compute_alignment_loss(self, lambda_T, lambda_S):
        """
        计算对齐损失L_A
        """
        delta_lambda = lambda_T - lambda_S  # [num_samples]
        abs_delta = torch.abs(delta_lambda)
        # L1和L2部分
        l2_part = 0.5 * delta_lambda.pow(2)
        l1_part = abs_delta - 0.5
        # 掩码
        mask = (abs_delta < 1).float()
        alignment_loss = (mask * l2_part + (1 - mask) * l1_part).mean()
        return alignment_loss

    def forward(self, z, query_idx, neg_idxs, neighbor_idxs, edge_times, current_time, G):
        """
        计算总损失函数
        """
        # 归一化嵌入
        z = F.normalize(z, dim=1)

        # 计算λ^T
        lambda_T = self.compute_lambda_T(z, query_idx, neighbor_idxs, edge_times, current_time, G)  # [num_neighbors]

        # 计算λ^S
        lambda_S = self.compute_lambda_S(z, query_idx, neighbor_idxs, G)  # [num_neighbors]

        # 计算紧密性损失
        pos_lambda_S = lambda_S  # 正样本
        # 负样本的λ^S
        neg_lambda_S = self.compute_lambda_S(z, query_idx, neg_idxs, G)  # [num_negative]
        tightness_loss = self.compute_tightness_loss(pos_lambda_S, neg_lambda_S)

        # 计算核心接近损失
        query_core = G.core_values[query_idx]
        neighbor_cores = G.core_values[neighbor_idxs]
        core_loss = self.compute_core_proximity_loss(neighbor_cores, query_core)

        # 计算对齐损失
        alignment_loss = self.compute_alignment_loss(lambda_T, lambda_S)

        # 总损失（可根据需要调整权重）
        total_loss = tightness_loss + 0.1 * core_loss + 0.1 * alignment_loss

        return total_loss