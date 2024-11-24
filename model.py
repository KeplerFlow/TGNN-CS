import torch
import torch.nn as nn
from layer import *
from utils import *
import torch.nn.functional as F

class TemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_features_dim, num_layers):
        super(TemporalGNN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_set = LayerSet(in_channels, hidden_channels, temporal_features_dim)
            self.layers.append(layer_set)
    
    def forward(self, x, edge_index, timestamps, time_diffs):
        z = x
        temporal_features = encode_temporal_features(timestamps)
        for layer in self.layers:
            z = layer(z, edge_index, temporal_features, time_diffs)
        return z

class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, z, query_idx, neighbor_idx, edge_times, current_time, G):
        # 1. 对输入进行归一化
        z = F.normalize(z, dim=1)
        
        # 获取查询节点的嵌入
        query_emb = z[query_idx]  # [1, D]
        all_emb = z  # [N, D]
        
        # 2. 计算时序相似度
        temporal_sim = torch.mm(query_emb, all_emb.t()) / self.temperature  # [1, N]
        
        # 3. 计算core相似度 - 直接使用图中的core值
        query_core = G.x[query_idx, -1]  # 假设特征的最后一维是core值
        all_cores = G.x[:z.size(0), -1]  # 确保维度匹配
        
        # 归一化core差异
        core_diff = torch.abs(query_core - all_cores)
        max_diff = core_diff.max() + 1e-8
        core_diff = core_diff / max_diff
        core_sim = torch.exp(-core_diff / self.temperature)  # [N]
        core_sim = core_sim.unsqueeze(0)  # [1, N]
        
        # 4. 确保两个相似度矩阵维度匹配并归一化
        temporal_sim = F.softmax(temporal_sim, dim=1)  # [1, N]
        core_sim = F.softmax(core_sim, dim=1)  # [1, N]
        
        # 5. 组合两种相似度
        combined_sim = self.alpha * temporal_sim + (1 - self.alpha) * core_sim  # [1, N]
        
        # 6. 计算时间权重
        time_diffs = (current_time - edge_times).float()
        time_diffs = time_diffs / (time_diffs.max() + 1e-8)
        time_weights = torch.exp(-time_diffs)
        time_weights = F.normalize(time_weights, dim=0, p=1)
        
        # 仅选择邻居节点的时间权重
        neighbor_time_weights = time_weights[neighbor_idx]
        
        # 7. 计算对比损失
        log_prob = torch.log(combined_sim + 1e-8)  # 防止 log(0)
        positive_probs = log_prob[0, neighbor_idx]
        contrast_loss = -(positive_probs * neighbor_time_weights).sum()
        
        # 8. 添加core对齐损失
        neighbor_cores = G.x[neighbor_idx, -1]
        core_alignment_loss = F.mse_loss(neighbor_cores, query_core.expand_as(neighbor_cores))
        
        # 9. 组合损失
        total_loss = contrast_loss + 0.1 * core_alignment_loss
        
        # 10. 添加L2正则化
        l2_reg = 0.0001 * torch.norm(z, p=2)
        total_loss = total_loss + l2_reg
        
        return total_loss
