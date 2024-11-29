import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops

class AdapterLayer(nn.Module):
    def __init__(self, in_channels, bottleneck_dim):
        super(AdapterLayer, self).__init__()
        self.down_proj = nn.Linear(in_channels, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.scaling_factor = nn.Parameter(torch.tensor(0.01))  # 可学习的缩放因子

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.bn(x)
        x = x * self.scaling_factor  # 应用可学习的缩放因子
        x = x + residual  # 残差连接
        return x

