import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch.nn import Parameter, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GATConv, BatchNorm
from torch_scatter import scatter_mean  # 新添加的导入

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from typing import Union, Tuple, Optional

from torch_geometric.utils.sparse import set_sparse_value


# # GAT
# class TemporalGNN(nn.Module):
#     def __init__(self, node_in_channels, node_out_channels, edge_dim=None):
#         super(TemporalGNN, self).__init__()
#         self.conv1 = GATConv(node_in_channels, 64, edge_dim=edge_dim, heads=4, concat=True)
#         self.bn1 = nn.BatchNorm1d(64 * 4)
#         self.conv2 = GATConv(64 * 4, 32, heads=4, edge_dim=edge_dim, concat=True)
#         self.bn2 = nn.BatchNorm1d(32 * 4)
#         self.conv3 = GATConv(32 * 4, node_out_channels, heads=1, concat=False)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.elu(self.bn1(self.conv1(x, edge_index)))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.elu(self.bn2(self.conv2(x, edge_index)))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv3(x, edge_index)
#         return x


# TGAT
class TemporalGNN(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, edge_dim):
        super(TemporalGNN, self).__init__()
        # 第一层：使用8个注意力头，concat=True
        self.conv1 = GATWithEdgeChannelAttention(
            in_channels=node_in_channels,
            out_channels=64,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn1 = nn.BatchNorm1d(64 * 4)  # 因为concat=True，所以是64*8

        # 第二层：使用8个注意力头，concat=True
        self.conv2 = GATWithEdgeChannelAttention(
            in_channels=64 * 4,  # 输入维度是第一层的输出
            out_channels=32,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn2 = nn.BatchNorm1d(32 * 4)  # 因为concat=True，所以是32*8

        # 第三层：使用1个注意力头，concat=False
        self.conv3 = GATWithEdgeChannelAttention(
            in_channels=32 * 4,  # 输入维度是第二层的输出
            out_channels=node_out_channels,
            heads=1,
            concat=False,  # 最后一层不需要concat
            edge_dim=edge_dim,
        )

    def forward(self, data):
        # 获取节点特征、边索引和边特征
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 第一层
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第三层
        x = self.conv3(x, edge_index, edge_attr)

        return x
    
class AdapterTemporalGNN(nn.Module):
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_dim,
                 adapter_dim=16,
                 use_adapter3=False
                 ):
        super(AdapterTemporalGNN, self).__init__()
        
        # ---------------------
        # 第 1 层 GAT
        # ---------------------
        # Pre-aggregation adapter
        self.pre_adapter1 = Adapter(in_channels=node_in_channels, 
                                  adapter_dim=adapter_dim, 
                                  edge_dim=edge_dim)
        
        self.conv1 = GATWithEdgeChannelAttention(
            in_channels=node_in_channels,
            out_channels=64,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn1 = nn.BatchNorm1d(64 * 4)
        
        # Post-aggregation adapter
        self.post_adapter1 = Adapter(in_channels=64*4, 
                                   adapter_dim=adapter_dim, 
                                   edge_dim=edge_dim)
        
        # ---------------------
        # 第 2 层 GAT
        # ---------------------
        # Pre-aggregation adapter
        self.pre_adapter2 = Adapter(in_channels=64*4, 
                                  adapter_dim=adapter_dim, 
                                  edge_dim=edge_dim)
        
        self.conv2 = GATWithEdgeChannelAttention(
            in_channels=64 * 4,
            out_channels=32,
            heads=4,
            concat=True,
            edge_dim=edge_dim,
        )
        self.bn2 = nn.BatchNorm1d(32 * 4)
        
        # Post-aggregation adapter
        self.post_adapter2 = Adapter(in_channels=32*4, 
                                   adapter_dim=adapter_dim, 
                                   edge_dim=edge_dim)
        
        # ---------------------
        # 第 3 层 GAT
        # ---------------------
        
        self.conv3 = GATWithEdgeChannelAttention(
            in_channels=32 * 4,
            out_channels=node_out_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        
        # Gating parameters for all adapters
        self.gating_params = nn.ParameterDict({
            "pre_gating1": nn.Parameter(torch.tensor(0.1)),
            "post_gating1": nn.Parameter(torch.tensor(0.1)),
            "pre_gating2": nn.Parameter(torch.tensor(0.1)),
            "post_gating2": nn.Parameter(torch.tensor(0.1)),
        })

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # ---- 第1层 ----
        # Pre-aggregation adapter
        delta_pre1 = self.pre_adapter1(x, edge_index, edge_attr)
        x = x + self.gating_params["pre_gating1"] * (delta_pre1 - x)
        
        # GNN layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        
        # Post-aggregation adapter
        delta_post1 = self.post_adapter1(x, edge_index, edge_attr)
        x = x + self.gating_params["post_gating1"] * (delta_post1 - x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        # ---- 第2层 ----
        # Pre-aggregation adapter
        delta_pre2 = self.pre_adapter2(x, edge_index, edge_attr)
        x = x + self.gating_params["pre_gating2"] * (delta_pre2 - x)
        
        # GNN layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        
        # Post-aggregation adapter
        delta_post2 = self.post_adapter2(x, edge_index, edge_attr)
        x = x + self.gating_params["post_gating2"] * (delta_post2 - x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        # ---- 第3层 ----
        # GNN layer
        x = self.conv3(x, edge_index, edge_attr)
        
        return x
    
class Adapter(nn.Module):
    """
    对输入做轻量级映射，结合时间信息(傅里叶变换后的时间戳)，并通过残差方式加回
    """
    def __init__(self, in_channels, adapter_dim, edge_dim):
        super(Adapter, self).__init__()

        self.down = nn.Linear(in_channels, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, in_channels)

        # 时间信息处理 (针对傅里叶变换后的特征)
        self.time_proj = nn.Sequential(
            nn.Linear(edge_dim, adapter_dim),
            nn.ReLU()
        )

        # 注意：由于是频域信息，可以考虑使用不同的融合策略
        # 方案 1: 仍然使用简单的线性融合层
        self.fusion = nn.Linear(adapter_dim * 2, adapter_dim)
        # 方案 2: 使用更复杂的融合策略，例如注意力机制 (下面提供示例代码)
        # self.fusion = TimeAttentionFusion(adapter_dim)  # 下面提供了该模块的实现代码

    def forward(self, x, edge_index, edge_attr):
        # 1. 处理节点特征
        node_feat = self.down(x)
        node_feat = self.activation(node_feat)

        # 2. 处理时间特征 (已经是频域特征)
        time_feat = self.time_proj(edge_attr)  # [num_edges, adapter_dim]

        # 3. 聚合时间信息到节点
        node_time_feat = scatter_mean(
            time_feat,
            edge_index[0],  # 使用源节点进行聚合
            dim=0,
            dim_size=x.size(0)
        )  # [num_nodes, adapter_dim]

        # 4. 融合节点特征和时间特征
        combined = torch.cat([node_feat, node_time_feat], dim=-1)
        fused = self.fusion(combined)
        fused = self.activation(fused)
        
        # 5. 最终映射和残差连接
        out = self.up(fused)
        return x + out

# 方案 2 的补充：使用注意力机制进行融合 (可选)
class TimeAttentionFusion(nn.Module):
    def __init__(self, adapter_dim):
        super(TimeAttentionFusion, self).__init__()
        self.query = nn.Linear(adapter_dim, adapter_dim)
        self.key = nn.Linear(adapter_dim, adapter_dim)
        self.value = nn.Linear(adapter_dim, adapter_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(adapter_dim, adapter_dim)

    def forward(self, combined):
        node_feat = combined[:, :combined.shape[1] // 2]
        node_time_feat = combined[:, combined.shape[1] // 2:]

        q = self.query(node_feat)
        k = self.key(node_time_feat)
        v = self.value(node_time_feat)

        attn_weights = self.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5))  # Scaled dot-product attention
        attn_output = attn_weights @ v
        
        fused = self.out(attn_output)
        return fused
      
# 时态信息-特征拼接-再计算注意力系数
class GATWithEdgeChannelAttention(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # 节点特征转换层
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # 边特征通道注意力MLP
        if edge_dim is not None:
            self.edge_channel_attention = nn.Sequential(
                Linear(edge_dim, edge_dim // 2),
                nn.ReLU(),
                Linear(edge_dim // 2, edge_dim),
                nn.Sigmoid()
            )

            # 注意力计算的线性变换
            # [W*h_i || W*h_j || e_ij] 的维度为 2*out_channels*heads + edge_dim
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        else:
            self.edge_channel_attention = None
            # 没有边特征时,只用节点特征计算注意力
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = Parameter(torch.empty(heads * out_channels if concat else out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
        else:
            self.lin_src.reset_parameters()
            self.lin_dst.reset_parameters()

        if self.edge_channel_attention is not None:
            for layer in self.edge_channel_attention:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # 1. 节点特征线性变换
        if isinstance(x, Tensor):
            x_src = x_dst = self.lin(x).view(-1, H, C)
        else:
            x_src, x_dst = x[0], x[1]
            x_src = self.lin_src(x_src).view(-1, H, C)
            x_dst = self.lin_dst(x_dst).view(-1, H, C) if x_dst is not None else None

        x = (x_src, x_dst)

        # 2. 处理自环
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)

        # 3. 计算注意力权重
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        # 4. 处理输出
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            alpha = self._alpha
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            else:
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        """
        计算消息和注意力权重
        x_i: 目标节点特征
        x_j: 源节点特征
        edge_attr: 边特征
        """
        # 1. 计算边特征的通道注意力
        if edge_attr is not None and self.edge_channel_attention is not None:
            # 计算通道注意力系数
            channel_weights = self.edge_channel_attention(edge_attr)
            # 加权边特征
            edge_attr_weighted = edge_attr * channel_weights

            # 将节点特征和加权边特征拼接
            alpha = torch.cat([x_i, x_j, edge_attr_weighted.unsqueeze(1).expand(-1, self.heads, -1)], dim=-1)
        else:
            # 没有边特征时只用节点特征
            alpha = torch.cat([x_i, x_j], dim=-1)

        # 2. 计算注意力分数
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 3. softmax归一化
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # 保存用于返回

        # 4. dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 5. 加权消息
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


# # 时态信息-注意力系数直接相加
# class GATWithEdgeChannelAttention(MessagePassing):
#     """
#     带有边特征通道注意力的图注意力网络层

#     Args:
#         in_channels: 输入特征维度
#         out_channels: 输出特征维度
#         heads: 注意力头数
#         concat: 是否拼接多头注意力的结果
#         negative_slope: LeakyReLU的负斜率
#         dropout: Dropout概率
#         add_self_loops: 是否添加自环
#         edge_dim: 边特征维度
#         fill_value: 自环边特征的填充方式
#         bias: 是否使用偏置
#         residual: 是否使用残差连接
#     """

#     def __init__(
#             self,
#             in_channels: Union[int, Tuple[int, int]],
#             out_channels: int,
#             heads: int = 1,
#             concat: bool = True,
#             negative_slope: float = 0.2,
#             dropout: float = 0.0,
#             add_self_loops: bool = True,
#             edge_dim: Optional[int] = None,
#             fill_value: Union[float, Tensor, str] = 'mean',
#             bias: bool = True,
#             residual: bool = False,
#             **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value
#         self.residual = residual

#         # 节点特征转换层
#         if isinstance(in_channels, int):
#             self.lin = Linear(in_channels, heads * out_channels, bias=False,
#                               weight_initializer='glorot')
#             self.lin_src = self.lin_dst = None
#         else:
#             self.lin = None
#             self.lin_src = Linear(in_channels[0], heads * out_channels, False,
#                                   weight_initializer='glorot')
#             self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
#                                   weight_initializer='glorot')

#         # 节点注意力参数
#         self.att_src = Parameter(torch.empty(1, heads, out_channels))
#         self.att_dst = Parameter(torch.empty(1, heads, out_channels))

#         # 边特征相关层
#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.empty(1, heads, out_channels))

#             # 边特征通道注意力层
#             self.edge_channel_attention = nn.Sequential(
#                 Linear(edge_dim, edge_dim // 2),
#                 nn.ReLU(),
#                 Linear(edge_dim // 2, edge_dim),
#                 nn.Sigmoid()
#             )
#         else:
#             self.lin_edge = None
#             self.register_parameter('att_edge', None)
#             self.edge_channel_attention = None

#         # 残差连接
#         if residual:
#             self.res = Linear(
#                 in_channels if isinstance(in_channels, int) else in_channels[1],
#                 heads * out_channels if concat else out_channels,
#                 bias=False,
#                 weight_initializer='glorot',
#             )
#         else:
#             self.register_parameter('res', None)

#         # 偏置
#         if bias:
#             self.bias = Parameter(torch.empty(heads * out_channels if concat else out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         """重置所有可学习参数"""
#         if self.lin is not None:
#             self.lin.reset_parameters()
#         if self.lin_src is not None:
#             self.lin_src.reset_parameters()
#             self.lin_dst.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         if self.edge_channel_attention is not None:
#             for layer in self.edge_channel_attention:
#                 if hasattr(layer, 'reset_parameters'):
#                     layer.reset_parameters()
#         if self.res is not None:
#             self.res.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         if self.att_edge is not None:
#             glorot(self.att_edge)
#         if self.bias is not None:
#             zeros(self.bias)

#     def forward(
#             self,
#             x: Union[Tensor, OptPairTensor],
#             edge_index: Adj,
#             edge_attr: OptTensor = None,
#             size: Size = None,
#             return_attention_weights: Optional[bool] = None,
#     ):
#         """
#         前向传播

#         Args:
#             x: 节点特征
#             edge_index: 边索引
#             edge_attr: 边特征
#             size: 邻接矩阵的形状
#             return_attention_weights: 是否返回注意力权重
#         """

#         H, C = self.heads, self.out_channels

#         # 处理残差连接
#         res = None
#         if self.residual:
#             if isinstance(x, Tensor):
#                 res = self.res(x)
#             elif isinstance(x, tuple) and x[1] is not None:
#                 res = self.res(x[1])

#         # 节点特征转换
#         if isinstance(x, Tensor):
#             x_src = x_dst = self.lin(x).view(-1, H, C)
#         else:
#             x_src, x_dst = x[0], x[1]
#             x_src = self.lin_src(x_src).view(-1, H, C)
#             if x_dst is not None:
#                 x_dst = self.lin_dst(x_dst).view(-1, H, C)

#         x = (x_src, x_dst)

#         # 计算节点级别的注意力系数
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         # 处理自环
#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 num_nodes = x_src.size(0)
#                 if x_dst is not None:
#                     num_nodes = min(num_nodes, x_dst.size(0))
#                 num_nodes = min(size) if size is not None else num_nodes
#                 edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = torch_sparse.set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # 更新边的注意力系数
#         alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

#         # 消息传递
#         out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

#         # 处理多头注意力的输出
#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         # 添加残差连接
#         if res is not None:
#             out = out + res

#         # 添加偏置
#         if self.bias is not None:
#             out = out + self.bias

#         # 返回结果
#         if isinstance(return_attention_weights, bool):
#             if isinstance(edge_index, Tensor):
#                 if is_torch_sparse_tensor(edge_index):
#                     adj = set_sparse_value(edge_index, alpha)
#                     return out, (adj, alpha)
#                 else:
#                     return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out

#     def edge_update(
#             self,
#             alpha_j: Tensor,
#             alpha_i: OptTensor,
#             edge_attr: OptTensor,
#             index: Tensor,
#             ptr: OptTensor,
#             size_i: Optional[int]
#     ) -> Tensor:
#         """
#         更新边的注意力系数

#         Args:
#             alpha_j: 源节点的注意力分数
#             alpha_i: 目标节点的注意力分数
#             edge_attr: 边特征
#             index: 边的目标节点索引
#             ptr: 用于分段softmax的指针
#             size_i: 目标节点数量
#         """
#         # 合并源节点和目标节点的注意力分数
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

#         if index.numel() == 0:
#             return alpha

#         # 处理边特征
#         if edge_attr is not None and self.lin_edge is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)

#             # 应用通道注意力
#             if self.edge_channel_attention is not None:
#                 # 计算通道注意力权重
#                 channel_weights = self.edge_channel_attention(edge_attr)
#                 # 加权边特征
#                 edge_attr = edge_attr * channel_weights

#             # 边特征转换
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             # 计算边注意力分数
#             alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
#             alpha = alpha + alpha_edge

#         # 应用激活函数和归一化
#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)

#         return alpha

#     def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
#         """定义消息如何传递"""
#         return alpha.unsqueeze(-1) * x_j

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')

