import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# # MLP 模型定义
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         return x.squeeze()

# # MLP 模型定义
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 新增的隐藏层
#         self.relu2 = nn.ReLU()                       # 新增的 ReLU 激活函数
#         self.fc3 = nn.Linear(hidden_dim, 1)          # 输出层
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)                              # 通过新增的隐藏层
#         x = self.relu2(x)                            # 通过新增的 ReLU 激活函数
#         x = self.fc3(x)                              # 通过输出层
#         return x.squeeze()

# # MLP 模型定义
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim, dropout_rate=0.5):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout_rate)  # 新增的 Dropout 层
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout_rate)  # 新增的第二个 Dropout 层
#         self.fc3 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)  # 通过第一个 Dropout 层
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)  # 通过第二个 Dropout 层
#         x = self.fc3(x)
#         return x.squeeze()

# # MLP 模型定义 - 单值特征
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim, dropout_rate=0.5):
#         super(MLP, self).__init__()
#         self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.additional_fc = nn.Linear(1, hidden_dim)  # 处理单一值特征
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 融合层
#         self.output_fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, additional_feature):
#         x = x.view(x.size(0), -1)
#         x = self.sequence_fc(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         additional_feature = additional_feature.unsqueeze(1)  # 扩展维度
#         additional_feature = self.additional_fc(additional_feature)
#         additional_feature = self.relu(additional_feature)
#         additional_feature = self.dropout(additional_feature)
#
#         combined = torch.cat((x, additional_feature), dim=1)
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#         combined = self.dropout(combined)
#
#         output = self.output_fc(combined)
#         return output.squeeze()

# # 去掉 Dropout 层的 MLP 模型定义
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim):
#         super(MLP, self).__init__()
#         # 分别处理时间序列特征和单一值特征
#         self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.additional_fc = nn.Linear(1, hidden_dim)
#
#         self.relu = nn.ReLU()
#
#         # 融合层，减少了额外的隐藏层
#         self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层
#
#     def forward(self, x, additional_feature):
#         # 排序
#         sorted_indices = torch.argsort(-x[:, :, 0], dim=1)
#         b, n, c = x.shape
#         x = torch.gather(x, 1, sorted_indices.unsqueeze(2).expand(b, n, c))
#
#         # 处理数据
#         x = x.view(x.size(0), -1)
#         x = self.sequence_fc(x)
#         x = self.relu(x)
#
#         # 处理单一值特征
#         if additional_feature.dim() == 1:  # 检查是否是一维张量
#             additional_feature = additional_feature.unsqueeze(1)  # 扩展维度
#         additional_feature = self.additional_fc(additional_feature)
#         additional_feature = self.relu(additional_feature)
#
#         # 融合两部分特征
#         combined = torch.cat((x, additional_feature), dim=1)
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#
#         # 输出层
#         output = self.output_fc(combined)
#         return output.squeeze()

# 双重排序
class MLP(nn.Module):
    def __init__(self, input_dim, max_seq_length, hidden_dim):
        super(MLP, self).__init__()
        # 分别处理时间序列特征和单一值特征
        self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
        self.additional_fc = nn.Linear(1, hidden_dim)

        self.relu = nn.ReLU()

        # 融合层，减少了额外的隐藏层
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, x, additional_feature):
        # 排序
        first_col = x[:, :, 0]  # 第一列
        second_col = x[:, :, 1]  # 第二列
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = x.shape
        x = torch.gather(x, 1, sorted_indices.unsqueeze(2).expand(b, n, c))

        # 处理数据
        x = x.view(x.size(0), -1)
        x = self.sequence_fc(x)
        x = self.relu(x)

        # 处理单一值特征
        if additional_feature.dim() == 1:  # 检查是否是一维张量
            additional_feature = additional_feature.unsqueeze(1)  # 扩展维度
        additional_feature = self.additional_fc(additional_feature)
        additional_feature = self.relu(additional_feature)

        # 融合两部分特征
        combined = torch.cat((x, additional_feature), dim=1)
        combined = self.fusion_fc(combined)
        combined = self.relu(combined)

        # 输出层
        output = self.output_fc(combined)
        return output.squeeze()

# # 去掉 Dropout 层的 MLP 模型定义，使用门控机制进行特征融合
# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim):
#         super(MLP, self).__init__()
#         # 分别处理时间序列特征和单一值特征
#         self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.additional_fc = nn.Linear(1, hidden_dim)
#
#         self.relu = nn.ReLU()
#
#         # 门控层，用于控制每个特征的贡献
#         self.gate_sequence = nn.Linear(hidden_dim, hidden_dim)
#         self.gate_additional = nn.Linear(hidden_dim, hidden_dim)
#
#         # 融合层，结合门控后的特征
#         self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层
#
#     def forward(self, x, additional_feature):
#         # 处理时间序列特征
#         x = x.view(x.size(0), -1)
#         x = self.sequence_fc(x)
#         x = self.relu(x)
#
#         # 处理单一值特征
#         if additional_feature.dim() == 1:  # 检查是否是一维张量
#             additional_feature = additional_feature.unsqueeze(1)  # 扩展维度
#         additional_feature = self.additional_fc(additional_feature)
#         additional_feature = self.relu(additional_feature)
#
#         # 计算门控权重
#         gate_x = torch.sigmoid(self.gate_sequence(x))  # [batch_size, hidden_dim]
#         gate_additional = torch.sigmoid(self.gate_additional(additional_feature))  # [batch_size, hidden_dim]
#
#         # 加权特征
#         gated_x = x * gate_x  # [batch_size, hidden_dim]
#         gated_additional = additional_feature * gate_additional  # [batch_size, hidden_dim]
#
#         # 融合两部分特征
#         combined = torch.cat((gated_x, gated_additional), dim=1)  # [batch_size, hidden_dim * 2]
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#
#         # 输出层
#         output = self.output_fc(combined)
#         return output.squeeze()

# # MLP 模型定义 - 注意力

# class MLP(nn.Module):
#     def __init__(self, input_dim, max_seq_length, hidden_dim):
#         super(MLP, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.sequence_fc = nn.Linear(input_dim * max_seq_length, hidden_dim)
#         self.additional_fc = nn.Linear(1, hidden_dim)
#         self.relu = nn.ReLU()
#
#         # 注意力层 - 输出维度修改为 hidden_dim * 2
#         self.attention_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
#         self.output_fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, additional_feature):
#         # 处理时间序列特征
#         x = x.view(x.size(0), -1)  # x 形状：[batch_size, input_dim * max_seq_length]
#         x = self.sequence_fc(x)    # x 形状：[batch_size, hidden_dim]
#         x = self.relu(x)
#
#         # 处理单一值特征
#         if additional_feature.dim() == 1:
#             additional_feature = additional_feature.unsqueeze(1)  # 扩展维度
#         additional_feature = self.additional_fc(additional_feature)  # 形状：[batch_size, hidden_dim]
#         additional_feature = self.relu(additional_feature)
#
#         # 拼接特征
#         combined = torch.cat((x, additional_feature), dim=1)  # 形状：[batch_size, hidden_dim * 2]
#
#         # 计算注意力权重
#         attention_scores = self.attention_fc(combined)  # 形状：[batch_size, hidden_dim * 2]
#         attention_weights = torch.sigmoid(attention_scores)
#
#         # 分割注意力权重
#         weight_x = attention_weights[:, :self.hidden_dim]  # 形状：[batch_size, hidden_dim]
#         weight_additional = attention_weights[:, self.hidden_dim:]  # 形状：[batch_size, hidden_dim]
#
#         # 加权融合
#         fused_feature = x * weight_x + additional_feature * weight_additional
#
#         # 激活和输出
#         fused_feature = self.relu(fused_feature)
#         output = self.output_fc(fused_feature)
#         return output.squeeze()

# 针对非叶节点的 MLP 模型定义 drop_out
# class MLPNonleaf(nn.Module):
#     def __init__(self, input_dim, max_seq_length1, max_seq_length2, hidden_dim, dropout_rate=0.5):
#         super(MLPNonleaf, self).__init__()
#         # 定义两个时间序列特征的线性层
#         self.sequence_fc1 = nn.Linear(input_dim * max_seq_length1, hidden_dim)
#         self.sequence_fc2 = nn.Linear(input_dim * max_seq_length2, hidden_dim)
#
#         # 定义单一特征值的线性层
#         self.single_feature_fc = nn.Linear(1, hidden_dim)
#
#         # 激活函数和 Dropout 层
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # 融合层
#         self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
#         self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层
#
#     def forward(self, sequence1, sequence2, single_feature):
#         # 处理第一个时间序列特征
#         x1 = sequence1.view(sequence1.size(0), -1)
#         x1 = self.sequence_fc1(x1)
#         x1 = self.relu(x1)
#         x1 = self.dropout(x1)
#
#         # 处理第二个时间序列特征
#         x2 = sequence2.view(sequence2.size(0), -1)
#         x2 = self.sequence_fc2(x2)
#         x2 = self.relu(x2)
#         x2 = self.dropout(x2)
#
#         # 处理单一特征值
#         if single_feature.dim() == 1:
#             single_feature = single_feature.unsqueeze(1)
#         x3 = self.single_feature_fc(single_feature)  # 扩展维度并输入线性层
#         x3 = self.relu(x3)
#         x3 = self.dropout(x3)
#
#         # 将三个特征合并
#         combined = torch.cat((x1, x2, x3), dim=1)
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#         combined = self.dropout(combined)
#
#         # 输出层
#         output = self.output_fc(combined)
#         return output.squeeze()

# 针对非叶节点的 MLP 模型定义
# class MLPNonleaf(nn.Module):
#     def __init__(self, input_dim, max_seq_length1, max_seq_length2, hidden_dim):
#         super(MLPNonleaf, self).__init__()
#         # 定义两个时间序列特征的线性层
#         self.sequence_fc1 = nn.Linear(input_dim * max_seq_length1, hidden_dim)
#         self.sequence_fc2 = nn.Linear(input_dim * max_seq_length2, hidden_dim)
#
#         # 定义单一特征值的线性层
#         self.single_feature_fc = nn.Linear(1, hidden_dim)
#
#         # 激活函数
#         self.relu = nn.ReLU()
#
#         # 融合层
#         self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
#         self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层
#
#     def forward(self, sequence1, sequence2, single_feature):
#         # 处理第一个时间序列特征
#         sorted_indices = torch.argsort(-sequence1[:, :, 0], dim=1)
#         b, n, c = sequence1.shape
#         sequence1 = torch.gather(sequence1, 1, sorted_indices.unsqueeze(2).expand(b, n, c))
#
#         x1 = sequence1.view(sequence1.size(0), -1)
#         x1 = self.sequence_fc1(x1)
#         x1 = self.relu(x1)
#
#         # 处理第二个时间序列特征
#         sorted_indices = torch.argsort(-sequence2[:, :, 0], dim=1)
#         b, n, c = sequence2.shape
#         sequence2 = torch.gather(sequence2, 1, sorted_indices.unsqueeze(2).expand(b, n, c))
#
#         x2 = sequence2.view(sequence2.size(0), -1)
#         x2 = self.sequence_fc2(x2)
#         x2 = self.relu(x2)
#
#         # 处理单一特征值
#         if single_feature.dim() == 1:
#             single_feature = single_feature.unsqueeze(1)
#         x3 = self.single_feature_fc(single_feature)  # 扩展维度并输入线性层
#         x3 = self.relu(x3)
#
#         # 将三个特征合并
#         combined = torch.cat((x1, x2, x3), dim=1)
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#
#         # 输出层
#         output = self.output_fc(combined)
#         return output.squeeze()

# 双重排序
class MLPNonleaf(nn.Module):
    def __init__(self, input_dim, max_seq_length1, max_seq_length2, hidden_dim):
        super(MLPNonleaf, self).__init__()
        # 定义两个时间序列特征的线性层
        self.sequence_fc1 = nn.Linear(input_dim * max_seq_length1, hidden_dim)
        self.sequence_fc2 = nn.Linear(input_dim * max_seq_length2, hidden_dim)

        # 定义单一特征值的线性层
        self.single_feature_fc = nn.Linear(1, hidden_dim)

        # 激活函数
        self.relu = nn.ReLU()

        # 融合层
        self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)  # 输出层

    # @profile
    def forward(self, sequence1, sequence2, single_feature):
        # 处理第一个时间序列特征
        first_col = sequence1[:, :, 0]  # 第一列
        second_col = sequence1[:, :, 1]  # 第二列
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = sequence1.shape
        sequence1 = torch.gather(sequence1, 1, sorted_indices.unsqueeze(2).expand(b, n, c))

        x1 = sequence1.view(sequence1.size(0), -1)
        x1 = self.sequence_fc1(x1)
        x1 = self.relu(x1)

        # 处理第二个时间序列特征
        first_col = sequence2[:, :, 0]  # 第一列
        second_col = sequence2[:, :, 1]  # 第二列
        sorted_indices = torch.argsort(-first_col * 1e5 - second_col, dim=1)
        b, n, c = sequence2.shape
        sequence2 = torch.gather(sequence2, 1, sorted_indices.unsqueeze(2).expand(b, n, c))

        x2 = sequence2.view(sequence2.size(0), -1)
        x2 = self.sequence_fc2(x2)
        x2 = self.relu(x2)

        # 处理单一特征值
        if single_feature.dim() == 1:
            single_feature = single_feature.unsqueeze(1)
        x3 = self.single_feature_fc(single_feature)  # 扩展维度并输入线性层
        x3 = self.relu(x3)

        # 将三个特征合并
        combined = torch.cat((x1, x2, x3), dim=1)
        combined = self.fusion_fc(combined)
        combined = self.relu(combined)

        # 输出层
        output = self.output_fc(combined)
        return output.squeeze()

# # 再多一个隐藏层
# class MLPNonleaf(nn.Module):
#     def __init__(self, input_dim, max_seq_length1, max_seq_length2, hidden_dim):
#         super(MLPNonleaf, self).__init__()
#         # 定义两个时间序列特征的线性层
#         self.sequence_fc1 = nn.Linear(input_dim * max_seq_length1, hidden_dim)
#         self.sequence_fc2 = nn.Linear(input_dim * max_seq_length2, hidden_dim)
#
#         # 定义单一特征值的线性层
#         self.single_feature_fc = nn.Linear(1, hidden_dim)
#
#         # 激活函数
#         self.relu = nn.ReLU()
#
#         # 融合层
#         self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
#
#         # 新增隐藏层
#         self.hidden_fc = nn.Linear(hidden_dim, hidden_dim // 2)
#
#         # 输出层
#         self.output_fc = nn.Linear(hidden_dim // 2, 1)
#
#     def forward(self, sequence1, sequence2, single_feature):
#         # 处理第一个时间序列特征
#         x1 = sequence1.view(sequence1.size(0), -1)
#         x1 = self.sequence_fc1(x1)
#         x1 = self.relu(x1)
#
#         # 处理第二个时间序列特征
#         x2 = sequence2.view(sequence2.size(0), -1)
#         x2 = self.sequence_fc2(x2)
#         x2 = self.relu(x2)
#
#         # 处理单一特征值
#         if single_feature.dim() == 1:
#             single_feature = single_feature.unsqueeze(1)
#         x3 = self.single_feature_fc(single_feature)
#         x3 = self.relu(x3)
#
#         # 将三个特征合并
#         combined = torch.cat((x1, x2, x3), dim=1)
#         combined = self.fusion_fc(combined)
#         combined = self.relu(combined)
#
#         # 新增隐藏层
#         combined = self.hidden_fc(combined)
#         combined = self.relu(combined)
#
#         # 输出层
#         output = self.output_fc(combined)
#         return output.squeeze()

# 序列预处理
def preprocess_sequences(sequences, max_length):
    sequences = [torch.tensor(seq[:max_length], dtype=torch.float32) for seq in sequences]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    if sequences_padded.size(1) < max_length:
        pad_size = max_length - sequences_padded.size(1)
        padding = torch.zeros((sequences_padded.size(0), pad_size, sequences_padded.size(2)), dtype=torch.float32)
        sequences_padded = torch.cat((sequences_padded, padding), dim=1)

    return sequences_padded


# def preprocess_sequences(sequences, max_length, device='cpu'):
#     batch_size = len(sequences)
#
#     # 确保所有序列都是 PyTorch 张量，并在指定设备上
#     sequences = [seq.to(device) if torch.is_tensor(seq) else torch.tensor(seq, dtype=torch.float32, device=device) for
#                  seq in sequences]
#
#     # 计算所有序列中的最大特征维度
#     max_feature_dim = 0
#     for seq in sequences:
#         if seq.dim() == 1:
#             current_dim = 1
#         else:
#             current_dim = seq.size(1)
#         max_feature_dim = max(max_feature_dim, current_dim)
#
#     # 预先分配零张量
#     sequences_padded = torch.zeros((batch_size, max_length, max_feature_dim), dtype=torch.float32, device=device)
#
#     for idx, seq in enumerate(sequences):
#         seq = seq[:max_length]
#         seq_length = seq.size(0)
#
#         # 确保序列的形状正确
#         if seq.dim() == 1:
#             # 序列为一维，转换为二维
#             seq = seq.view(seq_length, 1)
#
#         # 获取当前序列的特征维度
#         seq_feature_dim = seq.size(1)
#
#         # 如果当前序列的特征维度小于最大特征维度，则进行填充
#         if seq_feature_dim < max_feature_dim:
#             padding_width = max_feature_dim - seq_feature_dim
#             padding = torch.zeros((seq_length, padding_width), dtype=torch.float32, device=device)
#             seq = torch.cat((seq, padding), dim=1)
#
#         # 部分赋值
#         sequences_padded[idx, :seq_length, :] = seq
#
#     return sequences_padded


# def preprocess_sequences(sequences, max_length):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     sequences = [torch.tensor(seq[:max_length], dtype=torch.float32, device=device) for seq in sequences]
#     sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
#
#     if sequences_padded.size(1) < max_length:
#         pad_size = max_length - sequences_padded.size(1)
#         padding = torch.zeros(
#             (sequences_padded.size(0), pad_size, sequences_padded.size(2)),
#             dtype=torch.float32,
#             device=device
#         )
#         sequences_padded = torch.cat((sequences_padded, padding), dim=1)
#
#     return sequences_padded

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, additional_features, max_seq_length):
        self.sequences = preprocess_sequences(sequences, max_seq_length)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.additional_features = torch.tensor(additional_features, dtype=torch.float32)  # 新增的单一值特征

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.additional_features[idx], self.labels[idx]

class SequenceDatasetNonleaf(Dataset):
    def __init__(self, sequences, sequences_range, labels, additional_features, max_seq_length1, max_seq_length2):
        self.sequences = preprocess_sequences(sequences, max_seq_length1)
        self.sequences_range = preprocess_sequences(sequences_range, max_seq_length2)  # New
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.additional_features = torch.tensor(additional_features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences_range[idx], self.additional_features[idx], self.labels[idx]