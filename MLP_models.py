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

# 序列预处理
def preprocess_sequences(sequences, max_length):
    sequences = [torch.tensor(seq[:max_length], dtype=torch.float32) for seq in sequences]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    if sequences_padded.size(1) < max_length:
        pad_size = max_length - sequences_padded.size(1)
        padding = torch.zeros((sequences_padded.size(0), pad_size, sequences_padded.size(2)), dtype=torch.float32)
        sequences_padded = torch.cat((sequences_padded, padding), dim=1)

    return sequences_padded


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