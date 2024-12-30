import torch
import torch.nn.functional as F

def margin_triplet_loss(anchor, positives, negatives, margin=1):
    """
        计算三元组损失。

        参数：
        - anchor: 张量，形状为 (embedding_dim,)
        - positives: 列表，包含一个或多个张量，每个形状为 (embedding_dim,)
        - negatives: 列表，包含多个张量，每个形状为 (embedding_dim,)
        - margin: float,边距超参数

        返回：
        - loss: 标量，损失值
        """

    # 计算锚点与正样本的欧氏距离平方
    D_ap = torch.sum((anchor.unsqueeze(0) - positives) ** 2, dim=1)  # 形状：(num_positives,)

    # 计算锚点与负样本的欧氏距离平方
    D_an = torch.sum((anchor.unsqueeze(0) - negatives) ** 2, dim=1)  # 形状：(num_negatives,)

    # 扩展维度以便进行广播
    D_ap_expanded = D_ap.unsqueeze(1)  # 形状：(num_positives, 1)
    D_an_expanded = D_an.unsqueeze(0)  # 形状：(1, num_negatives)

    # 计算所有正负样本对的损失矩阵
    losses = torch.clamp(D_ap_expanded - D_an_expanded + margin, min=0.0)  # 形状：(num_positives, num_negatives)

    # 计算平均损失
    loss = losses.mean()
    return loss