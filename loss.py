import torch

def margin_triplet_loss(anchor, positives, negatives, margin=1):

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


def compute_link_loss(embeddings, vertex_map, node_indices, t_start, t_end, anchor,time_range_link_samples_cache, margin=0.2):

    # 获取当前anchor的所有link samples
    link_samples_dict = time_range_link_samples_cache[(t_start, t_end)].get(anchor, {})
    if not link_samples_dict:
        return torch.tensor(0.0, device=embeddings.device)

    # 收集所有有效的样本
    vertices = []
    positives = []
    negatives = []

    # 一次性收集所有有效样本
    for vertex, samples in link_samples_dict.items():
        if vertex_map[vertex].item() != -1:  # 顶点在子图中
            for pos, neg in samples:
                if vertex_map[pos].item() != -1 and vertex_map[neg].item() != -1:
                    vertices.append(vertex)
                    positives.append(pos)
                    negatives.append(neg)

    if not vertices:  # 如果没有有效样本
        return torch.tensor(0.0, device=embeddings.device)

    # 转换为张量并一次性获取所有嵌入
    vertex_indices = node_indices[vertex_map[vertices]]
    pos_indices = node_indices[vertex_map[positives]]
    neg_indices = node_indices[vertex_map[negatives]]

    vertex_embs = embeddings[vertex_indices]
    pos_embs = embeddings[pos_indices]
    neg_embs = embeddings[neg_indices]

    # 批量计算距离
    d_pos = torch.sum((vertex_embs - pos_embs) ** 2, dim=1)
    d_neg = torch.sum((vertex_embs - neg_embs) ** 2, dim=1)

    # 批量计算损失
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)

    return losses.mean()

def quadruplet_collate_fn(batch):
    anchors = torch.tensor([item[0] for item in batch], dtype=torch.long)  # 锚点
    positives = [item[1] for item in batch]  # 正样本列表
    negatives = [item[2] for item in batch]  # 负样本列表
    time_ranges = [item[3] for item in batch]  # 时间范围
    return anchors, positives, negatives, time_ranges