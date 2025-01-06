import torch

from utils import *

def model_out_put_for_any_range_vertex_set(
    vertex_set,
    time_start,
    time_end,
    max_layer_id,
    max_time_range_layers,
    device,
    sequence_features1_matrix,
    partition,
    num_timestamp, 
    root
):
    # 直接用tensor
    # inter_time  # 假设不再需要这个变量，或者它应该从其他地方传入或计算
    node = tree_query(time_start, time_end,num_timestamp, root, max_layer_id)
    if node is None:
        print("Error: node not found.")
        return torch.zeros(len(vertex_set), device=device)  # 确保返回的 tensor 在正确的设备上
    model = node.model  # 设置模型为评估模式
    # model.eval()
    if node.layer_id == max_layer_id:
        max_length = max_time_range_layers[node.layer_id]

        vertex_indices = torch.tensor(vertex_set, device=device).sort().values

        # 获取 indices 和 values
        indices = sequence_features1_matrix.indices()  # (n, nnz)
        values = sequence_features1_matrix.values()  # (nnz,)

        start_idx = torch.searchsorted(indices[0], vertex_indices, side='left')
        end_idx = torch.searchsorted(indices[0], vertex_indices, side='right')

        range_lengths = end_idx - start_idx  # 每个范围的长度
        total_indices = range_lengths.sum()  # 总共的索引数

        range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
        flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

        mask_indices = start_idx.repeat_interleave(range_lengths) + flat_indices

        vertex_mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=device)
        vertex_mask[mask_indices] = True

        filtered_indices = indices[:, vertex_mask]
        filtered_values = values[vertex_mask]

        # 筛选时间范围
        time_mask = (
                (filtered_indices[1] >= time_start) &
                (filtered_indices[1] <= time_end)
        )
        final_indices = filtered_indices[:, time_mask]
        final_values = filtered_values[time_mask]

        # 构建 vertex_map 的张量版本
        vertex_map = torch.zeros(vertex_indices.max() + 1, dtype=torch.long, device=device)
        vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

        # 通过张量索引映射 final_indices[0]
        final_indices[0] = vertex_map[final_indices[0]]

        final_indices[1] -= time_start

        # 构造筛选后的稀疏张量
        result_size = (
            len(vertex_indices), max_length, sequence_features1_matrix.size(2)
        )
        result_sparse_tensor = torch.sparse_coo_tensor(
            final_indices, final_values, size=result_size, device=device
        )

        sequence_features = result_sparse_tensor.to_dense()

        # 设置single_value
        single_value = model_output_for_path(time_start, time_end, vertex_set, sequence_features, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition)

        # 将特征格式化为模型输入
        sequence_features = sequence_features.to(device)
        single_value = single_value.to(device)
        with torch.no_grad():
            output = model(sequence_features, single_value)

        return output
    else:
        # 直接截取特征
        covered_nodes = []
        sequence_features2 = torch.zeros(len(vertex_set), partition, 2, device=device)

        for child_node in node.children:
            if child_node.time_start >= time_start and child_node.time_end <= time_end:
                covered_nodes.append(child_node)

        for idx, v in enumerate(vertex_set):
            for idx2, temp_node in enumerate(covered_nodes):
                core_number = temp_node.vertex_core_number.get(v, 0)
                num_neighbor = temp_node.vertex_degree[v]
                sequence_features2[idx, idx2, 0] = core_number
                sequence_features2[idx, idx2, 1] = num_neighbor

        # 处理feature1

        max_length = max_time_range_layers[node.layer_id + 1] * 2

        vertex_indices = torch.tensor(vertex_set, device=device)
        vertex_indices = torch.sort(vertex_indices).values

        # 获取 indices 和 values
        indices = sequence_features1_matrix.indices()  # (n, nnz)
        values = sequence_features1_matrix.values()  # (nnz,)

        start_idx = torch.searchsorted(indices[0], vertex_indices, side='left')
        end_idx = torch.searchsorted(indices[0], vertex_indices, side='right')

        range_lengths = end_idx - start_idx  # 每个范围的长度
        total_indices = range_lengths.sum()  # 总共的索引数

        range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
        flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

        mask_indices = start_idx.repeat_interleave(range_lengths) + flat_indices

        vertex_mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=device)
        vertex_mask[mask_indices] = True

        filtered_indices = indices[:, vertex_mask]
        filtered_values = values[vertex_mask]

        # 构建 vertex_map 的张量版本
        vertex_map = torch.zeros(vertex_indices.max() + 1, dtype=torch.long, device=device)
        vertex_map[vertex_indices] = torch.arange(len(vertex_indices), device=device)

        if len(covered_nodes) == 0:
            # 筛选时间范围
            time_mask = (
                    (filtered_indices[1] >= time_start) &
                    (filtered_indices[1] <= time_end)
            )
            final_indices = filtered_indices[:, time_mask]
            final_values = filtered_values[time_mask]

            final_indices[0] = vertex_map[final_indices[0]]

            final_indices[1] -= time_start

            # 构造筛选后的稀疏张量
            result_size = (
                len(vertex_indices), max_length, sequence_features1_matrix.size(2)
            )
            result_sparse_tensor = torch.sparse_coo_tensor(
                final_indices, final_values, size=result_size, device=device
            )

            sequence_features1 = result_sparse_tensor.to_dense()

            single_value = model_output_for_path(time_start, time_end, vertex_set, sequence_features1_matrix, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition)
        else:
            # 改变time mask
            time_mask = (
                    (filtered_indices[1] >= time_start) &
                    (filtered_indices[1] <= time_end) & (filtered_indices[1]) < covered_nodes[0].time_start & (filtered_indices[1] > covered_nodes[-1].time_end)
            )
            final_indices = filtered_indices[:, time_mask]
            final_values = filtered_values[time_mask]
            for i in range(len(final_indices[0])):
                rank = vertex_map.get(final_indices[0][i].item(), 0)
                final_indices[0][i] = rank
            final_indices[1] -= time_start
            result_size = (
                len(vertex_indices), max_length, sequence_features1_matrix.size(2)
            )
            result_sparse_tensor = torch.sparse_coo_tensor(
                final_indices, final_values, size=result_size, device=device
            )
            sequence_features1 = result_sparse_tensor.to_dense()

            # 额外构造一个张量
            time_mask2 = (
                    (filtered_indices[1] >= time_start) &
                    (filtered_indices[1] <= time_end)
            )
            final_indices2 = filtered_indices[:, time_mask2]
            final_values2 = filtered_values[time_mask2]

            final_indices2[0] = vertex_map[final_indices2[0]]

            final_indices2[1] -= time_start
            result_size2 = (
                len(vertex_indices), time_end-time_start+1, sequence_features1_matrix.size(2)
            )
            result_sparse_tensor2 = torch.sparse_coo_tensor(
                final_indices2, final_values2, size=result_size2, device=device
            )
            sequence_features1_extra = result_sparse_tensor2.to_dense()

            single_value = model_output_for_path(time_start, time_end, vertex_set, sequence_features1_extra, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition)
            # single_value = torch.zeros(len(vertex_set), 1, device=device)

        # 将特征格式化为模型输入
        sequence_features1 = sequence_features1.to(device)
        sequence_features2 = sequence_features2.to(device)
        single_value = single_value.to(device)
        with torch.no_grad():
            output = model(sequence_features1, sequence_features2, single_value)
        return output