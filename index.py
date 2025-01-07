import torch

from utils import *

def tree_query(time_start, time_end, num_timestamp, root, max_layer_id):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None

    node = root
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                move_to_next = True
                break
        if not move_to_next:
            break
    return node

def model_output_for_path(time_start, time_end, vertex_set, sequence_features, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return torch.zeros(len(vertex_set), 1, device=device)
    sequence_features = sequence_features.to(device)
    node = root
    path = [node]
    while node.layer_id < max_layer_id:
        move_to_next = False
        for child in node.children:
            if child.time_start <= time_start and child.time_end >= time_end:
                node = child
                path.append(node)
                move_to_next = True
                break
        if not move_to_next:
            break
    if len(path) == 1:
        return torch.zeros(len(vertex_set), 1, device=device)

    # 计算模型输出
    path.pop()
    output = torch.zeros(len(vertex_set), 1, dtype=torch.float32, device=device)
    # output = output.to(device)
    sequence_input1 = torch.zeros(len(vertex_set), max_time_range_layers[0], 2, device=device)
    sequence_input1[:, 0:sequence_features.shape[1], :] = sequence_features
    for node in path:
        max_length1 = max_time_range_layers[node.layer_id + 1] * 2
        max_length2 = partition
        # 构造模型输入
        sequence_input1 = sequence_input1[:, :max_length1, :]
        sequence_input2 = torch.zeros(len(vertex_set), max_length2, 2, device=device)

        single_value = output

        # 所有张量已经在 device 上，无需再次调用 .to(device)
        model = node.model
        # model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            output = model(sequence_input1, sequence_input2, single_value)
            if output.dim() == 0:
                output = output.reshape(len(vertex_set), 1)
    return output

def model_out_put_for_any_range_vertex_set(vertex_set, time_start, time_end, max_layer_id, max_time_range_layers, device, sequence_features1_matrix, partition, num_timestamp, root):
    # 直接用tensor
    # global inter_time
    node = tree_query(time_start, time_end,num_timestamp, root, max_layer_id)
    if node is None:
        print("Error: node not found.")
        return torch.zeros(len(vertex_set), 1, device=device)
    model = node.model  # 设置模型为评估模式
    # model.eval()
    if node.layer_id == max_layer_id:
        max_length = max_time_range_layers[node.layer_id]

        vertex_indices = torch.tensor(vertex_set, device=device)
        vertex_indices = torch.sort(vertex_indices).values

         # 获取 indices 和 values
        indices = sequence_features1_matrix.indices()  # (n, nnz)
        values = sequence_features1_matrix.values()  # (nnz,)

        start_idx = torch.searchsorted(indices[0], vertex_indices, side='left')
        end_idx = torch.searchsorted(indices[0], vertex_indices, side='right')

         # mask_indices = torch.cat([torch.arange(start, end, device=device) for start, end in zip(start_idx, end_idx)])

         # 利用 start_idx 和 end_idx 构建 mask_indices，而不是逐个遍历生成
        range_lengths = end_idx - start_idx  # 每个范围的长度
        total_indices = range_lengths.sum()  # 总共的索引数

         # 创建一个平铺的范围张量，直接表示所有 mask_indices
        range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
        flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

         # 映射 flat_indices 到实际的 mask_indices 范围
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
        # 先处理feature2
        covered_nodes = []
        sequence_features2 = torch.zeros(len(vertex_set), partition, 2, device=device)

        for child_node in node.children:
            if child_node.time_start >= time_start and child_node.time_end <= time_end:
                covered_nodes.append(child_node)

        for idx, v in enumerate(vertex_set):
            for idx2, temp_node in enumerate(covered_nodes):
                core_number = temp_node.vertex_core_number.get(v, 0)
                num_neighbor = temp_node.vertex_degree.get(v, 0)
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

        # mask_indices = torch.cat([torch.arange(start, end, device=device) for start, end in zip(start_idx, end_idx)])

        # 利用 start_idx 和 end_idx 构建 mask_indices，而不是逐个遍历生成
        range_lengths = end_idx - start_idx  # 每个范围的长度
        total_indices = range_lengths.sum()  # 总共的索引数

        # 创建一个平铺的范围张量，直接表示所有 mask_indices
        range_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), range_lengths.cumsum(dim=0)[:-1]])
        flat_indices = torch.arange(total_indices, device=device) - range_offsets.repeat_interleave(range_lengths)

        # 映射 flat_indices 到实际的 mask_indices 范围
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
            single_value = model_output_for_path(time_start, time_end, vertex_set, sequence_features1, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition)


        else:
            # 改变time mask
            time_mask = (
                    (filtered_indices[1] >= time_start) &
                    (filtered_indices[1] <= time_end) & (filtered_indices[1]) < covered_nodes[0].time_start & (filtered_indices[1] > covered_nodes[-1].time_end)
            )
            final_indices = filtered_indices[:, time_mask]
            final_values = filtered_values[time_mask]

            final_indices[0] = vertex_map[final_indices[0]]

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
                len(vertex_indices), time_end - time_start + 1, sequence_features1_matrix.size(2)
            )
            result_sparse_tensor2 = torch.sparse_coo_tensor(
                final_indices2, final_values2, size=result_size2, device=device
            )
            sequence_features1_extra = result_sparse_tensor2.to_dense()

            single_value = model_output_for_path(time_start, time_end, vertex_set, sequence_features1_extra, 
                          num_timestamp, root, max_layer_id, device, max_time_range_layers, partition)


        # 将特征格式化为模型输入
        sequence_features1 = sequence_features1.to(device)
        sequence_features2 = sequence_features2.to(device)
        single_value = single_value.to(device)
        with torch.no_grad():
            output = model(sequence_features1, sequence_features2, single_value)
        return output