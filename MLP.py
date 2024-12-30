import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from pympler import asizeof
import time

from MLP_models import MLP, MLPNonleaf, preprocess_sequences, SequenceDataset, SequenceDatasetNonleaf
from Tree import TreeNode

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 全局常量和变量
dataset_name = 'mathoverflow'
# dataset_name = 'wikitalk'
num_vertex = 0
num_edge = 0
num_timestamp = 0
time_edge = {}
num_core_number = 0
vertex_core_numbers = []
time_range_core_number = defaultdict(dict)
temporal_graph = []
time_range_layers = []
time_range_set = set()
max_time_range_layers = []
min_time_range_layers = []
max_layer_id = 0

model_time_range_layers = []
partition = 4
max_range = 40
max_degree = 0
root = None


# 读取时间序列图数据-优化内存
def read_temporal_graph():
    print("Loading the graph...")
    filename = f'../datasets/{dataset_name}.txt'
    global num_vertex, num_edge, num_timestamp, time_edge, temporal_graph, max_degree
    time_edge = defaultdict(list)
    temporal_graph = defaultdict(lambda: defaultdict(list))

    with open(filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                num_vertex, num_edge, num_timestamp = map(int, line.strip().split())
                first_line = False
                continue
            v1, v2, t = map(int, line.strip().split())
            time_edge[t].append((v1, v2))
            # 构建temporal_graph
            temporal_graph[v1][t].append(v2)
            temporal_graph[v2][t].append(v1)

    total_size = asizeof.asizeof(temporal_graph)
    print(f"temporal_graph 占用的内存大小为 {total_size / (1024 ** 2):.2f} MB")

# 读取节点的核心数
def read_core_number():
    print("Loading the core number...")
    global num_core_number, vertex_core_numbers
    vertex_core_numbers = [{} for _ in range(num_vertex)]
    core_number_filename = f'../datasets/{dataset_name}-core_number.txt'
    with open(core_number_filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                num_core_number = int(line.strip())
                first_line = False
                continue
            range_part, core_numbers_part = line.split(' ', 1)
            range_start, range_end = map(int, range_part.strip('[]').split(','))
            is_node_range = False
            if (range_start, range_end) in time_range_set:
                is_node_range = True
            for pair in core_numbers_part.split():
                vertex, core_number = map(int, pair.split(':'))
                vertex_core_numbers[vertex][(range_start, range_end)] = core_number
                if is_node_range:
                    time_range_core_number[(range_start, range_end)][vertex] = core_number

# 获取时间范围层次
def get_timerange_layers():
    global time_range_layers, min_time_range_layers, max_time_range_layers
    temp_range = num_timestamp
    while temp_range > max_range:
        temp_range = temp_range // partition
    layer_id = 0
    while temp_range < num_timestamp:
        range_start = 0
        time_range_layers.append([])
        temp_max_range = 0
        temp_min_range = num_timestamp
        while range_start < num_timestamp:
            range_end = range_start + temp_range - 1
            range_end = min(range_end, num_timestamp - 1)
            if range_end + temp_range > num_timestamp - 1:
                range_end = num_timestamp - 1
            time_range_layers[layer_id].append((range_start, range_end))
            time_range_set.add((range_start, range_end))
            if range_end - range_start + 1 > temp_max_range:
                temp_max_range = range_end - range_start + 1
            if range_end - range_start + 1 < temp_min_range:
                temp_min_range = range_end - range_start + 1
            range_start = range_end + 1
        max_time_range_layers.append(temp_max_range)
        min_time_range_layers.append(temp_min_range)
        temp_range = temp_range * partition
        layer_id = layer_id + 1
    time_range_layers.reverse()
    max_time_range_layers.reverse()
    min_time_range_layers.reverse()

def build_tree():
    node_stack = []
    root_node = TreeNode((0, num_timestamp-1), 0)
    global max_layer_id
    node_stack.append(root_node)
    while len(node_stack) > 0:
        current_node = node_stack.pop()
        current_node.vertex_core_number = time_range_core_number[(current_node.time_start, current_node.time_end)]
        if current_node.layer_id < len(time_range_layers) - 1:
            for i in range(len(time_range_layers[current_node.layer_id + 1])):
                temp_time_start = time_range_layers[current_node.layer_id +1][i][0]
                temp_time_end = time_range_layers[current_node.layer_id + 1][i][1]
                if temp_time_start >= current_node.time_start and temp_time_end <= current_node.time_end:
                    child_node = TreeNode((temp_time_start, temp_time_end), current_node.layer_id + 1)
                    current_node.add_child(child_node)
                    node_stack.append(child_node)
    max_layer_id = len(time_range_layers) - 1
    return root_node

def tree_query(time_start, time_end):
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



def get_node_path(time_start, time_end):
    node = root
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return None
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
    return path

def model_output_for_path(time_start, time_end, v):
    if time_start < 0 or time_end >= num_timestamp or time_start > time_end:
        return 0
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
        return 0

    # 计算模型输出
    path.pop()
    output = 0
    for node in path:
        # 构造模型输入
        sequence_features1 = []
        sequence_features2 = [(0, 0)]
        for t in range(time_start, time_end + 1):
            core_number = vertex_core_numbers[v].get((t, t), 0)
            num_neighbors = len(temporal_graph[v][t])  # 获取该节点在时间 t 的邻居个数
            sequence_features1.append((core_number, num_neighbors))  # 添加核心数和邻居数信息
        sequence_features1 = sorted(sequence_features1, key=lambda x: x[0], reverse=True)

        single_value = output

        # 将特征格式化为模型输入
        sequence_input1 = preprocess_sequences([sequence_features1],
                                               max_time_range_layers[node.layer_id + 1] * 2)  # 保留批次维度
        sequence_input2 = preprocess_sequences([sequence_features2], partition)  # 保留批次维度
        single_value_input = torch.tensor([single_value],
                                          dtype=torch.float32)  # 保持形状为 (1, 1)
        sequence_input1 = sequence_input1.to(device)
        sequence_input2 = sequence_input2.to(device)
        single_value_input = single_value_input.to(device)
        model = node.model  # 设置模型为评估模式
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            model_output = model(sequence_input1, sequence_input2, single_value_input)
            output = model_output.item()

    return output

# 获取序列和标签 叶子节点
def get_sequence_and_label_leaf(layer_id=0, timerange_id=0):
    print("Creating training samples...")
    # global h_index_error, valid_h_index
    sequence_features1_list, label_list, single_feature_list = [], [], []
    time_start = time_range_layers[layer_id][timerange_id][0]
    time_end = time_range_layers[layer_id][timerange_id][1]
    num_sample = 1500
    while len(sequence_features1_list) < num_sample:
        # random select a vertex
        v = random.randint(0, num_vertex - 1)
        if len(vertex_core_numbers[v].keys()) == 0:
            continue
        temp_time_range = random.choice(list(vertex_core_numbers[v].keys()))
        temp_start = temp_time_range[0]
        temp_end = temp_time_range[1]
        if temp_start != temp_end and temp_start >= time_start and temp_end <= time_end:
            sequence_features1 = []
            neighbor_projected_set = set()
            for t in range(temp_start, temp_end + 1):
                core_number = vertex_core_numbers[v].get((t, t), 0)
                num_neighbors = len(temporal_graph[v][t])  # 获取该节点在时间 t 的邻居个数
                num_better_neighbors = 0
                average_neighbor_core_number = 0
                average_neighbor_degree = 0
                for neighbor in temporal_graph[v][t]:
                    average_neighbor_core_number += vertex_core_numbers[neighbor].get((t, t), 0)
                    average_neighbor_degree += len(temporal_graph[neighbor][t])
                    neighbor_projected_set.add(neighbor)
                    if len(temporal_graph[neighbor][t]) >= num_neighbors:
                        num_better_neighbors = num_better_neighbors + 1
                if core_number != 0 or num_neighbors != 0:
                    sequence_features1.append((core_number, num_neighbors))  # 添加核心数和邻居数信息
            # sequence_features1 = sorted(sequence_features1, key=lambda x: x[0], reverse=True)

            single_feature = model_output_for_path(time_start, time_end, v)
            # single_feature = 0
            single_feature_list.append(single_feature)

            temp_label = vertex_core_numbers[v][(temp_start, temp_end)]
            if len(sequence_features1) == 0:
                sequence_features1 = [(0, 0)]

            # 如果 sequence_features1 中的元素为0，则跳过
            if all(x == (0, 0) for x in sequence_features1) or temp_label < 1:
                if random.random() > 0.1:
                    continue

            sequence_features1_list.append(sequence_features1)
            label_list.append(temp_label)
            if len(sequence_features1_list) % 100 == 0:
                print(f"{len(sequence_features1_list)}/{num_sample}")

    return sequence_features1_list, label_list, single_feature_list

def get_sequence_and_label_nonleaf(layer_id=1, timerange_id=0):
    print("Creating training samples...")
    # global valid_h_index, h_index_error
    sequence_features1_list, sequences_features2_list, label_list, single_feature_list = [], [], [], []
    time_start = time_range_layers[layer_id][timerange_id][0]
    time_end = time_range_layers[layer_id][timerange_id][1]
    num_sample = 1500
    while len(sequence_features1_list) < num_sample:
        # random select a vertex
        v = random.randint(0, num_vertex - 1)
        if len(vertex_core_numbers[v].keys()) == 0:
            continue
        temp_time_range = random.choice(list(vertex_core_numbers[v].keys()))
        temp_start = temp_time_range[0]
        temp_end = temp_time_range[1]
        if temp_start != temp_end and temp_start >= time_start and temp_end <= time_end and (temp_end - temp_start) > (
                (time_end - time_start + 1) // partition):
            node = tree_query(temp_start, temp_end)
            if node is None or node.layer_id > layer_id:
                continue
            # print(len(sequence_features1_list))
            covered_nodes = []
            for child_node in node.children:
                if child_node.time_start >= temp_start and child_node.time_end <= temp_end:
                    covered_nodes.append(child_node)
            sequence_features1 = []
            neighbor_projected_set = set()
            for t in range(temp_start, temp_end + 1):
                neighbor_projected_set.update(temporal_graph[v][t])
                if len(covered_nodes) > 0 and covered_nodes[0].time_start <= t <= covered_nodes[-1].time_end:
                    continue
                core_number = vertex_core_numbers[v].get((t, t), 0)
                num_neighbors = len(temporal_graph[v][t])  # 获取该节点在时间 t 的邻居个数
                if core_number != 0 or num_neighbors != 0:
                    sequence_features1.append((core_number, num_neighbors))  # 添加核心数和邻居数信息
            # sequence_features1 = sorted(sequence_features1, key=lambda x: x[0], reverse=True)
            # sequence_features1 = sorted(sequence_features1, key=lambda x: x[0] + x[1], reverse=True)

            sequence_features2 = []
            for child_node in covered_nodes:
                time_start1 = child_node.time_start
                time_end1 = child_node.time_end
                covered_neighbor_projected_set = set()
                for t in range(time_start1, time_end1 + 1):
                    for neighbor in temporal_graph[v][t]:
                        covered_neighbor_projected_set.add(neighbor)
                core_number = child_node.vertex_core_number.get(v, 0)
                if core_number != 0 or len(covered_neighbor_projected_set) != 0:
                    sequence_features2.append((core_number, len(covered_neighbor_projected_set)))
            # sequence_features2 = sorted(sequence_features2, key=lambda x: x[0], reverse=True)


            # 计算投影图上单一特征值 这个特征值为其父亲节点模型的输出
            single_feature = 0

            if layer_id > 0:
                single_feature = model_output_for_path(time_start, time_end, v)
            single_feature_list.append(single_feature)

            temp_label = vertex_core_numbers[v][(temp_start, temp_end)]

            if len(sequence_features2) == 0:
                continue

            # 如果 sequence_features1 中的元素为0，则跳过
            if (all(x == (0, 0) for x in sequence_features1) and all(x == (0, 0) for x in sequence_features2)) or temp_label < 1:
                if random.random() > 0.1:
                    continue

            if len(sequence_features1) == 0:
                sequence_features1.append((0, 0))
            sequence_features1_list.append(sequence_features1)
            if len(sequence_features2) == 0:
                sequence_features2.append((0, 0))
            sequences_features2_list.append(sequence_features2)
            label_list.append(temp_label)
            if len(sequence_features1_list) % 100 == 0:
                print(f"{len(sequence_features1_list)}/{num_sample}")

    return sequence_features1_list,sequences_features2_list, label_list, single_feature_list


# 训练和评价函数
def train_and_evaluate_leaf(layer_id=0, timerange_id=0):
    input_dim = 2
    max_seq_length = max_time_range_layers[layer_id]
    hidden_dim = 64
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    patience = 15

    sequences, labels, additional_features = get_sequence_and_label_leaf(layer_id, timerange_id)
    num_samples = len(sequences)
    print(f"Sample Number: {num_samples}")

    train_size = int(0.7 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size

    train_sequences = sequences[:train_size]
    train_labels = labels[:train_size]
    train_additional_features = additional_features[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    val_additional_features = additional_features[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    test_additional_features = additional_features[train_size + val_size:]

    train_dataset = SequenceDataset(train_sequences, train_labels, train_additional_features, max_seq_length)
    val_dataset = SequenceDataset(val_sequences, val_labels, val_additional_features, max_seq_length)
    test_dataset = SequenceDataset(test_sequences, test_labels, test_additional_features, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLP(input_dim, max_seq_length, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences_batch, additional_features_batch, labels_batch in train_loader:
            sequences_batch = sequences_batch.to(device)
            additional_features_batch = additional_features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(sequences_batch, additional_features_batch)
                loss = criterion(outputs, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_sequences_batch, val_additional_features_batch, val_labels_batch in val_loader:
                val_sequences_batch = val_sequences_batch.to(device)
                val_additional_features_batch = val_additional_features_batch.to(device)
                val_labels_batch = val_labels_batch.to(device)

                with autocast():
                    val_outputs = model(val_sequences_batch, val_additional_features_batch)
                    val_loss += criterion(val_outputs, val_labels_batch).item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    with torch.no_grad():
        test_loss = 0
        total_percentage_error = 0
        total_inference_time = 0
        num_batches = 0
        epsilon = 0.5  # 防止除以零
        for test_sequences_batch, test_additional_features_batch, test_labels_batch in test_loader:
            test_sequences_batch = test_sequences_batch.to(device)
            test_additional_features_batch = test_additional_features_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            start_time = time.time()
            with autocast():
                test_outputs = model(test_sequences_batch, test_additional_features_batch)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            num_batches += 1

            test_loss += criterion(test_outputs, test_labels_batch).item()

            # # 使用MAPE计算误差
            # percentage_error = torch.abs((test_outputs - test_labels_batch) / test_labels_batch) * 100
            # total_percentage_error += percentage_error.sum().item()

            # 使用SMAPE计算误差
            numerator = torch.abs(torch.round(test_outputs) - test_labels_batch)
            denominator = torch.abs(torch.round(test_outputs)) + torch.abs(test_labels_batch) + epsilon
            percentage_error = (2 * numerator / denominator) * 100
            total_percentage_error += percentage_error.sum().item()

        test_loss /= len(test_loader)
        avg_percentage_error = total_percentage_error / len(test_dataset)
        avg_inference_time = total_inference_time / (num_batches * batch_size)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Average Percentage Error: {avg_percentage_error:.2f}%")
        print(f"Average Inference Time per Sequence: {avg_inference_time:.6f} ms")

    torch.save(model.state_dict(), f'models/{dataset_name}/model_{layer_id}_{timerange_id}.pth')
    return avg_percentage_error, avg_inference_time

def train_and_evaluate_nonleaf(layer_id=1, timerange_id=0):
    input_dim = 2
    max_seq_length1 = max_time_range_layers[layer_id + 1] * 2
    max_seq_length2 = partition
    hidden_dim = 64
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    patience = 15

    sequences, sequences_range, labels, additional_features = get_sequence_and_label_nonleaf(layer_id, timerange_id)
    # sequences, sequences_range, labels, additional_features = get_sequence_and_label_nonleaf_table(layer_id, timerange_id)
    num_samples = len(sequences)
    # if len(sequences) > 0:
    #     print("Shape of first element in sequences:", torch.tensor(sequences[0]).shape)

    train_size = int(0.7 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size

    train_sequences = sequences[:train_size]
    train_sequences_range = sequences_range[:train_size]
    train_labels = labels[:train_size]
    train_additional_features = additional_features[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    val_sequences_range = sequences_range[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    val_additional_features = additional_features[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    test_sequences_range = sequences_range[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    test_additional_features = additional_features[train_size + val_size:]

    # Update the datasets to include sequences_range
    train_dataset = SequenceDatasetNonleaf(train_sequences, train_sequences_range, train_labels, train_additional_features, max_seq_length1, max_seq_length2)
    val_dataset = SequenceDatasetNonleaf(val_sequences, val_sequences_range, val_labels, val_additional_features, max_seq_length1, max_seq_length2)
    test_dataset = SequenceDatasetNonleaf(test_sequences, test_sequences_range, test_labels, test_additional_features, max_seq_length1, max_seq_length2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLPNonleaf(input_dim, max_seq_length1, max_seq_length2, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences_batch, sequences_range_batch, additional_features_batch, labels_batch in train_loader:
            sequences_batch = sequences_batch.to(device)
            sequences_range_batch = sequences_range_batch.to(device)
            additional_features_batch = additional_features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(sequences_batch, sequences_range_batch, additional_features_batch)
                loss = criterion(outputs, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_sequences_batch, val_sequences_range_batch, val_additional_features_batch, val_labels_batch in val_loader:
                val_sequences_batch = val_sequences_batch.to(device)
                val_sequences_range_batch = val_sequences_range_batch.to(device)
                val_additional_features_batch = val_additional_features_batch.to(device)
                val_labels_batch = val_labels_batch.to(device)

                with autocast():
                    val_outputs = model(val_sequences_batch, val_sequences_range_batch,
                                        val_additional_features_batch)
                    val_loss += criterion(val_outputs, val_labels_batch).item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    with torch.no_grad():
        test_loss = 0
        total_percentage_error = 0
        total_inference_time = 0
        num_batches = 0
        epsilon = 0.5  # 防止除以零

        for test_sequences_batch, test_sequences_range_batch, test_additional_features_batch, test_labels_batch in test_loader:
            test_sequences_batch = test_sequences_batch.to(device)
            test_sequences_range_batch = test_sequences_range_batch.to(device)
            test_additional_features_batch = test_additional_features_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            start_time = time.time()
            with autocast():
                test_outputs = model(test_sequences_batch, test_sequences_range_batch,
                                     test_additional_features_batch)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            num_batches += 1

            test_loss += criterion(test_outputs, test_labels_batch).item()

            # # 使用MAPE计算误差
            # percentage_error = torch.abs((test_outputs - test_labels_batch) / test_labels_batch) * 100
            # total_percentage_error += percentage_error.sum().item()

            # 使用SMAPE计算误差
            numerator = torch.abs(torch.round(test_outputs) - test_labels_batch)
            denominator = torch.abs(torch.round(test_outputs)) + torch.abs(test_labels_batch) + epsilon
            percentage_error = (2 * numerator / denominator) * 100
            total_percentage_error += percentage_error.sum().item()

        test_loss /= len(test_loader)
        avg_percentage_error = total_percentage_error / len(test_dataset)
        avg_inference_time = total_inference_time / (num_batches * batch_size)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Average Percentage Error: {avg_percentage_error:.2f}%")
        print(f"Average Inference Time per Sequence: {avg_inference_time:.6f} ms")

    torch.save(model.state_dict(), f'models/{dataset_name}/model_{layer_id}_{timerange_id}.pth')
    return avg_percentage_error, avg_inference_time

def load_models(depth_id=0):
    # 读取节点的模型
    global model_time_range_layers
    model_time_range_layers = [[] for _ in range(len(time_range_layers))]
    print("Loading models...")
    # load models in a tree
    for layer_id in range(0, depth_id):
        for i in range(len(time_range_layers[layer_id])):
            print(f"{i+1}/{len(time_range_layers[layer_id])}")
            time_start = time_range_layers[layer_id][i][0]
            time_end = time_range_layers[layer_id][i][1]
            model = None
            if layer_id != max_layer_id:
                model = MLPNonleaf(2, max_time_range_layers[layer_id + 1] * 2, partition, 64).to(device)
                model.load_state_dict(torch.load(f'models/{dataset_name}/model_{layer_id}_{i}.pth'))
                model.eval()
            else:
                model = MLP(2, max_time_range_layers[0], 64).to(device)
                model.load_state_dict(torch.load(f'models/{dataset_name}/model_{layer_id}_{i}.pth'))
                model.eval()
            node = tree_query(time_start, time_end)
            if node is None:
                print("Error: node not found.")
            else:
                node.set_model(model)

    # # load models in a table
    # for i in range(len(time_range_layers[0])):
    #     print(f"{i+1}/{len(time_range_layers[0])}")
    #     model_time_range_layers[0].append(MLP(2, max_time_range_layers[0], 64).to(device))
    #     model_time_range_layers[0][i].load_state_dict(torch.load(f'models/model_0_{i}.pth'))
    #     model_time_range_layers[0][i].eval()
    print("Loading finished.")

# 主函数
def main():
    global root
    read_temporal_graph()
    get_timerange_layers()
    read_core_number()
    root = build_tree()
    avg_percentage_error = 0
    avg_inference_time = 0

    # # 测试 10 个时间范围
    # layer_id = 0
    # num_elements = len(time_range_layers[layer_id])
    # num_test = min(10, num_elements)
    # selected_indices = random.sample(range(num_elements), num_test)  # 从范围中随机选择10 个唯一的元素
    # for t in selected_indices:
    #     # 对随机选择的 t 进行操作
    #     temp_avg_percentage_error, temp_avg_inference_time = train_and_evaluate_nonleaf(layer_id, t)
    #     avg_percentage_error += temp_avg_percentage_error
    #     avg_inference_time += temp_avg_inference_time
    # avg_percentage_error = float(avg_percentage_error) / num_test
    # avg_inference_time = float(avg_inference_time) / num_test

    avg_percentage_error_list = []
    avg_inference_time_list = []

    # 训练并保存第0-x层的模型
    for layer_id in range(0, max_layer_id+1):
        load_models(layer_id)
        avg_percentage_error = 0
        avg_inference_time = 0
        for i in range(len(time_range_layers[layer_id])):
            print(f"{i}/{len(time_range_layers[layer_id])}")
            if layer_id == max_layer_id:
                temp_avg_percentage_error, temp_avg_inference_time = train_and_evaluate_leaf(layer_id, i)
            else:
                temp_avg_percentage_error, temp_avg_inference_time = train_and_evaluate_nonleaf(layer_id, i)
            avg_percentage_error += temp_avg_percentage_error
            avg_inference_time += temp_avg_inference_time
        avg_percentage_error = float(avg_percentage_error) / len(time_range_layers[layer_id])
        avg_inference_time = float(avg_inference_time) / len(time_range_layers[layer_id])
        print(avg_percentage_error)
        print(avg_inference_time)
        avg_percentage_error_list.append(avg_percentage_error)
        avg_inference_time_list.append(avg_inference_time)

    print(avg_percentage_error_list)
    print(avg_inference_time_list)
    # print(h_index_error)
    # print(valid_h_index)
    # var = float(h_index_error) / valid_h_index
    # print(var)


    # # 训练并保存叶子结点的模型
    # layer_id = 0
    # for i in range(len(time_range_layers[layer_id])):
    #     print(f"{i}/{len(time_range_layers[layer_id])}")
    #     temp_avg_percentage_error, temp_avg_inference_time = train_and_evaluate(layer_id, i)
    #     avg_percentage_error += temp_avg_percentage_error
    #     avg_inference_time += temp_avg_inference_time
    # avg_percentage_error = float(avg_percentage_error) / len(time_range_layers[layer_id])
    # avg_inference_time = float(avg_inference_time) / len(time_range_layers[layer_id])
    # print(avg_percentage_error)
    # print(avg_inference_time)



if __name__ == "__main__":
    main()
