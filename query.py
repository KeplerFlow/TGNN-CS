import torch
import heapq
import numpy as np

from metric import *

def temporal_test_GNN_query_time(distances, vertex_map, query_vertex, t_start, t_end, temporal_graph, device,num_vertex):

    visited = set()
    visited.add(query_vertex)
    result = set()
    queue = []
    heapq.heappush(queue, (0, query_vertex))

    # Ensure distances is on the correct device
    distances = distances.to(device)


    mask = (distances != 0)
    temp_distances = distances[mask]

    distances = (distances - temp_distances.min()) / (temp_distances.max() - temp_distances.min() + 1e-6)
    result_distance = []

    threshold = distances.mean().item()
    less_num = torch.sum(distances < threshold).item()

    while queue:

        distance, top_vertex = heapq.heappop(queue)
        result.add(top_vertex)
        result_distance.append(distance)


        if distance > threshold:
            break

        if len(result) > 1:
            tau = 0.9
            alpha = np.cos((np.pi / 2) * (len(result) / (less_num ** tau if less_num > 0 else 1))) # Handle less_num = 0

            threshold = alpha * threshold + (1-alpha) * (sum(result_distance) / (len(result_distance)))


        if top_vertex not in temporal_graph: #check if top_vertex is in temporal_graph
            continue

        for t, neighbors in temporal_graph[top_vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if vertex_map[neighbor] != -1:
                          # Check if neighbor is within the bounds of distances
                          if vertex_map[neighbor] < distances.shape[0]:
                            heapq.heappush(queue, (distances[vertex_map[neighbor]].item(), neighbor))



    if len(result) == 0:
        return 0, 0, result

    temporal_density = compute_density(result, t_start, t_end, temporal_graph) # Pass temporal_graph to compute_density
    temporal_conductance = compute_conductance(result, t_start, t_end, temporal_graph,num_vertex) # Pass temporal_graph to compute_conductance
    print(f"Result Number: {len(result)}")
    return temporal_density, temporal_conductance, result