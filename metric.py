def compute_density(vertex_set, t_start, t_end,temporal_graph):
    time_edges_count = 0
    time_stamps = set()
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor in vertex_set:
                        time_edges_count += 1
                        time_stamps.add(t)
    temporal_density = time_edges_count / ((len(vertex_set) * (len(vertex_set) - 1)) * len(time_stamps))
    return temporal_density

def compute_conductance(vertex_set, t_start, t_end,temporal_graph,num_vertex):
    time_edges_count = 0
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                for neighbor in neighbors:
                    if neighbor not in vertex_set:
                        time_edges_count += 1
    degree_community = 0
    for vertex in vertex_set:
        for t, neighbors in temporal_graph[vertex].items():
            if t_start <= t <= t_end:
                degree_community += len(neighbors)
    degree_not_in_community = 0
    for vertex in range(num_vertex):
        if vertex not in vertex_set:
            for t, neighbors in temporal_graph[vertex].items():
                if t_start <= t <= t_end:
                    degree_not_in_community += len(neighbors)
    temporal_conductance = time_edges_count / min(degree_community, degree_not_in_community)
    return temporal_conductance
