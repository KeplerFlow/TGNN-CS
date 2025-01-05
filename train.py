import torch
from torch_geometric.data import Batch
import torch.nn.functional as F

from utils import *
from loss import *
from extract_subgraph import *


def validate_model(model, val_loader, device, alpha, subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex,time_range_link_samples_cache):
    model.eval()
    val_loss = 0.0
    top_num = 0.0
    with torch.no_grad():
        for batch in val_loader:
            anchors, positives, negatives, time_ranges = batch
            anchors = torch.tensor(anchors, device=device).clone()
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            subgraphs = []
            vertex_maps = []

            for anchor, pos_samples, neg_samples, time_range in zip(anchors, positives, negatives, time_ranges):
              # 传入所有需要的参数
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1], subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex, device)
                if len(vertex_map) == 0 or vertex_map[anchor.item()] == -1 : # 添加anchor是否存在的判断
                    continue

                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)

            if not subgraphs:
                continue

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)
            embeddings = model(batched_subgraphs)
            batch_indices = batched_subgraphs.batch

            del batched_subgraphs
            torch.cuda.empty_cache()


            batch_loss = 0.0
            for i, (anchor, pos_samples, neg_samples, vertex_map, subgraph_pyg,time_range) in enumerate(zip(anchors, positives, negatives, vertex_maps, subgraphs,time_ranges)):
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]


                anchor_idx = node_indices[vertex_map[anchor.item()]] if vertex_map[anchor.item()] !=-1 else -1 #check if -1

                pos_indices = [node_indices[vertex_map[p.item()]].item() for p in pos_samples if vertex_map[p.item()] != -1]
                neg_indices = [node_indices[vertex_map[n.item()]].item() for n in neg_samples if vertex_map[n.item()] != -1]

                if  anchor_idx == -1 or not pos_indices or not neg_indices:
                    continue


                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]


                loss = margin_triplet_loss(anchor_emb, positive_emb, negative_emb)

                link_loss_value = compute_link_loss(embeddings, vertex_map, node_indices,
                                                        time_range[0], time_range[1], anchor.item(),time_range_link_samples_cache, margin=0.2)

                loss += alpha * link_loss_value
                batch_loss += loss


            if vertex_maps:
                batch_loss /= len(vertex_maps)


            val_loss += batch_loss.item()


    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def test_model(model, test_loader, device, subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex):
    model.eval()
    avg_sim_pos, avg_sim_neg = 0.0, 0.0
    total_pos_samples, total_neg_samples = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            anchors, positives, negatives, time_ranges = batch #add cache
            anchors = torch.tensor(anchors, device=device)
            positives = [torch.tensor(pos, device=device) for pos in positives]
            negatives = [torch.tensor(neg, device=device) for neg in negatives]

            subgraphs = []
            vertex_maps = []

            for anchor, pos_samples, neg_samples, time_range in zip(anchors, positives, negatives, time_ranges):
                subgraph_pyg, vertex_map = extract_subgraph_for_anchor(anchor.item(), time_range[0], time_range[1], subgraph_pyg_cache, subgraph_k_hop_cache, subgraph_vertex_map_cache, num_vertex, device)
                if len(vertex_map) == 0 or vertex_map[anchor.item()] == -1:  #check if -1
                    continue
                subgraphs.append(subgraph_pyg)
                vertex_maps.append(vertex_map)


            if not subgraphs:
                continue

            batched_subgraphs = Batch.from_data_list(subgraphs).to(device)
            embeddings = model(batched_subgraphs)
            batch_indices = batched_subgraphs.batch

            del batched_subgraphs
            torch.cuda.empty_cache()  # 清理缓存


            for i, (anchor, pos_samples, neg_samples, vertex_map) in enumerate(zip(anchors, positives, negatives, vertex_maps)):
                node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

                anchor_idx = node_indices[vertex_map[anchor.item()]] if vertex_map[anchor.item()] !=-1 else -1 #check if -1
                pos_indices = [node_indices[vertex_map[p.item()]].item() for p in pos_samples if
                               vertex_map[p.item()] != -1]
                neg_indices = [node_indices[vertex_map[n.item()]].item() for n in neg_samples if
                               vertex_map[n.item()] != -1]
                if anchor_idx == -1 or not pos_indices or not neg_indices:
                    continue


                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[pos_indices]
                negative_emb = embeddings[neg_indices]


                sim_pos = F.cosine_similarity(anchor_emb.unsqueeze(0), positive_emb).mean().item()
                sim_neg = F.cosine_similarity(anchor_emb.unsqueeze(0), negative_emb).mean().item()

                avg_sim_pos += sim_pos * len(pos_indices)
                avg_sim_neg += sim_neg * len(neg_indices)
                total_pos_samples += len(pos_indices)
                total_neg_samples += len(neg_indices)


    if total_pos_samples > 0 and total_neg_samples > 0 :
      avg_sim_pos /= total_pos_samples
      avg_sim_neg /= total_neg_samples


    print(f"Test Avg Positive Similarity: {avg_sim_pos:.4f}")
    print(f"Test Avg Negative Similarity: {avg_sim_neg:.4f}")

    return avg_sim_pos, avg_sim_neg