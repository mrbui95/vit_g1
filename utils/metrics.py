import torch
import numpy as np
from sklearn.metrics import average_precision_score

def evaluate_retrieval(query_feats, query_labels, gallery_feats, gallery_labels, topk=(1, 5, 10)):
    """
    Args:
        query_feats: (N, D) Tensor
        query_labels: (N,) Tensor or array
        gallery_feats: (M, D) Tensor
        gallery_labels: (M,) Tensor or array
    Returns:
        metrics: dict with R@K, mP@K, mAP
    """
    query_feats = torch.nn.functional.normalize(query_feats, dim=1)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)

    sim_matrix = query_feats @ gallery_feats.T  # cosine similarity
    indices = sim_matrix.topk(max(topk), dim=1, largest=True).indices  # (N, max_k)

    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()
    indices = indices.cpu().numpy()

    recalls = {f'R@{k}': 0.0 for k in topk}
    precisions = {f'mP@{k}': 0.0 for k in topk}
    APs = []

    #for i in range(len(query_labels)):
    for i in range(10):
        true_label = query_labels[i]
        print(true_label)
        retrieved_labels = gallery_labels[indices[i]]  # Top-K predicted
        print(retrieved_labels)

        hits = (retrieved_labels == true_label).astype(np.int32)
        print(hits)
        for k in topk:
            retrieved_at_k = hits[:k]
            recalls[f'R@{k}'] += (retrieved_at_k.sum() > 0)
            precisions[f'mP@{k}'] += np.mean(retrieved_at_k)

        # For mAP: compute binary relevance for full gallery
        relevance = (gallery_labels == true_label).astype(np.int32)
        sim_scores = sim_matrix[i].cpu().numpy()
        print(sim_scores)
        ap = average_precision_score(relevance, sim_scores)
        print(ap)
        if not np.isnan(ap):
            APs.append(ap)

    N = len(query_labels)
    metrics = {k: v / N for k, v in {**recalls, **precisions}.items()}
    metrics['mAP'] = np.mean(APs) if APs else 0.0

    return metrics