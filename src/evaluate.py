# Offline Evaluation Metrics (Phase 7)

import numpy as np


# ---------------------------------------------------------------------------
# Core metrics (implemented from scratch — interviewers ask about these)
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    rels = np.array(relevances[:k], dtype=float)
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rels) + 2))  # log2(2), log2(3), ...
    return float(np.sum(rels / discounts))


def ndcg_at_k(recommended: list, relevant_set: set, k: int) -> float:
    """Normalized DCG at K. relevant_set contains ground-truth item IDs."""
    relevances = [1 if item in relevant_set else 0 for item in recommended[:k]]
    ideal      = sorted(relevances, reverse=True)
    dcg        = dcg_at_k(relevances, k)
    idcg       = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: list, relevant_set: set, k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant_set:
        return 0.0
    hits = len(set(recommended[:k]) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(recommended: list, relevant_set: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant_set)
    return hits / k


def average_precision(recommended: list, relevant_set: set) -> float:
    """Average Precision (area under precision-recall curve)."""
    if not relevant_set:
        return 0.0
    hits, precision_sum = 0, 0.0
    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_set)


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(recommend_fn, val_df, k_list=(5, 10), n_users=500) -> dict:
    """
    Evaluate a recommendation function against the validation set.

    recommend_fn(user_id) -> list of movieIds (ordered, no scores)

    Returns dict of metric -> value for each K in k_list.
    E.g. {'ndcg@5': 0.12, 'ndcg@10': 0.09, 'map': 0.07, ...}
    """
    val_items   = val_df.groupby('userId')['movieId'].apply(set).to_dict()
    sample_users = list(val_items.keys())[:n_users]

    accum = {f'ndcg@{k}': []    for k in k_list}
    accum.update({f'recall@{k}': [] for k in k_list})
    accum.update({f'precision@{k}': [] for k in k_list})
    accum['map'] = []

    for uid in sample_users:
        relevant = val_items.get(uid, set())
        if not relevant:
            continue
        recs = recommend_fn(uid)            # ordered list of movieIds

        for k in k_list:
            accum[f'ndcg@{k}'].append(ndcg_at_k(recs, relevant, k))
            accum[f'recall@{k}'].append(recall_at_k(recs, relevant, k))
            accum[f'precision@{k}'].append(precision_at_k(recs, relevant, k))
        accum['map'].append(average_precision(recs, relevant))

    return {metric: float(np.mean(vals)) for metric, vals in accum.items() if vals}
