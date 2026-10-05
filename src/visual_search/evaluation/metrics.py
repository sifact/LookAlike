"""Ranking metrics from the book: MRR, recall@k, precision@k, mAP, nDCG.

`rels` is the ranked list of graded relevance scores the model returned.
Binary metrics treat rel == REL_SAME_PRODUCT as relevant.
"""

import numpy as np


def dcg(rels: np.ndarray) -> float:
    rels = np.asarray(rels, dtype=float)
    # Book form: sum_i rel_i / log2(i + 1), with i starting at 1.
    return float((rels / np.log2(np.arange(2, len(rels) + 2))).sum())


def ndcg(rels: np.ndarray, ideal_rels: np.ndarray) -> float:
    """`ideal_rels` are the relevance scores of *all* candidates; the best k of them form IDCG."""
    k = len(rels)
    ideal = np.sort(np.asarray(ideal_rels))[::-1][:k]
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def reciprocal_rank(hits: np.ndarray) -> float:
    idx = np.flatnonzero(hits)
    return 1.0 / (idx[0] + 1) if len(idx) else 0.0


def precision_at_k(hits: np.ndarray) -> float:
    return float(np.mean(hits)) if len(hits) else 0.0


def recall_at_k(hits: np.ndarray, total_relevant: int) -> float:
    return float(np.sum(hits)) / total_relevant if total_relevant else 0.0


def average_precision(hits: np.ndarray, total_relevant: int) -> float:
    """AP@k: mean of precision@i at each relevant position, over min(k, total relevant)."""
    hits = np.asarray(hits, dtype=float)
    denom = min(len(hits), total_relevant)
    if denom == 0:
        return 0.0
    precisions = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precisions * hits).sum() / denom)
