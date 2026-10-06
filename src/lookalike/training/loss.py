"""Contrastive loss (book: query, 1 positive, n-1 negatives, softmax + cross-entropy)."""

import torch
import torch.nn.functional as F


def info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float) -> torch.Tensor:
    """Row i: similarities of query i to every positive in the batch; label = i.

    The other positives in the batch are the negatives, so each query gets
    batch_size - 1 of them. Symmetrized over queries and positives.
    """
    logits = q @ p.T / temperature
    labels = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
