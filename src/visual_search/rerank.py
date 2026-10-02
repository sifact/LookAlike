"""Re-ranking service: business rules applied after nearest-neighbor retrieval.

- drop blocked ids (private / moderated images)
- drop near-duplicates of the query (the user already has that image)
- drop near-duplicates of results already kept
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    image_id: int
    score: float


def rerank(query_vec: np.ndarray, candidates: list[Candidate], vectors: np.ndarray, k: int,
           blocked: set[int] = frozenset(), dup_threshold: float = 0.97) -> list[Candidate]:
    """`vectors[i]` is the normalized embedding of `candidates[i]`."""
    kept: list[Candidate] = []
    kept_vecs = [np.asarray(query_vec, dtype=np.float32)]
    for cand, vec in zip(candidates, vectors):
        if cand.image_id < 0 or cand.image_id in blocked:
            continue
        if max(float(vec @ v) for v in kept_vecs) >= dup_threshold:
            continue
        kept.append(cand)
        kept_vecs.append(vec)
        if len(kept) == k:
            break
    return kept
