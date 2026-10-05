import numpy as np
import torch
import torch.nn.functional as F

from visual_search.retrieval import Candidate, VectorIndex, rerank
from visual_search.training import info_nce


def _unit(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_info_nce_rewards_aligned_pairs():
    q = F.normalize(torch.randn(16, 32), dim=-1)
    aligned = info_nce(q, q, 0.05)
    shuffled = info_nce(q, q[torch.randperm(16)], 0.05)
    assert aligned < shuffled


def test_indexes_find_self_and_map_ids():
    x = _unit(3000, 64)
    ids = np.arange(3000) + 1000
    for kind in ("flat", "hnsw", "ivfpq"):
        idx = VectorIndex.build(x, ids, kind, pq_m=16)
        _, got = idx.search(x[:100], 5)
        hit = (got[:, 0] == ids[:100]).mean() if kind != "ivfpq" else np.mean([i in r for i, r in zip(ids[:100], got)])
        assert hit > (0.99 if kind == "flat" else 0.8), kind


def test_index_add_save_load(tmp_path):
    x = _unit(500, 32)
    idx = VectorIndex.build(x[:400], np.arange(400), "hnsw")
    idx.add(x[400:], np.arange(400, 500) + 10_000)
    assert idx.search(x[450], 1)[1][0, 0] == 10_450
    idx.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert len(loaded) == 500
    assert np.allclose(loaded.vectors_for([10_450]), x[450:451])


def test_rerank_blocks_and_dedupes():
    q = _unit(1, 16, seed=1)[0]
    a, b = _unit(2, 16, seed=2)
    vecs = np.stack([q, a, a, b])  # exact copy of query, a, near-dup of a, b
    cands = [Candidate(i, 1.0) for i in range(4)]
    kept = rerank(q, cands, vecs, k=10)
    assert [c.image_id for c in kept] == [1, 3]
    kept = rerank(q, cands, vecs, k=10, blocked={3})
    assert [c.image_id for c in kept] == [1]
