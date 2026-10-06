"""Exact vs approximate nearest-neighbor search on the same embeddings."""

import time

import numpy as np

from ..data import load_catalog
from ..modeling import catalog_embeddings
from ..retrieval import KINDS, VectorIndex
from .scoring import score_rankings, split_queries_gallery


def ann_benchmark(spec: str, k: int = 10, latency_queries: int = 500) -> list[dict]:
    """Per index kind: recall vs exact, nDCG, single-query latency, build time, memory."""
    queries, gallery = split_queries_gallery(load_catalog())
    qv = catalog_embeddings(spec, queries)
    gv = catalog_embeddings(spec, gallery)
    gids = gallery["image_id"].to_numpy()
    exact_ids = None
    out = []
    for kind in KINDS:  # "flat" first: it is the exact reference
        t0 = time.perf_counter()
        index = VectorIndex.build(gv, gids, kind)
        build_s = time.perf_counter() - t0
        _, ids = index.search(qv, k)
        if kind == "flat":
            exact_ids = ids
        overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, exact_ids)])
        lat = []
        for v in qv[:latency_queries]:
            t = time.perf_counter()
            index.search(v, k)
            lat.append((time.perf_counter() - t) * 1000)
        ndcg = score_rankings(queries, gallery, ids)[f"ndcg@{k}"].mean()
        out.append({"index": kind, "n": len(index), f"recall@{k}_vs_exact": round(float(overlap), 4),
                    f"ndcg@{k}": round(float(ndcg), 4), "p50_ms": round(float(np.percentile(lat, 50)), 3),
                    "p95_ms": round(float(np.percentile(lat, 95)), 3), "build_s": round(build_s, 2),
                    "index_mb": round(index.memory_bytes() / 2**20, 2)})
    return out
