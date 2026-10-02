"""Offline evaluation.

Queries: one image per held-out product (products never seen in training).
Gallery: every other held-out image. Graded relevance per (query, result):
    2 same product, 1 same category, 0 otherwise.
nDCG@k uses the graded scores (the book's primary metric). mAP, MRR, precision@k
and recall@k use binary relevance (same product). hit@1 is the usual SOP "Recall@1".
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import metrics as M
from .config import REL_SAME_CATEGORY, REL_SAME_PRODUCT, RUNS
from .data import load_catalog
from .embed import catalog_embeddings
from .index import KINDS, VectorIndex
from .tracking import log_metrics


def _split(cat: pd.DataFrame):
    q = cat[cat["role"] == "query"].reset_index(drop=True)
    g = cat[cat["role"] == "gallery"].reset_index(drop=True)
    return q, g


def score_rankings(queries: pd.DataFrame, gallery: pd.DataFrame, ranked_ids: np.ndarray) -> pd.DataFrame:
    """Per-query metrics for ranked gallery image_ids (n_queries, k)."""
    g = gallery.set_index("image_id")
    item_counts = gallery["item_id"].value_counts()
    cat_counts = gallery["category"].value_counts()
    k = ranked_ids.shape[1]
    rows = []
    for (_, q), ids in zip(queries.iterrows(), ranked_ids):
        ids = ids[ids >= 0]
        hit = g.loc[ids]
        same_item = (hit["item_id"] == q["item_id"]).to_numpy()
        same_cat = (hit["category"] == q["category"]).to_numpy()
        rels = np.where(same_item, REL_SAME_PRODUCT, np.where(same_cat, REL_SAME_CATEGORY, 0))
        rels = np.pad(rels, (0, k - len(rels)))
        hits = np.pad(same_item, (0, k - len(same_item))).astype(int)
        n_item = int(item_counts.get(q["item_id"], 0))
        n_cat = int(cat_counts.get(q["category"], 0)) - n_item
        ideal = np.array([REL_SAME_PRODUCT] * min(n_item, k) + [REL_SAME_CATEGORY] * min(n_cat, k))
        rows.append({
            "image_id": q["image_id"], "category": q["category"],
            f"ndcg@{k}": M.ndcg(rels, ideal),
            f"map@{k}": M.average_precision(hits, n_item),
            "mrr": M.reciprocal_rank(hits),
            f"precision@{k}": M.precision_at_k(hits),
            f"recall@{k}": M.recall_at_k(hits, n_item),
            "hit@1": float(hits[0]),
            f"category_precision@{k}": float(np.mean(rels > 0)),
        })
    return pd.DataFrame(rows)


def evaluate_model(spec: str, k: int = 10, kind: str = "flat", cat: pd.DataFrame | None = None) -> dict:
    cat = load_catalog() if cat is None else cat
    queries, gallery = _split(cat)
    qv = catalog_embeddings(spec, queries)
    gv = catalog_embeddings(spec, gallery)
    index = VectorIndex.build(gv, gallery["image_id"].to_numpy(), kind)
    _, ids = index.search(qv, k)
    per_query = score_rankings(queries, gallery, ids)
    summary = per_query.drop(columns=["image_id", "category"]).mean().round(4).to_dict()
    by_cat = per_query.groupby("category")[f"ndcg@{k}"].mean().round(4).to_dict()
    return {"model": spec, "index": kind, "k": k, "queries": len(queries), "gallery": len(gallery),
            "dim": int(gv.shape[1]), "metrics": summary, f"ndcg@{k}_by_category": by_cat}


def ann_benchmark(spec: str, k: int = 10, latency_queries: int = 500) -> list[dict]:
    """Exact vs approximate search: quality, latency and memory on the same embeddings."""
    cat = load_catalog()
    queries, gallery = _split(cat)
    qv = catalog_embeddings(spec, queries)
    gv = catalog_embeddings(spec, gallery)
    gids = gallery["image_id"].to_numpy()
    exact_ids = None
    out = []
    for kind in KINDS:
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


def _table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def run(models: list[str], k: int = 10, ann_model: str | None = None) -> str:
    cat = load_catalog()
    results = [evaluate_model(m, k, cat=cat) for m in models]
    log_metrics(results[-1]["metrics"])  # the last model listed is the one being trained
    flat = [{"model": r["model"], "dim": r["dim"], **r["metrics"]} for r in results]
    report = [f"queries={results[0]['queries']} gallery={results[0]['gallery']} (held-out products)\n",
              "## Retrieval quality (exact search)\n", _table(flat)]
    cats = sorted(results[0][f"ndcg@{k}_by_category"])
    by_cat = [{"model": r["model"], **{c: r[f"ndcg@{k}_by_category"][c] for c in cats}} for r in results]
    report += [f"\n## nDCG@{k} by category\n", _table(by_cat)]
    ann = None
    if ann_model:
        ann = ann_benchmark(ann_model, k)
        report += [f"\n## Exact vs ANN ({ann_model})\n", _table(ann)]
    text = "\n".join(report)

    out = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"results": results, "ann": ann}, indent=2))
    (out / "eval.md").write_text(text + "\n")
    return text + f"\n\nsaved to {out}"
