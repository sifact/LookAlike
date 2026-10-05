"""Offline retrieval quality for one model.

Queries: one image per held-out product (products never seen in training).
Gallery: every other held-out image. Graded relevance per (query, result):
    2 same product, 1 same category, 0 otherwise.
nDCG@k uses the graded scores (the book's primary metric). mAP, MRR, precision@k
and recall@k use binary relevance (same product). hit@1 is the usual SOP "Recall@1".
"""

import numpy as np
import pandas as pd

from ..config import REL_SAME_CATEGORY, REL_SAME_PRODUCT
from ..data import load_catalog
from ..modeling import catalog_embeddings
from ..retrieval import VectorIndex
from . import metrics as M


def split_queries_gallery(cat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    queries, gallery = split_queries_gallery(cat)
    qv = catalog_embeddings(spec, queries)
    gv = catalog_embeddings(spec, gallery)
    index = VectorIndex.build(gv, gallery["image_id"].to_numpy(), kind)
    _, ids = index.search(qv, k)
    per_query = score_rankings(queries, gallery, ids)
    summary = per_query.drop(columns=["image_id", "category"]).mean().round(4).to_dict()
    by_cat = per_query.groupby("category")[f"ndcg@{k}"].mean().round(4).to_dict()
    return {"model": spec, "index": kind, "k": k, "queries": len(queries), "gallery": len(gallery),
            "dim": int(gv.shape[1]), "metrics": summary, f"ndcg@{k}_by_category": by_cat}
