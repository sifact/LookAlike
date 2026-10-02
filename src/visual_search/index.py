"""Nearest-neighbor index over normalized embeddings (inner product == cosine).

kinds:
    flat   exact linear scan, O(N*D)
    hnsw   graph-based ANN
    ivfpq  clustering (IVF) + product quantization, the memory-saving option
"""

import json
from math import sqrt
from pathlib import Path

import numpy as np

from ._faiss import faiss

KINDS = ("flat", "hnsw", "ivfpq")


class VectorIndex:
    def __init__(self, index, ids: np.ndarray, vectors: np.ndarray, kind: str):
        self.index = index
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vectors = vectors  # kept for re-ranking (near-duplicate checks)
        self.kind = kind
        self._pos = {int(i): p for p, i in enumerate(self.ids)}

    @classmethod
    def build(cls, vectors: np.ndarray, ids: np.ndarray, kind: str = "hnsw",
              hnsw_m: int = 32, ef_search: int = 64, pq_m: int = 32, nprobe: int = 16) -> "VectorIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape
        if kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = ef_search
        elif kind == "ivfpq":
            if d % pq_m:
                raise ValueError(f"dim {d} not divisible by pq_m {pq_m}")
            nlist = max(1, min(int(4 * sqrt(n)), n // 39))  # faiss wants ~39 points per centroid
            index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = min(nprobe, nlist)
        else:
            raise ValueError(f"unknown index kind {kind!r}; choose from {KINDS}")
        index.add(vectors)
        return cls(index, ids, vectors, kind)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, image_ids), both (n_queries, k). Missing slots have id -1."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        scores, pos = self.index.search(queries, k)
        ids = np.where(pos >= 0, self.ids[np.clip(pos, 0, None)], -1)
        return scores, ids

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        dup = [int(i) for i in ids if int(i) in self._pos]
        if dup:
            raise ValueError(f"ids already indexed: {dup[:5]}")
        start = len(self.ids)
        self.index.add(vectors)
        self.ids = np.concatenate([self.ids, ids])
        self.vectors = np.concatenate([self.vectors, vectors])
        self._pos.update({int(i): start + p for p, i in enumerate(ids)})

    def vectors_for(self, ids) -> np.ndarray:
        return self.vectors[[self._pos[int(i)] for i in ids]]

    def memory_bytes(self) -> int:
        return int(faiss.serialize_index(self.index).nbytes)

    def save(self, directory: Path, meta: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "ids.npy", self.ids)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "meta.json").write_text(json.dumps({"kind": self.kind, "n": len(self), **(meta or {})}, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        meta = json.loads((directory / "meta.json").read_text())
        return cls(faiss.read_index(str(directory / "index.faiss")),
                   np.load(directory / "ids.npy"), np.load(directory / "vectors.npy"), meta["kind"])
