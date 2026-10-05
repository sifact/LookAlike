# benchmark.py

Compares exact and approximate nearest-neighbor search on the same embeddings, the book's ANN deep dive.

- **Tests every index type.** It builds `flat` (exact), `hnsw` and `ivfpq` indexes from the same gallery vectors.
- **Uses exact search as the reference.** Flat search runs first, and each approximate method's results are compared with it (recall@10 vs exact).
- **Measures the effect on quality.** It also computes nDCG@10 for each index, showing how much an approximation actually hurts the final metric.
- **Measures speed.** It times 500 single-query searches per index and reports p50 and p95 latency in milliseconds, plus the build time.
- **Measures memory.** It reports each index's size in MB, which shows the tradeoff: HNSW is about 6× faster than exact at 99.8% recall, and IVF-PQ is about 12× smaller but less accurate.
