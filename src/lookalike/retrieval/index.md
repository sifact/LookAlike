# index.py

The vector index: stores every image's embedding and finds the nearest ones fast.

- **Offers three search methods.** `flat` is exact (checks every vector), `hnsw` is a graph-based approximate search (fast and nearly exact), and `ivfpq` uses clustering plus compression (much smaller in memory, less accurate).
- **Uses cosine similarity.** It uses FAISS inner-product indexes on unit-length vectors, which is the same as cosine similarity.
- **Maps results to catalog ids.** It keeps a parallel `ids` array, so search returns real `image_id`s instead of FAISS's internal row numbers, and missing results come back as `-1`.
- **Supports live additions.** `add` appends new images without rebuilding, and it rejects duplicate ids. `vectors_for` returns stored vectors for re-ranking.
- **Persists to disk.** `save` and `load` write `index.faiss`, `ids.npy`, `vectors.npy` and `meta.json` (which model made the vectors), and `memory_bytes` reports index size for the benchmark.
