# rerank.py

The re-ranking step: business rules applied after nearest-neighbor search, before showing results.

- **Runs after retrieval.** It takes the over-fetched nearest neighbors (about 3× k) and filters them down to the final k results.
- **Removes blocked images.** Ids listed as private or moderated (`data/blocked_ids.txt`) never appear in results.
- **Drops copies of the query.** Results that are near-identical to the query image (cosine similarity ≥ 0.97) are removed, since the user already has that photo.
- **Removes near-duplicates.** A result too similar to one already kept is skipped, so the results show variety rather than the same image several times.
- **Keeps the ranking order.** It walks candidates from best to worst and stops as soon as k results are kept, so the best matches stay at the top.
