# metrics.py

The ranking metrics from the book, as small pure functions on one ranked result list.

- **nDCG (main metric).** `dcg` and `ndcg` score a ranking using graded relevance (2 = same product, 1 = same category, 0 = other), comparing it with the best possible ordering of *all* candidates.
- **Reciprocal rank.** `reciprocal_rank` gives 1 / position of the first correct result, the building block of MRR.
- **Precision@k.** `precision_at_k` is the fraction of the top-k results that are correct.
- **Recall@k.** `recall_at_k` is the fraction of all correct items in the gallery that appear in the top k.
- **Average precision.** `average_precision` averages precision at each correct position (AP@k, divided by min(k, total relevant)), the building block of mAP. It rewards putting *all* matches near the top.
