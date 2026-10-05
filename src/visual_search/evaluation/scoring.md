# scoring.py

Measures how good one model's search results are on products it has never seen.

- **Splits queries and gallery.** `split_queries_gallery` takes the held-out products' query images and the gallery images they are searched against.
- **Grades every result.** `score_rankings` labels each returned image 2 (same product), 1 (same category) or 0, and builds the ideal ranking from all matching gallery images.
- **Computes all metrics per query.** For every query it records nDCG@10, mAP@10, MRR, precision@10, recall@10, hit@1 and category precision@10.
- **Evaluates a whole model end to end.** `evaluate_model` embeds queries and gallery (using the cache), builds an index, searches, scores and averages.
- **Breaks results down by category.** It also reports nDCG per product category, which shows where the model is weak (tables, chairs) and strong (mugs, bicycles).
