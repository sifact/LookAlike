queries=2000 gallery=17324 (held-out products)

## Retrieval quality (exact search)

| model | dim | ndcg@10 | map@10 | mrr | precision@10 | recall@10 | hit@1 | category_precision@10 |
|---|---|---|---|---|---|---|---|---|
| resnet50.a1_in1k | 2048 | 0.6484 | 0.3576 | 0.6006 | 0.1495 | 0.4228 | 0.5535 | 0.7475 |
| vit_small_patch14_dinov2.lvd142m | 384 | 0.7112 | 0.3978 | 0.6301 | 0.1694 | 0.4675 | 0.5825 | 0.8278 |
| artifacts/models/interactions.pt | 256 | 0.7469 | 0.618 | 0.8172 | 0.2716 | 0.6902 | 0.7765 | 0.7537 |

## nDCG@10 by category

| model | bicycle | cabinet | chair | coffee maker | fan | kettle | lamp | mug | sofa | stapler | table | toaster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50.a1_in1k | 0.6738 | 0.6437 | 0.5566 | 0.6277 | 0.6551 | 0.6752 | 0.6755 | 0.7103 | 0.6926 | 0.6797 | 0.5902 | 0.5993 |
| vit_small_patch14_dinov2.lvd142m | 0.6768 | 0.7056 | 0.6545 | 0.6911 | 0.7261 | 0.7404 | 0.7418 | 0.7654 | 0.7141 | 0.7585 | 0.6812 | 0.6831 |
| artifacts/models/interactions.pt | 0.8159 | 0.7597 | 0.6809 | 0.6927 | 0.7289 | 0.7836 | 0.7609 | 0.8304 | 0.797 | 0.748 | 0.6652 | 0.6898 |

## Exact vs ANN (artifacts/models/interactions.pt)

| index | n | recall@10_vs_exact | ndcg@10 | p50_ms | p95_ms | build_s | index_mb |
|---|---|---|---|---|---|---|---|
| flat | 17324 | 1.0 | 0.7469 | 0.236 | 0.246 | 0.0 | 16.92 |
| hnsw | 17324 | 0.9978 | 0.7469 | 0.041 | 0.06 | 0.51 | 21.41 |
| ivfpq | 17324 | 0.7285 | 0.7283 | 0.039 | 0.045 | 1.86 | 1.35 |
