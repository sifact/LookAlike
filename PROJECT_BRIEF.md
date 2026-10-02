# Visual Search System: Find Visually Similar Images from a Photo or Crop

Source: *Machine Learning System Design Interview* (Aminian & Xu), Chapter 2.

## What problem we're actually solving

- Users see something they like in an image (a lamp, a bike, a mug) but can't describe it in words, so text search fails them.
- The platform has a huge catalog, and users can only find most of it by browsing. Every "I can't find it" is lost engagement or a lost sale.
- Results must be ranked: the most similar items have to appear first, because users rarely scroll.
- Users need to search on a *part* of an image (a crop), not just the whole photo.
- Search has to stay fast as the catalog grows to billions of images, so scanning everything per query isn't an option.

## What solution we'll provide (non-technical)

- Upload a photo (or drag a box around one object in it) and instantly see the most similar images in the catalog, best match first.
- Click any result to keep exploring from there.
- New images become searchable as soon as they are added, with no retraining needed.
- Private or blocked images never appear, and duplicate results are removed.
- A report showing, with numbers, how often the right product shows up near the top and how fast each search is, compared against off-the-shelf models.

## How we'll approach it (technical)

- **Data:** Stanford Online Products (120k images, 22.6k products, 12 categories) stands in for the platform catalog. Train and test use disjoint products, so evaluation is on products the model has never seen. Graded relevance: same product = 2, same category = 1, else 0.
- **Model:** a pretrained backbone (DINOv2 ViT-S/14, with ResNet-50 as a CNN baseline) plus a linear projection to L2-normalized 256-d embeddings. Images are resized, scaled, z-score normalized and converted to RGB.
- **Training:** contrastive InfoNCE (query, 1 positive, batch−1 negatives, softmax + cross-entropy). We compare the book's labeling options: self-supervised augmentation positives (SimCLR) vs. interaction-style positives (another photo of the same product, a stand-in for clicks), with injected label noise to simulate noisy clicks.
- **Evaluation:** offline nDCG@10 on graded relevance is the primary metric, alongside mAP@10, MRR, precision@10, recall@10 and hit@1, broken down by category. An exact-vs-ANN benchmark (flat, HNSW, IVF-PQ) reports recall against exact search, p50/p95 latency and index memory.
- **Serving:** a FastAPI prediction pipeline (embedding service → FAISS ANN → re-ranking that removes blocked ids, query duplicates and near-duplicate results) and an indexing pipeline that embeds and adds new uploads live. Crop search is supported, and a demo page is included. Online metrics (CTR, time spent) are noted as the production follow-up.
