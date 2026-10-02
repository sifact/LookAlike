# Project guide: the visual search system in 10 chapters

The chapters follow the order the data flows through the system. Each one says what
happens, why, and which file to read.

**Suggested path:** read Chapters 0 → 3 → 4 → 8 first (the core ML idea), then
6 → 7 → 9 (the engineering).

---

## Chapter 0: The big picture

**Goal:** a user gives a photo (or a crop of one) and gets back the most similar images
from a catalog, best match first.

**Core trick:** turn every image into a list of numbers (an **embedding**) so that similar
images get similar numbers. Search then becomes "find the closest lists of numbers".

The project has three phases (the three lanes in [flowchart.excalidraw](flowchart.excalidraw)):

1. **Train** a model that produces good embeddings (offline, once).
2. **Index**: embed every catalog image and store the results in a fast search structure.
3. **Search**: embed the user's photo, find its nearest neighbors, clean up the results,
   and return the top 10.

---

## Chapter 1: The data ([data.py](../src/visual_search/data.py))

- **Dataset:** Stanford Online Products, eBay photos of 22k products in 12 categories
  (bicycle, mug, lamp, …), with several photos per product.
- **Train / test split:** the two splits share **no products**. So the test measures
  whether the model finds products it has never seen, which is the real-world situation.
- **Query / gallery:** from the test products, one photo per product becomes a **query**
  (2,000 queries). Every other test photo goes into the **gallery** (17,324 images) that
  we search in.
- **Relevance score:** this is how we know what counts as "correct":
  - 2 = same product (a perfect match)
  - 1 = same category (for example, a different mug: okay)
  - 0 = anything else (wrong)

The output is `data/catalog.parquet`, a table listing every image with its product,
category and role.

---

## Chapter 2: Preparing images ([preprocess.py](../src/visual_search/preprocess.py))

Models need consistent input. Every image is:

- converted to RGB (`load_rgb`),
- resized and center-cropped to 224×224 (`eval_transform`),
- scaled to 0–1 and normalized (shifted and scaled per color channel).

`crop()` cuts out the box the user drew, before embedding. This is how "search a part of
the image" works.

Two **augmentation** recipes are used only in training:

- `selfsup_augment`: heavy random crops, color changes and blur.
- `light_augment`: small random crops and flips.

---

## Chapter 3: The model ([model.py](../src/visual_search/model.py))

`Embedder` has three steps:

1. **Backbone:** a pretrained network (DINOv2 or ResNet-50) that already understands
   images. It outputs 384 or 2048 numbers.
2. **Projection:** one linear layer that compresses the backbone output to 256 numbers.
3. **Normalize:** scales the vector to length 1, so `a · b` (the dot product) equals
   cosine similarity. A value of 1 means identical; around 0 means unrelated.

`load_embedder()` loads either a checkpoint you trained or an off-the-shelf model by name.

---

## Chapter 4: Training ([train.py](../src/visual_search/train.py))

This is the most important idea in the project: **contrastive learning**.

- Take a batch of 64 pairs, each a **query** plus its **positive** (a matching image).
- For each query, its own positive should be the most similar, and the other 63
  positives in the batch act as **negatives**.
- `info_nce()` computes all 64×64 similarities, applies softmax, and uses cross-entropy
  with the "correct answer" at position *i*. In effect, it's a 64-way multiple-choice
  question where the model must pick the right match.

Where do positives come from? That's the book's key tradeoff, and we test all of it:

| Mode | Positive | Pros / cons |
|---|---|---|
| `selfsup` | augmented copy of the same photo | free and never wrong, but never shows the model two *different* photos of the same thing |
| `interactions` | another photo of the same product (like a user click) | realistic, but needs labels |
| `--noise 0.3` | 30% of positives are random | simulates noisy clicks |

---

## Chapter 5: Embedding the catalog ([embed.py](../src/visual_search/embed.py))

- `embed_paths()` runs images through the model in batches on the Mac GPU (MPS).
- `catalog_embeddings()` caches the results in `artifacts/embeddings/`, so evaluating
  again doesn't recompute 19k images. The cache is keyed by image path, because image ids
  change whenever the catalog is rebuilt.

---

## Chapter 6: Fast search ([index.py](../src/visual_search/index.py))

Comparing a query with every image doesn't scale to billions of images, so we use FAISS
with three options:

| Index | How it works | Measured result (DINOv2, 17k images) |
|---|---|---|
| `flat` | compares with everything (exact) | 100% accurate, 0.37 ms |
| `hnsw` | walks a graph of neighbors | 99.5% of the exact results, **8× faster** |
| `ivfpq` | clusters the vectors and compresses them | **15× smaller**, but only 68% of the exact results |

[_faiss.py](../src/visual_search/_faiss.py) exists only to stop torch and faiss from
crashing each other on macOS.

---

## Chapter 7: Re-ranking ([rerank.py](../src/visual_search/rerank.py))

Raw nearest neighbors aren't ready to show users. `rerank()`:

- removes **blocked** images (private or moderated),
- removes results that are near-identical to the **query** (the user already has that photo),
- removes **near-duplicates** of results already kept.

We fetch about 3× more candidates than needed, so after filtering there are still 10 left.

---

## Chapter 8: Measuring quality ([metrics.py](../src/visual_search/metrics.py), [evaluate.py](../src/visual_search/evaluate.py))

The main metric is **nDCG@10**: "how good is the top 10, compared with the best possible
top 10?"

Worked example. Say the model returns relevance scores `[1, 0, 2]`:

- **DCG** = 1/log₂2 + 0/log₂3 + 2/log₂4 = 1 + 0 + 1 = **2.0**
- **Ideal order** `[2, 1, 0]` gives 2/1 + 1/1.58 + 0 ≈ **2.63**
- **nDCG** = 2.0 / 2.63 = **0.76**. Higher positions count more, so the best match should
  be first.

Other metrics:

- **hit@1:** was the first result the exact product?
- **mAP, MRR, precision, recall:** alternative views of the same question, each with the
  weakness the book describes.

Results on 2,000 queries for unseen products ([results.md](results.md)):

| Model | nDCG@10 | hit@1 | mAP@10 | same-category in top 10 |
|---|---|---|---|---|
| ResNet-50 (ImageNet, off the shelf) | 0.648 | 0.554 | 0.358 | 0.748 |
| DINOv2 ViT-S (off the shelf) | 0.711 | 0.583 | 0.398 | **0.828** |
| **Ours: DINOv2 + contrastive fine-tune** | **0.747** | **0.777** | **0.618** | 0.754 |

Fine-tuning made the first result the exact product 78% of the time, up from 58%.
The tradeoff: the model learned "same product" so well that the *rest* of the top 10
is less often the same category (0.83 → 0.75). Tables and staplers got slightly worse
than off-the-shelf DINOv2. A fix would be to add category-level pairs to training.

---

## Chapter 9: Serving ([service.py](../src/visual_search/service.py), [frontend/](../frontend/src/App.tsx))

A FastAPI app ties everything together:

- `POST /search`: image (+ optional crop box) → embed → index → re-rank → top k, with a
  timing for each stage.
- `POST /images`: adds a new image to the index instantly, with no retraining.
- `GET /`: the React web UI (Vite + shadcn/ui, dark theme). Drag a box to search a crop,
  click a result to search from it, or paste an image with ⌘V. The main files are
  [App.tsx](../frontend/src/App.tsx), [crop-image.tsx](../frontend/src/components/crop-image.tsx)
  and [api.ts](../frontend/src/lib/api.ts).

---

## Chapter 10: Running it ([cli.py](../src/visual_search/cli.py))

```bash
uv run visual-search data       # Chapter 1
uv run visual-search train      # Chapter 4
uv run visual-search evaluate   # Chapter 8 (+ Chapter 6 with --ann)
uv run visual-search index      # Chapters 5 + 6
uv run visual-search serve      # Chapter 9
```

Full commands and options are in the [README](../README.md).
