# Visual search system

Upload an image, or drag a box around part of one, and get the most visually similar
catalog images, best match first. This implements Chapter 2 of *Machine Learning System
Design Interview* (Aminian & Xu) end to end. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md)
for the problem and the approach.

## Pipeline

New to the code? Start with [docs/GUIDE.md](docs/GUIDE.md), which walks through it in 10 chapters.

Visual overview: [docs/flowchart.excalidraw](docs/flowchart.excalidraw). Open it at
excalidraw.com (File → Open) or with the Excalidraw VS Code extension.

```
                     ┌──────────── indexing pipeline ────────────┐
catalog / uploads ──>│ preprocess -> embedder -> FAISS index.add │
                     └───────────────────────────────────────────┘
                     ┌──────────────────────── prediction pipeline ─────────────────────────┐
query image (+crop)─>│ preprocess -> embedder -> ANN top 3k+5 -> re-rank -> top k results  │
                     └──────────────────────────────────────────────────────────────────────┘
```

| Module | Role |
|---|---|
| `data.py` | Downloads Stanford Online Products, samples products per category, and assigns query/gallery roles on held-out products |
| `preprocess.py` | Resize, scale, z-score, force RGB, crop; SimCLR and light augmentations |
| `model.py` | timm backbone + linear projection, L2-normalized embeddings |
| `train.py` | InfoNCE with in-batch negatives; `selfsup` or `interactions` positives, with optional click noise |
| `embed.py` | Batch embedding, cached per model in `artifacts/embeddings/` |
| `index.py` | FAISS `flat` (exact), `hnsw`, `ivfpq`; add, save and load |
| `rerank.py` | Drops blocked ids, duplicates of the query and near-duplicate results |
| `metrics.py`, `evaluate.py` | nDCG, mAP, MRR, P@k, R@k, hit@1; per-category nDCG; exact-vs-ANN benchmark |
| `service.py` | FastAPI `/search`, `/images`, `/health`; serves the web UI |
| `frontend/` | React + Vite + shadcn/ui web UI (see below) |

## Data and evaluation protocol

Stanford Online Products has 120k images of 22.6k products in 12 categories, and its
train and test splits share **no products**. By default `data` samples 300 products per
category from each split (about 19k images each) so everything runs on a laptop.

- **Queries:** one image from each of up to 2,000 held-out products.
- **Gallery:** all other held-out images. Each query's product has at least one other image there.
- **Relevance:** 2 = same product, 1 = same category, 0 = other. nDCG@10 uses these graded
  scores (the book's choice). The binary metrics count only "same product" as relevant.

## Setup

```bash
uv sync
uv run pytest          # metrics, loss, indexes, re-ranking, API; no dataset needed
```

## Run

```bash
uv run visual-search data                                   # download (~3 GB, resumable) + build catalog
uv run visual-search train --mode interactions              # same-product positives
uv run visual-search train --mode selfsup                   # augmentation positives
uv run visual-search train --mode interactions --noise 0.3  # 30% noisy "clicks"

uv run visual-search evaluate resnet50.a1_in1k vit_small_patch14_dinov2.lvd142m \
    artifacts/models/selfsup.pt artifacts/models/interactions.pt \
    --ann artifacts/models/interactions.pt                  # -> runs/<timestamp>/eval.md

uv run visual-search index --model artifacts/models/interactions.pt --kind hnsw
uv run visual-search serve --index artifacts/indexes/<name> # http://127.0.0.1:8000
```

### Web UI

[frontend/](frontend/) is a React + Vite + TypeScript app with shadcn/ui in a dark theme.
It has drag, drop and paste upload, drag-to-crop search, click-a-result-to-search,
same-product highlighting, per-stage timings, recent queries, and "add to index".

```bash
cd frontend && npm install && npm run build   # FastAPI then serves frontend/dist at /
npm run dev                                    # hot-reload dev server on :5173 (proxies the API on :8000)
```

Without a build, `/` falls back to the dependency-free page in `src/visual_search/static/`.

API:

```bash
curl -F file=@mug.jpg -F k=10 localhost:8000/search
curl -F file=@mug.jpg -F box=0.2,0.1,0.7,0.8 localhost:8000/search   # crop (fractions or pixels)
curl -F file=@new.jpg -F category=mug localhost:8000/images            # index a new image live
```

## Train on Azure Machine Learning

[azureml/](azureml/) runs the same training (same data sample, seeds and
hyperparameters) on a cloud T4 GPU. The job downloads the dataset itself and saves the
model and eval report as job outputs. Loss and metrics appear as charts in Azure ML
Studio through MLflow.

1. **GPU quota** (one time): Azure portal → *Quotas* → *Compute* → region East US →
   request **Standard NCASv3_T4 Family vCPUs = 4** (or **Low-priority vCPUs = 4** for
   spot pricing, which `compute-gpu.yml` uses by default), and make sure **Total Regional
   vCPUs** has 4 free.
2. **Create resources and submit:** `bash azureml/setup.sh` creates the resource group, the
   workspace and a GPU cluster that scales to zero when idle, then submits
   [azureml/train-job.yml](azureml/train-job.yml).
3. **Follow and download:** use the `az ml job stream` / `az ml job download` commands the
   script prints, then serve the downloaded model as usual.

Delete everything afterwards with `az group delete -n visual-search-rg`.

To keep private or moderated images out of results, put their image ids (one per line)
in `data/blocked_ids.txt`.

## Notes

- **macOS:** torch and faiss each bundle their own OpenMP runtime. `_faiss.py` imports
  them in a safe order and runs faiss single-threaded. Always import faiss through it.
- **Scale:** at about 19k gallery images, exact search is already fast. The ANN benchmark
  shows the quality, memory and latency tradeoff, which only starts to matter at much larger scale.
- **Not covered offline:** CTR and time spent are the online metrics to A/B test after launch.
