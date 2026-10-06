import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("VS_DATA", ROOT / "data"))
RAW = DATA / "raw"
IMAGES = DATA / "images"
CATALOG = DATA / "catalog.parquet"
BLOCKED_IDS = DATA / "blocked_ids.txt"
ARTIFACTS = Path(os.getenv("VS_ARTIFACTS", ROOT / "artifacts"))
MODELS = ARTIFACTS / "models"
EMBEDDINGS = ARTIFACTS / "embeddings"
INDEXES = ARTIFACTS / "indexes"
RUNS = ROOT / "runs"

# Stanford Online Products: 120k images of 22.6k products in 12 categories.
# Train and test splits contain disjoint products.
HF_REPO = "nyris/stanford-online-products-v1"

DEFAULT_BACKBONE = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE = 224

# Graded relevance used for nDCG, mirroring the book's 0..N similarity score.
REL_SAME_PRODUCT = 2
REL_SAME_CATEGORY = 1


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
