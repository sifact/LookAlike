"""Batch embedding of catalog images, cached per model."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import EMBEDDINGS, IMAGES, device
from .model import Embedder, load_embedder, model_tag
from .preprocess import eval_transform, load_rgb


class _Paths(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.tf = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.tf(load_rgb(self.paths[i]))


@torch.no_grad()
def embed_paths(model: Embedder, paths: list[Path], batch_size: int = 64, workers: int = 4) -> np.ndarray:
    dev = device()
    model = model.to(dev).eval()
    loader = DataLoader(_Paths(paths), batch_size=batch_size, num_workers=workers,
                        persistent_workers=workers > 0)
    out = [model(x.to(dev)).float().cpu() for x in tqdm(loader, desc="embed", leave=False)]
    return torch.cat(out).numpy().astype(np.float32)


def catalog_embeddings(spec: str, catalog: pd.DataFrame, model: Embedder | None = None) -> np.ndarray:
    """Embeddings aligned to `catalog` rows. Cached by model tag and image path, since
    image_ids are reassigned whenever the catalog is rebuilt."""
    cache = EMBEDDINGS / f"{model_tag(spec)}.npz"
    cached: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        cached = dict(zip(z["paths"].tolist(), z["vectors"]))
    missing = catalog.loc[~catalog["path"].isin(cached.keys())]
    if len(missing):
        model = model or load_embedder(spec)
        vecs = embed_paths(model, [IMAGES / p for p in missing["path"]])
        cached.update(zip(missing["path"].tolist(), vecs))
        cache.parent.mkdir(parents=True, exist_ok=True)
        paths = list(cached)
        np.savez(cache, paths=np.array(paths), vectors=np.stack([cached[p] for p in paths]))
    return np.stack([cached[p] for p in catalog["path"].tolist()])
