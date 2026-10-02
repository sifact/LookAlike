"""Contrastive fine-tuning (book: query, 1 positive, n-1 negatives, softmax + cross-entropy).

Positive sources (the book's labeling options):
    selfsup       positive = augmented view of the query image (SimCLR). Free, clean, but
                  never shows the model two *different* photos of the same thing.
    interactions  positive = another photo of the same product, a stand-in for "user clicked
                  this result". `noise` swaps the positive for a random image with that
                  probability, simulating noisy clicks.

Negatives are the other positives in the batch, so batch_size - 1 negatives per query.
"""

import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DEFAULT_BACKBONE, IMAGES, MODELS, device
from .data import load_catalog
from .model import Embedder
from .preprocess import light_augment, load_rgb, selfsup_augment
from .tracking import log_metrics, log_params


class SelfSupPairs(Dataset):
    def __init__(self, train: pd.DataFrame):
        self.paths = [IMAGES / p for p in train["path"]]
        self.tf = selfsup_augment()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_rgb(self.paths[i])
        return self.tf(img), self.tf(img)


class InteractionPairs(Dataset):
    def __init__(self, train: pd.DataFrame, noise: float = 0.0):
        groups = train.groupby("item_id")["path"].apply(list)
        self.groups = [[IMAGES / p for p in g] for g in groups if len(g) >= 2]
        self.all_paths = [IMAGES / p for p in train["path"]]
        self.noise = noise
        self.tf = light_augment()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int):
        q, p = random.sample(self.groups[i], 2)
        if random.random() < self.noise:
            p = random.choice(self.all_paths)
        return self.tf(load_rgb(q)), self.tf(load_rgb(p))


def info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float) -> torch.Tensor:
    """Row i: similarities of query i to every positive in the batch; label = i. Symmetrized."""
    logits = q @ p.T / temperature
    labels = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(mode: str = "interactions", backbone: str = DEFAULT_BACKBONE, proj_dim: int = 256,
          epochs: int = 3, batch_size: int = 64, lr: float = 2e-5, temperature: float = 0.05,
          noise: float = 0.0, name: str | None = None, max_steps: int | None = None,
          workers: int = 4, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = device()

    cat = load_catalog()
    train_df = cat[cat["role"] == "train"]
    if mode == "selfsup":
        ds = SelfSupPairs(train_df)
    elif mode == "interactions":
        ds = InteractionPairs(train_df, noise)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0)

    model = Embedder(backbone, proj_dim).to(dev).train()
    # Recompute activations in backward instead of storing them. Without this, a
    # 64-pair ViT batch needs ~15 GB and thrashes swap on a 16 GB Mac. The batch
    # size is kept because in-batch negatives are what the loss learns from.
    model.backbone.set_grad_checkpointing(True)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.proj.parameters(), "lr": lr * 50},
    ], weight_decay=0.05)
    total = epochs * len(loader) if max_steps is None else min(max_steps, epochs * len(loader))
    warmup = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    name = name or f"{mode}" + (f"_noise{noise:g}" if noise else "")
    history, step, t0 = [], 0, time.time()
    bar = tqdm(total=total, desc=f"train {name}")
    while step < total:
        for q, p in loader:
            q, p = q.to(dev, non_blocking=True), p.to(dev, non_blocking=True)
            loss = info_nce(model(q), model(p), temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            bar.update(1)
            bar.set_postfix(loss=f"{loss.item():.3f}")
            if step % 20 == 0 or step == total:
                history.append({"step": step, "loss": round(loss.item(), 4)})
                log_metrics({"loss": loss.item(), "lr": sched.get_last_lr()[0]}, step)
            if step >= total:
                break
    bar.close()

    config = {"mode": mode, "backbone": backbone, "proj_dim": proj_dim, "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "temperature": temperature, "noise": noise,
              "steps": step, "train_images": len(train_df), "train_pairs_per_epoch": len(ds),
              "seconds": round(time.time() - t0, 1), "device": str(dev)}
    log_params(config)
    out = MODELS / f"{name}.pt"
    model.cpu().save(out, meta={"config": config, "history": history})
    (MODELS / f"{name}.json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    return out
