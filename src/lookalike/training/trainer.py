"""Contrastive fine-tuning loop: pairs -> embedder -> InfoNCE -> checkpoint."""

import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import DEFAULT_BACKBONE, MODELS, device
from ..data import load_catalog
from ..modeling import Embedder
from ..tracking import log_metrics, log_params
from .loss import info_nce
from .pairs import make_pairs


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _optimizer(model: Embedder, lr: float) -> torch.optim.Optimizer:
    # Pretrained backbone moves gently; the new projection head learns 50x faster.
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.proj.parameters(), "lr": lr * 50},
    ], weight_decay=0.05)


def _warmup_cosine(opt: torch.optim.Optimizer, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, total // 20)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))


def train(mode: str = "interactions", backbone: str = DEFAULT_BACKBONE, proj_dim: int = 256,
          epochs: int = 3, batch_size: int = 64, lr: float = 2e-5, temperature: float = 0.05,
          noise: float = 0.0, name: str | None = None, max_steps: int | None = None,
          workers: int = 4, seed: int = 0):
    _seed_everything(seed)
    dev = device()

    cat = load_catalog()
    train_df = cat[cat["role"] == "train"]
    ds = make_pairs(train_df, mode, noise)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0)

    model = Embedder(backbone, proj_dim).to(dev).train()
    # Recompute activations in backward instead of storing them. Without this, a
    # 64-pair ViT batch needs ~15 GB and thrashes swap on a 16 GB Mac. The batch
    # size is kept because in-batch negatives are what the loss learns from.
    model.backbone.set_grad_checkpointing(True)
    opt = _optimizer(model, lr)
    total = epochs * len(loader) if max_steps is None else min(max_steps, epochs * len(loader))
    sched = _warmup_cosine(opt, total)

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
