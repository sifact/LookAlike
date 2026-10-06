"""Embedding model: pretrained backbone + optional linear projection, L2-normalized output."""

from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn

from ..config import IMG_SIZE


class Embedder(nn.Module):
    def __init__(self, backbone: str, proj_dim: int | None = None, pretrained: bool = True):
        super().__init__()
        kwargs = {"img_size": IMG_SIZE} if backbone.startswith(("vit", "deit", "eva")) else {}
        self.backbone_name = backbone
        self.proj_dim = proj_dim
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kwargs)
        feat = self.backbone.num_features
        self.proj = nn.Linear(feat, proj_dim) if proj_dim else nn.Identity()
        self.dim = proj_dim or feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": self.backbone_name,
            "proj_dim": self.proj_dim,
            "state_dict": self.state_dict(),
            "meta": meta or {},
        }, path)


def load_embedder(spec: str) -> Embedder:
    """`spec` is a checkpoint path or a timm model name (used off the shelf)."""
    path = Path(spec)
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = Embedder(ckpt["backbone"], ckpt["proj_dim"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = Embedder(spec)
    return model.eval()


def model_tag(spec: str) -> str:
    """Filesystem-safe name used to cache embeddings and indexes per model."""
    path = Path(spec)
    if path.suffix == ".pt":
        # Include mtime so retraining to the same path invalidates cached embeddings.
        name = f"{path.stem}_{int(path.stat().st_mtime)}"
    else:
        name = spec
    return name.replace("/", "_").replace(".", "_")
