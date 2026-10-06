"""Contrastive fine-tuning: training pairs, InfoNCE loss and the training loop."""

from .loss import info_nce
from .trainer import train

__all__ = ["info_nce", "train"]
