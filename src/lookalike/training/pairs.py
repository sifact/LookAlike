"""Training pairs: (query, positive). The positive source is the book's labeling choice.

    SelfSupPairs      positive = augmented view of the same image (SimCLR). Free and
                      clean, but never shows two *different* photos of the same thing.
    InteractionPairs  positive = another photo of the same product, a stand-in for
                      "user clicked this result". `noise` swaps the positive for a
                      random image with that probability, simulating noisy clicks.
"""

import random

import pandas as pd
from torch.utils.data import Dataset

from ..config import IMAGES
from ..data.preprocess import light_augment, load_rgb, selfsup_augment


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


def make_pairs(train: pd.DataFrame, mode: str, noise: float = 0.0) -> Dataset:
    if mode == "selfsup":
        return SelfSupPairs(train)
    if mode == "interactions":
        return InteractionPairs(train, noise)
    raise ValueError(f"unknown mode {mode!r}")
