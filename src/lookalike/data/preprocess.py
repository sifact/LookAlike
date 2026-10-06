"""Image preprocessing (book: resize, scale, z-score normalize, consistent color mode)."""

from pathlib import Path
from typing import BinaryIO

from PIL import Image
from torchvision import transforms as T

from ..config import IMG_SIZE

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_rgb(src: str | Path | BinaryIO) -> Image.Image:
    """Open any image and force RGB (drops alpha, converts CMYK / grayscale / palette)."""
    img = Image.open(src)
    img.load()
    return img.convert("RGB")


def crop(img: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
    """Crop to (x0, y0, x1, y1). Values <= 1 are treated as fractions of width/height."""
    if box is None:
        return img
    x0, y0, x1, y1 = box
    if max(box) <= 1.0:
        w, h = img.size
        x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"crop box too small: {box}")
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def eval_transform(size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize(int(size * 256 / 224)),  # resize
        T.CenterCrop(size),
        T.ToTensor(),  # scale to [0, 1]
        T.Normalize(MEAN, STD),  # z-score
    ])


def selfsup_augment(size: int = IMG_SIZE) -> T.Compose:
    """SimCLR-style views: the positive is an augmented copy of the query."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.25, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def light_augment(size: int = IMG_SIZE) -> T.Compose:
    """Used when positives are already distinct images, so heavy augmentation isn't needed."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
