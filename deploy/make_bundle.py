"""Collect everything the API needs at runtime into deploy/bundle/ for the container image.

    uv run python deploy/make_bundle.py --index artifacts/indexes/<name>

bundle/
    model.pt                 trained checkpoint
    index/                   FAISS index (meta.json points at /app/bundle/model.pt)
    data/catalog.parquet     image metadata
    data/images/...          catalog images downsized to thumbnails (display + click-to-search)
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "bundle"
CONTAINER_BUNDLE = "/app/bundle"


def thumb(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        img.save(dst, "JPEG", quality=85, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--thumb", type=int, default=320, help="max image side in pixels")
    a = p.parse_args()

    meta = json.loads((a.index / "meta.json").read_text())
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)

    shutil.copy(ROOT / meta["model"], OUT / "model.pt")
    shutil.copytree(a.index, OUT / "index")
    meta["model"] = f"{CONTAINER_BUNDLE}/model.pt"
    (OUT / "index" / "meta.json").write_text(json.dumps(meta, indent=2))

    data = ROOT / "data"
    shutil.copy(data / "catalog.parquet", OUT / "data" / "catalog.parquet")
    import pandas as pd

    paths = pd.read_parquet(data / "catalog.parquet")["path"].tolist()
    with ThreadPoolExecutor(8) as pool:
        list(tqdm(pool.map(lambda r: thumb(data / "images" / r, OUT / "data" / "images" / r, a.thumb), paths),
                  total=len(paths), desc="thumbnails"))
    print(f"bundle ready: {OUT}")


if __name__ == "__main__":
    main()
