"""Download Stanford Online Products and build a laptop-sized catalog.

Source: a parquet mirror of SOP on the Hugging Face Hub (columns: image, image_path,
item_id, category). Train and test splits contain disjoint products.

catalog.parquet columns:
    image_id  int, stable id used by the index and the API
    path      relative to data/images
    item_id   product id (images of the same product are "same product" relevant)
    category  one of 12 product categories
    split     train | test
    role      train | query | gallery
"""

import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from tqdm import tqdm

from ..config import CATALOG, HF_REPO, IMAGES, RAW

CHUNK = 8 * 2**20


def _fetch_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if attempt == retries - 1:
                raise
    raise OSError(f"short read for bytes {start}-{end}")


def _download_file(filename: str, size: int, workers: int) -> None:
    """Parallel ranged download. Resumable: finished chunks are recorded in a sidecar file."""
    out = RAW / filename
    if out.exists() and out.stat().st_size == size and not out.with_suffix(".parts").exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    parts_file = out.with_suffix(".parts")
    done = set(json.loads(parts_file.read_text())) if parts_file.exists() and out.exists() else set()
    if not out.exists():
        with open(out, "wb") as f:
            f.truncate(size)
    url = hf_hub_url(HF_REPO, filename, repo_type="dataset")
    starts = [s for s in range(0, size, CHUNK) if s not in done]
    with open(out, "r+b") as f, ThreadPoolExecutor(workers) as pool, \
            tqdm(total=size, initial=len(done) * CHUNK, unit="B", unit_scale=True, desc=filename) as bar:
        futures = {pool.submit(_fetch_range, url, s, min(s + CHUNK, size) - 1): s for s in starts}
        for fut in as_completed(futures):
            s = futures[fut]
            data = fut.result()
            f.seek(s)
            f.write(data)
            done.add(s)
            parts_file.write_text(json.dumps(sorted(done)))
            bar.update(len(data))
    parts_file.unlink()


def download(workers: int = 16) -> None:
    api = HfApi()
    info = api.dataset_info(HF_REPO, files_metadata=True)
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        if s.rfilename.endswith(".parquet"):
            _download_file(s.rfilename, s.size, workers)


def _shards(split: str) -> list:
    files = sorted((RAW / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} shards in {RAW / 'data'}; run `lookalike data` first")
    return files


def _sample_items(split: str, per_category: int, rng: np.random.Generator) -> set[str]:
    meta = pd.concat(pq.read_table(f, columns=["item_id", "category"]).to_pandas() for f in _shards(split))
    meta["item_id"] = meta["item_id"].astype(str)
    chosen = set()
    for _, items in meta.groupby("category")["item_id"]:
        unique = items.unique()
        chosen.update(rng.choice(unique, size=min(per_category, len(unique)), replace=False))
    return chosen


def _write_images(split: str, items: set[str]) -> list[dict]:
    (IMAGES / split).mkdir(parents=True, exist_ok=True)
    rows, counts = [], defaultdict(int)
    for f in _shards(split):
        pf = pq.ParquetFile(f)
        for batch in tqdm(pf.iter_batches(batch_size=256, columns=["image", "item_id", "category"]),
                          total=-(-pf.metadata.num_rows // 256), desc=f"{split}:{f.stem}"):
            for rec in batch.to_pylist():
                item = str(rec["item_id"])
                if item not in items:
                    continue
                rel = f"{split}/{item}_{counts[item]}.jpg"
                counts[item] += 1
                (IMAGES / rel).write_bytes(rec["image"]["bytes"])
                rows.append({"path": rel, "item_id": item, "category": rec["category"], "split": split})
    return rows


def prepare(per_category_train: int = 300, per_category_test: int = 300,
            max_queries: int = 2000, seed: int = 0) -> pd.DataFrame:
    # Separate streams so the eval set is identical whether or not train data is included.
    rng = np.random.default_rng([seed, 1])
    test_items = _sample_items("test", per_category_test, rng)
    train_items = set()
    if per_category_train:
        train_items = _sample_items("train", per_category_train, np.random.default_rng([seed, 0]))
    assert not train_items & test_items, "train/test products overlap"

    rows = _write_images("train", train_items) if train_items else []
    df = pd.DataFrame(rows + _write_images("test", test_items))
    df.insert(0, "image_id", np.arange(len(df)))
    df["role"] = np.where(df["split"] == "train", "train", "gallery")

    # One query image per held-out product that has at least one other image to find.
    test = df[df["split"] == "test"]
    sizes = test.groupby("item_id").size()
    eligible = sizes[sizes >= 2].index.to_numpy()
    query_items = rng.choice(eligible, size=min(max_queries, len(eligible)), replace=False)
    pick = test.groupby("item_id")["image_id"].agg(lambda s: rng.choice(s.to_numpy()))
    df.loc[df["image_id"].isin(pick.loc[query_items]), "role"] = "query"

    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CATALOG, index=False)
    return df


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(CATALOG)
