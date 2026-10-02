"""End-to-end API test on a tiny synthetic catalog with a small off-the-shelf model."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def client(tmp_path, monkeypatch):
    from visual_search import config, data, service
    from visual_search.index import VectorIndex
    from visual_search.model import Embedder

    images = tmp_path / "images"
    (images / "test").mkdir(parents=True)
    rows, rng = [], np.random.default_rng(0)
    colors = [(220, 30, 30), (30, 200, 30), (30, 30, 220)]
    for i in range(12):
        c = np.array(colors[i % 3]) + rng.integers(-15, 15, 3)
        Image.new("RGB", (64, 64), tuple(int(v) for v in c)).save(images / f"test/{i}.jpg")
        rows.append({"image_id": i, "path": f"test/{i}.jpg", "item_id": str(i % 3),
                     "category": "c", "split": "test", "role": "gallery"})
    catalog = tmp_path / "catalog.parquet"
    pd.DataFrame(rows).to_parquet(catalog)
    for mod in (config, data, service):
        monkeypatch.setattr(mod, "CATALOG", catalog, raising=False)
        monkeypatch.setattr(mod, "IMAGES", images, raising=False)
    monkeypatch.setattr(service, "UPLOADS", images / "uploads")
    monkeypatch.setattr(service, "UPLOADS_LOG", images / "uploads/catalog.jsonl")
    monkeypatch.setattr(service, "BLOCKED_IDS", tmp_path / "none.txt")

    model = Embedder("resnet18.a1_in1k", pretrained=False).eval()
    ckpt = tmp_path / "tiny.pt"
    model.save(ckpt)
    from visual_search.embed import embed_paths
    vecs = embed_paths(model, [images / r["path"] for r in rows], workers=0)
    VectorIndex.build(vecs, np.arange(12), "flat").save(tmp_path / "index", {"model": str(ckpt)})

    from fastapi.testclient import TestClient
    return TestClient(service.create_app(tmp_path / "index"))


def _png(color):
    buf = io.BytesIO()
    Image.new("RGBA", (80, 60), color + (255,)).save(buf, "PNG")
    return buf.getvalue()


def test_search_crop_add(client):
    assert client.get("/health").json()["images"] == 12
    r = client.post("/search", files={"file": ("q.png", _png((220, 30, 30)))}, data={"k": "3"})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert 1 <= len(res) <= 3 and set(r.json()["timings_ms"]) == {"embed", "ann", "rerank"}
    assert client.get(res[0]["url"]).status_code == 200

    r = client.post("/search", files={"file": ("q.png", _png((30, 30, 220)))}, data={"k": "3", "box": "0.1,0.1,0.6,0.6"})
    assert r.status_code == 200
    bad = client.post("/search", files={"file": ("q.png", _png((30, 30, 220)))}, data={"box": "0,0,0.01,0.01"})
    assert bad.status_code == 422
    assert client.post("/search", files={"file": ("x.txt", b"not an image")}).status_code == 415

    added = client.post("/images", files={"file": ("n.png", _png((250, 250, 0)))}).json()
    assert added["image_id"] == 12
    assert client.get("/health").json()["images"] == 13
    assert client.get("/images/12").status_code == 200
