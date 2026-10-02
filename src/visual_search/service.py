"""Serving: prediction pipeline (embed -> ANN -> re-rank) and indexing pipeline (add new images).

Run with `visual-search serve --index artifacts/indexes/<name>`.
"""

import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import BLOCKED_IDS, IMAGES, ROOT, device
from .data import load_catalog
from .index import VectorIndex
from .model import load_embedder
from .preprocess import crop, eval_transform, load_rgb
from .rerank import Candidate, rerank

UPLOADS = IMAGES / "uploads"
UPLOADS_LOG = UPLOADS / "catalog.jsonl"
STATIC = Path(__file__).parent / "static"
FRONTEND_DIST = ROOT / "frontend" / "dist"  # React app (cd frontend && npm run build)


class SearchEngine:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.index = VectorIndex.load(index_dir)
        self.dev = device()
        self.model = load_embedder(self.meta["model"]).to(self.dev)
        self.tf = eval_transform()
        self.catalog = {int(r["image_id"]): r for r in load_catalog().to_dict("records")}
        if UPLOADS_LOG.exists():
            for line in UPLOADS_LOG.read_text().splitlines():
                r = json.loads(line)
                self.catalog[int(r["image_id"])] = r
        self.blocked = {int(x) for x in BLOCKED_IDS.read_text().split()} if BLOCKED_IDS.exists() else set()
        self.lock = threading.Lock()

    @torch.no_grad()
    def embed(self, img) -> np.ndarray:
        return self.model(self.tf(img).unsqueeze(0).to(self.dev)).cpu().numpy()[0]

    def search(self, img, k: int, box=None) -> dict:
        t0 = time.perf_counter()
        vec = self.embed(crop(img, box))
        t1 = time.perf_counter()
        with self.lock:
            # Over-fetch so re-ranking can drop items and still return k.
            scores, ids = self.index.search(vec, k * 3 + 5)
            ids, scores = ids[0], scores[0]
            ids, scores = ids[ids >= 0], scores[ids >= 0]
            vecs = self.index.vectors_for(ids)
        t2 = time.perf_counter()
        kept = rerank(vec, [Candidate(int(i), float(s)) for i, s in zip(ids, scores)], vecs, k, self.blocked)
        t3 = time.perf_counter()
        return {
            "results": [self._describe(c) for c in kept],
            "timings_ms": {"embed": round((t1 - t0) * 1000, 1), "ann": round((t2 - t1) * 1000, 2),
                           "rerank": round((t3 - t2) * 1000, 2)},
        }

    def add(self, img, item_id: str | None, category: str | None) -> dict:
        vec = self.embed(img)
        with self.lock:
            image_id = int(max(max(self.catalog), int(self.index.ids.max())) + 1)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            rel = f"uploads/{image_id}.jpg"
            img.save(IMAGES / rel, "JPEG", quality=92)
            self.index.add(vec, [image_id])
            self.index.save(self.index_dir, {k: v for k, v in self.meta.items() if k not in ("kind", "n")})
            row = {"image_id": image_id, "path": rel, "item_id": item_id or f"upload-{image_id}",
                   "category": category or "unknown", "split": "upload", "role": "upload"}
            self.catalog[image_id] = row
            with UPLOADS_LOG.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def _describe(self, c: Candidate) -> dict:
        r = self.catalog.get(c.image_id, {})
        return {"image_id": c.image_id, "score": round(c.score, 4), "item_id": r.get("item_id"),
                "category": r.get("category"), "url": f"/images/{c.image_id}"}


def _parse_box(box: str | None):
    if not box:
        return None
    try:
        vals = tuple(float(x) for x in box.split(","))
    except ValueError:
        raise HTTPException(422, "box must be 'x0,y0,x1,y1'")
    if len(vals) != 4:
        raise HTTPException(422, "box must be 'x0,y0,x1,y1'")
    return vals


async def _read_image(file: UploadFile):
    try:
        return load_rgb(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(415, "could not decode image")


def create_app(index_dir: Path | None = None) -> FastAPI:
    index_dir = Path(index_dir or os.environ["VS_INDEX"])
    engine = SearchEngine(index_dir)
    app = FastAPI(title="Visual search")
    app.state.engine = engine

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.meta["model"], "index": engine.index.kind,
                "images": len(engine.index), "device": str(engine.dev)}

    @app.post("/search")
    async def search(file: UploadFile = File(...), k: int = Form(10), box: str | None = Form(None)):
        if not 1 <= k <= 100:
            raise HTTPException(422, "k must be between 1 and 100")
        img = await _read_image(file)
        try:
            return engine.search(img, k, _parse_box(box))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/images")
    async def add_image(file: UploadFile = File(...), item_id: str | None = Form(None),
                        category: str | None = Form(None)):
        return engine.add(await _read_image(file), item_id, category)

    @app.get("/images/{image_id}")
    def get_image(image_id: int):
        row = engine.catalog.get(image_id)
        if row is None or image_id in engine.blocked:
            raise HTTPException(404, "image not found")
        return FileResponse(IMAGES / row["path"])

    # UI: the built React app if present, else the dependency-free fallback page.
    # Mounted last so the API routes above take precedence.
    if (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def home():
            return (STATIC / "index.html").read_text()

    return app
