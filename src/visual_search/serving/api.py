"""HTTP API and web UI on top of SearchEngine.

Run with `visual-search serve --index artifacts/indexes/<name>`.
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config import IMAGES, ROOT
from ..data.preprocess import load_rgb
from .engine import SearchEngine

STATIC = Path(__file__).parent / "static"
FRONTEND_DIST = ROOT / "frontend" / "dist"  # React app (cd frontend && npm run build)


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
