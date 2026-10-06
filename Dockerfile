# Visual search API + web UI in one image.
# Build for AKS (x86 nodes) from an Apple Silicon Mac:
#   docker build --platform linux/amd64 -t <registry>.azurecr.io/visual-search:v1 .
# Requires deploy/bundle/ (run: uv run python deploy/make_bundle.py --index artifacts/indexes/<name>).

# 1) Web UI: static files, so build it natively (fast) whatever the target platform.
FROM --platform=$BUILDPLATFORM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# 2) API
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VS_DATA=/app/bundle/data \
    VS_INDEX=/app/bundle/index \
    VS_ARTIFACTS=/app/bundle/artifacts
WORKDIR /app

# CPU-only PyTorch (~4x smaller than the default CUDA build). Own layer: rarely changes.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
# Editable install keeps the package at /app/src, so it finds /app/frontend/dist.
RUN pip install -e .

COPY --from=ui /ui/dist ./frontend/dist
COPY deploy/bundle ./bundle

EXPOSE 8000
CMD ["uvicorn", "lookalike.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
