"""The embedding model and batch embedding of catalog images."""

from .embed import catalog_embeddings, embed_paths
from .embedder import Embedder, load_embedder, model_tag

__all__ = ["Embedder", "catalog_embeddings", "embed_paths", "load_embedder", "model_tag"]
