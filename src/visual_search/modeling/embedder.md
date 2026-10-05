# embedder.py

The embedding model itself: turns an image tensor into a 256-number vector.

- **Wraps a pretrained backbone.** It loads any timm model by name, by default DINOv2 ViT-S/14 (self-supervised by Meta), with the classifier removed, so it outputs features.
- **Adds a projection layer.** A linear layer maps the backbone's 384 features to 256. It is trained during fine-tuning to adapt the features to product search.
- **Normalizes every output.** Vectors are scaled to length 1, so the dot product equals cosine similarity, which both FAISS and the loss rely on.
- **Saves and loads checkpoints.** `save` stores the weights with the backbone name, projection size and training metadata, and `load_embedder` rebuilds the model from a `.pt` file or a timm name for off-the-shelf baselines.
- **Names caches safely.** `model_tag` creates a filename-safe name for each model and includes the checkpoint's timestamp, so a retrained model never reuses stale cached embeddings.
