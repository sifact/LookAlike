# trainer.py

The fine-tuning loop: pairs → model → InfoNCE → saved checkpoint.

- **Sets up a reproducible run.** It seeds Python, NumPy and PyTorch, picks the device (Mac GPU, NVIDIA GPU or CPU), and loads the training rows from the catalog.
- **Fine-tunes gently.** AdamW trains the pretrained backbone at a small learning rate (2e-5) and the new projection layer 50× faster, with warm-up and then cosine decay.
- **Fits on a 16 GB laptop.** Gradient checkpointing recomputes activations instead of storing them, keeping the batch at 64 (more negatives) without running out of memory.
- **Tracks progress.** It shows loss in a progress bar, records loss history every 20 steps, and sends loss, learning rate and settings to MLflow when running in Azure ML.
- **Saves the result.** It writes `artifacts/models/<name>.pt` (weights plus config) and a matching `.json` with the full settings, timing and loss history.
