# embed.py

Turns catalog images into embedding vectors, efficiently and only once per model.

- **Turns image files into embeddings.** Each image becomes a 256-number vector from the
  fine-tuned DINOv2 model, which is what search and evaluation compare.
- **Prepares every image the same way.** It opens the file as RGB, resizes, crops to
  224×224 and normalizes. This is the exact same preprocessing the server applies to a
  user's photo, so the vectors are comparable.
- **Works in batches, quickly.** It feeds 64 images at a time to the GPU (or CPU), with 4
  background workers loading the next images in parallel, and gradients turned off.
- **Caches the results.** Vectors are saved to `artifacts/embeddings/<model>.npz`, keyed
  by image path, so each image is embedded only once per model. Rerunning evaluation or
  building the index reuses them, and only new images are computed.
- **Returns vectors in catalog order.** Row *i* of the output always belongs to catalog
  row *i*, so evaluation and indexing can line up vectors with products and categories.
