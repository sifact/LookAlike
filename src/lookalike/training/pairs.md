# pairs.py

Builds training examples as (query, positive) image pairs — the book's "how do we label similarity?" choice.

- **Self-supervised pairs.** `SelfSupPairs` creates two augmented views of the same photo as query and positive. This needs no labels and is never wrong, but never shows the model two *different* photos of the same product.
- **Interaction pairs.** `InteractionPairs` uses two different photos of the same product, a stand-in for "a user clicked this result". This is the mode the deployed model was trained with.
- **Simulates noisy clicks.** With `noise > 0`, that fraction of positives is replaced by a random image, mimicking real click data where users sometimes click unrelated results.
- **Avoids false negatives.** Interaction pairs are sampled one per product, so the same product never appears twice in a batch and is never treated as a wrong match for itself.
- **Selects the mode by name.** `make_pairs(train, mode, noise)` returns the right dataset for `"selfsup"` or `"interactions"`, so the training loop doesn't need to know the difference.
