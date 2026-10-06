# loss.py

The contrastive loss that teaches the model which images belong together.

- **Implements InfoNCE.** It is the book's "similarity → softmax → cross-entropy" loss for contrastive training.
- **Compares every query with every positive.** It builds a batch × batch similarity matrix, so each row is one query scored against all positives in the batch.
- **Uses in-batch negatives.** For each query the correct answer is its own positive (the diagonal), and the other 63 positives in a batch of 64 act as negatives, so no separate negative sampling is needed.
- **Sharpens with temperature.** Dividing similarities by a small temperature (0.05) makes the softmax strict, forcing the model to separate near-matches clearly.
- **Is symmetric.** It averages the loss in both directions (query → positive and positive → query), so both sides of every pair learn equally.
