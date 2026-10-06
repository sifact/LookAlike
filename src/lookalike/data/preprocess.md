# preprocess.py

Every image transformation: loading, cropping, model input preparation and training augmentations.

- **Loads any image consistently.** `load_rgb` opens a file or uploaded bytes and converts it to RGB, so PNG transparency, grayscale and CMYK images all work.
- **Supports crop search.** `crop` cuts out the box the user drew. It accepts fractions (0–1) or pixels and rejects boxes too small to be meaningful.
- **Prepares model input.** `eval_transform` resizes, center-crops to 224×224, scales to 0–1 and normalizes with ImageNet statistics. Catalog images and user queries go through the same steps, so their vectors are comparable.
- **Creates self-supervised training views.** `selfsup_augment` applies strong random crops, color jitter, grayscale and blur, so two views of the same photo look different but remain the same object (SimCLR-style).
- **Adds light augmentation for real pairs.** `light_augment` applies only mild crops and flips, because same-product pairs already differ naturally in angle and background.
