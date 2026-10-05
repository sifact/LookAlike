# catalog.py

Downloads Stanford Online Products and builds the laptop-sized image catalog every other stage uses.

- **Downloads the dataset in parallel.** It fetches the parquet files from the Hugging Face Hub in 8 MB chunks over 16 connections, and is resumable, because a single connection was too slow on a home network.
- **Samples a fair subset.** It picks the same number of products per category (300 by default) from each of the 12 categories, so no category dominates training or evaluation.
- **Keeps train and test products separate.** The two splits share no products (it asserts this), so evaluation measures how well search works on products the model has never seen.
- **Writes the images and the catalog table.** It extracts the selected JPEGs to `data/images/` and saves `data/catalog.parquet`, with columns `image_id`, `path`, `item_id`, `category`, `split` and `role`.
- **Chooses queries and gallery, reproducibly.** It picks one image per held-out product as a query and puts the rest in the gallery. Separate random seeds for train and test keep the test set identical whether or not training data is included.
