"""Import faiss safely alongside torch on macOS.

torch and faiss-cpu each ship their own libomp. Loading both aborts the process
unless KMP_DUPLICATE_LIB_OK is set, and multi-threaded faiss still segfaults
after torch has initialized OpenMP. Importing torch first and running faiss
single-threaded is stable. Always import faiss through this module.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402,F401  (must load before faiss)
import faiss  # noqa: E402

faiss.omp_set_num_threads(1)

__all__ = ["faiss"]
