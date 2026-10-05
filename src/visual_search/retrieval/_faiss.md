# _faiss.py

A small safety shim for importing FAISS together with PyTorch, especially on macOS.

- **Solves a crash.** torch and faiss-cpu each ship their own OpenMP runtime, and loading both in one process makes macOS abort ("OMP: Error #15").
- **Allows the duplicate runtime.** It sets `KMP_DUPLICATE_LIB_OK=TRUE` before anything loads, so the process doesn't abort on start.
- **Imports in the safe order.** PyTorch is imported first, then FAISS. The reverse order was found to segfault under load.
- **Runs FAISS single-threaded.** `faiss.omp_set_num_threads(1)` avoids the remaining crash under multi-threaded search. At tens of thousands of vectors, single-threaded search is still sub-millisecond.
- **Is the only way to import FAISS.** Every module does `from ._faiss import faiss` instead of `import faiss`, so the fix applies everywhere.
