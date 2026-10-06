# report.py

Runs the full comparison and writes the evaluation report.

- **Evaluates several models in one run.** It scores every model given (baselines and fine-tuned) on the same queries and gallery, so the numbers are directly comparable.
- **Builds markdown tables.** `markdown_table` turns result rows into tables for overall metrics and nDCG per category.
- **Optionally adds the ANN benchmark.** With `--ann <model>`, it appends the exact-vs-approximate comparison for that model.
- **Saves the report.** It writes `runs/<timestamp>/eval.md` (human-readable) and `eval.json` (all numbers), so each run is kept and can be compared later.
- **Logs to MLflow.** It sends the last model's metrics to MLflow, so they appear in Azure ML Studio when evaluation runs as a cloud job.
