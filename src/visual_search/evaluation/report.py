"""Compare models and write the evaluation report to runs/<timestamp>/."""

import json
from datetime import datetime

from ..config import RUNS
from ..data import load_catalog
from ..tracking import log_metrics
from .benchmark import ann_benchmark
from .scoring import evaluate_model


def markdown_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def run(models: list[str], k: int = 10, ann_model: str | None = None) -> str:
    cat = load_catalog()
    results = [evaluate_model(m, k, cat=cat) for m in models]
    log_metrics(results[-1]["metrics"])  # the last model listed is the one being trained
    flat = [{"model": r["model"], "dim": r["dim"], **r["metrics"]} for r in results]
    report = [f"queries={results[0]['queries']} gallery={results[0]['gallery']} (held-out products)\n",
              "## Retrieval quality (exact search)\n", markdown_table(flat)]
    cats = sorted(results[0][f"ndcg@{k}_by_category"])
    by_cat = [{"model": r["model"], **{c: r[f"ndcg@{k}_by_category"][c] for c in cats}} for r in results]
    report += [f"\n## nDCG@{k} by category\n", markdown_table(by_cat)]
    ann = None
    if ann_model:
        ann = ann_benchmark(ann_model, k)
        report += [f"\n## Exact vs ANN ({ann_model})\n", markdown_table(ann)]
    text = "\n".join(report)

    out = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"results": results, "ann": ann}, indent=2))
    (out / "eval.md").write_text(text + "\n")
    return text + f"\n\nsaved to {out}"
