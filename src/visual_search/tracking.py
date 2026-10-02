"""Optional MLflow logging. Active only when a tracking server is configured,
e.g. inside an Azure ML job (which sets MLFLOW_TRACKING_URI). No-op locally."""

import os

try:
    import mlflow
except ImportError:
    mlflow = None

ENABLED = mlflow is not None and bool(os.getenv("MLFLOW_TRACKING_URI"))


def log_params(params: dict) -> None:
    if ENABLED:
        mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if ENABLED:
        mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in metrics.items()}, step=step)
