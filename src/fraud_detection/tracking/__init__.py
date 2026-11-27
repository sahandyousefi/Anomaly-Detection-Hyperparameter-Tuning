"""Experiment tracking with MLflow.

MLflow is an optional dependency (``pip install -e ".[track]"``), so this
package imports it only when a run is actually logged.
"""

from fraud_detection.tracking.mlflow_run import (
    flatten_params,
    log_training_run,
    metric_name,
    run_metrics,
    run_tags,
)

__all__ = [
    "flatten_params",
    "log_training_run",
    "metric_name",
    "run_metrics",
    "run_tags",
]
