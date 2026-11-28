"""Log training runs to MLflow.

A tracked run records

* parameters: the resolved configuration, flattened with dotted section
  prefixes (``model.params.max_depth``, ``resampling.method``) so that runs of
  different model families line up in the MLflow comparison view;
* metrics: every finite validation and test metric of
  :class:`~fraud_detection.train.TrainingResult`, prefixed by fold;
* tags: package version, model family, whether the data was synthetic, and
  the SHA-256 digest of the raw file;
* artifacts: the files written by :func:`fraud_detection.train.save_artifacts`;
* model: optionally, a pyfunc model that returns scores and decisions at the
  threshold chosen on validation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from fraud_detection import __version__
from fraud_detection.config import Config, TrackingConfig

if TYPE_CHECKING:
    from fraud_detection.train import TrainingResult

MODEL_ARTIFACT_NAME = "scoring_model"

_INVALID_METRIC_CHARS = re.compile(r"[^\w\-. :/]")


def _import_mlflow() -> ModuleType:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            'MLflow tracking requires the "track" extra: pip install -e ".[track]"'
        ) from exc
    return mlflow


def _param_value(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ",".join(str(v) for v in value)
    return str(value)


def _flatten(prefix: str, values: Mapping[str, Any]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(name, value))
        else:
            flat[name] = _param_value(value)
    return flat


def flatten_params(config: Config) -> dict[str, str]:
    """Configuration as a flat mapping of MLflow parameters.

    Paths and tracking settings are left out: they describe the machine the
    run happened on, not the experiment.

    Parameters
    ----------
    config
        Configuration of the run.

    Returns
    -------
    dict
        Dotted parameter name mapped to its string value.
    """
    sections: dict[str, Mapping[str, Any]] = {
        "data": asdict(config.data),
        "features": asdict(config.features),
        "split": asdict(config.split),
        "resampling": asdict(config.resampling),
        "evaluation": asdict(config.evaluation),
        "model": {"name": config.model.name, "params": dict(config.model.params)},
    }
    params = {"seed": str(config.seed)}
    for section, values in sections.items():
        params.update(_flatten(section, values))
    return params


def metric_name(name: str) -> str:
    """Map a report key to a valid MLflow metric name.

    ``recall@p0.9`` becomes ``recall_at_p0.9``; other characters MLflow does
    not accept are replaced by underscores.
    """
    return _INVALID_METRIC_CHARS.sub("_", name.replace("@", "_at_"))


def run_metrics(result: TrainingResult) -> dict[str, float]:
    """Finite validation and test metrics, prefixed by fold.

    MLflow rejects NaN and infinite values on some backends, so they are
    dropped; ``metrics.json`` keeps the full record.
    """
    metrics: dict[str, float] = {}
    for fold, report in (("validation", result.validation), ("test", result.test)):
        for key, value in report.items():
            if isinstance(value, Real) and math.isfinite(value):
                metrics[metric_name(f"{fold}.{key}")] = float(value)
    return metrics


def run_tags(
    config: Config, synthetic: bool, data_sha256: str | None = None
) -> dict[str, str]:
    """Tags identifying the code version and the data source of a run."""
    tags = {
        "package_version": __version__,
        "model_family": config.model.name,
        "synthetic_data": str(synthetic).lower(),
    }
    if data_sha256 is not None:
        tags["data_sha256"] = data_sha256
    return tags


def _experiment_id(mlflow: ModuleType, tracking: TrackingConfig) -> str:
    experiment = mlflow.get_experiment_by_name(tracking.experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        tracking.experiment_name, artifact_location=tracking.artifact_location
    )


def log_training_run(
    result: TrainingResult,
    config: Config,
    artifact_paths: Mapping[str, Path],
    synthetic: bool = False,
    run_name: str | None = None,
    data_sha256: str | None = None,
) -> str:
    """Record a finished training run in MLflow.

    Parameters
    ----------
    result
        Output of :func:`fraud_detection.train.run_training`.
    config
        Configuration of the run; ``config.tracking`` selects the server and
        experiment.
    artifact_paths
        Files written by :func:`fraud_detection.train.save_artifacts`. The
        ``model``, ``features``, and ``metrics`` entries are required when
        ``config.tracking.log_model`` is set.
    synthetic
        Whether the data came from the synthetic generator.
    run_name
        Optional display name of the run.
    data_sha256
        Digest of the raw data file, logged as a tag.

    Returns
    -------
    str
        MLflow run ID.
    """
    mlflow = _import_mlflow()
    tracking = config.tracking
    if tracking.tracking_uri:
        mlflow.set_tracking_uri(tracking.tracking_uri)
    experiment_id = _experiment_id(mlflow, tracking)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(run_tags(config, synthetic, data_sha256))
        mlflow.log_params(flatten_params(config))
        mlflow.log_metrics(run_metrics(result))
        for path in artifact_paths.values():
            if Path(path).is_dir():
                mlflow.log_artifacts(str(path), artifact_path=Path(path).name)
            else:
                mlflow.log_artifact(str(path))
        if tracking.log_model:
            from fraud_detection.tracking.pyfunc_model import log_scoring_model

            log_scoring_model(artifact_paths, MODEL_ARTIFACT_NAME)
    return run.info.run_id
