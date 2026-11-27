"""MLflow pyfunc wrapper around a model saved by the training run.

The logged model bundles the pickled model, its feature list, and
``metrics.json`` (for the decision threshold), together with the source of
this package, so it can be loaded with ``mlflow.pyfunc.load_model`` without
installing the project.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

import fraud_detection
from fraud_detection.evaluation.thresholds import apply_threshold

_MODEL_FILES = ("model", "features", "metrics")


class FraudScoringModel(mlflow.pyfunc.PythonModel):
    """Score a feature table and flag rows at the validation threshold.

    The input holds the engineered feature columns listed in
    ``features.json``; raw transactions go through
    :func:`fraud_detection.features.build.build_features` first. Extra
    columns are ignored. The output has a ``score`` and a binary
    ``flagged`` column, indexed like the input.
    """

    def load_context(self, context: Any) -> None:
        with Path(context.artifacts["model"]).open("rb") as fh:
            self.model = pickle.load(fh)
        self.features = json.loads(Path(context.artifacts["features"]).read_text())
        metrics = json.loads(Path(context.artifacts["metrics"]).read_text())
        # metrics.json stores an infinite threshold as null
        threshold = metrics["threshold"]["threshold"]
        self.threshold = float("inf") if threshold is None else float(threshold)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        missing = [c for c in self.features if c not in model_input.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")
        scores = self.model.predict_scores(model_input[self.features])
        return pd.DataFrame(
            {"score": scores, "flagged": apply_threshold(scores, self.threshold)},
            index=model_input.index,
        )


def runtime_requirements() -> list[str]:
    """Core dependencies of the installed package, without optional extras."""
    try:
        declared = requires("fraud-detection") or []
    except PackageNotFoundError:
        return []
    return [r for r in declared if not re.search(r";\s*extra\s*==", r)]


def log_scoring_model(artifact_paths: Mapping[str, Path], name: str) -> Any:
    """Log :class:`FraudScoringModel` to the active MLflow run.

    Parameters
    ----------
    artifact_paths
        Must contain ``model``, ``features``, and ``metrics`` file paths.
    name
        Name of the model within the run.

    Returns
    -------
    mlflow.models.model.ModelInfo
        Metadata of the logged model, including its URI.
    """
    missing = [k for k in _MODEL_FILES if k not in artifact_paths]
    if missing:
        raise KeyError(f"Artifacts required for the scoring model: {missing}")
    return mlflow.pyfunc.log_model(
        name=name,
        python_model=FraudScoringModel(),
        artifacts={k: str(artifact_paths[k]) for k in _MODEL_FILES},
        code_paths=[str(Path(fraud_detection.__file__).parent)],
        pip_requirements=runtime_requirements(),
    )
