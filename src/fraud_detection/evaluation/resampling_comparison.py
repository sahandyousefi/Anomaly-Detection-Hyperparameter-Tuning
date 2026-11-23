r"""Compare class rebalancing strategies on the validation fold.

Fraud is rare, so the training fold is rebalanced in one of three ways
before fitting:

``undersample``
    Drop negatives at random until :math:`n_{pos} / n_{neg} = r`.
``smote``
    Add synthetic positives by interpolating between nearest positive
    neighbours (Chawla et al., 2002) until the same ratio is reached.
``class_weight``
    Keep every row and weight positives by
    :math:`w_{pos} = r \, n_{neg} / n_{pos}`, passed to the boosters as
    ``scale_pos_weight``.

Choosing among them is a model selection decision, so every candidate is
scored on the validation fold at its observed class ratio and the test fold
is not used. The comparison holds the feature set, model family,
hyperparameters, and seed fixed, so differences in validation PR-AUC and in
cost at the validation-optimal threshold are attributable to the rebalancing
step.

SMOTE interpolates feature vectors and cannot handle missing values, while
the point-in-time aggregates are missing for an account's first transaction.
When a SMOTE candidate is present, missing values in all folds are replaced
by ``fill_value`` for every candidate, so all methods see identical inputs.
A sentinel outside the observed range (the default, -1) keeps the "no
history" state separable for tree models.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from fraud_detection.config import EvaluationConfig, ResamplingConfig
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import TimeSplit
from fraud_detection.evaluation.metrics import pr_auc, roc_auc
from fraud_detection.evaluation.thresholds import select_threshold
from fraud_detection.models.base import fit_model

DEFAULT_CANDIDATES: dict[str, ResamplingConfig] = {
    "none": ResamplingConfig("none"),
    "undersample": ResamplingConfig("undersample", 0.1),
    "smote": ResamplingConfig("smote", 0.1),
    "class_weight": ResamplingConfig("class_weight", 0.1),
}


def fill_missing(split: TimeSplit, columns: Sequence[str], value: float) -> TimeSplit:
    """Return a copy of ``split`` with missing values in ``columns`` replaced.

    Parameters
    ----------
    split
        Chronological folds holding the feature columns.
    columns
        Columns to fill.
    value
        Constant replacement. A constant needs no fitting, so the same value
        can be applied to every fold without leakage.

    Returns
    -------
    TimeSplit
        Folds with the selected columns filled.
    """
    cols = list(columns)

    def _fill(fold: pd.DataFrame) -> pd.DataFrame:
        out = fold.copy()
        out[cols] = out[cols].fillna(value)
        return out

    return replace(
        split,
        train=_fill(split.train),
        validation=_fill(split.validation),
        test=_fill(split.test),
    )


def _evaluate_candidate(
    name: str,
    resampling: ResamplingConfig,
    split: TimeSplit,
    features: Sequence[str],
    target: str,
    model_name: str,
    params: Mapping[str, Any] | None,
    evaluation: EvaluationConfig,
    seed: int,
) -> dict[str, Any]:
    folds = prepare_modeling_folds(split, features, target, resampling, seed)
    start = time.perf_counter()
    model = fit_model(
        model_name,
        folds.train,
        folds.X_validation,
        folds.y_validation,
        params,
        seed=seed,
    )
    fit_seconds = time.perf_counter() - start
    scores = model.predict_scores(folds.X_validation)
    choice = select_threshold(folds.y_validation, scores, evaluation)
    return {
        "candidate": name,
        "method": resampling.method,
        "sampling_strategy": resampling.sampling_strategy,
        "train_rows": len(folds.train.y),
        "train_positives": int(folds.train.y.sum()),
        "scale_pos_weight": folds.train.scale_pos_weight,
        "fit_seconds": fit_seconds,
        "pr_auc": pr_auc(folds.y_validation, scores),
        "roc_auc": roc_auc(folds.y_validation, scores),
        "threshold": choice.threshold,
        "precision": choice.precision,
        "recall": choice.recall,
        "normalized_cost": choice.normalized_cost,
    }


def compare_resampling(
    split: TimeSplit,
    features: Sequence[str],
    target: str,
    evaluation: EvaluationConfig,
    model_name: str = "xgboost",
    params: Mapping[str, Any] | None = None,
    candidates: Mapping[str, ResamplingConfig] | None = None,
    fill_value: float = -1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit one model per rebalancing strategy and score it on validation.

    Parameters
    ----------
    split
        Chronological folds of a table that already holds the feature columns.
    features
        Model input columns.
    target
        Binary label column.
    evaluation
        Threshold strategy and unit costs used for the cost columns.
    model_name
        Model family from :data:`fraud_detection.models.base.MODEL_REGISTRY`.
    params
        Hyperparameters shared by every candidate.
    candidates
        Named rebalancing settings. Defaults to :data:`DEFAULT_CANDIDATES`.
    fill_value
        Replacement for missing feature values, applied to all candidates
        when any of them uses SMOTE.
    seed
        Random state for the samplers and the model.

    Returns
    -------
    pandas.DataFrame
        One row per candidate, indexed by name and sorted by validation
        PR-AUC in decreasing order. Cost columns refer to the threshold chosen
        on the validation fold with ``evaluation``.
    """
    candidates = dict(DEFAULT_CANDIDATES if candidates is None else candidates)
    if not candidates:
        raise ValueError("At least one candidate is required.")
    if any(c.method == "smote" for c in candidates.values()):
        split = fill_missing(split, features, fill_value)
    rows = [
        _evaluate_candidate(
            name, cfg, split, features, target, model_name, params, evaluation, seed
        )
        for name, cfg in candidates.items()
    ]
    table = pd.DataFrame.from_records(rows).set_index("candidate")
    return table.sort_values("pr_auc", ascending=False)
