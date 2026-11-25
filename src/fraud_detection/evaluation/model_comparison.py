"""Side-by-side evaluation of supervised, unsupervised, and hybrid models.

Every candidate is fitted on the same chronological folds and evaluated with
the same protocol:

1. fit on the training fold (unsupervised detectors use its legitimate rows
   only), with early stopping on the validation fold where supported;
2. choose the decision threshold on the validation fold with the configured
   strategy;
3. score the test fold once and report ranking metrics and the cost at the
   validation threshold.

The table is sorted by validation PR-AUC, which is the quantity a model would
be selected on. Test metrics are reported for every candidate but must not be
used to pick one, since that would turn the test fold into a second
validation fold and bias the reported performance upward.

Unsupervised scores are empirical CDF values under legitimate training rows
rather than fraud probabilities. Ranking metrics and validation-chosen
thresholds are invariant to this difference in scale, so the columns are
comparable across families.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from fraud_detection.config import EvaluationConfig, ResamplingConfig
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import TimeSplit
from fraud_detection.evaluation.metrics import ranking_report
from fraud_detection.evaluation.thresholds import evaluate_threshold, select_threshold
from fraud_detection.models.base import MODEL_REGISTRY, fit_model

Family = Literal["rule", "supervised", "unsupervised", "hybrid"]

MODEL_FAMILIES: dict[str, Family] = {
    "logistic_regression": "supervised",
    "xgboost": "supervised",
    "lightgbm": "supervised",
    "catboost": "supervised",
    "feedforward": "supervised",
    "gru": "supervised",
    "isolation_forest": "unsupervised",
    "local_outlier_factor": "unsupervised",
    "autoencoder": "unsupervised",
    "variational_autoencoder": "unsupervised",
    "hybrid": "hybrid",
}


@dataclass(frozen=True)
class Candidate:
    """One entry of the comparison.

    Attributes
    ----------
    model
        Key of :data:`fraud_detection.models.base.MODEL_REGISTRY`.
    params
        Hyperparameters passed to the fit function.
    features
        Input columns; ``None`` uses the comparison's shared feature list.
    extra_features
        Columns appended to the inputs, for example the history columns that
        only the sequence model reads.
    resampling
        Training fold rebalancing; ``None`` uses the shared setting.
    """

    model: str
    params: Mapping[str, Any] | None = None
    features: Sequence[str] | None = None
    extra_features: Sequence[str] = ()
    resampling: ResamplingConfig | None = None

    def __post_init__(self) -> None:
        if self.model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {self.model!r}.")

    @property
    def family(self) -> Family:
        """Model family used to group the comparison table."""
        return MODEL_FAMILIES.get(self.model, "supervised")


def default_candidates(history_features: Sequence[str] = ()) -> dict[str, Candidate]:
    """One representative per model family with library default settings.

    Parameters
    ----------
    history_features
        History columns of :mod:`fraud_detection.features.sequence`. The GRU
        candidate is included only when they are given, and only it receives
        them, so the other candidates keep identical inputs.

    Returns
    -------
    dict
        Candidates keyed by display name.
    """
    candidates = {
        "logistic_regression": Candidate("logistic_regression"),
        "xgboost": Candidate("xgboost"),
        "feedforward_focal": Candidate("feedforward", {"loss": "focal"}),
        "isolation_forest": Candidate("isolation_forest"),
        "autoencoder": Candidate("autoencoder"),
        "variational_autoencoder": Candidate("variational_autoencoder"),
        "hybrid_autoencoder_xgboost": Candidate("hybrid"),
    }
    if history_features:
        candidates["gru"] = Candidate("gru", extra_features=tuple(history_features))
    return candidates


def _resolve_features(candidate: Candidate, shared: Sequence[str]) -> list[str]:
    base = shared if candidate.features is None else candidate.features
    return [*base, *candidate.extra_features]


def evaluate_candidate(
    name: str,
    candidate: Candidate,
    split: TimeSplit,
    features: Sequence[str],
    target: str,
    evaluation: EvaluationConfig,
    resampling: ResamplingConfig,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit one candidate and collect its validation and test metrics.

    Returns
    -------
    dict
        One row of the comparison table; see :func:`compare_models`.
    """
    columns = _resolve_features(candidate, features)
    folds = prepare_modeling_folds(
        split, columns, target, candidate.resampling or resampling, seed
    )
    start = time.perf_counter()
    model = fit_model(
        candidate.model,
        folds.train,
        folds.X_validation,
        folds.y_validation,
        candidate.params,
        seed=seed,
    )
    fit_seconds = time.perf_counter() - start

    validation_scores = model.predict_scores(folds.X_validation)
    choice = select_threshold(folds.y_validation, validation_scores, evaluation)
    test_scores = model.predict_scores(folds.X_test)
    validation = ranking_report(folds.y_validation, validation_scores)
    test = ranking_report(folds.y_test, test_scores, (0.5, 0.8), (0.5, 0.8))
    at_threshold = evaluate_threshold(folds.y_test, test_scores, choice, evaluation)
    return {
        "candidate": name,
        "family": candidate.family,
        "model": candidate.model,
        "n_features": len(columns),
        "fit_seconds": fit_seconds,
        "validation_pr_auc": validation["pr_auc"],
        "test_pr_auc": test["pr_auc"],
        "test_roc_auc": test["roc_auc"],
        "test_recall@p0.5": test["recall@p0.5"],
        "test_recall@p0.8": test["recall@p0.8"],
        "test_precision@r0.5": test["precision@r0.5"],
        "test_precision@r0.8": test["precision@r0.8"],
        "threshold": choice.threshold,
        "test_precision": at_threshold["precision"],
        "test_recall": at_threshold["recall"],
        "test_normalized_cost": at_threshold["normalized_cost"],
    }


def compare_models(
    split: TimeSplit,
    features: Sequence[str],
    target: str,
    evaluation: EvaluationConfig,
    candidates: Mapping[str, Candidate] | None = None,
    resampling: ResamplingConfig | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate every candidate on the same folds with the same protocol.

    Parameters
    ----------
    split
        Chronological folds of a table holding every candidate's features.
    features
        Shared input columns for candidates that do not set their own.
    target
        Binary label column.
    evaluation
        Threshold strategy and unit costs.
    candidates
        Named candidates; defaults to :func:`default_candidates` without the
        GRU.
    resampling
        Shared training fold rebalancing; defaults to none.
    seed
        Random state of resampling and every model.

    Returns
    -------
    pandas.DataFrame
        One row per candidate indexed by name, sorted by validation PR-AUC in
        decreasing order. ``threshold`` is chosen on validation; columns
        prefixed ``test_`` come from a single pass over the test fold.
    """
    candidates = dict(default_candidates() if candidates is None else candidates)
    if not candidates:
        raise ValueError("At least one candidate is required.")
    resampling = resampling or ResamplingConfig("none")
    rows = [
        evaluate_candidate(
            name, candidate, split, features, target, evaluation, resampling, seed
        )
        for name, candidate in candidates.items()
    ]
    table = pd.DataFrame.from_records(rows).set_index("candidate")
    return table.sort_values("validation_pr_auc", ascending=False)
