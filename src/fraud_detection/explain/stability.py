r"""Stability of SHAP importance rankings across seeds and time folds.

An explanation is only useful for model governance if it does not change
materially when the model is refitted under conditions that should not
matter. Two sources of variation are measured.

* Seed variation. The same training data is refitted with seeds
  :math:`s = 1, \dots, S`, which changes row and column subsampling in the
  boosters and the sampled rows of the rebalancing step. All refits are
  explained on the same held-out rows, so differences come from the model
  alone.
* Fold variation. The model is refitted on the expanding windows of
  :func:`fraud_detection.tuning.optuna_search.expanding_window_folds` and each
  fit is explained on its own next block. Differences combine model variance
  and genuine drift in the transactions over time.

For each run :math:`r` the global importance vector :math:`I^{(r)}` is the
mean absolute SHAP value per feature. Stability is reported per pair of runs
(Spearman's :math:`\rho`, Kendall's :math:`\tau_b`, top-:math:`k` Jaccard
overlap) and per feature (mean and coefficient of variation of
:math:`I^{(r)}_j`, range of ranks, and the share of runs in which the feature
is in the top :math:`k`). Features with a small mean importance can have a
large coefficient of variation without practical consequence, so the per
feature table should be read together with the mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import pandas as pd

from fraud_detection.config import ResamplingConfig
from fraud_detection.data.resample import resample_training_fold
from fraud_detection.explain.importance_comparison import compare_rankings
from fraud_detection.explain.shap_analysis import (
    ShapExplanation,
    explain_model,
    sample_rows,
)
from fraud_detection.models.base import fit_model
from fraud_detection.tuning.optuna_search import expanding_window_folds


def importance_matrix(explanations: Mapping[str, ShapExplanation]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (rows) and run (columns).

    Parameters
    ----------
    explanations
        Run label mapped to its explanation. All runs must share features.

    Returns
    -------
    pandas.DataFrame
        Importance matrix with features in the order of the first run.
    """
    if not explanations:
        raise ValueError("At least one explanation is required.")
    columns = {}
    features = None
    for label, explanation in explanations.items():
        if features is None:
            features = explanation.feature_names
        elif explanation.feature_names != features:
            raise ValueError(f"Run {label!r} has different features.")
        columns[label] = explanation.mean_abs().reindex(features)
    return pd.DataFrame(columns)


def pairwise_agreement(matrix: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Rank agreement statistics for every pair of runs.

    Parameters
    ----------
    matrix
        Output of :func:`importance_matrix` with at least two runs.
    k
        Size of the top sets compared.

    Returns
    -------
    pandas.DataFrame
        One row per pair with columns ``run_a``, ``run_b``, ``spearman``,
        ``kendall`` and ``top_k_jaccard``.
    """
    if matrix.shape[1] < 2:
        raise ValueError("At least two runs are needed.")
    rows = []
    for a, b in combinations(matrix.columns, 2):
        _, agreement = compare_rankings(matrix[a], matrix[b], k)
        rows.append(
            {
                "run_a": a,
                "run_b": b,
                "spearman": agreement.spearman,
                "kendall": agreement.kendall,
                "top_k_jaccard": agreement.top_k_jaccard,
            }
        )
    return pd.DataFrame(rows)


def feature_stability(matrix: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Per-feature variation of importance and rank across runs.

    Parameters
    ----------
    matrix
        Output of :func:`importance_matrix`.
    k
        Size of the top set used for ``top_k_share``.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature with columns ``mean``, ``std``, ``cv``,
        ``mean_rank``, ``min_rank``, ``max_rank`` and ``top_k_share``, sorted
        by decreasing mean importance.
    """
    ranks = matrix.rank(ascending=False, method="min")
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1) if matrix.shape[1] > 1 else mean * 0.0
    table = pd.DataFrame(
        {
            "mean": mean,
            "std": std,
            "cv": (std / mean).where(mean > 0),
            "mean_rank": ranks.mean(axis=1),
            "min_rank": ranks.min(axis=1),
            "max_rank": ranks.max(axis=1),
            "top_k_share": (ranks <= k).mean(axis=1),
        }
    )
    return table.sort_values("mean", ascending=False)


def seed_stability(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_explain: pd.DataFrame,
    seeds: Sequence[int],
    params: Mapping[str, Any] | None = None,
    resampling: ResamplingConfig | None = None,
    max_rows: int | None = 2_000,
) -> dict[str, ShapExplanation]:
    """Refit with several seeds and explain the same rows each time.

    Parameters
    ----------
    model_name
        Key of :data:`fraud_detection.models.base.MODEL_REGISTRY`.
    X_train, y_train
        Training fold before rebalancing; it is rebalanced per seed.
    X_validation, y_validation
        Validation fold for early stopping.
    X_explain
        Held-out rows explained by every refit.
    seeds
        Random states, at least two.
    params
        Model hyperparameters.
    resampling
        Rebalancing of the training fold; defaults to none.
    max_rows
        Rows of ``X_explain`` explained; the same sample is used for all
        seeds.

    Returns
    -------
    dict
        ``"seed=<s>"`` mapped to the explanation of that refit.
    """
    if len(seeds) < 2:
        raise ValueError("At least two seeds are needed.")
    resampling = resampling or ResamplingConfig("none")
    # A fixed sample keeps the explained rows identical across seeds
    rows = sample_rows(X_explain, max_rows, seed=0)
    explanations = {}
    for seed in seeds:
        train = resample_training_fold(X_train, y_train, resampling, seed)
        model = fit_model(model_name, train, X_validation, y_validation, params, seed)
        explanations[f"seed={seed}"] = explain_model(
            model, rows, background=train.X, max_rows=None, seed=seed
        )
    return explanations


def fold_stability(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    steps: pd.Series,
    n_splits: int = 3,
    params: Mapping[str, Any] | None = None,
    resampling: ResamplingConfig | None = None,
    seed: int = 0,
    max_rows: int | None = 2_000,
) -> dict[str, ShapExplanation]:
    """Refit on expanding windows and explain each fold's next block.

    The scored block of each fold doubles as its early-stopping set, as in
    the tuning objective. Use the training and validation periods only, so
    the test period stays untouched.

    Parameters
    ----------
    model_name
        Key of :data:`fraud_detection.models.base.MODEL_REGISTRY`.
    X, y
        Features and labels of the period to cut into folds.
    steps
        Time index aligned with ``X``.
    n_splits
        Number of folds.
    params
        Model hyperparameters.
    resampling
        Rebalancing of each training window; defaults to none.
    seed
        Random state, fixed across folds.
    max_rows
        Rows of each scored block explained.

    Returns
    -------
    dict
        ``"steps=<first>-<last>"`` of the scored block mapped to its
        explanation.
    """
    resampling = resampling or ResamplingConfig("none")
    explanations = {}
    for fold in expanding_window_folds(steps, n_splits):
        train = resample_training_fold(
            X.iloc[fold.train], y.iloc[fold.train], resampling, seed
        )
        X_block, y_block = X.iloc[fold.validation], y.iloc[fold.validation]
        model = fit_model(model_name, train, X_block, y_block, params, seed)
        first, last = fold.validation_steps
        explanations[f"steps={first}-{last}"] = explain_model(
            model, X_block, background=train.X, max_rows=max_rows, seed=seed
        )
    return explanations


__all__ = [
    "feature_stability",
    "fold_stability",
    "importance_matrix",
    "pairwise_agreement",
    "seed_stability",
]
