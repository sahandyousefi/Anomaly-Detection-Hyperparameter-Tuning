"""Gradient boosted trees with early stopping on the validation fold.

XGBoost (Chen and Guestrin, 2016) is trained on the training fold, after any
rebalancing, and the number of boosting rounds is chosen by early stopping on
the untouched validation fold. Missing values are passed through, since the
tree learner assigns them a default split direction; no imputation is needed
for the point-in-time aggregates.

The validation fold therefore serves two purposes: the stopping round and,
later, the decision threshold. Both are low-dimensional choices, and the test
fold is never used for either. The stopping metric is the area under the
precision-recall curve (``aucpr``), which matches the primary evaluation metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.data.resample import ResampledFold

XGBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
}


@dataclass(frozen=True)
class BoostedModel:
    """A fitted booster and its validation learning curve.

    Attributes
    ----------
    model
        Fitted classifier. Predictions use the rounds up to ``best_iteration``.
    best_iteration
        Zero-based index of the best round on the validation fold.
    best_score
        Validation value of the stopping metric at ``best_iteration``.
    validation_curve
        Validation metric after each round, up to the round that triggered the
        stop.
    """

    model: XGBClassifier
    best_iteration: int
    best_score: float
    validation_curve: np.ndarray

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability, using rounds up to the best iteration."""
        return self.model.predict_proba(X)[:, 1]


def xgboost_params(
    params: Mapping[str, Any] | None = None,
    scale_pos_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Merge user parameters over :data:`XGBOOST_DEFAULTS`.

    Parameters
    ----------
    params
        Overrides, typically ``config.model.params``.
    scale_pos_weight
        Positive class weight from the resampling step. An explicit
        ``scale_pos_weight`` in ``params`` takes precedence.
    seed
        Random state, unless ``params`` sets ``random_state``.

    Returns
    -------
    dict
        Keyword arguments for :class:`xgboost.XGBClassifier`.
    """
    merged = {
        **XGBOOST_DEFAULTS,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
    }
    merged.update(params or {})
    return merged


def fit_xgboost(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> BoostedModel:
    """Train XGBoost and stop when validation PR-AUC stops improving.

    Parameters
    ----------
    train
        Training rows after rebalancing; its ``scale_pos_weight`` is used unless
        ``params`` overrides it.
    X_validation, y_validation
        Validation fold at the observed class ratio.
    params
        Overrides of :data:`XGBOOST_DEFAULTS`. ``n_estimators`` is the upper
        bound on boosting rounds.
    early_stopping_rounds
        Rounds without improvement before training stops.
    seed
        Random state of the booster.

    Returns
    -------
    BoostedModel
        Fitted model, best round, and validation learning curve.
    """
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1.")
    if list(X_validation.columns) != list(train.X.columns):
        raise ValueError("Training and validation features must match in order.")

    kwargs = xgboost_params(params, train.scale_pos_weight, seed)
    model = XGBClassifier(early_stopping_rounds=early_stopping_rounds, **kwargs)
    model.fit(
        train.X,
        train.y,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    metric = kwargs["eval_metric"]
    # XGBoost stops on the last metric when several are given
    if not isinstance(metric, str):
        metric = metric[-1]
    curve = np.asarray(model.evals_result()["validation_0"][metric], dtype="float64")
    return BoostedModel(
        model=model,
        best_iteration=int(model.best_iteration),
        best_score=float(model.best_score),
        validation_curve=curve,
    )
