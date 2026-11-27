"""Gradient boosted trees with early stopping on the validation fold.

XGBoost (Chen and Guestrin, 2016), LightGBM (Ke et al., 2017), and CatBoost
(Prokhorenkova et al., 2018) are trained
on the training fold, after any rebalancing, and the number of boosting rounds
is chosen by early stopping on the untouched validation fold. Missing values
are passed through, since both learners assign them a default split direction;
no imputation is needed for the point-in-time aggregates.

The validation fold therefore serves two purposes: the stopping round and,
later, the decision threshold. Both are low-dimensional choices, and the test
fold is never used for either. The stopping metric is the area under the
precision-recall curve (``aucpr`` in XGBoost, ``average_precision`` in
LightGBM, ``PRAUC`` in CatBoost), which matches the primary evaluation metric.
The three libraries integrate the curve differently, so their stopping scores
are comparable only within one library; cross-model comparison uses
:func:`fraud_detection.evaluation.metrics.pr_auc` on the predicted scores.

LightGBM and CatBoost are optional dependencies, installed with the ``gbm``
extra.
"""

from __future__ import annotations

import inspect
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

LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
    # Same model for a given seed regardless of thread scheduling
    "deterministic": True,
    "force_row_wise": True,
}

CATBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "eval_metric": "PRAUC",
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
}

# Set as the only metric so early stopping does not also track the log loss
LIGHTGBM_METRIC = "average_precision"


@dataclass(frozen=True)
class BoostedModel:
    """A fitted booster and its validation learning curve.

    Attributes
    ----------
    model
        Fitted scikit-learn style classifier (XGBoost, LightGBM, or CatBoost).
        Its ``predict_proba`` uses the rounds up to ``best_iteration``.
    best_iteration
        Zero-based index of the best round on the validation fold.
    best_score
        Validation value of the stopping metric at ``best_iteration``.
    validation_curve
        Validation metric after each round, up to the round that triggered the
        stop.
    """

    model: Any
    best_iteration: int
    best_score: float
    validation_curve: np.ndarray

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability, using rounds up to the best iteration."""
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype="float64")


def _merge_params(
    defaults: Mapping[str, Any],
    params: Mapping[str, Any] | None,
    scale_pos_weight: float,
    seed: int,
) -> dict[str, Any]:
    merged = {**defaults, "scale_pos_weight": scale_pos_weight, "random_state": seed}
    merged.update(params or {})
    return merged


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
    return _merge_params(XGBOOST_DEFAULTS, params, scale_pos_weight, seed)


def lightgbm_params(
    params: Mapping[str, Any] | None = None,
    scale_pos_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Merge user parameters over :data:`LIGHTGBM_DEFAULTS`.

    Parameters
    ----------
    params
        Overrides of the LightGBM defaults.
    scale_pos_weight
        Positive class weight from the resampling step. An explicit
        ``scale_pos_weight`` in ``params`` takes precedence.
    seed
        Random state, unless ``params`` sets ``random_state``.

    Returns
    -------
    dict
        Keyword arguments for ``lightgbm.LGBMClassifier``.
    """
    return _merge_params(LIGHTGBM_DEFAULTS, params, scale_pos_weight, seed)


def check_fit_inputs(
    train: ResampledFold, X_validation: pd.DataFrame, early_stopping_rounds: int
) -> None:
    """Validate the arguments shared by the early-stopped fitters."""
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1.")
    if list(X_validation.columns) != list(train.X.columns):
        raise ValueError("Training and validation features must match in order.")


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
    check_fit_inputs(train, X_validation, early_stopping_rounds)
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


def _lightgbm_eval_kwargs(
    model: Any, X_validation: pd.DataFrame, y_validation: pd.Series
) -> dict[str, Any]:
    # LightGBM 4.7 replaced eval_set with eval_X and eval_y
    if "eval_X" in inspect.signature(model.fit).parameters:
        return {"eval_X": (X_validation,), "eval_y": (y_validation,)}
    return {"eval_set": [(X_validation, y_validation)]}


def fit_lightgbm(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> BoostedModel:
    """Train LightGBM and stop when validation average precision stops improving.

    Parameters
    ----------
    train
        Training rows after rebalancing; its ``scale_pos_weight`` is used unless
        ``params`` overrides it.
    X_validation, y_validation
        Validation fold at the observed class ratio.
    params
        Overrides of :data:`LIGHTGBM_DEFAULTS`. ``n_estimators`` is the upper
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
    import lightgbm as lgb

    check_fit_inputs(train, X_validation, early_stopping_rounds)
    kwargs = {
        "metric": LIGHTGBM_METRIC,
        **lightgbm_params(params, train.scale_pos_weight, seed),
    }
    history: dict[str, Any] = {}
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(
        train.X,
        train.y,
        **_lightgbm_eval_kwargs(model, X_validation, y_validation),
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(history),
        ],
    )
    curve = np.asarray(history["valid_0"][kwargs["metric"]], dtype="float64")
    # LightGBM counts iterations from 1
    best_iteration = int(model.best_iteration_) - 1
    return BoostedModel(
        model=model,
        best_iteration=best_iteration,
        best_score=float(curve[best_iteration]),
        validation_curve=curve,
    )


def catboost_params(
    params: Mapping[str, Any] | None = None,
    scale_pos_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Merge user parameters over :data:`CATBOOST_DEFAULTS`.

    Parameters
    ----------
    params
        Overrides of the CatBoost defaults. CatBoost names the thread count
        ``thread_count``, not ``n_jobs``.
    scale_pos_weight
        Positive class weight from the resampling step. An explicit
        ``scale_pos_weight`` in ``params`` takes precedence.
    seed
        Random state, unless ``params`` sets ``random_state``.

    Returns
    -------
    dict
        Keyword arguments for ``catboost.CatBoostClassifier``.
    """
    return _merge_params(CATBOOST_DEFAULTS, params, scale_pos_weight, seed)


def fit_catboost(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> BoostedModel:
    """Train CatBoost and stop when validation PR-AUC stops improving.

    The model is shrunk to the best iteration after training
    (``use_best_model``), so predictions use the early-stopped ensemble.

    Parameters
    ----------
    train
        Training rows after rebalancing; its ``scale_pos_weight`` is used unless
        ``params`` overrides it.
    X_validation, y_validation
        Validation fold at the observed class ratio.
    params
        Overrides of :data:`CATBOOST_DEFAULTS`. ``n_estimators`` is the upper
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
    from catboost import CatBoostClassifier

    check_fit_inputs(train, X_validation, early_stopping_rounds)
    kwargs = catboost_params(params, train.scale_pos_weight, seed)
    model = CatBoostClassifier(use_best_model=True, **kwargs)
    model.fit(
        train.X,
        train.y,
        eval_set=(X_validation, y_validation),
        early_stopping_rounds=early_stopping_rounds,
    )
    metric = kwargs["eval_metric"]
    curve = np.asarray(model.get_evals_result()["validation"][metric], dtype="float64")
    return BoostedModel(
        model=model,
        best_iteration=int(model.get_best_iteration()),
        best_score=float(model.get_best_score()["validation"][metric]),
        validation_curve=curve,
    )
