"""Common interface for every fraud scoring model.

Each model family is fitted by a function with the same signature,

``fit(train, X_validation, y_validation, params, seed) -> FraudModel``,

and the fitted object exposes ``predict_scores(X)``, returning one score per
row with higher meaning more likely fraud. Threshold selection, evaluation,
and explanation code only depend on this interface, so model families can be
swapped through ``config.model.name``.

Models that do not early-stop (the logistic regression baseline and the
unsupervised detectors) accept the validation fold and ignore it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from fraud_detection.data.resample import ResampledFold


@runtime_checkable
class FraudModel(Protocol):
    """A fitted model that scores transactions."""

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return one score per row of ``X``, higher meaning more likely fraud."""
        ...


FitFunction = Callable[..., FraudModel]


@dataclass(frozen=True)
class SklearnModel:
    """Adapter for a fitted scikit-learn classifier with ``predict_proba``."""

    estimator: Any

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability."""
        return np.asarray(self.estimator.predict_proba(X)[:, 1], dtype="float64")


def _fit_logistic_regression(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.baselines import fit_logistic_regression

    return SklearnModel(fit_logistic_regression(train, seed=seed, **(params or {})))


def _split_patience(
    params: Mapping[str, Any] | None, default: int = 50
) -> tuple[dict[str, Any], int]:
    rest = dict(params or {})
    return rest, int(rest.pop("early_stopping_rounds", default))


def _fit_xgboost(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.gbm import fit_xgboost

    params, patience = _split_patience(params)
    return fit_xgboost(train, X_validation, y_validation, params, patience, seed)


def _fit_lightgbm(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.gbm import fit_lightgbm

    params, patience = _split_patience(params)
    return fit_lightgbm(train, X_validation, y_validation, params, patience, seed)


def _fit_catboost(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.gbm import fit_catboost

    params, patience = _split_patience(params)
    return fit_catboost(train, X_validation, y_validation, params, patience, seed)


def _fit_isolation_forest(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.unsupervised import fit_isolation_forest

    params = dict(params or {})
    max_rows = params.pop("max_rows", None)
    return fit_isolation_forest(train, params, max_rows, seed)


def _fit_local_outlier_factor(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.unsupervised import (
        DEFAULT_MAX_REFERENCE_ROWS,
        fit_local_outlier_factor,
    )

    params = dict(params or {})
    max_rows = params.pop("max_rows", DEFAULT_MAX_REFERENCE_ROWS)
    return fit_local_outlier_factor(train, params, max_rows, seed)


def _fit_autoencoder(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.autoencoder import fit_autoencoder

    return fit_autoencoder(train, params, seed)


def _fit_variational_autoencoder(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.autoencoder import fit_variational_autoencoder

    return fit_variational_autoencoder(train, params, seed)


def _fit_gru(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.sequence import fit_sequence_model

    return fit_sequence_model(train, X_validation, y_validation, params, seed)


def _fit_feedforward(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.feedforward import fit_feedforward

    return fit_feedforward(train, X_validation, y_validation, params, seed)


def _fit_hybrid(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    from fraud_detection.models.hybrid import fit_hybrid

    return fit_hybrid(train, X_validation, y_validation, params, seed)


MODEL_REGISTRY: dict[str, FitFunction] = {
    "logistic_regression": _fit_logistic_regression,
    "xgboost": _fit_xgboost,
    "lightgbm": _fit_lightgbm,
    "catboost": _fit_catboost,
    "isolation_forest": _fit_isolation_forest,
    "local_outlier_factor": _fit_local_outlier_factor,
    "autoencoder": _fit_autoencoder,
    "variational_autoencoder": _fit_variational_autoencoder,
    "gru": _fit_gru,
    "feedforward": _fit_feedforward,
    "hybrid": _fit_hybrid,
}


def fit_model(
    name: str,
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> FraudModel:
    """Fit the model family registered under ``name``.

    Parameters
    ----------
    name
        Key of :data:`MODEL_REGISTRY`, typically ``config.model.name``.
    train
        Training rows after rebalancing.
    X_validation, y_validation
        Validation fold, used for early stopping where the family supports it.
    params
        Family-specific hyperparameters. The early-stopping patience of the
        boosters is passed as ``early_stopping_rounds`` here.
    seed
        Random state.

    Returns
    -------
    FraudModel
        Fitted model.
    """
    try:
        fit = MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}; expected one of: {known}.") from exc
    return fit(train, X_validation, y_validation, params, seed=seed)
