"""Reference models that every learned fraud model must beat.

Two rule baselines work on the raw transaction table.

* :func:`flagged_fraud_scores` returns the simulator's own ``isFlaggedFraud``
  output. It is the alerting rule that ships with PaySim and is kept out of
  the feature matrix for that reason.
* :func:`large_transfer_scores` applies the rule as stated in the dataset
  description: a transfer above 200,000 is flagged. In the published log
  ``isFlaggedFraud`` fires on far fewer rows than this rule would, so both
  versions are reported.

Both return binary scores. Their PR-AUC is the area of a single operating
point, which makes them directly comparable to the learned models.

The learned baseline is an L2-regularized logistic regression on the
engineered features. Missing point-in-time aggregates (the first transaction
of an account) are imputed with the training median, and a missingness
indicator is appended so the model can still tell "no history" apart from
an average history.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data.resample import ResampledFold
from fraud_detection.data.schema import FLAGGED_TRANSFER_THRESHOLD


def flagged_fraud_scores(df: pd.DataFrame) -> np.ndarray:
    """Scores of the simulator's built-in rule, read from ``isFlaggedFraud``.

    Parameters
    ----------
    df
        Raw transactions that still hold the ``isFlaggedFraud`` column.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of 0 and 1.
    """
    if "isFlaggedFraud" not in df.columns:
        raise KeyError("isFlaggedFraud is required for the simulator rule baseline.")
    return df["isFlaggedFraud"].to_numpy(dtype="float64")


def large_transfer_scores(
    df: pd.DataFrame, threshold: float = FLAGGED_TRANSFER_THRESHOLD
) -> np.ndarray:
    """Scores of the documented rule: a transfer whose amount exceeds ``threshold``.

    Parameters
    ----------
    df
        Raw transactions with ``type`` and ``amount``.
    threshold
        Amount above which a transfer is flagged.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of 0 and 1.
    """
    is_transfer = df["type"].astype(str).to_numpy() == "TRANSFER"
    is_large = df["amount"].to_numpy(dtype="float64") > threshold
    return (is_transfer & is_large).astype("float64")


def make_logistic_regression(
    C: float = 1.0, max_iter: int = 2000, seed: int = 0
) -> Pipeline:
    """Unfitted imputation, scaling, and logistic regression pipeline.

    Parameters
    ----------
    C
        Inverse L2 regularization strength.
    max_iter
        Iteration limit of the solver.
    seed
        Random state of the estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``impute``, ``scale``, and ``model``.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)),
        ]
    )


def fit_logistic_regression(
    train: ResampledFold, seed: int = 0, **params: Any
) -> Pipeline:
    """Fit the logistic regression baseline on a (possibly rebalanced) fold.

    Parameters
    ----------
    train
        Training rows. With the ``class_weight`` resampling method the positive
        weight is passed to the solver as sample weights.
    seed
        Random state of the estimator.
    **params
        Forwarded to :func:`make_logistic_regression`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    pipeline = make_logistic_regression(seed=seed, **params)
    pipeline.fit(train.X, train.y, model__sample_weight=train.sample_weight)
    return pipeline


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability of a fitted classifier."""
    return model.predict_proba(X)[:, 1]
