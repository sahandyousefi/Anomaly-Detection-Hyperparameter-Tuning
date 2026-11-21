"""Class rebalancing applied to the training fold only.

Resampling runs after feature construction and after the time split. The
validation and test folds keep the observed class ratio, so that precision,
PR-AUC, and cost estimates refer to the prevalence the model will face.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraud_detection.config import ResamplingConfig
from fraud_detection.data.split import TimeSplit


@dataclass(frozen=True)
class ResampledFold:
    """Training rows after rebalancing.

    Attributes
    ----------
    X, y
        Features and binary labels. Undersampling keeps the original index of
        the retained rows. SMOTE returns a fresh ``RangeIndex`` because the
        synthetic rows have no source row.
    scale_pos_weight
        Weight of a positive relative to a negative example. It differs from 1
        only for the ``class_weight`` method, where the rows are unchanged.
    """

    X: pd.DataFrame
    y: pd.Series
    scale_pos_weight: float = 1.0

    @property
    def sample_weight(self) -> np.ndarray:
        """Per-row weights implied by :attr:`scale_pos_weight`."""
        return np.where(self.y.to_numpy() == 1, self.scale_pos_weight, 1.0)


@dataclass(frozen=True)
class ModelingFolds:
    """Model inputs of the three folds, with resampling applied to train only."""

    train: ResampledFold
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _class_counts(y: pd.Series) -> tuple[int, int]:
    values = y.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Labels must be binary 0/1.")
    n_pos = int(values.sum())
    n_neg = len(values) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("The training fold must contain both classes.")
    return n_neg, n_pos


def _undersample(
    X: pd.DataFrame, y: pd.Series, ratio: float, seed: int
) -> ResampledFold:
    sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    sampler.fit_resample(X, y)
    # Restore chronological order so downstream time-aware code is not surprised
    keep = np.sort(sampler.sample_indices_)
    return ResampledFold(X=X.iloc[keep], y=y.iloc[keep])


def _smote(X: pd.DataFrame, y: pd.Series, ratio: float, seed: int) -> ResampledFold:
    missing = X.columns[X.isna().any()].tolist()
    if missing:
        raise ValueError(
            f"SMOTE needs complete features; impute these columns first: {missing}."
        )
    sampler = SMOTE(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    return ResampledFold(X=X_res.reset_index(drop=True), y=y_res.reset_index(drop=True))


def resample_training_fold(
    X: pd.DataFrame,
    y: pd.Series,
    config: ResamplingConfig,
    seed: int = 0,
) -> ResampledFold:
    r"""Rebalance the training fold according to ``config``.

    ``config.sampling_strategy`` is the target ratio
    :math:`n_{pos} / n_{neg}` after rebalancing. For ``class_weight`` the rows
    are kept and positives receive the weight that gives the same effective
    ratio, :math:`w_{pos} = r \, n_{neg} / n_{pos}`.

    Parameters
    ----------
    X
        Training features, already constructed on the full chronological log.
    y
        Binary training labels aligned with ``X``.
    config
        Resampling method and target ratio.
    seed
        Random seed for the samplers.

    Returns
    -------
    ResampledFold
        Rebalanced rows and the positive class weight.

    Raises
    ------
    ValueError
        If the labels are not binary, a class is absent, or SMOTE receives
        missing values.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index.")
    n_neg, n_pos = _class_counts(y)
    ratio = config.sampling_strategy

    if config.method == "none":
        return ResampledFold(X=X, y=y)
    if config.method == "class_weight":
        return ResampledFold(X=X, y=y, scale_pos_weight=ratio * n_neg / n_pos)
    if n_pos / n_neg >= ratio:
        raise ValueError(
            f"Observed ratio {n_pos / n_neg:.4f} already meets the target {ratio}."
        )
    if config.method == "undersample":
        return _undersample(X, y, ratio, seed)
    return _smote(X, y, ratio, seed)


def prepare_modeling_folds(
    split: TimeSplit,
    features: Sequence[str],
    target: str,
    config: ResamplingConfig,
    seed: int = 0,
) -> ModelingFolds:
    """Select model inputs from each fold and rebalance the training fold.

    Parameters
    ----------
    split
        Chronological folds of a table that already holds the feature columns.
    features
        Model input columns.
    target
        Binary label column.
    config
        Resampling settings for the training fold.
    seed
        Random seed for the samplers.

    Returns
    -------
    ModelingFolds
        Rebalanced training data and untouched validation and test data.
    """
    columns = list(features)
    train = resample_training_fold(
        split.train[columns], split.train[target], config, seed
    )
    return ModelingFolds(
        train=train,
        X_validation=split.validation[columns],
        y_validation=split.validation[target],
        X_test=split.test[columns],
        y_test=split.test[target],
    )
