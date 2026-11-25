r"""Gradient boosting with the autoencoder reconstruction error as a feature.

The hybrid model appends the reconstruction error :math:`a(x)` of an
autoencoder fitted on legitimate transactions (see
:mod:`fraud_detection.models.autoencoder`) to the tabular features and trains
a supervised booster on the augmented table. The error summarises how far a
transaction lies from the legitimate manifold in all features jointly, which
axis-aligned trees approximate only with many splits.

Cross-fitting
-------------
An autoencoder reconstructs its own training rows better than unseen rows. If
the booster were trained on in-sample errors, legitimate training rows would
carry lower errors than legitimate validation and test rows, while fraud rows
(never seen by the autoencoder) would not, and the booster would learn an
inflated contrast. The training rows are therefore split into :math:`K`
contiguous chronological blocks :math:`B_1, \dots, B_K`, and the error of a row
in :math:`B_k` comes from an autoencoder fitted on the legitimate rows outside
:math:`B_k`. Validation and test rows are scored by one autoencoder fitted on
all legitimate training rows. Every error the booster sees is thus out of
sample for the network that produced it. :math:`K = 1` disables cross-fitting
and is kept only to measure the effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.data.resample import ResampledFold
from fraud_detection.models.autoencoder import fit_autoencoder
from fraud_detection.models.base import FraudModel, fit_model
from fraud_detection.models.unsupervised import AnomalyDetector

RECONSTRUCTION_FEATURE = "ae_reconstruction_error"


@dataclass(frozen=True)
class HybridParams:
    """Settings of the hybrid model.

    Attributes
    ----------
    booster
        Registry name of the supervised model trained on the augmented table.
    booster_params
        Hyperparameters of the booster, including ``early_stopping_rounds``.
    autoencoder
        Fields of :class:`fraud_detection.models.autoencoder.AutoencoderParams`.
    cross_fit_folds
        Number of chronological blocks :math:`K` for the training errors.
    """

    booster: str = "xgboost"
    booster_params: Mapping[str, Any] | None = None
    autoencoder: Mapping[str, Any] | None = None
    cross_fit_folds: int = 3

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None) -> HybridParams:
        """Build from a configuration mapping, rejecting unknown keys."""
        values = dict(params or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown hybrid parameters: {sorted(unknown)}.")
        result = cls(**values)
        if result.cross_fit_folds < 1:
            raise ValueError("cross_fit_folds must be at least 1.")
        if result.booster == "hybrid":
            raise ValueError("The booster of a hybrid model cannot be a hybrid.")
        return result


def augment(X: pd.DataFrame, errors: np.ndarray) -> pd.DataFrame:
    """Copy of ``X`` with the reconstruction error appended."""
    if RECONSTRUCTION_FEATURE in X.columns:
        raise ValueError(f"{RECONSTRUCTION_FEATURE!r} is already a column.")
    return X.assign(**{RECONSTRUCTION_FEATURE: errors})


def chronological_blocks(n_rows: int, n_blocks: int) -> np.ndarray:
    """Block label of each row for ``n_blocks`` contiguous, near-equal blocks."""
    if n_blocks > n_rows:
        raise ValueError("More cross-fitting blocks than training rows.")
    edges = np.linspace(0, n_rows, n_blocks + 1).round().astype(int)
    return np.repeat(np.arange(n_blocks), np.diff(edges))


def out_of_fold_errors(
    train: ResampledFold,
    n_blocks: int,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Reconstruction error of each training row from a network not fitted on it.

    Parameters
    ----------
    train
        Training fold in chronological order.
    n_blocks
        Number of blocks :math:`K`. With one block the errors are in-sample.
    params
        Autoencoder settings.
    seed
        Random state of every autoencoder.

    Returns
    -------
    numpy.ndarray
        One error per training row.
    """
    if n_blocks == 1:
        return fit_autoencoder(train, params, seed).anomaly_scores(train.X)
    blocks = chronological_blocks(len(train.X), n_blocks)
    errors = np.empty(len(train.X))
    for block in range(n_blocks):
        held_out = blocks == block
        rest = ResampledFold(X=train.X.iloc[~held_out], y=train.y.iloc[~held_out])
        detector = fit_autoencoder(rest, params, seed)
        errors[held_out] = detector.anomaly_scores(train.X.iloc[held_out])
    return errors


@dataclass(frozen=True)
class HybridModel:
    """Autoencoder feeding its reconstruction error to a fitted booster."""

    autoencoder: AnomalyDetector
    booster: FraudModel

    def augment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Feature table with the reconstruction error appended."""
        return augment(X, self.autoencoder.anomaly_scores(X))

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Booster score on the augmented features."""
        return self.booster.predict_scores(self.augment(X))


def fit_hybrid(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> HybridModel:
    """Fit the cross-fitted autoencoders and the booster on augmented features.

    Parameters
    ----------
    train
        Training fold in chronological order, after rebalancing.
    X_validation, y_validation
        Validation fold, used for the booster's early stopping.
    params
        Fields of :class:`HybridParams`.
    seed
        Random state of the autoencoders and the booster.

    Returns
    -------
    HybridModel
        Fitted model with ``predict_scores``.
    """
    settings = HybridParams.from_mapping(params)
    train_errors = out_of_fold_errors(
        train, settings.cross_fit_folds, settings.autoencoder, seed
    )
    final = fit_autoencoder(train, settings.autoencoder, seed)
    augmented = ResampledFold(
        X=augment(train.X, train_errors),
        y=train.y,
        scale_pos_weight=train.scale_pos_weight,
    )
    booster = fit_model(
        settings.booster,
        augmented,
        augment(X_validation, final.anomaly_scores(X_validation)),
        y_validation,
        settings.booster_params,
        seed=seed,
    )
    return HybridModel(final, booster)
