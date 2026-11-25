"""Feed-forward network for supervised comparison with the boosted trees.

The network is a multilayer perceptron on the same tabular features as the
gradient boosting models, preprocessed by median imputation with missingness
indicators and a per-feature normal-score transform. It is trained with the
focal loss by default, or with the weighted cross-entropy for an ablation of
the loss; see :mod:`fraud_detection.models.neural` for both objectives and the
early-stopping protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal

import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from fraud_detection.data.resample import ResampledFold
from fraud_detection.models.neural import (
    Inputs,
    LossFunction,
    NeuralClassifier,
    TrainingSettings,
    binary_cross_entropy,
    focal_loss,
    mlp,
    params_from_mapping,
    train_classifier,
)
from fraud_detection.models.unsupervised import make_gaussianizer

LossName = Literal["focal", "cross_entropy"]


@dataclass(frozen=True)
class FeedForwardParams:
    """Architecture, loss, and optimisation settings.

    Attributes
    ----------
    hidden_dims
        Widths of the hidden layers.
    dropout
        Dropout probability after each hidden activation.
    loss
        ``"focal"`` or ``"cross_entropy"``.
    gamma, alpha
        Focal loss parameters; ignored for the cross-entropy.
    learning_rate, weight_decay, batch_size, max_epochs, patience
        See :class:`fraud_detection.models.neural.TrainingSettings`.
    """

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    loss: LossName = "focal"
    gamma: float = 2.0
    alpha: float | None = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5

    def training(self) -> TrainingSettings:
        """Optimisation subset of the parameters."""
        names = {f.name for f in fields(TrainingSettings)}
        return TrainingSettings(**{n: getattr(self, n) for n in names})

    def loss_function(self) -> LossFunction:
        """Per-row loss selected by :attr:`loss`."""
        if self.loss == "focal":
            return partial(focal_loss, gamma=self.gamma, alpha=self.alpha)
        if self.loss == "cross_entropy":
            return binary_cross_entropy
        raise ValueError(f"Unknown loss {self.loss!r}.")


@dataclass
class TabularEncoder:
    """Fitted normal-score preprocessing producing a single input tensor."""

    transform: Pipeline

    @classmethod
    def fit(cls, X: pd.DataFrame, seed: int = 0) -> TabularEncoder:
        """Fit imputation and the quantile transform on training rows."""
        return cls(make_gaussianizer(len(X), seed).fit(X))

    @property
    def n_features(self) -> int:
        """Input width, including missingness indicators."""
        return len(self.transform.get_feature_names_out())

    def __call__(self, X: pd.DataFrame) -> Inputs:
        return (torch.as_tensor(self.transform.transform(X), dtype=torch.float32),)


class FeedForwardClassifier(nn.Module):
    """Multilayer perceptron returning one logit per row."""

    def __init__(self, n_features: int, params: FeedForwardParams) -> None:
        super().__init__()
        self.layers = mlp(
            [n_features, *params.hidden_dims, 1], params.dropout, activate_last=False
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Logit per row."""
        return self.layers(X).squeeze(1)


def fit_feedforward(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | FeedForwardParams | None = None,
    seed: int = 0,
) -> NeuralClassifier:
    """Fit the feed-forward classifier with early stopping on validation PR-AUC.

    Parameters
    ----------
    train
        Training fold after rebalancing; class weights enter the loss.
    X_validation, y_validation
        Validation fold at the observed class ratio.
    params
        :class:`FeedForwardParams` or a mapping of its fields.
    seed
        Seed of preprocessing, initialisation, and batch order.

    Returns
    -------
    NeuralClassifier
        Fitted model with ``predict_scores``.
    """
    params = params_from_mapping(FeedForwardParams, params)
    loss_fn = params.loss_function()
    torch.manual_seed(seed)
    encoder = TabularEncoder.fit(train.X, seed)
    network = FeedForwardClassifier(encoder.n_features, params)
    history = train_classifier(
        network,
        encoder(train.X),
        train.y.to_numpy(),
        train.sample_weight,
        encoder(X_validation),
        y_validation.to_numpy(),
        params.training(),
        loss_fn=loss_fn,
        seed=seed,
    )
    return NeuralClassifier(encoder, network, history)
