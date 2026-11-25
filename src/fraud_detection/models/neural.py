r"""Shared training code for the supervised neural network classifiers.

A network maps preprocessed inputs to a logit :math:`f_\theta(x)`, and the
fraud score is :math:`\sigma(f_\theta(x))`. Training minimises the weighted
binary cross-entropy

.. math::

    \mathcal{L}(\theta) = \frac{\sum_i w_i \, \ell\big(y_i, f_\theta(x_i)\big)}
                               {\sum_i w_i},

where :math:`w_i` is the positive class weight of the training fold for
positives and 1 otherwise. Training stops when the average precision (PR-AUC)
on the validation fold has not improved for ``patience`` epochs, and the
weights of the best epoch are restored. The validation fold keeps the observed
class ratio, so the stopping criterion refers to the deployment prevalence.

Scores are not calibrated probabilities when the training fold was rebalanced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from itertools import pairwise
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

Inputs = tuple[torch.Tensor, ...]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

ParamsT = TypeVar("ParamsT")


def params_from_mapping(
    cls: type[ParamsT], params: Mapping[str, Any] | ParamsT | None
) -> ParamsT:
    """Build a frozen parameter dataclass, rejecting unknown keys.

    List values are converted to tuples so the result stays hashable.
    """
    if isinstance(params, cls):
        return params
    values = dict(params or {})  # type: ignore[call-overload]
    unknown = set(values) - {f.name for f in fields(cls)}  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}.")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**values)


def mlp(widths: list[int], dropout: float, activate_last: bool) -> nn.Sequential:
    """Linear layers of the given widths with GELU activations and dropout.

    The last layer has no activation unless ``activate_last`` is true.
    """
    layers: list[nn.Module] = []
    n_layers = len(widths) - 1
    for i, (w_in, w_out) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(w_in, w_out))
        if i < n_layers - 1 or activate_last:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row binary cross-entropy on logits."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


@dataclass(frozen=True)
class TrainingSettings:
    """Optimisation settings shared by the neural classifiers."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5


def predict_logits(
    network: nn.Module, inputs: Inputs, batch_size: int = 4096
) -> np.ndarray:
    """Logits of every row, computed in evaluation mode without gradients."""
    network.eval()
    n_rows = len(inputs[0])
    out = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            batch = tuple(t[start : start + batch_size] for t in inputs)
            out.append(network(*batch).numpy())
    return np.concatenate(out).astype("float64") if out else np.empty(0)


def _epoch(
    network: nn.Module,
    inputs: Inputs,
    targets: torch.Tensor,
    weights: torch.Tensor,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    network.train()
    order = torch.randperm(len(targets), generator=generator)
    total = 0.0
    for start in range(0, len(targets), batch_size):
        rows = order[start : start + batch_size]
        optimizer.zero_grad()
        per_row = loss_fn(network(*(t[rows] for t in inputs)), targets[rows])
        loss = (weights[rows] * per_row).sum() / weights[rows].sum()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(rows)
    return total / len(targets)


def train_classifier(
    network: nn.Module,
    train_inputs: Inputs,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    validation_inputs: Inputs,
    y_validation: np.ndarray,
    settings: TrainingSettings,
    loss_fn: LossFunction = binary_cross_entropy,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit ``network`` in place with early stopping on validation PR-AUC.

    Parameters
    ----------
    network
        Module whose ``forward`` takes the tensors of ``train_inputs`` and
        returns one logit per row.
    train_inputs, validation_inputs
        Preprocessed input tensors, aligned on the first dimension.
    y_train, y_validation
        Binary labels.
    sample_weight
        Per-row training weights.
    settings
        Optimiser and stopping settings.
    loss_fn
        Per-row loss on logits and float targets.
    seed
        Seed of the mini-batch order.

    Returns
    -------
    list of dict
        Per-epoch training loss and validation average precision.
    """
    y_validation = np.asarray(y_validation)
    if y_validation.sum() == 0:
        raise ValueError("Early stopping needs positives in the validation fold.")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    targets = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    weights = torch.tensor(np.asarray(sample_weight), dtype=torch.float32)

    history: list[dict[str, float]] = []
    best_score, best_state, stale = -np.inf, None, 0
    for epoch in range(settings.max_epochs):
        train_loss = _epoch(
            network,
            train_inputs,
            targets,
            weights,
            loss_fn,
            optimizer,
            settings.batch_size,
            generator,
        )
        logits = predict_logits(network, validation_inputs)
        score = float(average_precision_score(y_validation, logits))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_pr_auc": score}
        )
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return history


@dataclass
class NeuralClassifier:
    """Fitted preprocessing and network behind the ``FraudModel`` interface.

    Attributes
    ----------
    encode
        Maps a feature frame to the network's input tensors.
    network
        Trained module returning logits.
    history
        Per-epoch training record.
    """

    encode: Callable[[pd.DataFrame], Inputs]
    network: nn.Module
    history: list[dict[str, float]] = field(default_factory=list)

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Sigmoid of the logit."""
        return expit(predict_logits(self.network, self.encode(X)))
