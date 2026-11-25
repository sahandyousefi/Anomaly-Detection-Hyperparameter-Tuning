r"""GRU classifier over each account's recent transaction history.

For every account role :math:`r` (originator, destination) the history columns
of :mod:`fraud_detection.features.sequence` and the current transaction form a
chronological sequence

.. math::

    S_r(t) = \big(e_{t_{n_r}}, \dots, e_{t_1}, e_t\big), \qquad n_r \le L,

where :math:`e` holds the element features and :math:`n_r` is the number of
earlier transactions available. A gated recurrent unit (Cho et al., 2014) per
role reads the sequence, and its final hidden state :math:`h_r` summarises the
account's behaviour up to and including the transaction being scored. The
remaining row-level and aggregate features :math:`z_t` go through a small
multilayer perceptron, and the logit is

.. math::

    f(t) = \operatorname{MLP}\big([\,h_{orig};\ h_{dest};\
           \operatorname{MLP}_z(z_t)\,]\big).

Sequences are packed by their true length, so missing history elements do not
enter the recurrence. Element features are standardised with moments of the
training rows, with the element age entering as :math:`\log(1 + \Delta\tau)`.
The static features use the median-imputation and normal-score transform of
the unsupervised detectors.

Training and early stopping follow :mod:`fraud_detection.models.neural`.

References
----------
Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using RNN
encoder-decoder for statistical machine translation. EMNLP.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from fraud_detection.data.resample import ResampledFold
from fraud_detection.features.build import ACCOUNT_ROLES
from fraud_detection.features.sequence import (
    HISTORY_ELEMENT_FEATURES,
    HISTORY_SOURCE_FEATURES,
    history_column,
)
from fraud_detection.models.neural import (
    Inputs,
    NeuralClassifier,
    TrainingSettings,
    mlp,
    params_from_mapping,
    train_classifier,
)
from fraud_detection.models.unsupervised import make_gaussianizer

_HISTORY_PATTERN = re.compile(rf"^({'|'.join(ACCOUNT_ROLES)})_hist(\d+)_")
_AGE_INDEX = HISTORY_ELEMENT_FEATURES.index("steps_before")


@dataclass(frozen=True)
class SequenceParams:
    """Architecture and optimisation settings of the GRU classifier.

    Attributes
    ----------
    hidden_size, num_layers
        Width and depth of each role's GRU.
    static_dims
        Hidden widths of the MLP on the non-sequence features.
    head_dims
        Hidden widths of the output MLP.
    dropout
        Dropout after each hidden MLP activation, and between GRU layers.
    learning_rate, weight_decay, batch_size, max_epochs, patience
        See :class:`fraud_detection.models.neural.TrainingSettings`.
    """

    hidden_size: int = 32
    num_layers: int = 1
    static_dims: tuple[int, ...] = (64,)
    head_dims: tuple[int, ...] = (32,)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5

    def training(self) -> TrainingSettings:
        """Optimisation subset of the parameters."""
        names = {f.name for f in fields(TrainingSettings)}
        return TrainingSettings(**{n: getattr(self, n) for n in names})


def history_layout(columns: Sequence[str]) -> tuple[list[str], int]:
    """Roles and history length present in a feature list.

    Raises
    ------
    ValueError
        If no history columns are present or a role lacks some of them.
    """
    found: dict[str, int] = {}
    for column in columns:
        match = _HISTORY_PATTERN.match(column)
        if match:
            role, lag = match.group(1), int(match.group(2))
            found[role] = max(found.get(role, 0), lag)
    if not found:
        raise ValueError(
            "The GRU model needs history columns; set features.history_length > 0."
        )
    roles = [r for r in ACCOUNT_ROLES if r in found]
    length = max(found.values())
    present = set(columns)
    for role in roles:
        expected = [
            history_column(role, lag, f)
            for lag in range(1, length + 1)
            for f in HISTORY_ELEMENT_FEATURES
        ]
        missing = [c for c in expected if c not in present]
        if missing:
            raise ValueError(f"Incomplete history columns, missing {missing[:3]}.")
    return roles, length


@dataclass
class SequenceEncoder:
    """Turns a feature frame into static and per-role sequence tensors."""

    roles: list[str]
    length: int
    static_columns: list[str]
    static_transform: Pipeline
    element_mean: np.ndarray
    element_std: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame, seed: int = 0) -> SequenceEncoder:
        """Fit the static transform and element moments on training rows."""
        missing = [c for c in HISTORY_SOURCE_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"The current transaction needs {missing}.")
        roles, length = history_layout(list(X.columns))
        static = [c for c in X.columns if not _HISTORY_PATTERN.match(c)]
        transform = make_gaussianizer(len(X), seed).fit(X[static])
        elements = np.concatenate(
            [_raw_sequences(X, role, length)[0] for role in roles], axis=1
        ).reshape(-1, len(HISTORY_ELEMENT_FEATURES))
        mean = np.nanmean(elements, axis=0)
        std = np.nanstd(elements, axis=0)
        return cls(roles, length, static, transform, mean, np.where(std > 0, std, 1.0))

    @property
    def n_static(self) -> int:
        """Width of the static input, including missingness indicators."""
        return len(self.static_transform.get_feature_names_out())

    def __call__(self, X: pd.DataFrame) -> Inputs:
        static = self.static_transform.transform(X[self.static_columns])
        tensors = [torch.as_tensor(static, dtype=torch.float32)]
        for role in self.roles:
            values, lengths = _raw_sequences(X, role, self.length)
            values = np.nan_to_num((values - self.element_mean) / self.element_std)
            tensors.append(torch.as_tensor(values, dtype=torch.float32))
            tensors.append(torch.as_tensor(lengths, dtype=torch.int64))
        return tuple(tensors)


def _raw_sequences(
    X: pd.DataFrame, role: str, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Left-aligned sequences, oldest element first, and their lengths.

    Returns an array of shape ``(n, length + 1, n_features)``; positions at or
    beyond a row's length hold NaN or stale values and are ignored by packing.
    """
    n_rows, n_features = len(X), len(HISTORY_ELEMENT_FEATURES)
    # Right-aligned first: lag L at position 0, current transaction last
    padded = np.empty((n_rows, length + 1, n_features))
    for position, lag in enumerate(range(length, 0, -1)):
        names = [history_column(role, lag, f) for f in HISTORY_ELEMENT_FEATURES]
        padded[:, position] = X[names].to_numpy(dtype="float64")
    current = X[list(HISTORY_SOURCE_FEATURES)].to_numpy(dtype="float64")
    padded[:, length] = np.column_stack([current, np.zeros(n_rows)])
    padded[..., _AGE_INDEX] = np.log1p(padded[..., _AGE_INDEX])

    lengths = 1 + (~np.isnan(padded[:, :length, _AGE_INDEX])).sum(axis=1)
    shift = length + 1 - lengths
    index = np.minimum(np.arange(length + 1)[None, :] + shift[:, None], length)
    aligned = np.take_along_axis(padded, index[:, :, None], axis=1)
    return aligned, lengths


class SequenceClassifier(nn.Module):
    """Per-role GRUs and a static MLP joined by an output MLP."""

    def __init__(self, n_static: int, roles: Sequence[str], params: SequenceParams):
        super().__init__()
        n_element = len(HISTORY_ELEMENT_FEATURES)
        self.grus = nn.ModuleDict(
            {
                role: nn.GRU(
                    n_element,
                    params.hidden_size,
                    num_layers=params.num_layers,
                    batch_first=True,
                    dropout=params.dropout if params.num_layers > 1 else 0.0,
                )
                for role in roles
            }
        )
        self.static = mlp([n_static, *params.static_dims], params.dropout, True)
        width = len(roles) * params.hidden_size + params.static_dims[-1]
        self.head = mlp([width, *params.head_dims, 1], params.dropout, False)

    def forward(self, static: torch.Tensor, *sequences: torch.Tensor) -> torch.Tensor:
        """Logit per row; ``sequences`` alternates values and lengths per role."""
        parts = [self.static(static)]
        for role, values, lengths in zip(
            self.grus, sequences[::2], sequences[1::2], strict=True
        ):
            packed = pack_padded_sequence(
                values, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.grus[role](packed)
            parts.append(hidden[-1])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def fit_sequence_model(
    train: ResampledFold,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    params: Mapping[str, Any] | SequenceParams | None = None,
    seed: int = 0,
) -> NeuralClassifier:
    """Fit the GRU classifier with early stopping on the validation fold.

    Parameters
    ----------
    train
        Training fold whose features include the history columns.
    X_validation, y_validation
        Validation fold at the observed class ratio.
    params
        :class:`SequenceParams` or a mapping of its fields.
    seed
        Seed of preprocessing, initialisation, and batch order.

    Returns
    -------
    NeuralClassifier
        Fitted model with ``predict_scores``.
    """
    params = params_from_mapping(SequenceParams, params)
    if not params.static_dims:
        raise ValueError("static_dims must have at least one layer.")
    torch.manual_seed(seed)
    encoder = SequenceEncoder.fit(train.X, seed)
    network = SequenceClassifier(encoder.n_static, encoder.roles, params)
    history = train_classifier(
        network,
        encoder(train.X),
        train.y.to_numpy(),
        train.sample_weight,
        encoder(X_validation),
        y_validation.to_numpy(),
        params.training(),
        seed=seed,
    )
    return NeuralClassifier(encoder, network, history)
