r"""Tabular autoencoder scored by reconstruction error.

A feed-forward autoencoder :math:`g_\phi \circ f_\psi` is trained on
legitimate transactions to minimise the mean squared reconstruction error

.. math::

    \mathcal{L}(\psi, \phi) = \frac{1}{n_0 d} \sum_{i=1}^{n_0}
        \lVert z_i - g_\phi(f_\psi(z_i)) \rVert_2^2,

where :math:`z_i \in \mathbb{R}^d` is the preprocessed feature vector and the
bottleneck :math:`f_\psi(z_i)` has dimension much smaller than :math:`d`. A
transaction the network cannot reconstruct well lies away from the manifold
of legitimate behaviour, so the per-row error

.. math::

    a(x) = \frac{1}{d} \lVert z - g_\phi(f_\psi(z)) \rVert_2^2

serves as the anomaly score.

Preprocessing is shared with the LOF detector: median imputation with
missingness indicators, then a per-feature normal-score transform, which
puts every input on a comparable scale and bounds the influence of extreme
amounts on the loss.

Early stopping uses the most recent ``validation_fraction`` of the
legitimate training rows, kept out of gradient updates. The same rows
provide the reference distribution for the empirical-CDF score of
:class:`fraud_detection.models.unsupervised.AnomalyDetector`, so the
reference errors are out of sample. Fraud labels are only used to discard
positive training rows.

References
----------
Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the dimensionality of
data with neural networks. Science, 313(5786), 504-507.

Sakurada, M., and Yairi, T. (2014). Anomaly detection using autoencoders with
nonlinear dimensionality reduction. MLSDA Workshop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from fraud_detection.data.resample import ResampledFold
from fraud_detection.models.unsupervised import (
    AnomalyDetector,
    legitimate_rows,
    make_gaussianizer,
)

if TYPE_CHECKING:
    import torch
    from torch import nn


@dataclass(frozen=True)
class AutoencoderParams:
    """Architecture and optimisation settings.

    Attributes
    ----------
    hidden_dims
        Widths of the encoder's hidden layers; the decoder mirrors them.
    latent_dim
        Width of the bottleneck.
    dropout
        Dropout probability after each hidden activation.
    learning_rate, weight_decay
        Adam step size and decoupled L2 penalty (AdamW).
    batch_size, max_epochs
        Mini-batch size and epoch limit.
    patience
        Epochs without improvement of the holdout loss before stopping.
    validation_fraction
        Share of the most recent legitimate rows held out for early stopping
        and for the reference error distribution.
    max_rows
        Optional cap on the number of legitimate training rows.
    """

    hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 8
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 5
    validation_fraction: float = 0.1
    max_rows: int | None = None

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None) -> AutoencoderParams:
        """Build from a configuration mapping, rejecting unknown keys."""
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"Unknown autoencoder parameters: {sorted(unknown)}.")
        if "hidden_dims" in params:
            params["hidden_dims"] = tuple(params["hidden_dims"])
        return cls(**params)


def build_autoencoder(
    n_features: int, params: AutoencoderParams | None = None
) -> nn.Module:
    """Symmetric multilayer perceptron autoencoder.

    Parameters
    ----------
    n_features
        Input and output width :math:`d`.
    params
        Architecture settings.

    Returns
    -------
    torch.nn.Module
        Module with ``encoder`` and ``decoder`` submodules.
    """
    from torch import nn

    params = params or AutoencoderParams()

    def stack(widths: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i, (w_in, w_out) in enumerate(pairwise(widths)):
            layers.append(nn.Linear(w_in, w_out))
            if i < len(widths) - 2:
                layers.append(nn.GELU())
                if params.dropout > 0:
                    layers.append(nn.Dropout(params.dropout))
        return nn.Sequential(*layers)

    widths = [n_features, *params.hidden_dims, params.latent_dim]
    model = nn.Module()
    model.encoder = stack(widths)
    model.decoder = stack(widths[::-1])
    return model


@dataclass
class ReconstructionScorer:
    """Wraps a trained autoencoder with a scikit-learn style ``score_samples``.

    ``score_samples`` returns the negated reconstruction error, matching the
    convention of scikit-learn outlier detectors (higher is more normal).
    """

    network: Any
    batch_size: int = 4096
    history: list[dict[str, float]] = field(default_factory=list)

    def reconstruction_error(self, Z: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per row of preprocessed input."""
        import torch

        self.network.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(Z), self.batch_size):
                batch = torch.as_tensor(
                    Z[start : start + self.batch_size], dtype=torch.float32
                )
                recon = self.network.decoder(self.network.encoder(batch))
                errors.append(((batch - recon) ** 2).mean(dim=1).numpy())
        return np.concatenate(errors).astype("float64")

    def score_samples(self, Z: np.ndarray) -> np.ndarray:
        """Negated reconstruction error."""
        return -self.reconstruction_error(Z)


def _epoch(
    network: nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    import torch

    network.train()
    order = torch.randperm(len(data), generator=generator)
    total = 0.0
    for start in range(0, len(data), batch_size):
        batch = data[order[start : start + batch_size]]
        optimizer.zero_grad()
        loss = torch.mean((network.decoder(network.encoder(batch)) - batch) ** 2)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(data)


def train_autoencoder(
    Z_train: np.ndarray,
    Z_holdout: np.ndarray,
    params: AutoencoderParams | None = None,
    seed: int = 0,
) -> ReconstructionScorer:
    """Fit the autoencoder with early stopping on a holdout set.

    Parameters
    ----------
    Z_train, Z_holdout
        Preprocessed legitimate rows for gradient updates and for stopping.
    params
        Architecture and optimisation settings.
    seed
        Seed of weight initialisation and mini-batch order.

    Returns
    -------
    ReconstructionScorer
        Scorer holding the weights of the best holdout epoch and the
        per-epoch loss history.
    """
    import torch

    params = params or AutoencoderParams()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    network = build_autoencoder(Z_train.shape[1], params)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    train = torch.as_tensor(Z_train, dtype=torch.float32)
    scorer = ReconstructionScorer(network)

    best_loss, best_state, stale = np.inf, None, 0
    for epoch in range(params.max_epochs):
        train_loss = _epoch(network, train, optimizer, params.batch_size, generator)
        holdout_loss = float(scorer.reconstruction_error(Z_holdout).mean())
        scorer.history.append(
            {"epoch": epoch, "train_loss": train_loss, "holdout_loss": holdout_loss}
        )
        if holdout_loss < best_loss - 1e-7:
            best_loss, stale = holdout_loss, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= params.patience:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return scorer


def fit_autoencoder(
    train: ResampledFold,
    params: Mapping[str, Any] | AutoencoderParams | None = None,
    seed: int = 0,
) -> AnomalyDetector:
    """Fit preprocessing and autoencoder on the legitimate training rows.

    Parameters
    ----------
    train
        Training fold in chronological order; positive rows are discarded.
    params
        :class:`AutoencoderParams` or a mapping of its fields.
    seed
        Random state of the subsample, preprocessing, and network.

    Returns
    -------
    AnomalyDetector
        Detector whose ``predict_scores`` is the empirical CDF of the
        reconstruction error under the holdout rows.
    """
    if not isinstance(params, AutoencoderParams):
        params = AutoencoderParams.from_mapping(params)
    if not 0 < params.validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1).")
    X = legitimate_rows(train, params.max_rows, seed)
    n_holdout = max(1, round(params.validation_fraction * len(X)))
    if n_holdout >= len(X):
        raise ValueError("Too few legitimate rows to hold out a stopping set.")
    X_fit, X_holdout = X.iloc[:-n_holdout], X.iloc[-n_holdout:]

    preprocess = make_gaussianizer(len(X_fit), seed).fit(X_fit)
    Z_holdout = preprocess.transform(X_holdout)
    scorer = train_autoencoder(preprocess.transform(X_fit), Z_holdout, params, seed)
    reference = np.sort(scorer.reconstruction_error(Z_holdout))
    return AnomalyDetector(preprocess, scorer, reference)
