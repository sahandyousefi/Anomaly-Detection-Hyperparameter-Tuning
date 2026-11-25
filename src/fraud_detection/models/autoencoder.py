r"""Tabular autoencoders scored by reconstruction error.

Deterministic autoencoder
-------------------------
A feed-forward autoencoder :math:`g_\phi \circ f_\psi` is trained on
legitimate transactions to minimise the mean squared reconstruction error

.. math::

    \mathcal{L}_{\text{AE}}(\psi, \phi) = \frac{1}{n_0 d} \sum_{i=1}^{n_0}
        \lVert z_i - g_\phi(f_\psi(z_i)) \rVert_2^2,

where :math:`z_i \in \mathbb{R}^d` is the preprocessed feature vector and the
bottleneck :math:`f_\psi(z_i)` has dimension much smaller than :math:`d`. A
transaction the network cannot reconstruct well lies away from the manifold
of legitimate behaviour, so the per-row error
:math:`a(x) = d^{-1} \lVert z - g_\phi(f_\psi(z)) \rVert_2^2` serves as the
anomaly score.

Variational autoencoder
-----------------------
The variational variant (Kingma and Welling, 2014) has a Gaussian encoder
:math:`q_\psi(u \mid z) = \mathcal{N}(\mu_\psi(z), \operatorname{diag}
\sigma^2_\psi(z))`, a standard normal prior :math:`p(u) = \mathcal{N}(0, I)`
and a Gaussian decoder with mean :math:`g_\phi(u)` and fixed variance. Up to
constants, the per-row loss is the negative evidence lower bound

.. math::

    \ell(z) = \frac{1}{d} \Big( \mathbb{E}_{q_\psi(u \mid z)}
        \lVert z - g_\phi(u) \rVert_2^2
        + \beta\, \mathrm{KL}\big(q_\psi(u \mid z) \,\Vert\, p(u)\big) \Big),

with the expectation estimated by one reparameterised sample
:math:`u = \mu_\psi(z) + \sigma_\psi(z) \odot \varepsilon`,
:math:`\varepsilon \sim \mathcal{N}(0, I)` during training, and the closed-form
KL term :math:`\tfrac{1}{2} \sum_j (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)`.
The weight :math:`\beta` follows Higgins et al. (2017); :math:`\beta = 1`
gives the standard bound. Scoring replaces the sample by the posterior mean
:math:`u = \mu_\psi(z)`, which makes the anomaly score deterministic. The KL
term adds a penalty for rows whose code lies in a low-density region of the
prior, which the deterministic autoencoder cannot express.

Shared protocol
---------------
Preprocessing is shared with the LOF detector: median imputation with
missingness indicators, then a per-feature normal-score transform, which
puts every input on a comparable scale and bounds the influence of extreme
amounts on the loss.

Early stopping uses the most recent ``validation_fraction`` of the
legitimate training rows, kept out of gradient updates. The same rows
provide the reference distribution for the empirical-CDF score of
:class:`fraud_detection.models.unsupervised.AnomalyDetector`, so the
reference scores are out of sample. Fraud labels are only used to discard
positive training rows.

References
----------
Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M.,
Mohamed, S., and Lerchner, A. (2017). beta-VAE: Learning basic visual concepts
with a constrained variational framework. ICLR.

Hinton, G. E., and Salakhutdinov, R. R. (2006). Reducing the dimensionality of
data with neural networks. Science, 313(5786), 504-507.

Kingma, D. P., and Welling, M. (2014). Auto-encoding variational Bayes. ICLR.

Sakurada, M., and Yairi, T. (2014). Anomaly detection using autoencoders with
nonlinear dimensionality reduction. MLSDA Workshop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from torch import nn

from fraud_detection.data.resample import ResampledFold
from fraud_detection.models.neural import mlp
from fraud_detection.models.unsupervised import (
    AnomalyDetector,
    legitimate_rows,
    make_gaussianizer,
)


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
    variational
        Fit the variational autoencoder instead of the deterministic one.
    beta
        Weight of the KL term; used only when ``variational`` is true.
    learning_rate, weight_decay
        Adam step size and decoupled L2 penalty (AdamW).
    batch_size, max_epochs
        Mini-batch size and epoch limit.
    patience
        Epochs without improvement of the holdout loss before stopping.
    validation_fraction
        Share of the most recent legitimate rows held out for early stopping
        and for the reference score distribution.
    max_rows
        Optional cap on the number of legitimate training rows.
    """

    hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 8
    dropout: float = 0.0
    variational: bool = False
    beta: float = 1.0
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


class Autoencoder(nn.Module):
    """Symmetric multilayer perceptron autoencoder."""

    def __init__(self, n_features: int, params: AutoencoderParams) -> None:
        super().__init__()
        widths = [n_features, *params.hidden_dims, params.latent_dim]
        self.encoder = mlp(widths, params.dropout, activate_last=False)
        self.decoder = mlp(widths[::-1], params.dropout, activate_last=False)

    def row_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error of each row."""
        recon = self.decoder(self.encoder(batch))
        return ((batch - recon) ** 2).mean(dim=1)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Batch objective minimised during training."""
        return self.row_loss(batch).mean()


class VariationalAutoencoder(nn.Module):
    """Gaussian-encoder autoencoder trained on the (beta-weighted) ELBO."""

    def __init__(self, n_features: int, params: AutoencoderParams) -> None:
        super().__init__()
        trunk = [n_features, *params.hidden_dims]
        self.encoder = mlp(trunk, params.dropout, activate_last=True)
        self.mean = nn.Linear(trunk[-1], params.latent_dim)
        self.log_var = nn.Linear(trunk[-1], params.latent_dim)
        self.decoder = mlp(
            [params.latent_dim, *params.hidden_dims[::-1], n_features],
            params.dropout,
            activate_last=False,
        )
        self.beta = params.beta

    def encode(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and log-variance of the latent code."""
        hidden = self.encoder(batch)
        return self.mean(hidden), self.log_var(hidden)

    def _row_loss(self, batch: torch.Tensor, sample: bool) -> torch.Tensor:
        mean, log_var = self.encode(batch)
        code = mean
        if sample:
            code = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        recon = ((batch - self.decoder(code)) ** 2).sum(dim=1)
        kl = 0.5 * (mean**2 + log_var.exp() - log_var - 1.0).sum(dim=1)
        return (recon + self.beta * kl) / batch.shape[1]

    def row_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Negative ELBO per row, evaluated at the posterior mean."""
        return self._row_loss(batch, sample=False)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Single-sample Monte Carlo estimate of the mean negative ELBO."""
        return self._row_loss(batch, sample=True).mean()


def build_autoencoder(
    n_features: int, params: AutoencoderParams | None = None
) -> Autoencoder | VariationalAutoencoder:
    """Network selected by ``params.variational``.

    Parameters
    ----------
    n_features
        Input and output width :math:`d`.
    params
        Architecture settings.

    Returns
    -------
    Autoencoder or VariationalAutoencoder
        Untrained network.
    """
    params = params or AutoencoderParams()
    if params.variational:
        return VariationalAutoencoder(n_features, params)
    return Autoencoder(n_features, params)


@dataclass
class ReconstructionScorer:
    """Wraps a trained network with a scikit-learn style ``score_samples``.

    ``score_samples`` returns the negated per-row loss, matching the
    convention of scikit-learn outlier detectors (higher is more normal).
    """

    network: Autoencoder | VariationalAutoencoder
    batch_size: int = 4096
    history: list[dict[str, float]] = field(default_factory=list)

    def reconstruction_error(self, Z: np.ndarray) -> np.ndarray:
        """Per-row loss of preprocessed input (error, or negative ELBO for a VAE)."""
        self.network.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(Z), self.batch_size):
                batch = torch.as_tensor(
                    Z[start : start + self.batch_size], dtype=torch.float32
                )
                errors.append(self.network.row_loss(batch).numpy())
        return np.concatenate(errors).astype("float64")

    def score_samples(self, Z: np.ndarray) -> np.ndarray:
        """Negated per-row loss."""
        return -self.reconstruction_error(Z)


def _epoch(
    network: nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    network.train()
    order = torch.randperm(len(data), generator=generator)
    total = 0.0
    for start in range(0, len(data), batch_size):
        batch = data[order[start : start + batch_size]]
        optimizer.zero_grad()
        loss = network.training_loss(batch)
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
    """Fit the network with early stopping on a holdout set.

    Parameters
    ----------
    Z_train, Z_holdout
        Preprocessed legitimate rows for gradient updates and for stopping.
    params
        Architecture and optimisation settings.
    seed
        Seed of weight initialisation, mini-batch order, and VAE noise.

    Returns
    -------
    ReconstructionScorer
        Scorer holding the weights of the best holdout epoch and the
        per-epoch loss history.
    """
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
    """Fit preprocessing and network on the legitimate training rows.

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
        per-row loss under the holdout rows.
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


def fit_variational_autoencoder(
    train: ResampledFold,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> AnomalyDetector:
    """:func:`fit_autoencoder` with ``variational=True``."""
    return fit_autoencoder(train, {**(params or {}), "variational": True}, seed)
