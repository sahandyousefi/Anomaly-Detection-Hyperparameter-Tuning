r"""Probability calibration and its assessment.

A score :math:`s(x)` is calibrated when :math:`P(Y = 1 \mid s(x) = p) = p`.
Ranking metrics such as PR-AUC are invariant to monotone transformations of
the score, so a model can rank well and still output probabilities that are
far from the observed fraud rate. Calibration matters whenever the score is
read as a probability: expected-loss decisions, alert prioritization across
models, and monitoring of the predicted fraud rate.

Rebalancing the training fold shifts the base rate the model learns. If
positives are given an effective weight :math:`w` relative to negatives
(class weighting with :math:`w = w_{pos}`, or undersampling that keeps a
share :math:`\beta` of negatives, :math:`w = 1 / \beta`), the predicted odds
are inflated by :math:`w`, and the prior-shift correction

.. math::

    p = \frac{p_w}{p_w + w \, (1 - p_w)}

recovers probabilities at the original prevalence (Dal Pozzolo et al., 2015).
This holds when rebalancing changes the class prior only, which is exact for
random undersampling and approximate for SMOTE.

Two post-hoc calibrators are fitted on held-out scores:

``platt``
    Logistic regression on the logit of the score (Platt, 1999),
    :math:`p = \sigma(a \, \mathrm{logit}(s) + b)`. Two parameters, monotone,
    and stable with few positives.
``isotonic``
    Non-decreasing step function fitted by the pool-adjacent-violators
    algorithm (Zadrozny and Elkan, 2002). More flexible, but it needs more
    positives and turns ranges of scores into ties.

Calibration is assessed with the Brier score (Brier, 1950),
:math:`\mathrm{BS} = n^{-1} \sum_i (p_i - y_i)^2`, its skill relative to
predicting the prevalence :math:`\pi` for every row,
:math:`\mathrm{BSS} = 1 - \mathrm{BS} / (\pi (1 - \pi))`, and the expected
calibration error over bins :math:`b`,
:math:`\mathrm{ECE} = \sum_b (n_b / n) \, |\bar{y}_b - \bar{p}_b|`. With
fraud prevalence near 0.1 %, equal-width bins place almost every row in the
first bin, so bins of equal count are the default.

The calibrator is fitted on the validation fold and assessed on the test
fold. Platt scaling is strictly monotone, so the cost-optimal threshold on
validation selects the same rows before and after calibration; isotonic
scaling can merge neighbouring scores and move the optimum slightly.

References
----------
Brier, G. W. (1950). Verification of forecasts expressed in terms of
probability. Monthly Weather Review, 78(1), 1-3.

Dal Pozzolo, A., Caelen, O., Johnson, R. A., and Bontempi, G. (2015).
Calibrating probability with undersampling for unbalanced classification.
IEEE Symposium Series on Computational Intelligence.

Niculescu-Mizil, A., and Caruana, R. (2005). Predicting good probabilities
with supervised learning. ICML.

Platt, J. (1999). Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods. Advances in Large Margin
Classifiers.

Zadrozny, B., and Elkan, C. (2002). Transforming classifier scores into
accurate multiclass probability estimates. KDD.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from fraud_detection.models.base import FraudModel

CalibrationMethod = Literal["platt", "isotonic"]
BinStrategy = Literal["quantile", "uniform"]

# Keeps logit finite for scores of exactly 0 or 1
_EPS = 1e-12


def _validate(y_true: ArrayLike, y_prob: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError(
            "y_true and y_prob must be non-empty 1-D arrays of equal size."
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1.")
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must lie in [0, 1].")
    return y.astype("int8"), p


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.log(q) - np.log1p(-q)


def prior_shift_correction(y_prob: ArrayLike, positive_weight: float) -> np.ndarray:
    """Undo the base-rate shift caused by rebalancing the training fold.

    Parameters
    ----------
    y_prob
        Probabilities from a model trained on rebalanced data.
    positive_weight
        Effective weight of a positive relative to a negative during
        training: ``scale_pos_weight`` for class weighting, or the inverse of
        the share of negatives kept by undersampling.

    Returns
    -------
    numpy.ndarray
        Probabilities whose odds are divided by ``positive_weight``.
    """
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive.")
    p = np.asarray(y_prob, dtype="float64")
    return p / (p + positive_weight * (1 - p))


def undersampling_weight(n_pos: int, n_neg: int, sampling_strategy: float) -> float:
    r"""Effective positive weight of undersampling to a target ratio.

    Undersampling to :math:`n_{pos} / n_{neg}' = r` keeps a share
    :math:`\beta = n_{pos} / (r \, n_{neg})` of the negatives, which weights
    positives by :math:`1 / \beta` relative to the original data.

    Parameters
    ----------
    n_pos, n_neg
        Class counts of the training fold before undersampling.
    sampling_strategy
        Target ratio :math:`r` after undersampling.

    Returns
    -------
    float
        :math:`1 / \beta`, for use with :func:`prior_shift_correction`.
    """
    if min(n_pos, n_neg) < 1 or sampling_strategy <= 0:
        raise ValueError("Class counts and sampling_strategy must be positive.")
    return sampling_strategy * n_neg / n_pos


@dataclass(frozen=True)
class Calibrator:
    """A fitted map from raw scores to calibrated probabilities."""

    method: CalibrationMethod
    model: Any

    def transform(self, y_score: ArrayLike) -> np.ndarray:
        """Calibrated probabilities for ``y_score``."""
        s = np.asarray(y_score, dtype="float64")
        if self.method == "platt":
            z = _logit(s).reshape(-1, 1)
            return np.asarray(self.model.predict_proba(z)[:, 1], dtype="float64")
        return np.asarray(self.model.predict(s), dtype="float64")


def fit_calibrator(
    y_true: ArrayLike, y_score: ArrayLike, method: CalibrationMethod = "platt"
) -> Calibrator:
    """Fit a calibrator on held-out scores.

    Parameters
    ----------
    y_true
        Binary labels of the calibration fold.
    y_score
        Model scores in ``[0, 1]`` on the same rows.
    method
        ``platt`` or ``isotonic``.

    Returns
    -------
    Calibrator
        Fitted calibrator.
    """
    y, s = _validate(y_true, y_score)
    if y.min() == y.max():
        raise ValueError("The calibration fold must contain both classes.")
    if method == "platt":
        # Weak regularization; the fit has only two parameters
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(_logit(s).reshape(-1, 1), y)
    elif method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(s, y)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}.")
    return Calibrator(method=method, model=model)


@dataclass(frozen=True)
class CalibratedModel:
    """A fraud model whose scores pass through a fitted calibrator."""

    base: FraudModel
    calibrator: Calibrator

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of fraud."""
        return self.calibrator.transform(self.base.predict_scores(X))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared difference between probabilities and outcomes."""
    y, p = _validate(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def _bin_edges(p: np.ndarray, n_bins: int, strategy: BinStrategy) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # Heavy ties at small scores collapse quantiles into one edge
        return np.unique(edges)
    raise ValueError(f"Unknown bin strategy: {strategy!r}.")


def reliability_table(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.DataFrame:
    """Observed fraud rate against mean predicted probability per bin.

    Parameters
    ----------
    y_true
        Binary labels.
    y_prob
        Predicted probabilities.
    n_bins
        Requested number of bins. Quantile bins can be fewer when many scores
        are tied.
    strategy
        ``quantile`` for bins of equal count, ``uniform`` for equal width.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin with ``lower``, ``upper``, ``count``,
        ``positives``, ``mean_predicted``, and ``observed_rate``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    y, p = _validate(y_true, y_prob)
    edges = _bin_edges(p, n_bins, strategy)
    # Right-closed last bin, so a score equal to the top edge is counted
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    frame = pd.DataFrame({"bin": index, "y": y, "p": p})
    grouped = frame.groupby("bin")
    table = pd.DataFrame(
        {
            "count": grouped.size(),
            "positives": grouped["y"].sum(),
            "mean_predicted": grouped["p"].mean(),
            "observed_rate": grouped["y"].mean(),
        }
    )
    table.insert(0, "lower", edges[table.index])
    table.insert(1, "upper", edges[table.index + 1])
    return table.reset_index(drop=True)


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> float:
    """Count-weighted mean absolute gap between observed and predicted rates."""
    table = reliability_table(y_true, y_prob, n_bins, strategy)
    gap = (table["observed_rate"] - table["mean_predicted"]).abs()
    return float((gap * table["count"]).sum() / table["count"].sum())


def calibration_report(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> dict[str, float]:
    """Calibration summary of one set of probabilities.

    Parameters
    ----------
    y_true
        Binary labels.
    y_prob
        Predicted probabilities.
    n_bins, strategy
        Binning for the expected calibration error.

    Returns
    -------
    dict
        ``brier``, ``brier_reference`` (constant prediction at the observed
        prevalence), ``brier_skill``, ``log_loss``, ``ece``,
        ``mean_predicted``, and ``prevalence``.
    """
    y, p = _validate(y_true, y_prob)
    prevalence = float(y.mean())
    reference = prevalence * (1 - prevalence)
    brier = brier_score(y, p)
    return {
        "brier": brier,
        "brier_reference": reference,
        "brier_skill": 1 - brier / reference if reference > 0 else np.nan,
        "log_loss": float(log_loss(y, np.clip(p, _EPS, 1 - _EPS), labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins, strategy),
        "mean_predicted": float(p.mean()),
        "prevalence": prevalence,
    }


def plot_reliability(
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]],
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
    log_scale: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Reliability diagram for one or more sets of probabilities.

    Parameters
    ----------
    curves
        Label mapped to ``(y_true, y_prob)``.
    n_bins, strategy
        Binning passed to :func:`reliability_table`.
    log_scale
        Plot both axes on a log scale, which spreads out the low-probability
        region where most transactions lie.
    ax
        Axes to draw on. A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the diagram.
    """
    if ax is None:
        fig = Figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    lowest = 1.0
    for label, (y_true, y_prob) in curves.items():
        table = reliability_table(y_true, y_prob, n_bins, strategy)
        brier = brier_score(y_true, y_prob)
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            label=f"{label} (Brier {brier:.4g})",
        )
        positive = table[["mean_predicted", "observed_rate"]].to_numpy()
        positive = positive[positive > 0]
        if positive.size:
            lowest = min(lowest, float(positive.min()))
    ax.plot([lowest, 1.0], [lowest, 1.0], linestyle="--", color="grey", lw=1)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=lowest)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left", fontsize="small")
    return fig
