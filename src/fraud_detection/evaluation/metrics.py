r"""Threshold-free and fixed-operating-point metrics for imbalanced scoring.

With fraud prevalence near 0.1 %, ROC-AUC is dominated by the large pool of
easy negatives, so PR-AUC is the primary ranking metric. A random scorer has
an expected PR-AUC equal to the prevalence :math:`\pi`, which is reported next
to it as the reference value.

PR-AUC is estimated by average precision,

.. math::

    \mathrm{AP} = \sum_k (R_k - R_{k-1}) \, P_k,

the step-wise sum over score thresholds. It is preferred to the trapezoidal
area, which interpolates linearly in precision-recall space and overstates the
area (Davis and Goadrich, 2006).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Precision and recall when scores at or above ``threshold`` are flagged.

    ``threshold`` is ``inf`` and ``precision`` is NaN when no threshold meets
    the requested constraint.
    """

    threshold: float
    precision: float
    recall: float


def _validate(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype="float64")
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("y_true and y_score must be 1-D arrays of equal length.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1.")
    if y.min() == y.max():
        raise ValueError("y_true must contain both classes.")
    if not np.isfinite(s).all():
        raise ValueError("y_score must be finite.")
    return y.astype("int8"), s


def _curve(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, ...]:
    precision, recall, thresholds = precision_recall_curve(y, s)
    # The last curve point (precision 1, recall 0) has no threshold
    return precision[:-1], recall[:-1], thresholds


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve, estimated by average precision."""
    y, s = _validate(y_true, y_score)
    return float(average_precision_score(y, s))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve."""
    y, s = _validate(y_true, y_score)
    return float(roc_auc_score(y, s))


def recall_at_precision(
    y_true: ArrayLike, y_score: ArrayLike, min_precision: float
) -> OperatingPoint:
    """Highest recall among thresholds whose precision is at least the target.

    Parameters
    ----------
    y_true
        Binary labels.
    y_score
        Scores, higher meaning more likely fraud.
    min_precision
        Required precision in ``(0, 1]``.

    Returns
    -------
    OperatingPoint
        The feasible threshold with the highest recall. Ties are broken by
        higher precision, then by higher threshold.
    """
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    precision, recall, thresholds = _curve(*_validate(y_true, y_score))
    feasible = np.flatnonzero(precision >= min_precision)
    if feasible.size == 0:
        return OperatingPoint(threshold=np.inf, precision=np.nan, recall=0.0)
    order = np.lexsort((thresholds[feasible], precision[feasible], recall[feasible]))
    best = feasible[order[-1]]
    return OperatingPoint(
        float(thresholds[best]), float(precision[best]), float(recall[best])
    )


def precision_at_recall(
    y_true: ArrayLike, y_score: ArrayLike, min_recall: float
) -> OperatingPoint:
    """Highest precision among thresholds whose recall is at least the target.

    Parameters
    ----------
    y_true
        Binary labels.
    y_score
        Scores, higher meaning more likely fraud.
    min_recall
        Required recall in ``(0, 1]``.

    Returns
    -------
    OperatingPoint
        The feasible threshold with the highest precision. Ties are broken by
        higher recall, then by higher threshold.
    """
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1].")
    precision, recall, thresholds = _curve(*_validate(y_true, y_score))
    # The lowest threshold flags everything, so recall 1 is always reachable
    feasible = np.flatnonzero(recall >= min_recall)
    order = np.lexsort((thresholds[feasible], recall[feasible], precision[feasible]))
    best = feasible[order[-1]]
    return OperatingPoint(
        float(thresholds[best]), float(precision[best]), float(recall[best])
    )


def ranking_report(
    y_true: ArrayLike,
    y_score: ArrayLike,
    precision_targets: tuple[float, ...] = (0.5, 0.8, 0.9),
    recall_targets: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, float]:
    """Collect the ranking metrics of one scored fold in a flat mapping.

    Parameters
    ----------
    y_true
        Binary labels.
    y_score
        Scores, higher meaning more likely fraud.
    precision_targets
        Precision levels for :func:`recall_at_precision`.
    recall_targets
        Recall levels for :func:`precision_at_recall`.

    Returns
    -------
    dict
        ``n``, ``positives``, ``prevalence``, ``pr_auc``, ``roc_auc``, and one
        ``recall@p<level>`` and ``precision@r<level>`` entry per target, plus
        the matching ``threshold@...`` entries.
    """
    y, s = _validate(y_true, y_score)
    report: dict[str, float] = {
        "n": float(len(y)),
        "positives": float(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
    }
    for level in precision_targets:
        point = asdict(recall_at_precision(y, s, level))
        report[f"recall@p{level:g}"] = point["recall"]
        report[f"threshold@p{level:g}"] = point["threshold"]
    for level in recall_targets:
        point = asdict(precision_at_recall(y, s, level))
        report[f"precision@r{level:g}"] = point["precision"]
        report[f"threshold@r{level:g}"] = point["threshold"]
    return report
