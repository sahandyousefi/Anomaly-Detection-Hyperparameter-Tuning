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

Decisions are evaluated by misclassification cost. With unit costs
:math:`c_{FP}` for reviewing a legitimate transaction and :math:`c_{FN}` for a
missed fraud, flagging every score at or above :math:`t` costs

.. math::

    C(t) = c_{FN} \, FN(t) + c_{FP} \, FP(t).

It is normalized by the cheaper of the two trivial policies, flag nothing
(:math:`c_{FN} P`) and flag everything (:math:`c_{FP} N`), so a value below 1
means the model beats both. Costs are taken as constant per error; a missed
fraud of a large amount costs the same as a small one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_detection.config import EvaluationConfig


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


def _check_costs(cost_fp: float, cost_fn: float) -> None:
    if cost_fp < 0 or cost_fn < 0 or cost_fp + cost_fn == 0:
        raise ValueError("Costs must be non-negative and not both zero.")


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """True and false positive and negative counts of binary decisions."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return {
        "tp": int((y & p).sum()),
        "fp": int((~y & p).sum()),
        "fn": int((y & ~p).sum()),
        "tn": int((~y & ~p).sum()),
    }


def misclassification_cost(
    y_true: ArrayLike, y_pred: ArrayLike, cost_fp: float, cost_fn: float
) -> float:
    """Total cost :math:`c_{FN} FN + c_{FP} FP` of binary decisions."""
    _check_costs(cost_fp, cost_fn)
    counts = confusion_counts(y_true, y_pred)
    return cost_fn * counts["fn"] + cost_fp * counts["fp"]


def cost_curve(
    y_true: ArrayLike, y_score: ArrayLike, cost_fp: float, cost_fn: float
) -> pd.DataFrame:
    """Cost of every distinct decision threshold in one pass over sorted scores.

    Parameters
    ----------
    y_true
        Binary labels.
    y_score
        Scores, higher meaning more likely fraud. A row is flagged when its
        score is at or above the threshold.
    cost_fp, cost_fn
        Unit costs of a false positive and a false negative.

    Returns
    -------
    pandas.DataFrame
        One row per threshold in decreasing order, starting with ``inf`` (flag
        nothing). Columns ``threshold``, ``tp``, ``fp``, ``fn``, ``tn``,
        ``cost``, ``cost_per_transaction``, and ``normalized_cost``.
    """
    _check_costs(cost_fp, cost_fn)
    y, s = _validate(y_true, y_score)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Last position of each run of tied scores, so ties are flagged together
    last_of_run = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]
    tp = np.r_[0, np.cumsum(y_sorted)[last_of_run]]
    fp = np.r_[0, last_of_run + 1 - tp[1:]]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    fn = n_pos - tp
    cost = cost_fn * fn + cost_fp * fp
    baseline = min(cost_fn * n_pos, cost_fp * n_neg)
    return pd.DataFrame(
        {
            "threshold": np.r_[np.inf, s_sorted[last_of_run]],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": n_neg - fp,
            "cost": cost,
            "cost_per_transaction": cost / len(y),
            "normalized_cost": cost / baseline if baseline > 0 else np.nan,
        }
    )


def cost_report(
    y_true: ArrayLike, y_pred: ArrayLike, config: EvaluationConfig
) -> dict[str, float]:
    """Confusion counts and costs of binary decisions under configured costs.

    Parameters
    ----------
    y_true
        Binary labels.
    y_pred
        Binary decisions, 1 meaning flagged.
    config
        Source of ``cost_false_positive`` and ``cost_false_negative``.

    Returns
    -------
    dict
        ``tp``, ``fp``, ``fn``, ``tn``, ``precision``, ``recall``, ``cost``,
        ``cost_per_transaction``, and ``normalized_cost`` (relative to the
        cheaper trivial policy).
    """
    c_fp, c_fn = config.cost_false_positive, config.cost_false_negative
    counts = confusion_counts(y_true, y_pred)
    n_pos = counts["tp"] + counts["fn"]
    n_neg = counts["fp"] + counts["tn"]
    flagged = counts["tp"] + counts["fp"]
    cost = misclassification_cost(y_true, y_pred, c_fp, c_fn)
    baseline = min(c_fn * n_pos, c_fp * n_neg)
    return {
        **{k: float(v) for k, v in counts.items()},
        "precision": counts["tp"] / flagged if flagged else np.nan,
        "recall": counts["tp"] / n_pos if n_pos else np.nan,
        "cost": float(cost),
        "cost_per_transaction": cost / (n_pos + n_neg),
        "normalized_cost": cost / baseline if baseline > 0 else np.nan,
    }
