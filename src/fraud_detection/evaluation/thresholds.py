r"""Decision threshold selection on the validation fold.

The threshold is a model parameter like any other and is chosen on the
validation fold only. The test fold is then scored once at the chosen value,
so the reported cost and recall are not optimistically biased by the choice.

Two strategies are supported.

``cost``
    Minimize the validation misclassification cost

    .. math::

        t^\ast = \arg\min_t \; c_{FN} \, FN(t) + c_{FP} \, FP(t),

    searched over every distinct score (plus :math:`t = \infty`, flag nothing).
    Among equal-cost thresholds the highest is kept, since it raises fewer
    alerts.

``target_recall``
    Take the highest threshold whose validation recall is at least
    :math:`r^\ast`. Lowering the threshold can only add false positives, so
    this is the threshold with the fewest false positives that meets the
    recall requirement.

Both strategies assume that the score distribution and the prevalence on the
test period resemble those on the validation period. Drift between the two
periods shifts the realized recall and cost away from their validation values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud_detection.config import EvaluationConfig
from fraud_detection.evaluation.metrics import cost_curve, cost_report


@dataclass(frozen=True)
class ThresholdChoice:
    """A decision threshold and its operating point on the fold it was fitted on.

    ``threshold`` is ``inf`` when flagging nothing is optimal. ``precision`` is
    NaN when no row is flagged.
    """

    strategy: str
    threshold: float
    precision: float
    recall: float
    cost: float
    normalized_cost: float
    flagged: int


def apply_threshold(y_score: ArrayLike, threshold: float) -> np.ndarray:
    """Binary decisions, 1 where the score is at or above ``threshold``."""
    return (np.asarray(y_score, dtype="float64") >= threshold).astype("int8")


def _choice_from_row(row: pd.Series, strategy: str) -> ThresholdChoice:
    flagged = int(row["tp"] + row["fp"])
    positives = row["tp"] + row["fn"]
    return ThresholdChoice(
        strategy=strategy,
        threshold=float(row["threshold"]),
        precision=row["tp"] / flagged if flagged else np.nan,
        recall=float(row["tp"] / positives),
        cost=float(row["cost"]),
        normalized_cost=float(row["normalized_cost"]),
        flagged=flagged,
    )


def cost_optimal_threshold(
    y_true: ArrayLike, y_score: ArrayLike, cost_fp: float, cost_fn: float
) -> ThresholdChoice:
    """Threshold with the lowest misclassification cost.

    Parameters
    ----------
    y_true
        Binary validation labels.
    y_score
        Validation scores, higher meaning more likely fraud.
    cost_fp, cost_fn
        Unit costs of a false positive and a false negative.

    Returns
    -------
    ThresholdChoice
        The cheapest threshold; the highest one among ties.
    """
    curve = cost_curve(y_true, y_score, cost_fp, cost_fn)
    # The curve is ordered by decreasing threshold, so argmin keeps the highest
    best = int(np.argmin(curve["cost"].to_numpy()))
    return _choice_from_row(curve.iloc[best], "cost")


def target_recall_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    target_recall: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> ThresholdChoice:
    """Highest threshold whose recall reaches ``target_recall``.

    Parameters
    ----------
    y_true
        Binary validation labels.
    y_score
        Validation scores, higher meaning more likely fraud.
    target_recall
        Required recall in ``(0, 1]``.
    cost_fp, cost_fn
        Unit costs, used only to report the cost of the chosen threshold.

    Returns
    -------
    ThresholdChoice
        The feasible threshold with the fewest flagged rows.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    curve = cost_curve(y_true, y_score, cost_fp, cost_fn)
    recall = curve["tp"].to_numpy() / (curve["tp"].iloc[0] + curve["fn"].iloc[0])
    # Recall is non-decreasing down the curve and reaches 1 on the last row;
    # the tolerance absorbs rounding in tp / positives
    first = int(np.argmax(recall >= target_recall - 1e-12))
    return _choice_from_row(curve.iloc[first], "target_recall")


def select_threshold(
    y_true: ArrayLike, y_score: ArrayLike, config: EvaluationConfig
) -> ThresholdChoice:
    """Choose the threshold on validation data with the configured strategy.

    Parameters
    ----------
    y_true
        Binary validation labels.
    y_score
        Validation scores.
    config
        Strategy, unit costs, and target recall.

    Returns
    -------
    ThresholdChoice
        Chosen threshold and its validation operating point.
    """
    c_fp, c_fn = config.cost_false_positive, config.cost_false_negative
    if config.threshold_strategy == "cost":
        return cost_optimal_threshold(y_true, y_score, c_fp, c_fn)
    return target_recall_threshold(y_true, y_score, config.target_recall, c_fp, c_fn)


def evaluate_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    choice: ThresholdChoice,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Cost report of a fixed threshold on a held-out fold.

    Parameters
    ----------
    y_true
        Binary labels of the held-out fold.
    y_score
        Scores of the held-out fold.
    choice
        Threshold fitted on the validation fold.
    config
        Source of the unit costs.

    Returns
    -------
    dict
        :func:`~fraud_detection.evaluation.metrics.cost_report` output plus
        the ``threshold`` used.
    """
    report = cost_report(y_true, apply_threshold(y_score, choice.threshold), config)
    return {"threshold": choice.threshold, **report}
