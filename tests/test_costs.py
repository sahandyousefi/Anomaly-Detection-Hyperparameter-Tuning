import numpy as np
import pytest

from fraud_detection.config import EvaluationConfig
from fraud_detection.evaluation.metrics import (
    confusion_counts,
    cost_curve,
    cost_report,
    misclassification_cost,
)

Y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])


def test_confusion_counts() -> None:
    pred = S >= 0.6
    assert confusion_counts(Y, pred) == {"tp": 3, "fp": 1, "fn": 0, "tn": 4}


def test_misclassification_cost() -> None:
    pred = S >= 0.8
    assert misclassification_cost(Y, pred, cost_fp=1.0, cost_fn=10.0) == 10.0


def test_cost_curve_by_hand() -> None:
    curve = cost_curve(Y, S, cost_fp=1.0, cost_fn=10.0)
    assert curve["threshold"].iloc[0] == np.inf
    assert curve["cost"].tolist() == [30, 20, 10, 11, 1, 2, 3, 4, 5]
    assert curve["tp"].tolist() == [0, 1, 2, 2, 3, 3, 3, 3, 3]
    assert (curve[["tp", "fp", "fn", "tn"]].sum(axis=1) == len(Y)).all()
    # Flag-everything costs 5, the cheaper trivial policy
    best = curve.loc[curve["cost"].idxmin()]
    assert best["threshold"] == 0.6
    assert best["normalized_cost"] == pytest.approx(0.2)


def test_tied_scores_share_one_threshold() -> None:
    curve = cost_curve([1, 0, 1, 0], [0.5, 0.5, 0.3, 0.3], cost_fp=1, cost_fn=1)
    assert curve["threshold"].tolist() == [np.inf, 0.5, 0.3]
    assert curve["tp"].tolist() == [0, 1, 2]
    assert curve["fp"].tolist() == [0, 1, 2]


def test_cost_curve_matches_direct_evaluation() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.1).astype(int)
    s = np.round(rng.random(500) + 0.3 * y, 2)
    curve = cost_curve(y, s, cost_fp=2.0, cost_fn=25.0)
    for _, row in curve.sample(n=20, random_state=0).iterrows():
        direct = misclassification_cost(y, s >= row["threshold"], 2.0, 25.0)
        assert row["cost"] == pytest.approx(direct)


def test_cost_report_uses_config() -> None:
    config = EvaluationConfig(cost_false_positive=1.0, cost_false_negative=10.0)
    report = cost_report(Y, S >= 0.6, config)
    assert report["cost"] == 1.0
    assert report["precision"] == pytest.approx(0.75)
    assert report["recall"] == 1.0
    assert report["cost_per_transaction"] == pytest.approx(1 / 8)
    assert report["normalized_cost"] == pytest.approx(0.2)


def test_no_flags_gives_undefined_precision() -> None:
    report = cost_report(Y, np.zeros_like(Y), EvaluationConfig())
    assert np.isnan(report["precision"])
    assert report["recall"] == 0.0


def test_invalid_costs_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cost_curve(Y, S, cost_fp=0.0, cost_fn=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        misclassification_cost(Y, Y, cost_fp=-1.0, cost_fn=1.0)
