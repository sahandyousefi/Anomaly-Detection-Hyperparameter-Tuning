import numpy as np
import pytest

from fraud_detection.config import EvaluationConfig
from fraud_detection.evaluation.metrics import misclassification_cost
from fraud_detection.evaluation.thresholds import (
    apply_threshold,
    cost_optimal_threshold,
    evaluate_threshold,
    select_threshold,
    target_recall_threshold,
)

Y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])


def test_apply_threshold_flags_at_or_above() -> None:
    assert apply_threshold(S, 0.6).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert apply_threshold(S, np.inf).sum() == 0


def test_cost_optimal_threshold_by_hand() -> None:
    choice = cost_optimal_threshold(Y, S, cost_fp=1.0, cost_fn=10.0)
    assert choice.threshold == 0.6
    assert choice.cost == 1.0
    assert choice.recall == 1.0
    assert choice.precision == pytest.approx(0.75)
    assert choice.flagged == 4


def test_cost_optimal_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(7)
    y = (rng.random(400) < 0.05).astype(int)
    s = np.round(rng.random(400) + 0.4 * y, 3)
    choice = cost_optimal_threshold(y, s, cost_fp=1.0, cost_fn=20.0)
    candidates = np.r_[np.inf, np.unique(s)]
    costs = [
        misclassification_cost(y, apply_threshold(s, t), 1.0, 20.0) for t in candidates
    ]
    assert choice.cost == min(costs)


def test_cost_ties_keep_highest_threshold() -> None:
    # Flagging the single negative or missing the single positive costs the same
    choice = cost_optimal_threshold([0, 1], [0.9, 0.1], cost_fp=1.0, cost_fn=1.0)
    assert choice.threshold == np.inf
    assert choice.flagged == 0
    assert np.isnan(choice.precision)


def test_expensive_false_positives_flag_nothing() -> None:
    choice = cost_optimal_threshold([0, 1, 1], [0.9, 0.5, 0.4], 10.0, 1.0)
    assert choice.threshold == np.inf
    assert choice.recall == 0.0
    assert choice.cost == 2.0


def test_target_recall_takes_highest_feasible_threshold() -> None:
    choice = target_recall_threshold(Y, S, target_recall=2 / 3)
    assert choice.threshold == 0.8
    assert choice.recall == pytest.approx(2 / 3)
    choice = target_recall_threshold(Y, S, target_recall=0.9)
    assert choice.threshold == 0.6
    assert choice.flagged == 4


def test_target_recall_one_is_always_feasible() -> None:
    choice = target_recall_threshold([1, 0, 0, 1], [0.1, 0.5, 0.6, 0.2], 1.0)
    assert choice.threshold == 0.1
    assert choice.recall == 1.0


def test_target_recall_rejects_invalid_target() -> None:
    with pytest.raises(ValueError):
        target_recall_threshold(Y, S, target_recall=0.0)


def test_select_threshold_dispatches_on_strategy() -> None:
    by_cost = EvaluationConfig(
        threshold_strategy="cost", cost_false_positive=1.0, cost_false_negative=10.0
    )
    by_recall = EvaluationConfig(threshold_strategy="target_recall", target_recall=0.5)
    assert select_threshold(Y, S, by_cost).strategy == "cost"
    choice = select_threshold(Y, S, by_recall)
    assert choice.strategy == "target_recall"
    assert choice.threshold == 0.8


def test_validation_threshold_is_applied_unchanged_to_test() -> None:
    config = EvaluationConfig(cost_false_positive=1.0, cost_false_negative=10.0)
    choice = select_threshold(Y, S, config)
    y_test = np.array([1, 0, 0, 1])
    s_test = np.array([0.65, 0.61, 0.1, 0.55])
    report = evaluate_threshold(y_test, s_test, choice, config)
    assert report["threshold"] == 0.6
    assert (report["tp"], report["fp"], report["fn"]) == (1.0, 1.0, 1.0)
    assert report["cost"] == 11.0
