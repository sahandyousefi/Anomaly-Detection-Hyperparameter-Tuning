import numpy as np
import pytest

from fraud_detection.evaluation.metrics import (
    OperatingPoint,
    pr_auc,
    precision_at_recall,
    ranking_report,
    recall_at_precision,
    roc_auc,
)

# Ranked list: hits at ranks 1, 2, and 4 out of 8
Y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])


def test_pr_auc_is_average_precision() -> None:
    assert pr_auc(Y, S) == pytest.approx((1 + 1 + 0.75) / 3)


def test_roc_auc_counts_ordered_pairs() -> None:
    assert roc_auc(Y, S) == pytest.approx(14 / 15)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (0.75, OperatingPoint(0.6, 0.75, 1.0)),
        (0.8, OperatingPoint(0.8, 1.0, 2 / 3)),
        (1.0, OperatingPoint(0.8, 1.0, 2 / 3)),
    ],
)
def test_recall_at_precision(target: float, expected: OperatingPoint) -> None:
    got = recall_at_precision(Y, S, target)
    assert got.threshold == expected.threshold
    assert got.precision == pytest.approx(expected.precision)
    assert got.recall == pytest.approx(expected.recall)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (0.5, OperatingPoint(0.8, 1.0, 2 / 3)),
        (0.7, OperatingPoint(0.6, 0.75, 1.0)),
        (1.0, OperatingPoint(0.6, 0.75, 1.0)),
    ],
)
def test_precision_at_recall(target: float, expected: OperatingPoint) -> None:
    got = precision_at_recall(Y, S, target)
    assert got.threshold == expected.threshold
    assert got.precision == pytest.approx(expected.precision)
    assert got.recall == pytest.approx(expected.recall)


def test_unreachable_precision_returns_empty_point() -> None:
    got = recall_at_precision([0, 1], [0.9, 0.1], 0.9)
    assert got.threshold == np.inf
    assert np.isnan(got.precision)
    assert got.recall == 0.0


def test_flagging_at_threshold_reproduces_point() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    s = rng.normal(size=2_000) + 1.5 * y
    point = precision_at_recall(y, s, 0.8)
    flagged = s >= point.threshold
    assert y[flagged].mean() == pytest.approx(point.precision)
    assert flagged[y == 1].mean() == pytest.approx(point.recall)
    assert point.recall >= 0.8


def test_random_scores_give_prevalence_pr_auc() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(200_000) < 0.01).astype(int)
    s = rng.random(200_000)
    assert pr_auc(y, s) == pytest.approx(y.mean(), abs=2e-3)
    assert roc_auc(y, s) == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    ("y", "s", "match"),
    [
        ([0, 0, 0], [0.1, 0.2, 0.3], "both classes"),
        ([0, 2], [0.1, 0.2], "binary"),
        ([0, 1], [0.1], "equal length"),
        ([0, 1], [0.1, np.nan], "finite"),
    ],
)
def test_invalid_inputs_rejected(y: list, s: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pr_auc(y, s)


def test_invalid_targets_rejected() -> None:
    with pytest.raises(ValueError, match="min_precision"):
        recall_at_precision(Y, S, 0.0)
    with pytest.raises(ValueError, match="min_recall"):
        precision_at_recall(Y, S, 1.5)


def test_ranking_report_keys_and_values() -> None:
    report = ranking_report(Y, S, precision_targets=(0.8,), recall_targets=(0.5,))
    assert report["positives"] == 3
    assert report["prevalence"] == pytest.approx(3 / 8)
    assert report["pr_auc"] == pytest.approx(pr_auc(Y, S))
    assert report["recall@p0.8"] == pytest.approx(2 / 3)
    assert report["threshold@p0.8"] == 0.8
    assert report["precision@r0.5"] == 1.0
    assert report["threshold@r0.5"] == 0.8
