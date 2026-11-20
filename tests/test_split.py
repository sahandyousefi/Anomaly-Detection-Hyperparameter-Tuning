import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.data.split import (
    compute_step_boundaries,
    split_summary,
    time_split,
)
from fraud_detection.data.synthetic import generate_paysim_like

SPLIT = SplitConfig(train_fraction=0.7, validation_fraction=0.15, test_fraction=0.15)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return generate_paysim_like(n_rows=5_000, fraud_rate=0.01, n_steps=200, seed=11)


def test_boundaries_follow_step_fractions() -> None:
    bounds = compute_step_boundaries(1, 100, SPLIT)
    assert bounds.ranges() == {
        "train": (1, 70),
        "validation": (71, 85),
        "test": (86, 100),
    }


def test_boundaries_with_gap() -> None:
    bounds = compute_step_boundaries(1, 104, SPLIT, gap_steps=2)
    assert bounds.validation_start == bounds.train_end + 3
    assert bounds.test_start == bounds.validation_end + 3
    assert bounds.test_end == 104


@pytest.mark.parametrize(("last", "gap"), [(2, 0), (10, 4)])
def test_boundaries_reject_short_ranges(last: int, gap: int) -> None:
    with pytest.raises(ValueError, match="cannot give every fold"):
        compute_step_boundaries(1, last, SPLIT, gap_steps=gap)


def test_boundaries_reject_negative_gap() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_step_boundaries(1, 100, SPLIT, gap_steps=-1)


def test_folds_are_ordered_and_disjoint(frame: pd.DataFrame) -> None:
    result = time_split(frame, SPLIT)
    assert result.train["step"].max() < result.validation["step"].min()
    assert result.validation["step"].max() < result.test["step"].min()

    indices = [set(fold.index) for fold in result.folds().values()]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert sum(len(i) for i in indices) == len(frame)


def test_gap_rows_are_dropped(frame: pd.DataFrame) -> None:
    result = time_split(frame, SPLIT, gap_steps=3)
    bounds = result.boundaries
    kept = pd.concat(result.folds().values())
    gap_steps = set(range(bounds.train_end + 1, bounds.validation_start)) | set(
        range(bounds.validation_end + 1, bounds.test_start)
    )
    assert len(gap_steps) == 6
    assert not set(kept["step"]) & gap_steps
    assert len(kept) == (~frame["step"].isin(gap_steps)).sum()


def test_fraud_event_stays_in_one_fold(frame: pd.DataFrame) -> None:
    # Transfer and cash-out of an event share a step, so they cannot be separated
    result = time_split(frame, SPLIT)
    for fold in result.folds().values():
        fraud = fold[fold["isFraud"] == 1]
        assert (fraud["type"] == "TRANSFER").sum() == (
            fraud["type"] == "CASH_OUT"
        ).sum()


def test_split_respects_custom_time_column(frame: pd.DataFrame) -> None:
    shifted = frame.rename(columns={"step": "hour"}).assign(
        hour=lambda d: d["hour"] + 500
    )
    result = time_split(shifted, SPLIT, time_column="hour")
    assert result.boundaries.train_start == shifted["hour"].min()


def test_split_rejects_empty_table(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        time_split(frame.iloc[:0], SPLIT)


def test_split_summary(frame: pd.DataFrame) -> None:
    result = time_split(frame, SPLIT)
    summary = split_summary(result)
    assert list(summary.index) == ["train", "validation", "test"]
    assert summary["rows"].sum() == len(frame)
    assert summary["positives"].sum() == frame["isFraud"].sum()
    np.testing.assert_allclose(
        summary["positive_rate"], summary["positives"] / summary["rows"]
    )
