"""Chronological train, validation, and test split on the hourly step index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.config import SplitConfig

FOLDS: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class StepBoundaries:
    """Inclusive step ranges of the three folds.

    Steps strictly between ``train_end`` and ``validation_start`` (and between
    ``validation_end`` and ``test_start``) form the gap and belong to no fold.
    """

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def ranges(self) -> dict[str, tuple[int, int]]:
        """Return the inclusive ``(start, end)`` step range of each fold."""
        return {
            "train": (self.train_start, self.train_end),
            "validation": (self.validation_start, self.validation_end),
            "test": (self.test_start, self.test_end),
        }


@dataclass(frozen=True)
class TimeSplit:
    """Result of :func:`time_split`."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: StepBoundaries

    def folds(self) -> dict[str, pd.DataFrame]:
        """Return the three folds keyed by name."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


def compute_step_boundaries(
    first_step: int,
    last_step: int,
    split: SplitConfig,
    gap_steps: int = 0,
) -> StepBoundaries:
    """Cut the step range ``[first_step, last_step]`` into three folds.

    Fractions apply to the number of distinct steps in the range, not to row
    counts. PaySim activity is concentrated in the first part of the simulated
    month, so row shares per fold differ from the step fractions.

    Parameters
    ----------
    first_step, last_step
        Inclusive bounds of the observed step range.
    split
        Fold fractions.
    gap_steps
        Number of steps left out between consecutive folds, as an embargo
        against short-range dependence across the boundary.

    Returns
    -------
    StepBoundaries
        Inclusive step range of each fold.

    Raises
    ------
    ValueError
        If the range is too short to give every fold at least one step.
    """
    if gap_steps < 0:
        raise ValueError("gap_steps must be non-negative.")
    if last_step < first_step:
        raise ValueError("last_step must not precede first_step.")

    n_steps = last_step - first_step + 1
    usable = n_steps - 2 * gap_steps
    n_train = round(usable * split.train_fraction)
    n_validation = round(usable * split.validation_fraction)
    n_test = usable - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError(
            f"{n_steps} steps with gap {gap_steps} cannot give every fold a step."
        )

    train_end = first_step + n_train - 1
    validation_start = train_end + 1 + gap_steps
    validation_end = validation_start + n_validation - 1
    test_start = validation_end + 1 + gap_steps
    return StepBoundaries(
        train_start=first_step,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=last_step,
    )


def time_split(
    df: pd.DataFrame,
    split: SplitConfig,
    time_column: str = "step",
    gap_steps: int = 0,
) -> TimeSplit:
    """Split transactions chronologically into train, validation, and test.

    Parameters
    ----------
    df
        Transaction table.
    split
        Fold fractions over the step range.
    time_column
        Integer time index used for ordering.
    gap_steps
        Steps dropped between consecutive folds, see
        :func:`compute_step_boundaries`.

    Returns
    -------
    TimeSplit
        Folds as copies of the input rows, with the original index kept so rows
        can be traced back to the source table.
    """
    if df.empty:
        raise ValueError("Cannot split an empty table.")
    steps = df[time_column].to_numpy()
    bounds = compute_step_boundaries(
        int(steps.min()), int(steps.max()), split, gap_steps
    )
    parts = {
        name: df.loc[(steps >= start) & (steps <= end)].copy()
        for name, (start, end) in bounds.ranges().items()
    }
    return TimeSplit(boundaries=bounds, **parts)


def split_summary(split: TimeSplit, target: str = "isFraud") -> pd.DataFrame:
    """Tabulate size, positive count, and step range of each fold.

    Parameters
    ----------
    split
        Output of :func:`time_split`.
    target
        Binary label column.

    Returns
    -------
    pandas.DataFrame
        One row per fold with ``rows``, ``positives``, ``positive_rate``,
        ``first_step``, and ``last_step``.
    """
    records = []
    for name, fold in split.folds().items():
        start, end = split.boundaries.ranges()[name]
        n_rows = len(fold)
        positives = int(fold[target].sum())
        records.append(
            {
                "fold": name,
                "rows": n_rows,
                "positives": positives,
                "positive_rate": positives / n_rows if n_rows else np.nan,
                "first_step": start,
                "last_step": end,
            }
        )
    return pd.DataFrame.from_records(records).set_index("fold")
