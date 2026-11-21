"""Leakage checks on the full feature, split, and resampling pipeline."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns

SPLIT = SplitConfig(0.6, 0.2, 0.2)
FEATURES = feature_columns()


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    # A small account pool so most accounts appear in more than one fold
    return generate_paysim_like(
        n_rows=3_000, fraud_rate=0.02, n_steps=90, n_customers=150, seed=21
    )


@pytest.fixture(scope="module")
def table(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw)


@pytest.mark.parametrize("gap_steps", [0, 3])
def test_folds_are_disjoint_and_ordered(table: pd.DataFrame, gap_steps: int) -> None:
    split = time_split(table, SPLIT, gap_steps=gap_steps)
    train, val, test = split.train, split.validation, split.test
    assert train.index.intersection(val.index).empty
    assert train.index.intersection(test.index).empty
    assert val.index.intersection(test.index).empty
    assert train["step"].max() + gap_steps < val["step"].min()
    assert val["step"].max() + gap_steps < test["step"].min()


@pytest.mark.parametrize("method", ["none", "undersample", "class_weight"])
def test_training_rows_never_come_from_later_folds(
    table: pd.DataFrame, method: str
) -> None:
    split = time_split(table, SPLIT)
    folds = prepare_modeling_folds(
        split, FEATURES, "isFraud", ResamplingConfig(method, 0.2), seed=0
    )
    later = split.validation.index.union(split.test.index)
    assert not folds.train.X.index.isin(later).any()
    assert folds.train.X.index.isin(split.train.index).all()


def test_shared_accounts_exist_across_folds(table: pd.DataFrame) -> None:
    # Guards the tests below against passing trivially on disjoint accounts
    split = time_split(table, SPLIT)
    shared = set(split.train["nameOrig"]) & set(split.test["nameOrig"])
    assert len(shared) > 50


@pytest.mark.parametrize("fold", ["train", "validation"])
def test_features_ignore_rows_after_the_fold(raw: pd.DataFrame, fold: str) -> None:
    full = time_split(build_features(raw), SPLIT)
    end = full.boundaries.ranges()[fold][1]
    truncated = build_features(raw[raw["step"] <= end])
    rows = full.folds()[fold].index
    pd.testing.assert_frame_equal(
        full.folds()[fold][FEATURES], truncated.loc[rows, FEATURES]
    )


def test_features_do_not_depend_on_labels(raw: pd.DataFrame) -> None:
    shuffled = raw.copy()
    rng = np.random.default_rng(0)
    shuffled["isFraud"] = rng.permutation(shuffled["isFraud"].to_numpy())
    shuffled["isFlaggedFraud"] = 0
    pd.testing.assert_frame_equal(
        build_features(raw)[FEATURES], build_features(shuffled)[FEATURES]
    )


def test_label_columns_are_not_features() -> None:
    for column in ("isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "step"):
        assert column not in FEATURES


def test_resampling_does_not_change_feature_values(table: pd.DataFrame) -> None:
    split = time_split(table, SPLIT)
    folds = prepare_modeling_folds(
        split, FEATURES, "isFraud", ResamplingConfig("undersample", 0.1), seed=5
    )
    kept = folds.train.X.index
    pd.testing.assert_frame_equal(folds.train.X, split.train.loc[kept, FEATURES])
    pd.testing.assert_series_equal(folds.train.y, split.train.loc[kept, "isFraud"])
