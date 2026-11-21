import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import (
    prepare_modeling_folds,
    resample_training_fold,
)
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns

SPLIT = SplitConfig(0.6, 0.2, 0.2)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=11)
    return build_features(df)


def _xy(n_neg: int = 200, n_pos: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = n_neg + n_pos
    index = pd.RangeIndex(1_000, 1_000 + n)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)
    y = pd.Series([0] * n_neg + [1] * n_pos, index=index)
    return X, y


def test_none_returns_input_unchanged() -> None:
    X, y = _xy()
    fold = resample_training_fold(X, y, ResamplingConfig("none"))
    assert fold.X is X
    assert fold.scale_pos_weight == 1.0


def test_undersample_hits_target_ratio_and_keeps_index() -> None:
    X, y = _xy()
    fold = resample_training_fold(X, y, ResamplingConfig("undersample", 0.1), seed=3)
    assert int(fold.y.sum()) == 10
    assert (fold.y == 0).sum() == 100
    assert fold.X.index.is_monotonic_increasing
    assert fold.X.index.isin(X.index).all()
    pd.testing.assert_frame_equal(fold.X, X.loc[fold.X.index])


def test_undersample_is_reproducible() -> None:
    X, y = _xy()
    config = ResamplingConfig("undersample", 0.2)
    first = resample_training_fold(X, y, config, seed=7)
    second = resample_training_fold(X, y, config, seed=7)
    assert first.X.index.equals(second.X.index)


def test_smote_adds_positives() -> None:
    X, y = _xy()
    fold = resample_training_fold(X, y, ResamplingConfig("smote", 0.5), seed=0)
    assert (fold.y == 0).sum() == 200
    assert int(fold.y.sum()) == 100
    assert isinstance(fold.X.index, pd.RangeIndex)


def test_smote_rejects_missing_values() -> None:
    X, y = _xy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="impute"):
        resample_training_fold(X, y, ResamplingConfig("smote", 0.5))


def test_class_weight_keeps_rows() -> None:
    X, y = _xy()
    fold = resample_training_fold(X, y, ResamplingConfig("class_weight", 1.0))
    assert fold.X is X
    assert fold.scale_pos_weight == pytest.approx(20.0)
    weights = fold.sample_weight
    assert weights[y.to_numpy() == 1].sum() == pytest.approx(weights[y == 0].sum())


def test_target_ratio_already_met_is_rejected() -> None:
    X, y = _xy(n_neg=20, n_pos=10)
    with pytest.raises(ValueError, match="already meets"):
        resample_training_fold(X, y, ResamplingConfig("undersample", 0.1))


def test_single_class_is_rejected() -> None:
    X, y = _xy()
    with pytest.raises(ValueError, match="both classes"):
        resample_training_fold(X, y * 0, ResamplingConfig("undersample", 0.1))


def test_only_training_fold_is_resampled(table: pd.DataFrame) -> None:
    split = time_split(table, SPLIT)
    folds = prepare_modeling_folds(
        split,
        feature_columns(),
        "isFraud",
        ResamplingConfig("undersample", 0.1),
        seed=0,
    )
    assert folds.train.X.index.isin(split.train.index).all()
    assert len(folds.train.X) < len(split.train)
    pd.testing.assert_series_equal(folds.y_validation, split.validation["isFraud"])
    pd.testing.assert_series_equal(folds.y_test, split.test["isFraud"])
    assert list(folds.X_test.columns) == feature_columns()


def test_features_survive_resampling(table: pd.DataFrame) -> None:
    # Engineered columns must reach the model, unlike in the original notebook
    split = time_split(table, SPLIT)
    folds = prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("undersample", 0.1)
    )
    pd.testing.assert_frame_equal(
        folds.train.X, table.loc[folds.train.X.index, feature_columns()]
    )
