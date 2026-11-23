import numpy as np
import pytest

from fraud_detection.config import EvaluationConfig, ResamplingConfig, SplitConfig
from fraud_detection.data.split import TimeSplit, time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.resampling_comparison import (
    DEFAULT_CANDIDATES,
    compare_resampling,
    fill_missing,
)
from fraud_detection.features.build import build_features, feature_columns

FAST = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}
EVALUATION = EvaluationConfig("cost", cost_false_positive=1.0, cost_false_negative=25.0)


@pytest.fixture(scope="module")
def split() -> TimeSplit:
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.02, n_steps=120, seed=4)
    return time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))


def test_fill_missing_leaves_input_untouched(split: TimeSplit) -> None:
    columns = feature_columns()
    assert split.train[columns].isna().any().any()
    filled = fill_missing(split, columns, -1.0)
    for name, fold in filled.folds().items():
        assert not fold[columns].isna().any().any()
        assert fold.index.equals(split.folds()[name].index)
    assert split.train[columns].isna().any().any()


def test_compare_all_default_candidates(split: TimeSplit) -> None:
    table = compare_resampling(
        split, feature_columns(), "isFraud", EVALUATION, params=FAST, seed=1
    )
    assert set(table.index) == set(DEFAULT_CANDIDATES)
    assert table["pr_auc"].is_monotonic_decreasing
    assert table[["pr_auc", "roc_auc"]].le(1).all().all()

    n_train = len(split.train)
    n_pos = int(split.train["isFraud"].sum())
    assert table.loc["none", "train_rows"] == n_train
    assert table.loc["class_weight", "train_rows"] == n_train
    assert table.loc["class_weight", "scale_pos_weight"] == pytest.approx(
        0.1 * (n_train - n_pos) / n_pos
    )
    assert table.loc["undersample", "train_rows"] < n_train
    assert table.loc["smote", "train_rows"] > n_train
    assert table.loc["smote", "train_positives"] > n_pos


def test_test_fold_is_not_used(split: TimeSplit) -> None:
    # Corrupting the test labels must not change any validation result
    corrupted = TimeSplit(
        train=split.train,
        validation=split.validation,
        test=split.test.assign(isFraud=1 - split.test["isFraud"]),
        boundaries=split.boundaries,
    )
    candidates = {"cw": ResamplingConfig("class_weight", 0.2)}
    args = (feature_columns(), "isFraud", EVALUATION)
    a = compare_resampling(split, *args, params=FAST, candidates=candidates)
    b = compare_resampling(corrupted, *args, params=FAST, candidates=candidates)
    np.testing.assert_allclose(
        a.drop(columns="fit_seconds").select_dtypes("number"),
        b.drop(columns="fit_seconds").select_dtypes("number"),
    )


def test_empty_candidates_rejected(split: TimeSplit) -> None:
    with pytest.raises(ValueError, match="candidate"):
        compare_resampling(
            split, feature_columns(), "isFraud", EVALUATION, candidates={}
        )
