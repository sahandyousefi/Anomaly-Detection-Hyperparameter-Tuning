import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.data.split import TimeSplit, time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.explain.shap_analysis import ShapExplanation
from fraud_detection.explain.stability import (
    feature_stability,
    fold_stability,
    importance_matrix,
    pairwise_agreement,
    seed_stability,
)
from fraud_detection.features.build import build_features, feature_columns

FAST = {
    "n_estimators": 80,
    "learning_rate": 0.3,
    "max_depth": 3,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": 1,
    "early_stopping_rounds": 10,
}


@pytest.fixture(scope="module")
def split() -> TimeSplit:
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.03, n_steps=120, seed=11)
    return time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))


def _explanation(importance: dict[str, float]) -> ShapExplanation:
    names = list(importance)
    values = np.array([[importance[n] for n in names], [-importance[n] for n in names]])
    return ShapExplanation(values, 0.0, pd.DataFrame(values, columns=names), "log_odds")


def test_matrix_and_feature_table() -> None:
    runs = {
        "a": _explanation({"x": 3.0, "y": 2.0, "z": 1.0}),
        "b": _explanation({"x": 3.0, "y": 1.0, "z": 2.0}),
    }
    matrix = importance_matrix(runs)
    assert list(matrix.columns) == ["a", "b"]
    assert matrix.loc["y", "b"] == pytest.approx(1.0)
    table = feature_stability(matrix, k=1)
    assert table.index[0] == "x"
    assert table.loc["x", "cv"] == pytest.approx(0.0)
    assert table.loc["x", "top_k_share"] == 1.0
    assert table.loc["y", "min_rank"] == 2 and table.loc["y", "max_rank"] == 3


def test_pairwise_agreement_counts_pairs() -> None:
    runs = {
        name: _explanation({"x": 3.0 + i, "y": 2.0, "z": 1.0})
        for i, name in enumerate("abc")
    }
    pairs = pairwise_agreement(importance_matrix(runs), k=2)
    assert len(pairs) == 3
    assert np.allclose(pairs["spearman"], 1.0)
    with pytest.raises(ValueError, match="two runs"):
        pairwise_agreement(importance_matrix({"a": runs["a"]}))


def test_mismatched_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="different features"):
        importance_matrix(
            {"a": _explanation({"x": 1.0}), "b": _explanation({"y": 1.0})}
        )


def test_seed_stability_explains_identical_rows(split: TimeSplit) -> None:
    columns = feature_columns()
    runs = seed_stability(
        "xgboost",
        split.train[columns],
        split.train["isFraud"],
        split.validation[columns],
        split.validation["isFraud"],
        split.test[columns],
        seeds=[0, 1, 2],
        params=FAST,
        max_rows=300,
    )
    assert list(runs) == ["seed=0", "seed=1", "seed=2"]
    indices = [tuple(e.data.index) for e in runs.values()]
    assert len(set(indices)) == 1 and len(indices[0]) == 300
    pairs = pairwise_agreement(importance_matrix(runs), k=5)
    # Subsampling changes the trees, but the leading features should persist
    assert pairs["spearman"].min() > 0.5


def test_seed_stability_needs_two_seeds(split: TimeSplit) -> None:
    X = split.train[feature_columns()]
    y = split.train["isFraud"]
    with pytest.raises(ValueError, match="two seeds"):
        seed_stability("xgboost", X, y, X, y, X, seeds=[0], params=FAST)


def test_fold_stability_uses_later_blocks(split: TimeSplit) -> None:
    columns = feature_columns()
    period = pd.concat([split.train, split.validation])
    runs = fold_stability(
        "xgboost",
        period[columns],
        period["isFraud"],
        period["step"],
        n_splits=3,
        params=FAST,
        max_rows=200,
    )
    assert len(runs) == 3
    firsts = [int(label.split("=")[1].split("-")[0]) for label in runs]
    assert firsts == sorted(firsts)
    for label, explanation in runs.items():
        first, last = (int(v) for v in label.split("=")[1].split("-"))
        block_steps = period.loc[explanation.data.index, "step"]
        assert block_steps.between(first, last).all()
