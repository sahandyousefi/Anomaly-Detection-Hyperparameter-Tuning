import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.explain.importance_comparison import (
    compare_rankings,
    permutation_importance,
    plot_rank_comparison,
)
from fraud_detection.explain.shap_analysis import tree_shap
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.gbm import BoostedModel, fit_xgboost


class _FirstColumn:
    """Scores equal to the first column, so only that column matters."""

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return X.iloc[:, 0].to_numpy(dtype="float64")


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.03, n_steps=100, seed=8)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


@pytest.fixture(scope="module")
def booster(folds: ModelingFolds) -> BoostedModel:
    params = {"n_estimators": 150, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}
    return fit_xgboost(folds.train, folds.X_validation, folds.y_validation, params)


def test_unused_features_have_zero_importance() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    X = pd.DataFrame(
        {"signal": y + rng.normal(0, 0.5, 400), "noise": rng.normal(size=400)}
    )
    table = permutation_importance(_FirstColumn(), X, y, n_repeats=3)
    assert list(table.index) == ["signal", "noise"]
    assert table.loc["signal", "importance"] > 0.2
    assert table.loc["noise", "importance"] == 0.0
    assert table.loc["noise", "std"] == 0.0


def test_permutation_does_not_modify_input() -> None:
    X = pd.DataFrame({"a": np.arange(50, dtype=float), "b": np.zeros(50)})
    y = pd.Series((np.arange(50) > 40).astype(int))
    before = X.copy()
    permutation_importance(_FirstColumn(), X, y, n_repeats=2, features=["a"])
    pd.testing.assert_frame_equal(X, before)
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(_FirstColumn(), X, y, n_repeats=0)


def test_compare_rankings_identical_orders() -> None:
    a = pd.Series({"x": 3.0, "y": 2.0, "z": 1.0})
    table, agreement = compare_rankings(a, a * 10, k=2)
    assert agreement.spearman == pytest.approx(1.0)
    assert agreement.kendall == pytest.approx(1.0)
    assert agreement.top_k_jaccard == 1.0
    assert (table["rank_difference"] == 0).all()


def test_compare_rankings_reversed_orders() -> None:
    a = pd.Series({"x": 3.0, "y": 2.0, "z": 1.0, "w": 0.5})
    b = pd.Series({"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4, "extra": 9.0})
    table, agreement = compare_rankings(a, b, k=2)
    assert "extra" not in table.index
    assert agreement.spearman == pytest.approx(-1.0)
    assert agreement.top_k_jaccard == 0.0
    with pytest.raises(ValueError, match="shared"):
        compare_rankings(a, pd.Series({"x": 1.0}))


def test_booster_rankings_agree_at_the_top(
    folds: ModelingFolds, booster: BoostedModel
) -> None:
    shap_importance = tree_shap(booster, folds.X_test).mean_abs()
    features = list(shap_importance.index[:6])
    permutation = permutation_importance(
        booster, folds.X_test, folds.y_test, n_repeats=2, features=features
    )
    table, agreement = compare_rankings(
        shap_importance.loc[features], permutation["importance"], k=3
    )
    assert len(table) == 6
    assert -1.0 <= agreement.spearman <= 1.0
    assert permutation["importance"].iloc[0] > 0
    fig = plot_rank_comparison(table, agreement)
    assert "Spearman" in fig.axes[0].get_title()
