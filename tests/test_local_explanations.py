from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.explain.local_explanations import (
    contribution_table,
    explain_flagged,
    plot_waterfall,
    save_waterfalls,
    select_flagged,
)
from fraud_detection.explain.shap_analysis import ShapExplanation
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.gbm import BoostedModel, fit_xgboost

FAST = {"n_estimators": 200, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.03, n_steps=100, seed=5)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


@pytest.fixture(scope="module")
def booster(folds: ModelingFolds) -> BoostedModel:
    return fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 15)


@pytest.fixture
def toy() -> ShapExplanation:
    data = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
    values = np.array([[0.2, -1.5, 0.05, 0.7]])
    return ShapExplanation(values, -3.0, data, "log_odds")


def test_select_flagged_orders_by_score() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.5, 0.2])
    np.testing.assert_array_equal(select_flagged(scores, 0.5), [3, 1, 2, 4])
    np.testing.assert_array_equal(select_flagged(scores, 0.5, max_rows=2), [3, 1])
    assert len(select_flagged(scores, 0.99)) == 0


def test_contribution_table_pools_the_remainder(toy: ShapExplanation) -> None:
    table = contribution_table(toy, 0, max_features=2)
    assert list(table["feature"]) == ["b", "d", "other features (2)"]
    assert table["contribution"].iloc[-1] == pytest.approx(0.25)
    assert np.isnan(table["value"].iloc[-1])
    assert table["contribution"].sum() == pytest.approx(toy.values.sum())


def test_contribution_table_without_pooling(toy: ShapExplanation) -> None:
    table = contribution_table(toy, 0, max_features=10)
    assert len(table) == 4
    assert list(table["feature"]) == ["b", "d", "a", "c"]
    with pytest.raises(ValueError, match="max_features"):
        contribution_table(toy, 0, max_features=0)


def test_waterfall_ends_at_model_output(toy: ShapExplanation) -> None:
    fig = plot_waterfall(toy, 0, max_features=2)
    bars = fig.axes[0].patches
    assert len(bars) == 3
    final = max(bars, key=lambda bar: bar.get_y())
    assert final.get_x() + final.get_width() == pytest.approx(toy.model_output()[0])


def test_explain_flagged_covers_rows_above_threshold(
    folds: ModelingFolds, booster: BoostedModel
) -> None:
    scores = booster.predict_scores(folds.X_test)
    threshold = float(np.quantile(scores, 0.99))
    explanation = explain_flagged(booster, folds.X_test, threshold, max_rows=5)
    assert 0 < len(explanation.data) <= 5
    explained_scores = booster.predict_scores(explanation.data)
    assert (explained_scores >= threshold).all()
    assert np.all(np.diff(explained_scores) <= 0)


def test_explain_flagged_with_no_alerts(
    folds: ModelingFolds, booster: BoostedModel
) -> None:
    explanation = explain_flagged(booster, folds.X_test, threshold=1.1)
    assert explanation.values.shape == (0, folds.X_test.shape[1])


def test_waterfalls_are_written(
    folds: ModelingFolds, booster: BoostedModel, tmp_path: Path
) -> None:
    scores = booster.predict_scores(folds.X_test)
    threshold = float(np.quantile(scores, 0.98))
    explanation = explain_flagged(booster, folds.X_test, threshold, max_rows=3)
    paths = save_waterfalls(explanation, tmp_path)
    assert len(paths) == len(explanation.data)
    assert all(path.stat().st_size > 0 for path in paths)
    with pytest.raises(ValueError, match="labels"):
        save_waterfalls(explanation, tmp_path, labels=["only one", "two", "x", "y"])
