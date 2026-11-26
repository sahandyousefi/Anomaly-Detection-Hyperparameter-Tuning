from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.explain.shap_analysis import (
    ShapExplanation,
    explain_model,
    interaction_strength,
    sample_rows,
    save_shap_figures,
    strongest_partner,
    tree_shap,
    tree_shap_interactions,
)
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import fit_model
from fraud_detection.models.gbm import BoostedModel, fit_xgboost

FAST = {"n_estimators": 200, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.03, n_steps=100, seed=4)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


@pytest.fixture(scope="module")
def booster(folds: ModelingFolds) -> BoostedModel:
    return fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 15)


def test_tree_shap_is_additive_in_log_odds(
    folds: ModelingFolds, booster: BoostedModel
) -> None:
    X = folds.X_test.iloc[:300]
    explanation = tree_shap(booster, X)
    assert explanation.output == "log_odds"
    assert explanation.values.shape == X.shape
    expected = logit(np.clip(booster.predict_scores(X), 1e-12, 1 - 1e-12))
    np.testing.assert_allclose(explanation.model_output(), expected, atol=1e-4)


def test_interactions_sum_to_shap_values(
    folds: ModelingFolds, booster: BoostedModel
) -> None:
    X = folds.X_test.iloc[:50]
    interactions = tree_shap_interactions(booster, X)
    assert interactions.shape == (len(X), X.shape[1], X.shape[1])
    np.testing.assert_allclose(
        interactions.sum(axis=2), tree_shap(booster, X).values, atol=1e-4
    )
    np.testing.assert_allclose(interactions, interactions.transpose(0, 2, 1))


def test_lightgbm_tree_path(folds: ModelingFolds) -> None:
    pytest.importorskip("lightgbm")
    params = {"n_estimators": 100, "learning_rate": 0.2, "n_jobs": 1}
    model = fit_model(
        "lightgbm",
        folds.train,
        folds.X_validation,
        folds.y_validation,
        {**params, "early_stopping_rounds": 10},
    )
    X = folds.X_test.iloc[:100]
    explanation = tree_shap(model, X)
    expected = logit(np.clip(model.predict_scores(X), 1e-12, 1 - 1e-12))
    np.testing.assert_allclose(explanation.model_output(), expected, atol=1e-6)


def test_permutation_path_for_non_tree_models(folds: ModelingFolds) -> None:
    model = fit_model("logistic_regression", folds.train, folds.X_validation, None)
    X = folds.X_test.iloc[:20]
    explanation = explain_model(model, X, background=folds.train.X, seed=1)
    assert explanation.output == "probability"
    # With an independent masker the permutation estimator is exactly additive
    np.testing.assert_allclose(
        explanation.model_output(), model.predict_scores(X), atol=1e-6
    )


def test_non_tree_model_requires_background(folds: ModelingFolds) -> None:
    model = fit_model("logistic_regression", folds.train, folds.X_validation, None)
    with pytest.raises(ValueError, match="background"):
        explain_model(model, folds.X_test.iloc[:5])


def test_sample_rows_keeps_order_and_size() -> None:
    X = pd.DataFrame({"a": np.arange(100)})
    sampled = sample_rows(X, 10, seed=3)
    assert len(sampled) == 10
    assert sampled["a"].is_monotonic_increasing
    assert sample_rows(X, None) is X
    assert sample_rows(X, 500) is X


def test_mean_abs_ranks_informative_features_first() -> None:
    data = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    values = np.array([[0.1, -2.0], [0.1, 1.0]])
    explanation = ShapExplanation(values, 0.0, data, "log_odds")
    assert list(explanation.mean_abs().index) == ["b", "a"]
    assert explanation.subset([1]).values.tolist() == [[0.1, 1.0]]
    with pytest.raises(ValueError, match="shape"):
        ShapExplanation(values[:, :1], 0.0, data, "log_odds")


def test_strongest_partner_ignores_the_diagonal() -> None:
    interactions = np.zeros((2, 3, 3))
    interactions[:, 0, 0] = 9.0
    interactions[:, 0, 2] = interactions[:, 2, 0] = 1.0
    interactions[:, 0, 1] = interactions[:, 1, 0] = 0.5
    strength = interaction_strength(interactions, ["a", "b", "c"])
    assert strongest_partner(strength, "a") == "c"


def test_figures_are_written(
    folds: ModelingFolds, booster: BoostedModel, tmp_path: Path
) -> None:
    X = folds.X_test.iloc[:200]
    explanation = tree_shap(booster, X)
    interactions = tree_shap_interactions(booster, X.iloc[:40])
    paths = save_shap_figures(
        explanation.subset(np.arange(40)), tmp_path, interactions, n_dependence=2
    )
    assert {"importance", "summary", "interactions"} <= set(paths)
    assert sum(kind.startswith("dependence_") for kind in paths) == 2
    assert all(path.stat().st_size > 0 for path in paths.values())
