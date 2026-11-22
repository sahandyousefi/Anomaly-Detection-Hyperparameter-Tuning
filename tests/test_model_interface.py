import numpy as np
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import (
    MODEL_REGISTRY,
    FraudModel,
    SklearnModel,
    fit_model,
)
from fraud_detection.models.gbm import BoostedModel


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=21)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_registry_lists_model_families() -> None:
    expected = {"logistic_regression", "xgboost", "lightgbm", "catboost"}
    assert expected <= set(MODEL_REGISTRY)


def test_logistic_regression_through_interface(folds: ModelingFolds) -> None:
    model = fit_model(
        "logistic_regression",
        folds.train,
        folds.X_validation,
        folds.y_validation,
        {"C": 0.5},
    )
    assert isinstance(model, SklearnModel)
    assert isinstance(model, FraudModel)
    assert model.estimator.named_steps["model"].C == 0.5
    scores = model.predict_scores(folds.X_test)
    assert scores.dtype == np.float64
    assert scores.shape == (len(folds.X_test),)


def test_xgboost_through_interface_routes_patience(folds: ModelingFolds) -> None:
    params = {"n_estimators": 200, "learning_rate": 0.3, "n_jobs": 1}
    model = fit_model(
        "xgboost",
        folds.train,
        folds.X_validation,
        folds.y_validation,
        {**params, "early_stopping_rounds": 7},
        seed=3,
    )
    assert isinstance(model, BoostedModel)
    assert isinstance(model, FraudModel)
    assert model.model.get_params()["early_stopping_rounds"] == 7
    assert model.model.get_params()["random_state"] == 3
    assert len(model.validation_curve) <= model.best_iteration + 8


def test_lightgbm_through_interface(folds: ModelingFolds) -> None:
    pytest.importorskip("lightgbm")
    model = fit_model(
        "lightgbm",
        folds.train,
        folds.X_validation,
        folds.y_validation,
        {"n_estimators": 50, "n_jobs": 1, "early_stopping_rounds": 5},
    )
    assert isinstance(model, FraudModel)
    assert model.predict_scores(folds.X_test).shape == (len(folds.X_test),)


def test_unknown_model_name_is_rejected(folds: ModelingFolds) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        fit_model("random_forest", folds.train, folds.X_validation, folds.y_validation)


def test_catboost_through_interface(folds: ModelingFolds) -> None:
    pytest.importorskip("catboost")
    model = fit_model(
        "catboost",
        folds.train,
        folds.X_validation,
        folds.y_validation,
        {"n_estimators": 50, "thread_count": 1, "early_stopping_rounds": 5},
    )
    assert isinstance(model, FraudModel)
    assert model.predict_scores(folds.X_test).shape == (len(folds.X_test),)
