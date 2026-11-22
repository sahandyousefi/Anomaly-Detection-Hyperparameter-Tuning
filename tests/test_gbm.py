import numpy as np
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import (
    ModelingFolds,
    ResampledFold,
    prepare_modeling_folds,
)
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.gbm import XGBOOST_DEFAULTS, fit_xgboost, xgboost_params

FAST = {"n_estimators": 300, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=6_000, fraud_rate=0.02, n_steps=120, seed=9)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_params_merge_order() -> None:
    merged = xgboost_params({"max_depth": 3}, scale_pos_weight=4.0, seed=7)
    assert merged["max_depth"] == 3
    assert merged["scale_pos_weight"] == 4.0
    assert merged["random_state"] == 7
    assert merged["eval_metric"] == XGBOOST_DEFAULTS["eval_metric"]
    explicit = xgboost_params({"scale_pos_weight": 2.0}, scale_pos_weight=4.0)
    assert explicit["scale_pos_weight"] == 2.0


def test_early_stopping_on_validation(folds: ModelingFolds) -> None:
    fitted = fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 20)
    assert fitted.best_iteration < FAST["n_estimators"] - 1
    assert len(fitted.validation_curve) == fitted.best_iteration + 21
    assert fitted.best_score == pytest.approx(
        fitted.validation_curve[fitted.best_iteration]
    )
    assert fitted.best_score == pytest.approx(fitted.validation_curve.max())


def test_predictions_use_best_iteration(folds: ModelingFolds) -> None:
    fitted = fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 20)
    scores = fitted.predict_scores(folds.X_validation)
    truncated = fitted.model.predict_proba(
        folds.X_validation, iteration_range=(0, fitted.best_iteration + 1)
    )[:, 1]
    np.testing.assert_allclose(scores, truncated)
    # XGBoost's aucpr interpolates differently from average precision
    assert pr_auc(folds.y_validation, scores) == pytest.approx(
        fitted.best_score, abs=0.02
    )


def test_model_ranks_test_fold_above_prevalence(folds: ModelingFolds) -> None:
    fitted = fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 20)
    scores = fitted.predict_scores(folds.X_test)
    assert pr_auc(folds.y_test, scores) > 5 * folds.y_test.mean()


def test_class_weight_reaches_the_booster(folds: ModelingFolds) -> None:
    weighted = ResampledFold(folds.train.X, folds.train.y, scale_pos_weight=12.5)
    fitted = fit_xgboost(weighted, folds.X_validation, folds.y_validation, FAST, 10)
    assert fitted.model.get_params()["scale_pos_weight"] == 12.5


def test_feature_mismatch_is_rejected(folds: ModelingFolds) -> None:
    with pytest.raises(ValueError, match="match"):
        fit_xgboost(
            folds.train,
            folds.X_validation.iloc[:, ::-1],
            folds.y_validation,
            FAST,
        )
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        fit_xgboost(folds.train, folds.X_validation, folds.y_validation, FAST, 0)


def test_lightgbm_early_stopping_on_validation(folds: ModelingFolds) -> None:
    pytest.importorskip("lightgbm")
    from fraud_detection.models.gbm import fit_lightgbm

    params = {"n_estimators": 300, "learning_rate": 0.3, "num_leaves": 7, "n_jobs": 1}
    fitted = fit_lightgbm(
        folds.train, folds.X_validation, folds.y_validation, params, 20
    )
    assert fitted.best_iteration < params["n_estimators"] - 1
    assert len(fitted.validation_curve) == fitted.best_iteration + 21
    assert fitted.best_score == pytest.approx(fitted.validation_curve.max())
    scores = fitted.predict_scores(folds.X_validation)
    truncated = fitted.model.predict_proba(
        folds.X_validation, num_iteration=fitted.best_iteration + 1
    )[:, 1]
    np.testing.assert_allclose(scores, truncated)
    # LightGBM's average_precision matches the sklearn estimator used here
    assert pr_auc(folds.y_validation, scores) == pytest.approx(
        fitted.best_score, abs=1e-6
    )
    assert pr_auc(folds.y_test, fitted.predict_scores(folds.X_test)) > (
        5 * folds.y_test.mean()
    )


def test_lightgbm_receives_class_weight(folds: ModelingFolds) -> None:
    pytest.importorskip("lightgbm")
    from fraud_detection.models.gbm import fit_lightgbm

    weighted = ResampledFold(folds.train.X, folds.train.y, scale_pos_weight=8.0)
    fitted = fit_lightgbm(
        weighted,
        folds.X_validation,
        folds.y_validation,
        {"n_estimators": 20, "n_jobs": 1},
        5,
    )
    assert fitted.model.get_params()["scale_pos_weight"] == 8.0
