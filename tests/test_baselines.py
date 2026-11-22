import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.baselines import (
    fit_logistic_regression,
    flagged_fraud_scores,
    large_transfer_scores,
    predict_scores,
)


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=6_000, fraud_rate=0.02, n_steps=120, seed=5)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_flagged_fraud_scores_read_the_column() -> None:
    df = pd.DataFrame({"isFlaggedFraud": [0, 1, 0]})
    assert flagged_fraud_scores(df).tolist() == [0.0, 1.0, 0.0]


def test_flagged_fraud_scores_require_the_column() -> None:
    with pytest.raises(KeyError):
        flagged_fraud_scores(pd.DataFrame({"amount": [1.0]}))


def test_large_transfer_rule() -> None:
    df = pd.DataFrame(
        {
            "type": pd.Categorical(["TRANSFER", "TRANSFER", "CASH_OUT", "TRANSFER"]),
            "amount": [250_000.0, 200_000.0, 900_000.0, 10.0],
        }
    )
    assert large_transfer_scores(df).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert large_transfer_scores(df, threshold=5.0).tolist() == [1.0, 1.0, 0.0, 1.0]


def test_logistic_regression_handles_missing_history(folds: ModelingFolds) -> None:
    assert folds.train.X.isna().any().any()
    model = fit_logistic_regression(folds.train)
    scores = predict_scores(model, folds.X_validation)
    assert scores.shape == (len(folds.X_validation),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_logistic_regression_beats_prevalence(folds: ModelingFolds) -> None:
    model = fit_logistic_regression(folds.train)
    scores = predict_scores(model, folds.X_test)
    assert pr_auc(folds.y_test, scores) > 5 * folds.y_test.mean()


def test_class_weight_is_passed_as_sample_weight(folds: ModelingFolds) -> None:
    plain = fit_logistic_regression(folds.train)
    weighted_fold = type(folds.train)(folds.train.X, folds.train.y, 20.0)
    weighted = fit_logistic_regression(weighted_fold)
    mean_plain = predict_scores(plain, folds.X_validation).mean()
    mean_weighted = predict_scores(weighted, folds.X_validation).mean()
    # Upweighting positives shifts the intercept and raises every score
    assert mean_weighted > mean_plain
