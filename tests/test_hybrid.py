import numpy as np
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import ModelingFolds, prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import fit_model

pytest.importorskip("torch")

from fraud_detection.models.hybrid import (
    RECONSTRUCTION_FEATURE,
    HybridParams,
    augment,
    chronological_blocks,
    out_of_fold_errors,
)

AE = {"hidden_dims": [16], "latent_dim": 4, "max_epochs": 5, "batch_size": 256}
BOOSTER = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}
PARAMS = {"autoencoder": AE, "booster_params": BOOSTER, "cross_fit_folds": 2}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.03, n_steps=80, seed=23)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_blocks_are_contiguous_and_balanced() -> None:
    blocks = chronological_blocks(10, 3)
    np.testing.assert_array_equal(blocks, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert np.all(np.diff(blocks) >= 0)
    with pytest.raises(ValueError, match="blocks"):
        chronological_blocks(2, 3)


def test_params_are_validated() -> None:
    assert HybridParams.from_mapping(None).booster == "xgboost"
    with pytest.raises(ValueError, match="Unknown hybrid"):
        HybridParams.from_mapping({"folds": 2})
    with pytest.raises(ValueError, match="cross_fit_folds"):
        HybridParams.from_mapping({"cross_fit_folds": 0})
    with pytest.raises(ValueError, match="cannot be a hybrid"):
        HybridParams.from_mapping({"booster": "hybrid"})


def test_augment_appends_one_column(folds) -> None:
    X = folds.X_test
    out = augment(X, np.arange(len(X), dtype=float))
    assert list(out.columns) == [*X.columns, RECONSTRUCTION_FEATURE]
    assert RECONSTRUCTION_FEATURE not in X.columns
    with pytest.raises(ValueError, match="already"):
        augment(out, np.zeros(len(X)))


def test_cross_fitted_errors_are_out_of_sample(folds) -> None:
    in_sample = out_of_fold_errors(folds.train, 1, AE, seed=0)
    cross_fit = out_of_fold_errors(folds.train, 3, AE, seed=0)
    assert cross_fit.shape == in_sample.shape == (len(folds.train.X),)
    assert np.all(np.isfinite(cross_fit))
    legit = folds.train.y.to_numpy() == 0
    # Networks reconstruct their own training rows better than unseen rows
    assert np.median(cross_fit[legit]) > np.median(in_sample[legit])


def test_hybrid_through_interface(folds) -> None:
    model = fit_model(
        "hybrid", folds.train, folds.X_validation, folds.y_validation, PARAMS
    )
    augmented = model.augment(folds.X_test)
    assert augmented.columns[-1] == RECONSTRUCTION_FEATURE
    np.testing.assert_allclose(
        augmented[RECONSTRUCTION_FEATURE],
        model.autoencoder.anomaly_scores(folds.X_test),
    )
    scores = model.predict_scores(folds.X_test)
    assert scores.shape == (len(folds.X_test),)
    assert pr_auc(folds.y_test, scores) > 5 * folds.y_test.mean()
