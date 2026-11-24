import numpy as np
import pandas as pd
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
from fraud_detection.models.base import FraudModel, fit_model
from fraud_detection.models.unsupervised import (
    AnomalyDetector,
    fit_isolation_forest,
    fit_local_outlier_factor,
    legitimate_rows,
)

DETECTORS = {
    "isolation_forest": {"n_estimators": 100, "n_jobs": 1},
    "local_outlier_factor": {"n_neighbors": 20, "n_jobs": 1},
}


@pytest.fixture(scope="module")
def folds() -> ModelingFolds:
    df = generate_paysim_like(n_rows=6_000, fraud_rate=0.03, n_steps=100, seed=8)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    return prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )


def test_legitimate_rows_drop_positives_and_subsample(folds) -> None:
    X = legitimate_rows(folds.train)
    assert len(X) == int((folds.train.y == 0).sum())
    assert folds.train.y.loc[X.index].eq(0).all()
    capped = legitimate_rows(folds.train, max_rows=500, seed=1)
    assert len(capped) == 500
    assert capped.index.is_monotonic_increasing


def test_no_legitimate_rows_raises() -> None:
    train = ResampledFold(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1, 1]))
    with pytest.raises(ValueError, match="no legitimate rows"):
        legitimate_rows(train)


def test_detectors_ignore_fraud_rows(folds) -> None:
    # Fitting on the full fold and on legitimate rows only must agree
    legit = folds.train.y == 0
    only_legit = ResampledFold(folds.train.X[legit], folds.train.y[legit])
    a = fit_isolation_forest(folds.train, {"n_estimators": 50, "n_jobs": 1}, seed=2)
    b = fit_isolation_forest(only_legit, {"n_estimators": 50, "n_jobs": 1}, seed=2)
    np.testing.assert_allclose(
        a.predict_scores(folds.X_test), b.predict_scores(folds.X_test)
    )


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_through_interface(folds, name) -> None:
    model = fit_model(
        name, folds.train, folds.X_validation, folds.y_validation, DETECTORS[name]
    )
    assert isinstance(model, AnomalyDetector)
    assert isinstance(model, FraudModel)
    scores = model.predict_scores(folds.X_test)
    assert scores.shape == (len(folds.X_test),)
    assert scores.min() >= 0
    assert scores.max() <= 1
    # Ranking is that of the raw anomaly score
    raw = model.anomaly_scores(folds.X_test)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(scores[order]) >= 0)
    assert 0 <= pr_auc(folds.y_test, scores) <= 1


def test_reference_scores_define_the_empirical_cdf(folds) -> None:
    model = fit_local_outlier_factor(
        folds.train, {"n_neighbors": 20, "n_jobs": 1}, max_rows=1_000, seed=0
    )
    assert model.reference_scores.size == 1_000
    assert np.all(np.diff(model.reference_scores) >= 0)
    top = model.reference_scores[-1] + 1.0
    lowest = model.reference_scores[0] - 1.0
    ranks = np.searchsorted(model.reference_scores, [lowest, top], side="right")
    assert list(ranks / model.reference_scores.size) == [0.0, 1.0]


def test_isolation_forest_is_reproducible(folds) -> None:
    params = {"n_estimators": 50, "n_jobs": 1}
    a = fit_isolation_forest(folds.train, params, seed=5)
    b = fit_isolation_forest(folds.train, params, seed=5)
    np.testing.assert_array_equal(
        a.predict_scores(folds.X_test), b.predict_scores(folds.X_test)
    )
