import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import (
    EvaluationConfig,
    FeaturesConfig,
    ResamplingConfig,
    SplitConfig,
)
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import TimeSplit, time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluation.model_comparison import (
    MODEL_FAMILIES,
    Candidate,
    compare_models,
    default_candidates,
)
from fraud_detection.evaluation.thresholds import select_threshold
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.features.sequence import history_columns
from fraud_detection.models.base import MODEL_REGISTRY, fit_model

EVALUATION = EvaluationConfig("cost", 1.0, 20.0)
FEATURES = feature_columns()
XGB = {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}
IFOREST = {"n_estimators": 50, "n_jobs": 1}


@pytest.fixture(scope="module")
def split() -> TimeSplit:
    df = generate_paysim_like(
        n_rows=4_000, fraud_rate=0.03, n_steps=80, n_customers=400, seed=31
    )
    table = build_features(df, FeaturesConfig(history_length=2))
    return time_split(table, SplitConfig(0.6, 0.2, 0.2))


CANDIDATES = {
    "logistic": Candidate("logistic_regression"),
    "xgboost": Candidate("xgboost", XGB),
    "iforest": Candidate("isolation_forest", IFOREST),
}


@pytest.fixture(scope="module")
def table(split: TimeSplit) -> pd.DataFrame:
    return compare_models(split, FEATURES, "isFraud", EVALUATION, CANDIDATES, seed=0)


def test_every_registered_model_has_a_family() -> None:
    assert set(MODEL_REGISTRY) == set(MODEL_FAMILIES)


def test_table_layout_and_order(table: pd.DataFrame) -> None:
    assert set(table.index) == set(CANDIDATES)
    assert table.loc["iforest", "family"] == "unsupervised"
    assert table.loc["xgboost", "family"] == "supervised"
    assert table["validation_pr_auc"].is_monotonic_decreasing
    assert (table["n_features"] == len(FEATURES)).all()
    for column in ("test_pr_auc", "test_roc_auc", "test_recall@p0.8"):
        assert table[column].between(0, 1).all()


def test_threshold_is_chosen_on_validation(split: TimeSplit, table) -> None:
    folds = prepare_modeling_folds(
        split, FEATURES, "isFraud", ResamplingConfig("none"), seed=0
    )
    model = fit_model(
        "xgboost", folds.train, folds.X_validation, folds.y_validation, XGB, seed=0
    )
    scores = model.predict_scores(folds.X_validation)
    expected = select_threshold(folds.y_validation, scores, EVALUATION)
    assert table.loc["xgboost", "threshold"] == pytest.approx(expected.threshold)


def test_extra_features_and_resampling_per_candidate(split: TimeSplit) -> None:
    extra = history_columns(2, ["dest"])[:3]
    candidates = {
        "lr_extra": Candidate(
            "logistic_regression",
            extra_features=extra,
            resampling=ResamplingConfig("class_weight", 0.5),
        )
    }
    table = compare_models(split, FEATURES, "isFraud", EVALUATION, candidates)
    assert table.loc["lr_extra", "n_features"] == len(FEATURES) + 3
    assert np.isfinite(table.loc["lr_extra", "test_normalized_cost"])


def test_default_candidates_cover_each_family() -> None:
    defaults = default_candidates()
    assert {c.family for c in defaults.values()} == {
        "supervised",
        "unsupervised",
        "hybrid",
    }
    assert "gru" not in defaults
    with_history = default_candidates(history_columns(2))
    assert with_history["gru"].extra_features == tuple(history_columns(2))
    assert with_history["xgboost"].extra_features == ()


def test_invalid_candidates_are_rejected(split: TimeSplit) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        Candidate("random_forest")
    with pytest.raises(ValueError, match="At least one"):
        compare_models(split, FEATURES, "isFraud", EVALUATION, {})


def test_neural_and_hybrid_candidates_run(split: TimeSplit) -> None:
    pytest.importorskip("torch")
    ae = {"hidden_dims": [16], "latent_dim": 4, "max_epochs": 3}
    nn_fast = {"batch_size": 64, "max_epochs": 3}
    candidates = {
        "feedforward": Candidate("feedforward", {**nn_fast, "hidden_dims": [16]}),
        "autoencoder": Candidate("autoencoder", ae),
        "hybrid": Candidate(
            "hybrid",
            {"autoencoder": ae, "booster_params": XGB, "cross_fit_folds": 2},
        ),
        "gru": Candidate(
            "gru",
            {**nn_fast, "hidden_size": 8, "static_dims": [8]},
            extra_features=history_columns(2),
        ),
    }
    table = compare_models(split, FEATURES, "isFraud", EVALUATION, candidates)
    assert set(table["family"]) == {"supervised", "unsupervised", "hybrid"}
    assert table["test_pr_auc"].notna().all()
