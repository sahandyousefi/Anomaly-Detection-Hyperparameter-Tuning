import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.tuning.optuna_search import expanding_window_folds

FIXED = {"n_estimators": 60, "n_jobs": 1}


def test_folds_are_expanding_and_forward_in_time() -> None:
    steps = np.repeat(np.arange(1, 51), 3)
    folds = expanding_window_folds(steps, n_splits=4)
    assert len(folds) == 4
    previous_train = 0
    for fold in folds:
        assert steps[fold.train].max() < steps[fold.validation].min()
        assert fold.train.size > previous_train
        assert np.intersect1d(fold.train, fold.validation).size == 0
        previous_train = fold.train.size
    # Scored blocks tile the range after the first block
    scored = np.concatenate([f.validation for f in folds])
    assert np.unique(steps[scored]).size == 40
    assert folds[0].train_steps == (1, 10)
    assert folds[-1].validation_steps == (41, 50)


def test_gap_separates_training_and_scored_block() -> None:
    steps = np.arange(1, 61)
    folds = expanding_window_folds(steps, n_splits=3, gap_steps=2)
    for fold in folds:
        assert steps[fold.validation].min() - steps[fold.train].max() == 3


def test_fold_validation() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        expanding_window_folds(np.arange(10), n_splits=0)
    with pytest.raises(ValueError, match="distinct steps"):
        expanding_window_folds(np.arange(5), n_splits=5)
    with pytest.raises(ValueError, match="gap_steps"):
        expanding_window_folds(np.arange(10), gap_steps=-1)


@pytest.fixture(scope="module")
def tuning_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = generate_paysim_like(n_rows=8_000, fraud_rate=0.03, n_steps=100, seed=11)
    split = time_split(build_features(df), SplitConfig(0.7, 0.15, 0.15))
    period = pd.concat([split.train, split.validation])
    return period[feature_columns()], period["isFraud"], period["step"]


def test_tune_xgboost_runs_and_records_folds(tuning_data) -> None:
    pytest.importorskip("optuna")
    from fraud_detection.tuning.optuna_search import tune_xgboost

    X, y, steps = tuning_data
    study = tune_xgboost(
        X,
        y,
        steps,
        ResamplingConfig("class_weight", 0.2),
        n_trials=3,
        n_splits=2,
        fixed_params=FIXED,
        early_stopping_rounds=10,
        seed=0,
    )
    assert len(study.trials) == 3
    best = study.best_trial
    assert 0 < best.value <= 1
    assert len(best.user_attrs["fold_pr_auc"]) == 2
    assert best.value == pytest.approx(np.mean(best.user_attrs["fold_pr_auc"]))
    assert {"learning_rate", "max_depth", "reg_lambda"} <= set(best.params)
    assert "n_estimators" not in best.params


def test_search_is_reproducible(tuning_data) -> None:
    pytest.importorskip("optuna")
    from fraud_detection.tuning.optuna_search import tune_xgboost

    X, y, steps = tuning_data
    kwargs = {
        "resampling": ResamplingConfig("none"),
        "n_trials": 2,
        "n_splits": 2,
        "fixed_params": FIXED,
        "early_stopping_rounds": 10,
        "seed": 3,
    }
    a = tune_xgboost(X, y, steps, **kwargs)
    b = tune_xgboost(X, y, steps, **kwargs)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == pytest.approx([t.value for t in b.trials])
