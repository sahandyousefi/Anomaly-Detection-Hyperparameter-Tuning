import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns

pytest.importorskip("optuna")

from fraud_detection.tuning import nested_cv
from fraud_detection.tuning.nested_cv import nested_evaluation

FIXED = {"n_estimators": 40, "n_jobs": 1}


@pytest.fixture(scope="module")
def period() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = generate_paysim_like(n_rows=9_000, fraud_rate=0.04, n_steps=120, seed=5)
    split = time_split(build_features(df), SplitConfig(0.7, 0.15, 0.15))
    rows = pd.concat([split.train, split.validation, split.test])
    return rows[feature_columns()], rows["isFraud"], rows["step"]


def _run(period, **kwargs):
    X, y, steps = period
    defaults = {
        "resampling": ResamplingConfig("class_weight", 0.2),
        "n_outer": 2,
        "n_inner": 2,
        "n_trials": 2,
        "fixed_params": FIXED,
        "early_stopping_rounds": 10,
        "seed": 0,
    }
    return nested_evaluation("xgboost", X, y, steps, **{**defaults, **kwargs})


def test_outer_test_block_never_reaches_the_search(period, monkeypatch) -> None:
    seen: list[tuple[int, int]] = []
    original = nested_cv.tune_model

    def spy(model_name, X, y, steps, *args, **kwargs):
        seen.append((int(np.min(steps)), int(np.max(steps))))
        return original(model_name, X, y, steps, *args, **kwargs)

    monkeypatch.setattr(nested_cv, "tune_model", spy)
    result = _run(period)
    assert len(seen) == len(result.folds) == 2
    for (_, last_tuned), fold in zip(seen, result.folds, strict=True):
        assert last_tuned == fold.tuning_steps[1]
        assert last_tuned < fold.test_steps[0]


def test_outer_folds_tile_forward_in_time(period) -> None:
    result = _run(period)
    first, second = result.folds
    assert first.test_steps[1] < second.test_steps[0]
    assert first.tuning_steps[1] < second.tuning_steps[1]
    assert second.tuning_steps[1] < second.test_steps[0]


def test_summary_statistics_match_folds(period) -> None:
    result = _run(period)
    outer = [f.outer_pr_auc for f in result.folds]
    assert all(0 < v <= 1 for v in outer)
    assert result.outer_pr_auc == pytest.approx(np.mean(outer))
    assert result.outer_pr_auc_std == pytest.approx(np.std(outer))
    gaps = [f.inner_pr_auc - f.outer_pr_auc for f in result.folds]
    assert result.mean_optimism == pytest.approx(np.mean(gaps))
    assert all(f.n_test_positive > 0 for f in result.folds)


def test_frame_lists_selected_parameters(period) -> None:
    frame = _run(period).to_frame()
    assert list(frame["fold"]) == [1, 2]
    assert {"param_learning_rate", "param_max_depth", "optimism"} <= set(frame)
    assert "param_n_estimators" not in frame


def test_nested_evaluation_is_reproducible(period) -> None:
    a = _run(period, seed=4).to_frame()
    b = _run(period, seed=4).to_frame()
    pd.testing.assert_frame_equal(a, b)


def test_short_tuning_window_is_rejected() -> None:
    X = pd.DataFrame({"x": np.arange(12.0)})
    y = pd.Series([0, 1] * 6)
    with pytest.raises(ValueError, match="distinct steps"):
        nested_evaluation(
            "xgboost",
            X,
            y,
            np.arange(12),
            ResamplingConfig("none"),
            n_outer=2,
            n_inner=4,
        )
