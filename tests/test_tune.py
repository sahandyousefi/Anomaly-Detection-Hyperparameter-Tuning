from pathlib import Path

import pytest
import yaml

from fraud_detection import cli
from fraud_detection.config import (
    ConfigError,
    TuningConfig,
    load_config,
    load_model_override,
)
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.tune import tuning_period

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"] = {
        "raw_data": "data/missing.csv",
        "artifacts": "artifacts",
        "figures": "figures",
    }
    raw["model"]["params"] = {"n_estimators": 30, "n_jobs": 1}
    raw["resampling"] = {"method": "class_weight", "sampling_strategy": 0.1}
    raw["tuning"] = {"n_trials": 3, "n_splits": 2, "early_stopping_rounds": 5}
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def synthetic_log():
    return generate_paysim_like(n_rows=6_000, fraud_rate=0.03, n_steps=120, seed=8)


def test_tuning_defaults() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.tuning == TuningConfig()


@pytest.mark.parametrize(
    "kwargs",
    [{"n_trials": 0}, {"gap_steps": -1}, {"early_stopping_rounds": 0}, {"timeout": 0}],
)
def test_invalid_tuning_settings_rejected(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        TuningConfig(**kwargs)


def test_tuning_period_excludes_test_steps(config_path: Path, synthetic_log) -> None:
    config = load_config(config_path)
    X, y, steps = tuning_period(synthetic_log, config)
    split = time_split(build_features(synthetic_log), config.split)
    assert len(X) == len(y) == len(split.train) + len(split.validation)
    assert steps.is_monotonic_increasing
    assert steps.max() < split.test["step"].min()
    assert list(X.columns) == feature_columns(config.features)


def test_model_override_requires_name(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"model": {"params": {"max_depth": 3}}}))
    with pytest.raises(ConfigError, match="name"):
        load_model_override(path)


def test_tune_then_train_with_best_params(config_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    out = tmp_path / "tuning"
    common = ["--config", str(config_path), "--synthetic-rows", "20000"]
    assert cli.main(["tune", *common, "--output", str(out)]) == 0

    document = yaml.safe_load((out / "best_params.yaml").read_text())
    assert document["model"]["name"] == "xgboost"
    assert document["model"]["params"]["n_estimators"] == 30
    assert {"max_depth", "learning_rate"} <= document["model"]["params"].keys()
    assert document["search"]["completed_trials"] >= 1
    assert (out / "trials.csv").exists()
    assert (out / "study.db").exists()
    assert (tmp_path / "figures" / "xgboost_pr_auc_history.png").exists()

    # A second run extends the stored study
    assert cli.main(["tune", *common, "--output", str(out), "--n-trials", "2"]) == 0
    rerun = yaml.safe_load((out / "best_params.yaml").read_text())
    assert rerun["search"]["completed_trials"] + rerun["search"]["pruned_trials"] == 5

    model = load_model_override(out / "best_params.yaml")
    assert model.params == rerun["model"]["params"]
    train_out = tmp_path / "model"
    args = ["--output", str(train_out), "--params", str(out / "best_params.yaml")]
    assert cli.main(["train", *common, *args]) == 0
    assert (train_out / "metrics.json").exists()
