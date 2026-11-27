import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraud_detection import cli
from fraud_detection.config import ConfigError, TrackingConfig, load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features
from fraud_detection.tracking import (
    flatten_params,
    metric_name,
    run_metrics,
    run_tags,
)
from fraud_detection.train import run_training, save_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"] = {
        "raw_data": "data/missing.csv",
        "artifacts": "artifacts",
        "figures": "figures",
    }
    raw["model"]["params"] = {
        "n_estimators": 40,
        "max_depth": 3,
        "n_jobs": 1,
        "early_stopping_rounds": 10,
    }
    raw["resampling"] = {"method": "class_weight", "sampling_strategy": 0.1}
    raw["tracking"] = {
        "enabled": False,
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "experiment_name": "unit-test",
        "artifact_location": (tmp_path / "mlruns").as_uri(),
    }
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def synthetic_log():
    return generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=11)


def test_tracking_defaults_to_disabled() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.tracking == TrackingConfig()
    assert not config.tracking.enabled


def test_empty_experiment_name_rejected() -> None:
    with pytest.raises(ConfigError, match="experiment_name"):
        TrackingConfig(experiment_name="")


def test_flatten_params_prefixes_sections(config_path: Path) -> None:
    params = flatten_params(load_config(config_path))
    assert params["seed"] == "42"
    assert params["model.name"] == "xgboost"
    assert params["model.params.max_depth"] == "3"
    assert params["resampling.method"] == "class_weight"
    assert params["data.id_columns"] == "nameOrig,nameDest"
    assert all(isinstance(v, str) for v in params.values())
    assert not any(k.startswith(("paths.", "tracking.")) for k in params)


def test_run_metrics_drop_non_finite(config_path: Path, synthetic_log) -> None:
    result = run_training(synthetic_log, load_config(config_path))
    patched = replace(result, test={**result.test, "pr_auc": float("nan")})
    metrics = run_metrics(patched)
    assert "test.pr_auc" not in metrics
    assert metrics["validation.pr_auc"] == pytest.approx(result.validation["pr_auc"])
    assert all(math.isfinite(v) for v in metrics.values())


def test_metric_names_are_valid_for_mlflow() -> None:
    assert metric_name("test.recall@p0.9") == "test.recall_at_p0.9"
    assert metric_name("validation.decision_cost") == "validation.decision_cost"
    assert metric_name("test.lift[top%]") == "test.lift_top__"


def test_run_tags_mark_synthetic_data(config_path: Path) -> None:
    tags = run_tags(load_config(config_path), synthetic=True)
    assert tags["synthetic_data"] == "true"
    assert tags["model_family"] == "xgboost"


def test_logged_run_round_trip(
    config_path: Path, synthetic_log, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from fraud_detection.tracking import log_training_run

    config = load_config(config_path)
    result = run_training(synthetic_log, config)
    paths = save_artifacts(result, config, tmp_path / "out", synthetic=True)
    run_id = log_training_run(result, config, paths, synthetic=True, run_name="t")

    run = mlflow.MlflowClient(config.tracking.tracking_uri).get_run(run_id)
    assert run.data.params["model.params.n_estimators"] == "40"
    assert run.data.tags["synthetic_data"] == "true"
    assert run.data.metrics["test.pr_auc"] == pytest.approx(result.test["pr_auc"])

    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/scoring_model")
    X = build_features(synthetic_log.head(300))
    scored = model.predict(X)
    expected = result.model.predict_scores(X[result.features])
    np.testing.assert_allclose(scored["score"].to_numpy(), expected)
    flags = (expected >= result.threshold.threshold).astype("int8")
    np.testing.assert_array_equal(scored["flagged"].to_numpy(), flags)


def test_scoring_model_rejects_missing_features(
    config_path: Path, synthetic_log, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from fraud_detection.tracking import log_training_run

    config = load_config(config_path)
    result = run_training(synthetic_log, config)
    paths = save_artifacts(result, config, tmp_path / "out")
    run_id = log_training_run(result, config, paths)

    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/scoring_model")
    X = build_features(synthetic_log.head(50)).drop(columns=result.features[0])
    with pytest.raises(Exception, match="missing feature columns"):
        model.predict(X)


def test_cli_track_flag_logs_run(config_path: Path, tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    out = tmp_path / "cli-out"
    code = cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--synthetic-rows",
            "20000",
            "--output",
            str(out),
            "--track",
            "--run-name",
            "cli",
        ]
    )
    assert code == 0
    config = load_config(config_path)
    client = mlflow.MlflowClient(config.tracking.tracking_uri)
    experiment = client.get_experiment_by_name("unit-test")
    runs = client.search_runs([experiment.experiment_id])
    assert [r.info.run_name for r in runs] == ["cli"]
    assert runs[0].data.tags["synthetic_data"] == "true"
