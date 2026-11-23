import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraud_detection import cli
from fraud_detection.config import load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features
from fraud_detection.models.base import FraudModel
from fraud_detection.train import load_dataset, run_training, save_artifacts

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
        "n_estimators": 60,
        "learning_rate": 0.3,
        "max_depth": 3,
        "n_jobs": 1,
        "early_stopping_rounds": 10,
    }
    raw["resampling"] = {"method": "class_weight", "sampling_strategy": 0.1}
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_run_training_on_synthetic_log(config_path: Path) -> None:
    config = load_config(config_path)
    df = generate_paysim_like(n_rows=6_000, fraud_rate=0.02, n_steps=120, seed=3)
    result = run_training(df, config)

    assert isinstance(result.model, FraudModel)
    assert list(result.splits.index) == ["train", "validation", "test"]
    assert result.splits["rows"].sum() == len(df)
    for report in (result.validation, result.test):
        assert 0 <= report["pr_auc"] <= 1
        assert report["decision_threshold"] == result.threshold.threshold
        assert "calibration_brier" in report
    # The threshold is fitted on validation, so its cost there is reproduced
    assert result.validation["decision_cost"] == pytest.approx(result.threshold.cost)


def test_save_artifacts(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=5)
    result = run_training(df, config)
    paths = save_artifacts(result, config, tmp_path / "out", synthetic=True)

    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["synthetic_data"] is True
    assert metrics["model"] == "xgboost"
    assert metrics["test"]["pr_auc"] == pytest.approx(result.test["pr_auc"])
    assert json.loads(paths["features"].read_text()) == result.features

    with paths["model"].open("rb") as fh:
        model = pickle.load(fh)
    X = build_features(df.head(200))[result.features]
    np.testing.assert_allclose(model.predict_scores(X), result.model.predict_scores(X))


def test_missing_data_file_reports_instructions(config_path: Path) -> None:
    config = load_config(config_path)
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_dataset(config)


def test_cli_train_with_synthetic_rows(config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli_out"
    code = cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--output",
            str(out),
            "--synthetic-rows",
            "20000",
        ]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["synthetic_data"] is True
    assert np.isfinite(metrics["validation"]["pr_auc"])
    assert (out / "model.pkl").stat().st_size > 0


def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
