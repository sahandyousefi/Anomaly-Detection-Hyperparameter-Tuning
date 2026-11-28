import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection import cli
from fraud_detection.config import ModelConfig, ResamplingConfig, load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.monitoring.drift import (
    drift_report,
    drift_statistics,
    markdown_report,
    population_stability_index,
    psi_status,
)
from fraud_detection.serving.artifact import ModelArtifact, save_artifact
from fraud_detection.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_identical_samples_have_zero_psi(rng) -> None:
    x = rng.normal(size=5_000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_shift(rng) -> None:
    reference = rng.normal(size=20_000)
    small = population_stability_index(reference, rng.normal(0.1, 1, 20_000))
    large = population_stability_index(reference, rng.normal(1.0, 1, 20_000))
    assert 0 < small < 0.1
    assert large > 0.25
    assert large > small


def test_psi_matches_hand_computation() -> None:
    reference = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    current = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    e = np.array([0.3, 0.7])
    a = np.array([0.5, 0.5])
    expected = np.sum((a - e) * np.log(a / e))
    assert population_stability_index(reference, current) == pytest.approx(expected)


def test_missing_values_form_their_own_bin() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0] * 50)
    current = reference.copy()
    current[:100] = np.nan
    result = drift_statistics(reference, current)
    assert result.psi > 0.25
    assert result.missing_reference == 0.0
    assert result.missing_current == pytest.approx(0.5)


def test_unseen_discrete_values_counted() -> None:
    reference = np.zeros(1_000)
    current = np.r_[np.zeros(500), np.ones(500)]
    assert population_stability_index(reference, current) > 0.25


def test_all_missing_reference_is_handled() -> None:
    result = drift_statistics(np.full(10, np.nan), np.arange(10.0))
    assert result.psi > 0.25
    assert np.isnan(result.ks_statistic)


def test_ks_detects_shape_change(rng) -> None:
    result = drift_statistics(rng.normal(size=3_000), rng.normal(0.5, 1, 3_000))
    assert result.ks_statistic > 0.1
    assert result.ks_pvalue < 1e-6


def test_empty_sample_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [1.0])


@pytest.mark.parametrize(
    ("psi", "status"),
    [(0.0, "stable"), (0.099, "stable"), (0.1, "moderate"), (0.3, "significant")],
)
def test_psi_status(psi: float, status: str) -> None:
    assert psi_status(psi) == status


def test_report_sorted_by_psi(rng) -> None:
    reference = pd.DataFrame({"a": rng.normal(size=2_000), "b": rng.normal(size=2_000)})
    current = pd.DataFrame(
        {"a": rng.normal(size=2_000), "b": rng.normal(2.0, 1, 2_000)}
    )
    table = drift_report(reference, current)
    assert list(table.index) == ["b", "a"]
    assert table.loc["b", "status"] == "significant"
    text = markdown_report(table, "Test")
    assert text.startswith("# Test")
    assert "| b |" in text


def test_report_requires_matching_columns() -> None:
    with pytest.raises(KeyError, match="b"):
        drift_report(pd.DataFrame({"b": [1.0]}), pd.DataFrame({"a": [1.0]}))


def _saved_artifact(directory: Path) -> Path:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config = replace(
        config,
        model=ModelConfig("xgboost", {"n_estimators": 30, "max_depth": 3, "n_jobs": 1}),
        resampling=ResamplingConfig(method="class_weight"),
    )
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=5)
    result = run_training(df, config)
    artifact = ModelArtifact(result.model, result.schema, result.threshold.threshold)
    save_artifact(artifact, directory)
    return directory


def test_drift_command(tmp_path: Path) -> None:
    artifact_dir = _saved_artifact(tmp_path / "artifact")
    reference = generate_paysim_like(n_rows=1_500, n_steps=40, seed=1)
    current = generate_paysim_like(n_rows=1_500, n_steps=40, seed=2)
    current["amount"] *= 5.0
    reference.to_csv(tmp_path / "reference.csv", index=False)
    current.to_csv(tmp_path / "current.csv", index=False)

    output = tmp_path / "drift"
    code = cli.main(
        [
            "drift",
            "--artifact",
            str(artifact_dir),
            "--reference",
            str(tmp_path / "reference.csv"),
            "--current",
            str(tmp_path / "current.csv"),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    table = pd.read_csv(output / "drift.csv", index_col="variable")
    assert "score" in table.index
    assert table.loc["log_amount", "status"] == "significant"
    summary = json.loads((output / "summary.json").read_text())
    assert "log_amount" in summary["significant_features"]
    assert "score" not in summary["significant_features"]
    assert (output / "drift.md").read_text().startswith("# Drift of current.csv")
