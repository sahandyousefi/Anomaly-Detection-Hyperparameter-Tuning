import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_detection import cli
from fraud_detection.config import ConfigError, load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.evaluate import (
    available_candidates,
    load_candidates,
    markdown_table,
    missing_dependencies,
    run_comparison,
)
from fraud_detection.evaluation.model_comparison import Candidate

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"] = {
        "raw_data": "data/missing.csv",
        "artifacts": "artifacts",
        "figures": "figures",
    }
    raw["resampling"] = {"method": "class_weight", "sampling_strategy": 0.1}
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def candidates_path(tmp_path: Path) -> Path:
    spec = {
        "logit": {"model": "logistic_regression"},
        "xgb_small": {
            "model": "xgboost",
            "params": {"n_estimators": 40, "max_depth": 3, "n_jobs": 1},
            "resampling": {"method": "undersample", "sampling_strategy": 0.2},
        },
    }
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_load_candidates(candidates_path: Path) -> None:
    candidates = load_candidates(candidates_path)
    assert list(candidates) == ["logit", "xgb_small"]
    assert candidates["xgb_small"].params["max_depth"] == 3
    assert candidates["xgb_small"].resampling.method == "undersample"
    assert candidates["logit"].resampling is None


@pytest.mark.parametrize(
    "spec",
    [
        {"a": {"params": {}}},
        {"a": {"model": "xgboost", "weights": 2}},
        {"a": "xgboost"},
        {},
    ],
)
def test_invalid_candidate_files_rejected(tmp_path: Path, spec: dict) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ConfigError):
        load_candidates(path)


def test_candidates_with_missing_dependencies_are_skipped(monkeypatch) -> None:
    import fraud_detection.evaluate as evaluate

    monkeypatch.setitem(evaluate.MODEL_DEPENDENCIES, "xgboost", ("no_such_module",))
    assert missing_dependencies("xgboost") == ["no_such_module"]
    kept = available_candidates(
        {"x": Candidate("xgboost"), "lr": Candidate("logistic_regression")}
    )
    assert list(kept) == ["lr"]


def test_markdown_table_layout() -> None:
    table = pd.DataFrame(
        {"family": ["supervised"], "test_pr_auc": [0.123456], "threshold": [None]},
        index=pd.Index(["xgboost"], name="candidate"),
    ).astype({"threshold": "float64"})
    lines = markdown_table(table).splitlines()
    assert lines[0] == "| candidate | family | test_pr_auc | threshold |"
    assert lines[1] == "|---|---|---|---|"
    assert lines[2] == "| xgboost | supervised | 0.1235 |  |"


def test_run_comparison_uses_config_folds(
    config_path: Path, candidates_path: Path
) -> None:
    config = load_config(config_path)
    df = generate_paysim_like(n_rows=6_000, fraud_rate=0.03, n_steps=120, seed=2)
    table = run_comparison(df, config, load_candidates(candidates_path))
    assert set(table.index) == {"logit", "xgb_small"}
    assert table["validation_pr_auc"].is_monotonic_decreasing
    assert (table["n_features"] == table["n_features"].iloc[0]).all()


def test_evaluate_command_writes_outputs(
    config_path: Path, candidates_path: Path, tmp_path: Path
) -> None:
    params = tmp_path / "best_params.yaml"
    params.write_text(
        yaml.safe_dump({"model": {"name": "xgboost", "params": {"n_estimators": 30}}})
    )
    out = tmp_path / "evaluation"
    code = cli.main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--candidates",
            str(candidates_path),
            "--params",
            str(params),
            "--synthetic-rows",
            "20000",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    table = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert set(table.index) == {"logit", "xgb_small", "xgboost_tuned"}
    assert (out / "model_comparison.md").read_text().startswith("| candidate |")
    run = json.loads((out / "comparison_run.json").read_text())
    assert run["synthetic_data"] is True
    assert sorted(run["candidates"]) == sorted(table.index)
