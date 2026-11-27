import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make missing")


def dry_run(*args: str) -> str:
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_help_lists_pipeline_targets() -> None:
    result = subprocess.run(
        ["make", "--no-print-directory", "help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    names = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
    assert {"data", "train", "tune", "evaluate", "test", "smoke"} <= names


def test_real_data_targets_depend_on_the_raw_file(tmp_path: Path) -> None:
    raw = tmp_path / "missing.csv"
    commands = dry_run("train", f"RAW_DATA={raw}")
    assert "kaggle datasets download" in commands
    assert "--synthetic-rows" not in commands


def test_synthetic_runs_skip_the_download() -> None:
    commands = dry_run(
        "tune", "SYNTHETIC_ROWS=500", "TRIALS=2", "FIGURES=out/fig", "ARTIFACTS=out"
    )
    assert "kaggle" not in commands
    assert (
        "fraud_detection.cli tune --config configs/default.yaml --output out/tuning"
        in commands
    )
    assert "--n-trials 2 --figures out/fig --synthetic-rows 500" in commands
