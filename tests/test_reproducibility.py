import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraud_detection.config import load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features
from fraud_detection.reproducibility import environment_info, seed_everything
from fraud_detection.train import run_training, save_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"] = {"raw_data": "missing.csv", "artifacts": "out", "figures": "fig"}
    raw["model"]["params"] = {
        "n_estimators": 50,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 10,
    }
    raw["resampling"] = {"method": "undersample", "sampling_strategy": 0.2}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _draws() -> tuple[float, float]:
    return random.random(), float(np.random.rand())  # noqa: NPY002


def test_seed_everything_repeats_global_draws() -> None:
    seed_everything(7)
    first = _draws()
    seed_everything(7)
    assert _draws() == first
    seed_everything(8)
    assert _draws() != first


def test_seed_everything_exports_hash_seed() -> None:
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_seed_everything_seeds_torch() -> None:
    torch = pytest.importorskip("torch")
    seeded = seed_everything(3)
    assert seeded["torch"]
    first = torch.rand(4)
    seed_everything(3)
    assert torch.equal(torch.rand(4), first)
    assert torch.are_deterministic_algorithms_enabled()
    seed_everything(3, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()


def test_environment_info_lists_installed_packages() -> None:
    info = environment_info()
    assert info["numpy"] == np.__version__
    assert {"python", "platform", "scikit-learn", "xgboost"} <= info.keys()
    assert "not-a-real-package" not in environment_info(("not-a-real-package",))


def test_training_is_repeatable(config_path: Path) -> None:
    config = load_config(config_path)
    df = generate_paysim_like(n_rows=5_000, fraud_rate=0.02, n_steps=120, seed=21)
    X = build_features(df.head(500))
    runs = [run_training(df, config) for _ in range(2)]
    first, second = (r.model.predict_scores(X[r.features]) for r in runs)
    np.testing.assert_array_equal(first, second)
    assert runs[0].test == pytest.approx(runs[1].test, nan_ok=True)
    assert runs[0].threshold == runs[1].threshold


def test_metrics_record_environment(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    df = generate_paysim_like(n_rows=3_000, fraud_rate=0.03, n_steps=90, seed=4)
    paths = save_artifacts(run_training(df, config), config, tmp_path / "out")
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["environment"]["xgboost"] == environment_info()["xgboost"]
