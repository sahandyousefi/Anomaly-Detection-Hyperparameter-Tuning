from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fraud_detection.config import ConfigError, config_from_dict, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def raw_config() -> dict:
    with DEFAULT_CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.data.target == "isFraud"
    assert cfg.data.id_columns == ("nameOrig", "nameDest")
    assert cfg.model.params["max_depth"] == 7
    assert (
        cfg.paths.raw_data
        == REPO_ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
    )


def test_relative_paths_kept_without_base_dir(raw_config: dict) -> None:
    cfg = config_from_dict(raw_config)
    assert not cfg.paths.artifacts.is_absolute()


def test_config_is_immutable() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    with pytest.raises(AttributeError):
        cfg.seed = 0  # type: ignore[misc]


def test_split_fractions_must_sum_to_one(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    raw["split"]["test_fraction"] = 0.30
    with pytest.raises(ConfigError, match="sum to 1"):
        config_from_dict(raw)


def test_split_fractions_must_be_positive(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    raw["split"].update(train_fraction=1.0, validation_fraction=0.0, test_fraction=0.0)
    with pytest.raises(ConfigError, match="strictly positive"):
        config_from_dict(raw)


def test_unknown_resampling_method_rejected(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    raw["resampling"]["method"] = "oversample_everything"
    with pytest.raises(ConfigError, match="resampling method"):
        config_from_dict(raw)


def test_negative_cost_rejected(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    raw["evaluation"]["cost_false_negative"] = -1
    with pytest.raises(ConfigError, match="non-negative"):
        config_from_dict(raw)


def test_missing_section_reported(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    del raw["split"]
    with pytest.raises(ConfigError, match="'split'"):
        config_from_dict(raw)


def test_unexpected_key_reported(raw_config: dict) -> None:
    raw = deepcopy(raw_config)
    raw["data"]["label"] = "isFraud"
    with pytest.raises(ConfigError, match="'data'"):
        config_from_dict(raw)
