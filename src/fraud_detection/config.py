"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

ResamplingMethod = Literal["none", "undersample", "smote", "class_weight"]
ThresholdStrategy = Literal["cost", "target_recall"]


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or has invalid values."""


@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    artifacts: Path
    figures: Path


@dataclass(frozen=True)
class DataConfig:
    target: str
    time_column: str
    id_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(f <= 0 for f in fractions):
            raise ConfigError("Split fractions must be strictly positive.")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ConfigError(
                f"Split fractions must sum to 1, got {sum(fractions):.6f}."
            )


@dataclass(frozen=True)
class ResamplingConfig:
    method: ResamplingMethod = "none"
    sampling_strategy: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in get_args(ResamplingMethod):
            raise ConfigError(f"Unknown resampling method: {self.method!r}.")
        if not 0 < self.sampling_strategy <= 1:
            raise ConfigError("sampling_strategy must be in (0, 1].")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    threshold_strategy: ThresholdStrategy = "cost"
    cost_false_positive: float = 1.0
    cost_false_negative: float = 1.0
    target_recall: float = 0.9

    def __post_init__(self) -> None:
        if self.threshold_strategy not in get_args(ThresholdStrategy):
            raise ConfigError(
                f"Unknown threshold strategy: {self.threshold_strategy!r}."
            )
        if self.cost_false_positive < 0 or self.cost_false_negative < 0:
            raise ConfigError("Misclassification costs must be non-negative.")
        if not 0 < self.target_recall <= 1:
            raise ConfigError("target_recall must be in (0, 1].")


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    resampling: ResamplingConfig
    model: ModelConfig
    evaluation: EvaluationConfig


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    try:
        value = raw[key]
    except KeyError as exc:
        raise ConfigError(f"Missing configuration section: {key!r}.") from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"Section {key!r} must be a mapping.")
    return value


def _build(cls: type, values: Mapping[str, Any], section: str) -> Any:
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"Invalid keys in section {section!r}: {exc}") from exc


def config_from_dict(raw: Mapping[str, Any], base_dir: Path | None = None) -> Config:
    """Build a :class:`Config` from a parsed mapping.

    Parameters
    ----------
    raw
        Mapping with the same structure as ``configs/default.yaml``.
    base_dir
        Directory that relative paths are resolved against. Paths are left
        relative when ``None``.

    Returns
    -------
    Config
        Validated, immutable configuration.
    """
    root = base_dir or Path()
    paths = {k: root / Path(v) for k, v in _section(raw, "paths").items()}

    data = dict(_section(raw, "data"))
    for key in ("id_columns", "drop_columns"):
        data[key] = tuple(data.get(key) or ())

    model = dict(_section(raw, "model"))
    model["params"] = dict(model.get("params") or {})

    return Config(
        seed=int(raw.get("seed", 42)),
        paths=_build(PathsConfig, paths, "paths"),
        data=_build(DataConfig, data, "data"),
        split=_build(SplitConfig, _section(raw, "split"), "split"),
        resampling=_build(ResamplingConfig, raw.get("resampling") or {}, "resampling"),
        model=_build(ModelConfig, model, "model"),
        evaluation=_build(EvaluationConfig, raw.get("evaluation") or {}, "evaluation"),
    )


def load_config(path: str | Path, base_dir: Path | None = None) -> Config:
    """Read and validate a YAML configuration file.

    Parameters
    ----------
    path
        Path to the YAML file.
    base_dir
        Directory for resolving relative paths. Defaults to the parent of the
        ``configs`` directory, which is the repository root for the default layout.

    Returns
    -------
    Config
        Validated, immutable configuration.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} does not contain a YAML mapping.")
    if base_dir is None:
        base_dir = path.resolve().parent.parent
    return config_from_dict(raw, base_dir=base_dir)
