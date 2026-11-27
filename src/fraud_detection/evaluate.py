"""Model comparison driven by a configuration file.

Every candidate is fitted and evaluated on the same chronological folds with
the protocol of :mod:`fraud_detection.evaluation.model_comparison`: early
stopping and threshold selection on validation, a single pass over the test
fold. The table is sorted by validation PR-AUC; test columns are reported for
information and must not be used to choose a model.

Candidates come from a YAML file (``--candidates``) of the form::

    xgboost_default:
      model: xgboost
    isolation_forest:
      model: isolation_forest
      params: {n_estimators: 300}
      resampling: {method: none}

or, by default, from
:func:`~fraud_detection.evaluation.model_comparison.default_candidates`.
Default candidates whose optional dependency (PyTorch, LightGBM, CatBoost) is
not installed are skipped with a warning. ``--params best_params.yaml`` adds
the tuned model as one more candidate.

Written to the output directory: ``model_comparison.csv``, the same table as
``model_comparison.md``, and ``comparison_run.json`` with the data source and
seed.

Usage::

    fraud-detection evaluate --config configs/default.yaml
    fraud-detection evaluate --params artifacts/tuning/best_params.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud_detection import __version__
from fraud_detection.config import (
    Config,
    ConfigError,
    ResamplingConfig,
    load_config,
    load_model_override,
)
from fraud_detection.data.split import time_split
from fraud_detection.evaluation.model_comparison import (
    Candidate,
    compare_models,
    default_candidates,
)
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.features.sequence import history_columns
from fraud_detection.reproducibility import seed_everything
from fraud_detection.train import DEFAULT_CONFIG, load_dataset

logger = logging.getLogger(__name__)

# Optional modules each model family needs at fit time
MODEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "feedforward": ("torch",),
    "gru": ("torch",),
    "autoencoder": ("torch",),
    "variational_autoencoder": ("torch",),
    "hybrid": ("torch",),
}


def missing_dependencies(model: str) -> list[str]:
    """Optional modules required by ``model`` that are not installed."""
    required = MODEL_DEPENDENCIES.get(model, ())
    return [m for m in required if importlib.util.find_spec(m) is None]


def available_candidates(
    candidates: Mapping[str, Candidate],
) -> dict[str, Candidate]:
    """Drop candidates whose optional dependencies are not installed."""
    kept = {}
    for name, candidate in candidates.items():
        missing = missing_dependencies(candidate.model)
        if missing:
            logger.warning("Skipping %s: %s not installed", name, ", ".join(missing))
            continue
        kept[name] = candidate
    return kept


def _candidate_from_mapping(name: str, spec: Any) -> Candidate:
    if not isinstance(spec, Mapping) or "model" not in spec:
        raise ConfigError(f"Candidate {name!r} needs a mapping with a model key.")
    unknown = set(spec) - {"model", "params", "resampling"}
    if unknown:
        raise ConfigError(f"Candidate {name!r} has unknown keys: {sorted(unknown)}.")
    resampling = spec.get("resampling")
    return Candidate(
        model=spec["model"],
        params=dict(spec.get("params") or {}),
        resampling=ResamplingConfig(**resampling) if resampling else None,
    )


def load_candidates(path: str | Path) -> dict[str, Candidate]:
    """Read named candidates from a YAML mapping.

    Parameters
    ----------
    path
        YAML file mapping each display name to ``model`` and optional
        ``params`` and ``resampling`` entries.

    Returns
    -------
    dict
        Candidates keyed by display name, in file order.
    """
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"{path} must map candidate names to settings.")
    return {str(k): _candidate_from_mapping(str(k), v) for k, v in raw.items()}


def run_comparison(
    df: pd.DataFrame,
    config: Config,
    candidates: Mapping[str, Candidate] | None = None,
) -> pd.DataFrame:
    """Build features, split on time, and compare candidates.

    Parameters
    ----------
    df
        Raw transactions in PaySim layout.
    config
        Experiment configuration; supplies features, split, resampling,
        threshold strategy, costs, and seed.
    candidates
        Named candidates. Defaults to :func:`default_candidates`, including the
        GRU when ``features.history_length`` is positive, restricted to the
        installed dependencies.

    Returns
    -------
    pandas.DataFrame
        Comparison table, see
        :func:`~fraud_detection.evaluation.model_comparison.compare_models`.
    """
    seed_everything(config.seed)
    shared = feature_columns(replace(config.features, history_length=0))
    history = history_columns(
        config.features.history_length, config.features.history_roles
    )
    if candidates is None:
        candidates = available_candidates(default_candidates(history))
    table = build_features(df, config.features, config.data.time_column)
    split = time_split(table, config.split, config.data.time_column)
    logger.info("Comparing %d candidates: %s", len(candidates), ", ".join(candidates))
    return compare_models(
        split,
        shared,
        config.data.target,
        config.evaluation,
        candidates,
        config.resampling,
        config.seed,
    )


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def markdown_table(table: pd.DataFrame) -> str:
    """Render the comparison table as GitHub-flavoured Markdown."""
    frame = table.reset_index()
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(_format_cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def save_comparison(
    table: pd.DataFrame, config: Config, output_dir: Path, synthetic: bool = False
) -> dict[str, Path]:
    """Write the comparison table as CSV and Markdown, plus run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "model_comparison.csv",
        "markdown": output_dir / "model_comparison.md",
        "run": output_dir / "comparison_run.json",
    }
    table.to_csv(paths["csv"])
    paths["markdown"].write_text(markdown_table(table))
    run = {
        "package_version": __version__,
        "synthetic_data": synthetic,
        "seed": config.seed,
        "resampling": config.resampling.method,
        "threshold_strategy": config.evaluation.threshold_strategy,
        "candidates": list(table.index),
    }
    paths["run"].write_text(json.dumps(run, indent=2) + "\n")
    return paths


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options of the ``evaluate`` command."""
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file (default: %(default)s).",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="YAML file of named candidates (default: one per model family).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="best_params.yaml from the tune command, added as a candidate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <paths.artifacts>/evaluation).",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on N synthetic rows instead of the PaySim file.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Execute the ``evaluate`` command from parsed arguments."""
    config = load_config(args.config)
    candidates = None
    if args.candidates is not None:
        candidates = available_candidates(load_candidates(args.candidates))
    if args.params is not None:
        history = history_columns(
            config.features.history_length, config.features.history_roles
        )
        candidates = dict(
            candidates or available_candidates(default_candidates(history))
        )
        tuned = load_model_override(args.params)
        candidates[f"{tuned.name}_tuned"] = Candidate(tuned.name, dict(tuned.params))
    if candidates is not None and not candidates:
        raise ConfigError("No candidate can run with the installed dependencies.")
    df = load_dataset(config, args.synthetic_rows)
    table = run_comparison(df, config, candidates)
    output = args.output or config.paths.artifacts / "evaluation"
    paths = save_comparison(table, config, output, args.synthetic_rows is not None)
    best = table.index[0]
    logger.info(
        "Best validation PR-AUC: %s (%.4f); table in %s",
        best,
        table.loc[best, "validation_pr_auc"],
        paths["markdown"],
    )
    return 0
