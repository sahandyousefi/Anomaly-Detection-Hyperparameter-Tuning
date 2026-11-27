"""End-to-end training run driven by a configuration file.

The run follows the evaluation protocol of the package:

1. build features on the full chronological log (row-level and point-in-time
   only, so no fold sees later information);
2. split on ``step`` into train, validation, and test;
3. rebalance the training fold only;
4. fit the configured model, early-stopping on validation where supported;
5. choose the decision threshold on validation;
6. score the test fold once, at the chosen threshold.

Global random generators are seeded with ``config.seed`` before step 1.

Artifacts written to the output directory are the pickled model, the
resolved feature list, ``metrics.json`` with validation and test results, and
``split_summary.csv``. Numbers produced with ``--synthetic-rows`` come from
the synthetic generator and say nothing about performance on PaySim. With
``--track`` (or ``tracking.enabled`` in the config) the run is also logged to
MLflow.

Usage::

    fraud-detection train --config configs/default.yaml
    fraud-detection train --config configs/default.yaml --synthetic-rows 50000
    fraud-detection train --config configs/default.yaml --track --run-name base
    fraud-detection train --params artifacts/tuning/best_params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection import __version__
from fraud_detection.config import Config, load_config, load_model_override
from fraud_detection.data.load import load_transactions
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import split_summary, time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.data.versioning import data_fingerprint
from fraud_detection.evaluation.calibration import calibration_report
from fraud_detection.evaluation.metrics import ranking_report
from fraud_detection.evaluation.thresholds import (
    ThresholdChoice,
    evaluate_threshold,
    select_threshold,
)
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.models.base import FraudModel, fit_model
from fraud_detection.reproducibility import environment_info, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass(frozen=True)
class TrainingResult:
    """Fitted model and its evaluation.

    Attributes
    ----------
    model
        Fitted model behind the common interface.
    features
        Model input columns, in training order.
    threshold
        Decision threshold chosen on the validation fold.
    validation, test
        Ranking metrics, calibration summary, and cost at the chosen
        threshold, per fold.
    splits
        Size, positives, and step range of each fold.
    """

    model: FraudModel
    features: list[str]
    threshold: ThresholdChoice
    validation: dict[str, float]
    test: dict[str, float]
    splits: pd.DataFrame


def load_dataset(config: Config, synthetic_rows: int | None = None) -> pd.DataFrame:
    """Read the configured PaySim file, or generate a synthetic log.

    Parameters
    ----------
    config
        Source of ``paths.raw_data`` and the seed.
    synthetic_rows
        When set, generate this many synthetic rows instead of reading the
        file. Used for smoke tests and CI.

    Returns
    -------
    pandas.DataFrame
        Transactions in PaySim layout.
    """
    if synthetic_rows is not None:
        logger.info("Generating %d synthetic transactions", synthetic_rows)
        return generate_paysim_like(n_rows=synthetic_rows, seed=config.seed)
    logger.info("Reading %s", config.paths.raw_data)
    return load_transactions(config.paths.raw_data)


def _fold_report(
    y: pd.Series, scores: Any, choice: ThresholdChoice, config: Config
) -> dict[str, float]:
    return {
        **ranking_report(y, scores),
        **{f"calibration_{k}": v for k, v in calibration_report(y, scores).items()},
        **{
            f"decision_{k}": v
            for k, v in evaluate_threshold(y, scores, choice, config.evaluation).items()
        },
    }


def run_training(df: pd.DataFrame, config: Config) -> TrainingResult:
    """Train and evaluate the configured model on a transaction log.

    Parameters
    ----------
    df
        Raw transactions in PaySim layout.
    config
        Experiment configuration.

    Returns
    -------
    TrainingResult
        Fitted model with validation and test results.
    """
    seed_everything(config.seed)
    target = config.data.target
    features = feature_columns(config.features)
    table = build_features(df, config.features, config.data.time_column)
    split = time_split(table, config.split, config.data.time_column)
    folds = prepare_modeling_folds(
        split, features, target, config.resampling, config.seed
    )
    logger.info(
        "Fitting %s on %d rows (%d positives)",
        config.model.name,
        len(folds.train.y),
        int(folds.train.y.sum()),
    )
    model = fit_model(
        config.model.name,
        folds.train,
        folds.X_validation,
        folds.y_validation,
        config.model.params,
        seed=config.seed,
    )
    validation_scores = model.predict_scores(folds.X_validation)
    choice = select_threshold(folds.y_validation, validation_scores, config.evaluation)
    test_scores = model.predict_scores(folds.X_test)
    return TrainingResult(
        model=model,
        features=features,
        threshold=choice,
        validation=_fold_report(folds.y_validation, validation_scores, choice, config),
        test=_fold_report(folds.y_test, test_scores, choice, config),
        splits=split_summary(split, target),
    )


def _json_safe(value: Any) -> Any:
    # JSON has no inf or NaN
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def save_artifacts(
    result: TrainingResult,
    config: Config,
    output_dir: Path,
    synthetic: bool = False,
    data_sha256: str | None = None,
) -> dict[str, Path]:
    """Write the model, features, metrics, and split summary to disk.

    Parameters
    ----------
    result
        Output of :func:`run_training`.
    config
        Configuration of the run, recorded in ``metrics.json``.
    output_dir
        Directory to write to; created if missing.
    synthetic
        Whether the data came from the synthetic generator, recorded so that
        the metrics are not mistaken for PaySim results.
    data_sha256
        Digest of the raw data file, see
        :func:`fraud_detection.data.versioning.data_fingerprint`.

    Returns
    -------
    dict
        Artifact name mapped to its path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / "model.pkl",
        "features": output_dir / "features.json",
        "metrics": output_dir / "metrics.json",
        "splits": output_dir / "split_summary.csv",
    }
    with paths["model"].open("wb") as fh:
        pickle.dump(result.model, fh)
    paths["features"].write_text(json.dumps(result.features, indent=2) + "\n")
    metrics = {
        "package_version": __version__,
        "synthetic_data": synthetic,
        "data_sha256": data_sha256,
        "model": config.model.name,
        "model_params": dict(config.model.params),
        "resampling": asdict(config.resampling),
        "seed": config.seed,
        "environment": environment_info(),
        "threshold": asdict(result.threshold),
        "validation": result.validation,
        "test": result.test,
    }
    paths["metrics"].write_text(json.dumps(_json_safe(metrics), indent=2) + "\n")
    result.splits.to_csv(paths["splits"])
    return paths


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options of the ``train`` command."""
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artifact directory (default: paths.artifacts from the config).",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=None,
        metavar="N",
        help="Train on N synthetic rows instead of the PaySim file.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="YAML model section, such as best_params.yaml from the tune command,"
        " that replaces the model section of the config.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log the run to MLflow, regardless of tracking.enabled.",
    )
    parser.add_argument(
        "--run-name", default=None, help="Display name of the MLflow run."
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Execute the ``train`` command from parsed arguments."""
    config = load_config(args.config)
    if args.params is not None:
        config = replace(config, model=load_model_override(args.params))
    df = load_dataset(config, args.synthetic_rows)
    result = run_training(df, config)
    output = args.output or config.paths.artifacts
    synthetic = args.synthetic_rows is not None
    digest = None if synthetic else data_fingerprint(config.paths.raw_data)
    paths = save_artifacts(result, config, output, synthetic, digest)
    test = result.test
    logger.info(
        "Test PR-AUC %.4f, recall %.3f, precision %.3f at threshold %.4g",
        test["pr_auc"],
        test["decision_recall"],
        test["decision_precision"],
        result.threshold.threshold,
    )
    logger.info("Artifacts written to %s", paths["metrics"].parent)
    if args.track or config.tracking.enabled:
        from fraud_detection.tracking import log_training_run

        run_id = log_training_run(
            result, config, paths, synthetic, args.run_name, data_sha256=digest
        )
        logger.info("Logged MLflow run %s", run_id)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraud_detection.train``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
