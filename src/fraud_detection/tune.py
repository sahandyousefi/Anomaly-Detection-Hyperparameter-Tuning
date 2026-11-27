"""Hyperparameter search driven by a configuration file.

Features are built on the full chronological log, the log is split on
``step``, and only the training and validation periods are passed to the
search; the test period never reaches a trial. The search itself is
:func:`fraud_detection.tuning.optuna_search.tune_model`: a TPE sampler, mean
PR-AUC over expanding-window folds as the objective, and median pruning.

Written to the output directory:

* ``best_params.yaml``: model family and parameters of the best trial,
  merged over ``config.model.params``. Pass it to
  ``fraud-detection train --params`` to refit on the full training period.
* ``trials.csv``: every trial with its state, objective, and parameters.
* ``study.db``: SQLite storage, so running the command again extends the
  same study instead of starting over (disable with ``--in-memory``).

Optimization history, parameter importance, and slice plots go to
``paths.figures`` unless ``--figures`` is given. The ``tune`` extra (Optuna)
is required.

Usage::

    fraud-detection tune --config configs/default.yaml
    fraud-detection tune --config configs/default.yaml --n-trials 200
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from fraud_detection.config import Config, load_config
from fraud_detection.data.split import time_split
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.reproducibility import seed_everything
from fraud_detection.train import DEFAULT_CONFIG, load_dataset

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# Patience is a fit argument, not a model parameter
_NON_MODEL_PARAMS = ("early_stopping_rounds",)


def tuning_period(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features, labels, and steps of the training and validation periods.

    Parameters
    ----------
    df
        Raw transactions in PaySim layout.
    config
        Experiment configuration.

    Returns
    -------
    tuple
        ``(X, y, steps)`` in chronological order, without any test row.
    """
    time_column = config.data.time_column
    table = build_features(df, config.features, time_column)
    split = time_split(table, config.split, time_column)
    period = pd.concat([split.train, split.validation]).sort_values(
        time_column, kind="stable"
    )
    features = feature_columns(config.features)
    return period[features], period[config.data.target], period[time_column]


def run_tuning(
    df: pd.DataFrame, config: Config, storage: str | None = None
) -> optuna.Study:
    """Search hyperparameters of ``config.model.name`` on the tuning period.

    Parameters
    ----------
    df
        Raw transactions in PaySim layout.
    config
        Experiment configuration; ``config.tuning`` sets the search budget and
        ``config.model.params`` the parameters held fixed.
    storage
        Optuna storage URL; in memory when ``None``.

    Returns
    -------
    optuna.Study
        Study after the new trials.
    """
    from fraud_detection.tuning.optuna_search import tune_model

    seed_everything(config.seed)
    X, y, steps = tuning_period(df, config)
    fixed = {k: v for k, v in config.model.params.items() if k not in _NON_MODEL_PARAMS}
    settings = config.tuning
    logger.info(
        "Tuning %s on %d rows (%d positives), %d trials over %d folds",
        config.model.name,
        len(y),
        int(y.sum()),
        settings.n_trials,
        settings.n_splits,
    )
    return tune_model(
        config.model.name,
        X,
        y,
        steps,
        config.resampling,
        n_trials=settings.n_trials,
        n_splits=settings.n_splits,
        gap_steps=settings.gap_steps,
        fixed_params=fixed,
        early_stopping_rounds=settings.early_stopping_rounds,
        timeout=settings.timeout,
        storage=storage,
        seed=config.seed,
    )


def best_params_document(study: optuna.Study, config: Config) -> dict[str, Any]:
    """Best trial as a ``model`` section that ``train --params`` accepts.

    Parameters
    ----------
    study
        Study with at least one completed trial.
    config
        Configuration whose ``model.params`` the trial parameters override.

    Returns
    -------
    dict
        ``model`` (name and merged parameters) and ``search`` (objective
        value, trial number, and trial counts) sections.
    """
    from optuna.trial import TrialState

    try:
        best = study.best_trial
    except ValueError as exc:
        raise ValueError("No trial of the study completed.") from exc
    states = [t.state for t in study.trials]
    return {
        "model": {
            "name": config.model.name,
            "params": {**config.model.params, **best.params},
        },
        "search": {
            "study_name": study.study_name,
            "objective": "mean expanding-window PR-AUC",
            "best_value": float(best.value),
            "best_trial": best.number,
            "completed_trials": states.count(TrialState.COMPLETE),
            "pruned_trials": states.count(TrialState.PRUNED),
        },
    }


def save_tuning_outputs(
    study: optuna.Study, config: Config, output_dir: Path, figures_dir: Path
) -> dict[str, Path]:
    """Write the best parameters, the trial table, and the study figures.

    Returns
    -------
    dict
        Output name mapped to its path; figures are keyed ``figure_<kind>``.
    """
    from fraud_detection.tuning.study_plots import save_study_figures

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "best_params": output_dir / "best_params.yaml",
        "trials": output_dir / "trials.csv",
    }
    document = best_params_document(study, config)
    paths["best_params"].write_text(yaml.safe_dump(document, sort_keys=False))
    study.trials_dataframe().to_csv(paths["trials"], index=False)
    figures = save_study_figures(study, figures_dir, seed=config.seed)
    paths.update({f"figure_{kind}": path for kind, path in figures.items()})
    return paths


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options of the ``tune`` command."""
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
        help="Output directory (default: <paths.artifacts>/tuning).",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="Directory for the study figures (default: paths.figures).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override tuning.n_trials from the config.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Keep the study in memory instead of extending study.db.",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=None,
        metavar="N",
        help="Tune on N synthetic rows instead of the PaySim file.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Execute the ``tune`` command from parsed arguments."""
    from fraud_detection.tuning.optuna_search import sqlite_storage

    config = load_config(args.config)
    if args.n_trials is not None:
        config = replace(config, tuning=replace(config.tuning, n_trials=args.n_trials))
    output = args.output or config.paths.artifacts / "tuning"
    storage = None if args.in_memory else sqlite_storage(output / "study.db")
    df = load_dataset(config, args.synthetic_rows)
    study = run_tuning(df, config, storage)
    figures = args.figures or config.paths.figures
    paths = save_tuning_outputs(study, config, output, figures)
    logger.info(
        "Best mean PR-AUC %.4f (trial %d); parameters in %s",
        study.best_value,
        study.best_trial.number,
        paths["best_params"],
    )
    return 0
