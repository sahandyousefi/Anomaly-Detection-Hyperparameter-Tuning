r"""Bayesian hyperparameter search for XGBoost with time-series cross-validation.

Hyperparameters are chosen with Optuna (Akiba et al., 2019) using the
tree-structured Parzen estimator (Bergstra et al., 2011). Only the training
and validation periods of the chronological split are used; the test period
is held out from every trial.

The tuning period is cut into :math:`K + 1` consecutive blocks of hourly
steps. Fold :math:`k = 1, \dots, K` trains on blocks :math:`1, \dots, k`
(expanding window) and is scored on block :math:`k + 1`, so every score is
computed on transactions that occur after all of the fold's training rows.
Optional gap steps between the training window and the scored block reduce
leakage through short-range dependence. The objective of a trial with
parameters :math:`\theta` is the mean fold PR-AUC,

.. math::

    J(\theta) = \frac{1}{K} \sum_{k=1}^{K}
        \mathrm{AP}\big(y^{(k+1)}, \hat{s}_\theta^{(k)}\big),

where :math:`\hat{s}_\theta^{(k)}` are the scores of the model fitted on fold
:math:`k`. Rebalancing is applied inside each fold to the training window
only, and the scored block keeps its observed class ratio.

The number of boosting rounds is set by early stopping on the scored block of
each fold. This makes the fold score mildly optimistic, by the same mechanism
for every trial, so it does not favour particular parameters strongly; the
final estimate of generalization comes from the untouched test period.

References
----------
Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. KDD.

Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). Algorithms for
hyper-parameter optimization. NeurIPS.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
KDD.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fraud_detection.config import ResamplingConfig
from fraud_detection.data.resample import resample_training_fold
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.models.gbm import fit_xgboost

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True)
class TimeFold:
    """Positional row indices of one expanding-window fold."""

    train: np.ndarray
    validation: np.ndarray
    train_steps: tuple[int, int]
    validation_steps: tuple[int, int]


def expanding_window_folds(
    steps: pd.Series | np.ndarray,
    n_splits: int = 4,
    gap_steps: int = 0,
) -> list[TimeFold]:
    """Expanding-window folds over consecutive blocks of distinct steps.

    Parameters
    ----------
    steps
        Time index of each row of the tuning period.
    n_splits
        Number of folds :math:`K`. The step range is cut into ``n_splits + 1``
        blocks of nearly equal numbers of distinct steps.
    gap_steps
        Steps at the end of each training window that are left out, so the
        last training step precedes the scored block by ``gap_steps + 1``.

    Returns
    -------
    list of TimeFold
        Folds in chronological order of their scored block.

    Raises
    ------
    ValueError
        If there are too few distinct steps for the requested folds.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    if gap_steps < 0:
        raise ValueError("gap_steps must be non-negative.")
    values = np.asarray(steps)
    unique = np.unique(values)
    if len(unique) < (n_splits + 1) * (gap_steps + 1):
        raise ValueError(
            f"{len(unique)} distinct steps cannot form {n_splits} folds "
            f"with a gap of {gap_steps}."
        )
    blocks = np.array_split(unique, n_splits + 1)
    folds = []
    for k in range(1, n_splits + 1):
        train_last = blocks[k - 1][-1] - gap_steps
        block = blocks[k]
        train_idx = np.flatnonzero(values <= train_last)
        val_idx = np.flatnonzero((values >= block[0]) & (values <= block[-1]))
        if train_idx.size == 0:
            raise ValueError(f"Fold {k} has no training rows after the gap.")
        folds.append(
            TimeFold(
                train=train_idx,
                validation=val_idx,
                train_steps=(int(unique[0]), int(train_last)),
                validation_steps=(int(block[0]), int(block[-1])),
            )
        )
    return folds


def suggest_xgboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one XGBoost configuration.

    Learning rate and regularization strengths are sampled on a log scale,
    since their effect is multiplicative. ``n_estimators`` is not tuned; it is
    an upper bound and early stopping picks the number of rounds.

    Parameters
    ----------
    trial
        Active Optuna trial.

    Returns
    -------
    dict
        Overrides of :data:`fraud_detection.models.gbm.XGBOOST_DEFAULTS`.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.5, 50.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


@dataclass(frozen=True)
class FoldScore:
    """Validation result of one fold within a trial."""

    pr_auc: float
    best_iteration: int


def score_fold(
    X: pd.DataFrame,
    y: pd.Series,
    fold: TimeFold,
    params: Mapping[str, Any],
    resampling: ResamplingConfig,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> FoldScore:
    """Fit XGBoost on a fold's training window and score its next block.

    Parameters
    ----------
    X, y
        Features and labels of the whole tuning period.
    fold
        Positional indices of the training window and the scored block.
    params
        XGBoost parameters for this trial.
    resampling
        Rebalancing applied to the training window only.
    early_stopping_rounds
        Patience of early stopping on the scored block.
    seed
        Random state of the sampler and the booster.

    Returns
    -------
    FoldScore
        PR-AUC on the scored block and the selected number of rounds.
    """
    train = resample_training_fold(
        X.iloc[fold.train], y.iloc[fold.train], resampling, seed
    )
    X_val, y_val = X.iloc[fold.validation], y.iloc[fold.validation]
    model = fit_xgboost(train, X_val, y_val, params, early_stopping_rounds, seed)
    return FoldScore(
        pr_auc=pr_auc(y_val, model.predict_scores(X_val)),
        best_iteration=model.best_iteration,
    )


def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[TimeFold],
    resampling: ResamplingConfig,
    fixed_params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    seed: int = 0,
    search_space: Callable[[optuna.Trial], dict[str, Any]] = suggest_xgboost_params,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective: mean PR-AUC over expanding-window folds.

    Parameters
    ----------
    X, y
        Features and labels of the tuning period, in chronological order.
    folds
        Output of :func:`expanding_window_folds` for the same rows.
    resampling
        Rebalancing applied inside each fold.
    fixed_params
        Parameters held constant across trials, such as ``n_estimators`` or
        ``n_jobs``. Sampled parameters take precedence.
    early_stopping_rounds
        Patience of early stopping.
    seed
        Random state for every fold fit.
    search_space
        Function sampling the tuned parameters from a trial.

    Returns
    -------
    callable
        Objective to maximize. Per-fold PR-AUC and best rounds are stored as
        trial user attributes.
    """
    if not folds:
        raise ValueError("At least one fold is required.")

    def objective(trial: optuna.Trial) -> float:
        params = {**(fixed_params or {}), **search_space(trial)}
        scores = [
            score_fold(X, y, fold, params, resampling, early_stopping_rounds, seed)
            for fold in folds
        ]
        values = [s.pr_auc for s in scores]
        trial.set_user_attr("fold_pr_auc", values)
        trial.set_user_attr("fold_best_iteration", [s.best_iteration for s in scores])
        trial.set_user_attr("pr_auc_std", float(np.std(values)))
        return float(np.mean(values))

    return objective


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    steps: pd.Series | np.ndarray,
    resampling: ResamplingConfig,
    n_trials: int = 50,
    n_splits: int = 4,
    gap_steps: int = 0,
    fixed_params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    timeout: float | None = None,
    study_name: str = "xgboost_pr_auc",
    seed: int = 0,
) -> optuna.Study:
    """Run a TPE search for XGBoost on the tuning period.

    Parameters
    ----------
    X, y
        Features and labels of the training and validation periods, sorted by
        time. Test rows must not be passed.
    steps
        Time index aligned with ``X``.
    resampling
        Rebalancing applied inside each fold.
    n_trials
        Number of trials.
    n_splits, gap_steps
        Fold layout, see :func:`expanding_window_folds`.
    fixed_params
        Parameters held constant across trials.
    early_stopping_rounds
        Patience of early stopping.
    timeout
        Optional wall-clock limit in seconds.
    study_name
        Name recorded on the study.
    seed
        Seed of the TPE sampler and of every fold fit.

    Returns
    -------
    optuna.Study
        Completed study; ``best_params`` holds the sampled parameters only.
    """
    import optuna

    folds = expanding_window_folds(steps, n_splits, gap_steps)
    objective = make_objective(
        X, y, folds, resampling, fixed_params, early_stopping_rounds, seed
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
