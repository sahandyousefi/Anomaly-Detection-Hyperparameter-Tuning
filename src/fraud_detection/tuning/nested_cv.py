r"""Nested time-series evaluation of the tuning procedure.

The mean fold PR-AUC of the best trial is a biased estimate of performance:
it is the maximum over many noisy estimates, all computed on the same blocks
that drove the search. Nested evaluation removes this bias by treating the
whole search as part of the estimator (Varma and Simon, 2006; Cawley and
Talbot, 2010).

The period is cut into :math:`M + 1` blocks of hourly steps and outer fold
:math:`m = 1, \dots, M` uses blocks :math:`1, \dots, m` as its tuning window
and block :math:`m + 1` as its outer test block. Within each tuning window an
independent Optuna search runs over :math:`K` inner expanding-window folds
(:func:`fraud_detection.tuning.optuna_search.tune_model`). The selected
parameters :math:`\hat{\theta}_m` are then refitted on the last inner fold,
training on the window minus its final block and early-stopping on that
block, and the refitted model scores the outer test block once. Rows of the
outer test block never reach the search, the refit, or early stopping.

The estimate reported is

.. math::

    \widehat{\mathrm{AP}}_{\text{nested}} = \frac{1}{M} \sum_{m=1}^{M}
        \mathrm{AP}\big(y^{(m+1)}, \hat{s}_{\hat{\theta}_m}\big),

and the optimism of the search in fold :math:`m` is the difference between
the best inner objective and the outer score. Variation of
:math:`\hat{\theta}_m` across folds indicates how sharply the data identify
the optimum.

References
----------
Cawley, G. C., and Talbot, N. L. C. (2010). On over-fitting in model
selection and subsequent selection bias in performance evaluation. Journal of
Machine Learning Research, 11, 2079-2107.

Varma, S., and Simon, R. (2006). Bias in error estimation when using
cross-validation for model selection. BMC Bioinformatics, 7, 91.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import ResamplingConfig
from fraud_detection.data.resample import resample_training_fold
from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.models.base import fit_model
from fraud_detection.tuning.optuna_search import expanding_window_folds, tune_model


@dataclass(frozen=True)
class OuterFoldResult:
    """Search and test outcome of one outer fold.

    Attributes
    ----------
    fold
        One-based index of the outer fold.
    tuning_steps, test_steps
        Inclusive step ranges of the tuning window and the outer test block.
    best_params
        Parameters selected by the inner search.
    inner_pr_auc
        Best mean inner-fold PR-AUC, the value the search maximized.
    outer_pr_auc
        PR-AUC of the refitted model on the outer test block.
    n_test, n_test_positive
        Size and number of frauds in the outer test block.
    """

    fold: int
    tuning_steps: tuple[int, int]
    test_steps: tuple[int, int]
    best_params: dict[str, Any]
    inner_pr_auc: float
    outer_pr_auc: float
    n_test: int
    n_test_positive: int

    @property
    def optimism(self) -> float:
        """Inner objective minus outer score; positive means overfitted selection."""
        return self.inner_pr_auc - self.outer_pr_auc


@dataclass(frozen=True)
class NestedResult:
    """Outer-fold results of a nested evaluation."""

    model_name: str
    folds: list[OuterFoldResult] = field(default_factory=list)

    @property
    def outer_pr_auc(self) -> float:
        """Mean outer PR-AUC, the nested estimate of the tuned model."""
        return float(np.mean([f.outer_pr_auc for f in self.folds]))

    @property
    def outer_pr_auc_std(self) -> float:
        """Standard deviation of the outer PR-AUC across folds."""
        return float(np.std([f.outer_pr_auc for f in self.folds]))

    @property
    def mean_optimism(self) -> float:
        """Average gap between the inner objective and the outer score."""
        return float(np.mean([f.optimism for f in self.folds]))

    def to_frame(self) -> pd.DataFrame:
        """One row per outer fold, with the selected parameters as columns."""
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "fold": f.fold,
                    "tuning_first_step": f.tuning_steps[0],
                    "tuning_last_step": f.tuning_steps[1],
                    "test_first_step": f.test_steps[0],
                    "test_last_step": f.test_steps[1],
                    "n_test": f.n_test,
                    "n_test_positive": f.n_test_positive,
                    "inner_pr_auc": f.inner_pr_auc,
                    "outer_pr_auc": f.outer_pr_auc,
                    "optimism": f.optimism,
                    **{f"param_{k}": v for k, v in f.best_params.items()},
                }
            )
        return pd.DataFrame(rows)


def _refit_and_score(
    model_name: str,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    steps_tune: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Mapping[str, Any],
    resampling: ResamplingConfig,
    n_inner: int,
    gap_steps: int,
    early_stopping_rounds: int,
    seed: int,
) -> float:
    # Early stopping uses the last block of the tuning window, never the test block
    last = expanding_window_folds(steps_tune, n_inner, gap_steps)[-1]
    train = resample_training_fold(
        X_tune.iloc[last.train], y_tune.iloc[last.train], resampling, seed
    )
    model = fit_model(
        model_name,
        train,
        X_tune.iloc[last.validation],
        y_tune.iloc[last.validation],
        {**params, "early_stopping_rounds": early_stopping_rounds},
        seed=seed,
    )
    return pr_auc(y_test, model.predict_scores(X_test))


def nested_evaluation(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    steps: pd.Series | np.ndarray,
    resampling: ResamplingConfig,
    n_outer: int = 3,
    n_inner: int = 3,
    n_trials: int = 30,
    gap_steps: int = 0,
    fixed_params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    timeout: float | None = None,
    seed: int = 0,
) -> NestedResult:
    """Estimate the performance of the full tuning procedure.

    Parameters
    ----------
    model_name
        Key of :data:`fraud_detection.tuning.optuna_search.SEARCH_SPACES`.
    X, y
        Features and labels of the evaluation period, sorted by time.
    steps
        Time index aligned with ``X``.
    resampling
        Rebalancing applied to every training window, inner and outer.
    n_outer
        Number of outer folds :math:`M`.
    n_inner
        Number of inner folds :math:`K` used by each search.
    n_trials
        Trials per inner search.
    gap_steps
        Gap between training window and scored block, at both levels.
    fixed_params
        Parameters held constant across trials.
    early_stopping_rounds
        Patience of early stopping.
    timeout
        Optional wall-clock limit in seconds for each inner search.
    seed
        Seed of each inner sampler and of every fit.

    Returns
    -------
    NestedResult
        Per-fold selected parameters, inner objective, and outer PR-AUC.

    Raises
    ------
    ValueError
        If an outer tuning window is too short for ``n_inner`` folds.
    """
    step_values = np.asarray(steps)
    results = []
    for m, outer in enumerate(
        expanding_window_folds(step_values, n_outer, gap_steps), start=1
    ):
        X_tune, y_tune = X.iloc[outer.train], y.iloc[outer.train]
        steps_tune = step_values[outer.train]
        X_test, y_test = X.iloc[outer.validation], y.iloc[outer.validation]

        study = tune_model(
            model_name,
            X_tune,
            y_tune,
            steps_tune,
            resampling,
            n_trials=n_trials,
            n_splits=n_inner,
            gap_steps=gap_steps,
            fixed_params=fixed_params,
            early_stopping_rounds=early_stopping_rounds,
            timeout=timeout,
            study_name=f"{model_name}_nested_outer{m}",
            seed=seed,
        )
        try:
            best = study.best_trial
        except ValueError as exc:
            raise ValueError(f"Outer fold {m}: no inner trial completed.") from exc
        params = {**(fixed_params or {}), **best.params}
        results.append(
            OuterFoldResult(
                fold=m,
                tuning_steps=outer.train_steps,
                test_steps=outer.validation_steps,
                best_params=dict(best.params),
                inner_pr_auc=float(best.value),
                outer_pr_auc=_refit_and_score(
                    model_name,
                    X_tune,
                    y_tune,
                    steps_tune,
                    X_test,
                    y_test,
                    params,
                    resampling,
                    n_inner,
                    gap_steps,
                    early_stopping_rounds,
                    seed,
                ),
                n_test=len(y_test),
                n_test_positive=int(y_test.sum()),
            )
        )
    return NestedResult(model_name=model_name, folds=results)
