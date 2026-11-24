r"""Diagnostic figures for a finished Optuna study.

Three views are produced from the trials of one study:

* the optimization history, showing each completed trial's objective and the
  running best :math:`\max_{t' \le t} J(\theta_{t'})`, with pruned trials
  placed at the last running mean they reported;
* hyperparameter importances, estimated with functional ANOVA (Hutter et al.,
  2014): a random forest is fitted to :math:`(\theta_t, J(\theta_t))` and the
  variance of its prediction is decomposed into per-parameter main effects;
* slice plots, one panel per parameter, showing the objective against the
  sampled value, coloured by trial number so the TPE sampler's concentration
  over time is visible.

Only completed trials enter the importances and slices. Pruned trials report
a partial-fold mean, which is not comparable with the full objective.
Importances from a few dozen trials are noisy and describe the response
surface inside the searched ranges only.

References
----------
Hutter, F., Hoos, H., and Leyton-Brown, K. (2014). An efficient approach for
assessing hyperparameter importance. ICML.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import optuna


def _complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    import optuna

    return study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """Completed trials as a table of trial number, objective, and parameters.

    Parameters
    ----------
    study
        Study to summarize.

    Returns
    -------
    pandas.DataFrame
        One row per completed trial, with columns ``number``, ``value`` and
        one column per sampled parameter.
    """
    rows = [
        {"number": t.number, "value": t.value, **t.params}
        for t in _complete_trials(study)
    ]
    return pd.DataFrame(rows, columns=None if rows else ["number", "value"])


def _is_log_scale(study: optuna.Study, param: str) -> bool:
    for trial in _complete_trials(study):
        distribution = trial.distributions.get(param)
        if distribution is not None:
            return bool(getattr(distribution, "log", False))
    return False


def plot_optimization_history(study: optuna.Study) -> Figure:
    """Objective of each trial and the running best.

    Parameters
    ----------
    study
        Study with at least one completed trial.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the history.
    """
    import optuna

    frame = trials_frame(study)
    if frame.empty:
        raise ValueError("The study has no completed trials.")
    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot()
    ax.scatter(frame["number"], frame["value"], s=18, label="Completed trial")
    best = frame["value"].cummax()
    ax.step(frame["number"], best, where="post", color="black", label="Best so far")

    pruned = [
        (t.number, t.intermediate_values[max(t.intermediate_values)])
        for t in study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)
        )
        if t.intermediate_values
    ]
    if pruned:
        numbers, values = zip(*pruned, strict=True)
        ax.scatter(
            numbers,
            values,
            s=18,
            marker="x",
            color="grey",
            label="Pruned (last running mean)",
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean fold PR-AUC")
    ax.set_title(f"Optimization history: {study.study_name}")
    ax.legend(loc="lower right", fontsize="small")
    return fig


def parameter_importances(study: optuna.Study, seed: int = 0) -> pd.Series:
    """fANOVA importances of the sampled parameters.

    Parameters
    ----------
    study
        Study with at least two completed trials.
    seed
        Random state of the forest used by the evaluator.

    Returns
    -------
    pandas.Series
        Importance per parameter, summing to one, in decreasing order.
    """
    import optuna

    if len(_complete_trials(study)) < 2:
        raise ValueError("Importances need at least two completed trials.")
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    scores = optuna.importance.get_param_importances(study, evaluator=evaluator)
    return pd.Series(scores, name="importance", dtype=float).sort_values(
        ascending=False
    )


def plot_parameter_importance(importances: pd.Series) -> Figure:
    """Horizontal bar chart of parameter importances.

    Parameters
    ----------
    importances
        Output of :func:`parameter_importances`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the chart, largest importance at the top.
    """
    ordered = importances.sort_values()
    fig = Figure(figsize=(6.0, 0.4 * len(ordered) + 1.5))
    ax = fig.add_subplot()
    ax.barh(ordered.index.astype(str), ordered.to_numpy())
    ax.set_xlabel("fANOVA importance")
    ax.set_title("Hyperparameter importance")
    return fig


def plot_slices(
    study: optuna.Study,
    params: list[str] | None = None,
    n_cols: int = 3,
) -> Figure:
    """Objective against each sampled parameter.

    Parameters
    ----------
    study
        Study with at least one completed trial.
    params
        Parameters to plot. Defaults to all sampled parameters, sorted.
    n_cols
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
        Grid of scatter plots coloured by trial number.
    """
    frame = trials_frame(study)
    if frame.empty:
        raise ValueError("The study has no completed trials.")
    names = params or sorted(c for c in frame.columns if c not in {"number", "value"})
    if not names:
        raise ValueError("The study has no sampled parameters.")
    n_rows = math.ceil(len(names) / n_cols)
    fig = Figure(figsize=(4.0 * min(n_cols, len(names)), 3.2 * n_rows))
    axes = fig.subplots(n_rows, min(n_cols, len(names)), squeeze=False).ravel()
    scatter: Any = None
    for ax, name in zip(axes, names, strict=False):
        values = frame[name]
        if not pd.api.types.is_numeric_dtype(values):
            codes, labels = pd.factorize(values, sort=True)
            ax.set_xticks(range(len(labels)), labels=[str(v) for v in labels])
            values = codes
        scatter = ax.scatter(
            values, frame["value"], c=frame["number"], cmap="viridis", s=16
        )
        if _is_log_scale(study, name):
            ax.set_xscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("Mean fold PR-AUC")
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    fig.colorbar(scatter, ax=list(axes[: len(names)]), label="Trial")
    fig.suptitle(f"Slice plots: {study.study_name}")
    return fig


def save_study_figures(
    study: optuna.Study,
    out_dir: str | Path,
    seed: int = 0,
    dpi: int = 150,
) -> dict[str, Path]:
    """Write the history, importance, and slice figures as PNG files.

    Parameters
    ----------
    study
        Finished study.
    out_dir
        Output directory, created if missing (``reports/figures`` in the
        default configuration).
    seed
        Random state of the importance evaluator.
    dpi
        Resolution of the saved images.

    Returns
    -------
    dict
        Figure kind mapped to the written path. The importance figure is
        skipped when fewer than two trials completed or no parameter varied.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "history": plot_optimization_history(study),
        "slices": plot_slices(study),
    }
    frame = trials_frame(study)
    sampled = frame.drop(columns=["number", "value"])
    varied = sampled.nunique(dropna=True).gt(1).any() if len(sampled) else False
    if len(frame) >= 2 and varied:
        figures["importance"] = plot_parameter_importance(
            parameter_importances(study, seed)
        )
    paths = {}
    for kind, fig in figures.items():
        path = out / f"{study.study_name}_{kind}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths[kind] = path
    return paths


__all__ = [
    "parameter_importances",
    "plot_optimization_history",
    "plot_parameter_importance",
    "plot_slices",
    "save_study_figures",
    "trials_frame",
]
