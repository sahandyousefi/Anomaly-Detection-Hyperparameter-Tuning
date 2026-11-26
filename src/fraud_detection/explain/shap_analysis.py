r"""Global SHAP analysis of a fitted fraud model.

For a model output :math:`f(x)` and features :math:`j = 1, \dots, p`, the SHAP
value :math:`\phi_j(x)` is the Shapley value of feature :math:`j` in the
cooperative game whose value function is the conditional expectation of
:math:`f` given a subset of features (Lundberg and Lee, 2017). The values are
additive:

.. math::

    f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j(x),

with :math:`\phi_0` the expected output over the background distribution.

Output space
------------
For the gradient-boosted models the values are exact TreeSHAP attributions of
the raw margin, i.e. the log-odds of fraud (Lundberg et al., 2020). Log-odds
are additive across trees and do not saturate near 0 or 1, so contributions of
different transactions are comparable. Every other model family is explained
with the model-agnostic permutation estimator on the probability scale, with
an independent masker over a background sample of training rows. The two
scales are not comparable, and :attr:`ShapExplanation.output` records which
one was used.

Interaction values
------------------
SHAP interaction values :math:`\Phi_{ij}(x)` split each attribution into a main
effect :math:`\Phi_{jj}` and pairwise terms, with
:math:`\phi_j = \Phi_{jj} + \sum_{i \neq j} \Phi_{ij}`. They are only computed
for tree models, where they are exact and polynomial in the tree depth.

Global views
------------
The global importance of feature :math:`j` over rows :math:`x_1, \dots, x_n`
is :math:`I_j = n^{-1} \sum_k |\phi_j(x_k)|`. It measures how much a feature
moves the output, not whether it improves ranking quality; permutation
importance on a held-out fold answers the second question.

References
----------
Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting
model predictions. NeurIPS.

Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations
to global understanding with explainable AI for trees. Nature Machine
Intelligence, 2, 56-67.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fraud_detection.models.base import FraudModel
from fraud_detection.models.gbm import BoostedModel

OutputSpace = Literal["log_odds", "probability"]

DEFAULT_BACKGROUND_ROWS = 200


@dataclass(frozen=True)
class ShapExplanation:
    """SHAP values of a set of rows.

    Attributes
    ----------
    values
        Array of shape ``(n_rows, n_features)``.
    base_value
        Expected model output :math:`\\phi_0`, in the same space as ``values``.
    data
        The explained rows, with the model's feature columns.
    output
        ``"log_odds"`` for tree models, ``"probability"`` otherwise.
    """

    values: np.ndarray
    base_value: float
    data: pd.DataFrame
    output: OutputSpace

    def __post_init__(self) -> None:
        if self.values.shape != self.data.shape:
            raise ValueError(
                f"SHAP values have shape {self.values.shape}, "
                f"data has shape {self.data.shape}."
            )

    @property
    def feature_names(self) -> list[str]:
        """Feature names in column order."""
        return list(self.data.columns)

    def model_output(self) -> np.ndarray:
        """Model output reconstructed from the attributions."""
        return self.base_value + self.values.sum(axis=1)

    def mean_abs(self) -> pd.Series:
        """Global importance :math:`I_j`, sorted in decreasing order."""
        importance = pd.Series(
            np.abs(self.values).mean(axis=0), index=self.feature_names
        )
        return importance.sort_values(ascending=False)

    def subset(self, rows: Sequence[int] | np.ndarray) -> ShapExplanation:
        """Explanation restricted to the given positional rows."""
        index = np.asarray(rows)
        return ShapExplanation(
            values=self.values[index],
            base_value=self.base_value,
            data=self.data.iloc[index],
            output=self.output,
        )


def sample_rows(X: pd.DataFrame, max_rows: int | None, seed: int = 0) -> pd.DataFrame:
    """Uniform sample of at most ``max_rows`` rows, kept in original order.

    Parameters
    ----------
    X
        Rows to sample from.
    max_rows
        Sample size; ``None`` keeps every row.
    seed
        Random state of the sample.

    Returns
    -------
    pandas.DataFrame
        The sampled rows.
    """
    if max_rows is None or len(X) <= max_rows:
        return X
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[chosen]


def _library(model: BoostedModel) -> str:
    return type(model.model).__module__.split(".")[0]


def _xgboost_predict(model: BoostedModel, X: pd.DataFrame, **kwargs) -> np.ndarray:
    import xgboost as xgb

    booster = model.model.get_booster()
    return booster.predict(
        xgb.DMatrix(X, feature_names=list(X.columns)),
        iteration_range=(0, model.best_iteration + 1),
        **kwargs,
    )


def tree_shap(model: BoostedModel, X: pd.DataFrame) -> ShapExplanation:
    """Exact TreeSHAP values of the log-odds, using the early-stopped ensemble.

    Parameters
    ----------
    model
        Booster fitted by :mod:`fraud_detection.models.gbm`.
    X
        Rows to explain, with the training feature columns.

    Returns
    -------
    ShapExplanation
        Attributions in log-odds space.
    """
    library = _library(model)
    if library == "xgboost":
        raw = _xgboost_predict(model, X, pred_contribs=True)
    elif library == "lightgbm":
        raw = model.model.predict(
            X, pred_contrib=True, num_iteration=model.best_iteration + 1
        )
    elif library == "catboost":
        from catboost import Pool

        # use_best_model already shrank the ensemble to the best iteration
        raw = model.model.get_feature_importance(Pool(X), type="ShapValues")
    else:
        raise TypeError(f"No TreeSHAP path for models from {library!r}.")
    raw = np.asarray(raw, dtype="float64")
    return ShapExplanation(
        values=raw[:, :-1],
        base_value=float(raw[0, -1]),
        data=X,
        output="log_odds",
    )


def tree_shap_interactions(model: BoostedModel, X: pd.DataFrame) -> np.ndarray:
    """SHAP interaction values of the log-odds.

    Parameters
    ----------
    model
        Booster fitted by :mod:`fraud_detection.models.gbm`.
    X
        Rows to explain. The cost grows with ``n_rows * n_features**2``, so
        pass a sample.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_rows, n_features, n_features)``; the sum over the
        last axis equals the SHAP values.
    """
    if _library(model) == "xgboost":
        raw = _xgboost_predict(model, X, pred_interactions=True)
        return np.asarray(raw, dtype="float64")[:, :-1, :-1]
    import shap

    explainer = shap.TreeExplainer(model.model)
    return np.asarray(explainer.shap_interaction_values(X), dtype="float64")


def permutation_shap(
    model: FraudModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
    max_background_rows: int = DEFAULT_BACKGROUND_ROWS,
    seed: int = 0,
) -> ShapExplanation:
    """Model-agnostic SHAP values of the fraud probability.

    Parameters
    ----------
    model
        Any fitted model exposing ``predict_scores``.
    X
        Rows to explain.
    background
        Reference rows, usually a sample of the training fold. Features are
        masked independently, so correlated features are attributed under an
        independence assumption.
    max_background_rows
        Size of the background sample; the cost is linear in it.
    seed
        Random state of the background sample and the permutations.

    Returns
    -------
    ShapExplanation
        Attributions in probability space.
    """
    import shap

    columns = list(X.columns)
    reference = sample_rows(background[columns], max_background_rows, seed)

    def predict(values: np.ndarray) -> np.ndarray:
        return model.predict_scores(pd.DataFrame(values, columns=columns))

    masker = shap.maskers.Independent(reference, max_samples=len(reference))
    explainer = shap.PermutationExplainer(predict, masker, seed=seed)
    result = explainer(X, max_evals=max(2 * len(columns) + 1, 500), silent=True)
    base = np.atleast_1d(np.asarray(result.base_values, dtype="float64"))
    return ShapExplanation(
        values=np.asarray(result.values, dtype="float64"),
        base_value=float(base[0]),
        data=X,
        output="probability",
    )


def explain_model(
    model: FraudModel,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
    max_rows: int | None = 2_000,
    seed: int = 0,
) -> ShapExplanation:
    """SHAP values with the exact tree path where available.

    Parameters
    ----------
    model
        Fitted model.
    X
        Rows to explain, typically the validation or test fold.
    background
        Reference rows for non-tree models. Required unless ``model`` is a
        :class:`~fraud_detection.models.gbm.BoostedModel`.
    max_rows
        Rows of ``X`` explained, sampled uniformly; ``None`` explains all.
    seed
        Random state of the sampling.

    Returns
    -------
    ShapExplanation
        Attributions of the sampled rows.
    """
    rows = sample_rows(X, max_rows, seed)
    if isinstance(model, BoostedModel):
        return tree_shap(model, rows)
    if background is None:
        raise ValueError("A background sample is required for non-tree models.")
    return permutation_shap(model, rows, background, seed=seed)


def interaction_strength(
    interactions: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Mean absolute interaction value of every feature pair.

    Parameters
    ----------
    interactions
        Output of :func:`tree_shap_interactions`.
    feature_names
        Names along the last two axes.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix; the diagonal holds the mean absolute main effects.
    """
    strength = np.abs(interactions).mean(axis=0)
    return pd.DataFrame(strength, index=feature_names, columns=feature_names)


def strongest_partner(strength: pd.DataFrame, feature: str) -> str:
    """Feature with the largest mean absolute interaction with ``feature``."""
    row = strength.loc[feature].drop(feature)
    return str(row.idxmax())


def _value_ranks(column: pd.Series) -> np.ndarray:
    # Percentile ranks keep heavy-tailed amounts from collapsing the colour scale
    return column.rank(pct=True).to_numpy(dtype="float64")


def _output_label(explanation: ShapExplanation) -> str:
    space = "log-odds" if explanation.output == "log_odds" else "probability"
    return f"SHAP value ({space})"


def plot_importance(explanation: ShapExplanation, max_features: int = 20) -> Figure:
    """Horizontal bar chart of the mean absolute SHAP value per feature.

    Parameters
    ----------
    explanation
        SHAP values to summarise.
    max_features
        Number of top features shown.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the chart.
    """
    importance = explanation.mean_abs().head(max_features).iloc[::-1]
    fig = Figure(figsize=(6.5, 0.3 * len(importance) + 1.2))
    ax = fig.add_subplot()
    ax.barh(importance.index, importance.to_numpy(), color="#4c72b0")
    ax.set_xlabel(f"Mean |{_output_label(explanation)}|")
    ax.set_title("Global feature importance")
    return fig


def plot_summary(
    explanation: ShapExplanation, max_features: int = 20, seed: int = 0
) -> Figure:
    """Beeswarm-style summary: one jittered strip of SHAP values per feature.

    Points are coloured by the percentile rank of the feature value within the
    explained rows; missing values are grey.

    Parameters
    ----------
    explanation
        SHAP values to summarise.
    max_features
        Number of top features shown, ordered by mean absolute SHAP value.
    seed
        Random state of the vertical jitter.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the summary.
    """
    order = list(explanation.mean_abs().head(max_features).index)[::-1]
    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(7.5, 0.35 * len(order) + 1.5))
    ax = fig.add_subplot()
    scatter = None
    for position, feature in enumerate(order):
        j = explanation.feature_names.index(feature)
        shap_values = explanation.values[:, j]
        ranks = _value_ranks(explanation.data[feature])
        jitter = rng.uniform(-0.3, 0.3, size=len(shap_values))
        missing = np.isnan(ranks)
        ax.scatter(
            shap_values[missing],
            position + jitter[missing],
            s=6,
            color="lightgrey",
        )
        scatter = ax.scatter(
            shap_values[~missing],
            position + jitter[~missing],
            c=ranks[~missing],
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=6,
        )
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(_output_label(explanation))
    ax.set_title("SHAP summary")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="Feature value (percentile)")
    return fig


def plot_dependence(
    explanation: ShapExplanation, feature: str, color_by: str | None = None
) -> Figure:
    """SHAP value of one feature against its value.

    Parameters
    ----------
    explanation
        SHAP values to plot.
    feature
        Feature on the horizontal axis.
    color_by
        Second feature used for colour, usually its strongest interaction
        partner from :func:`strongest_partner`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the dependence plot.
    """
    j = explanation.feature_names.index(feature)
    fig = Figure(figsize=(6.0, 4.5))
    ax = fig.add_subplot()
    x = explanation.data[feature].to_numpy(dtype="float64")
    y = explanation.values[:, j]
    if color_by is None:
        ax.scatter(x, y, s=6, color="#4c72b0")
    else:
        points = ax.scatter(
            x, y, s=6, c=_value_ranks(explanation.data[color_by]), cmap="coolwarm"
        )
        fig.colorbar(points, ax=ax, label=f"{color_by} (percentile)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel(_output_label(explanation))
    ax.set_title(f"SHAP dependence: {feature}")
    return fig


def plot_interaction_heatmap(strength: pd.DataFrame, max_features: int = 12) -> Figure:
    """Heatmap of mean absolute interaction values among the top features.

    Parameters
    ----------
    strength
        Output of :func:`interaction_strength`.
    max_features
        Features kept, ranked by their total interaction strength.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the heatmap. The diagonal (main effects) is masked so
        the off-diagonal structure is visible.
    """
    off_diagonal = strength.to_numpy(copy=True)
    np.fill_diagonal(off_diagonal, np.nan)
    totals = pd.Series(np.nansum(off_diagonal, axis=1), index=strength.index)
    top = list(totals.sort_values(ascending=False).head(max_features).index)
    matrix = strength.loc[top, top].to_numpy(copy=True)
    np.fill_diagonal(matrix, np.nan)
    fig = Figure(figsize=(0.45 * len(top) + 3.0, 0.45 * len(top) + 2.0))
    ax = fig.add_subplot()
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(top)), top, rotation=90)
    ax.set_yticks(range(len(top)), top)
    ax.set_title("Mean |SHAP interaction value|")
    fig.colorbar(image, ax=ax)
    return fig


def save_shap_figures(
    explanation: ShapExplanation,
    out_dir: str | Path,
    interactions: np.ndarray | None = None,
    n_dependence: int = 3,
    prefix: str = "shap",
    dpi: int = 150,
) -> dict[str, Path]:
    """Write the importance, summary, dependence, and interaction figures.

    Parameters
    ----------
    explanation
        SHAP values of a held-out sample.
    out_dir
        Output directory, created if missing.
    interactions
        Interaction values of the same rows. When given, each dependence plot
        is coloured by the feature's strongest partner and a heatmap is added.
    n_dependence
        Dependence plots for the top features by mean absolute SHAP value.
    prefix
        File name prefix.
    dpi
        Resolution of the saved images.

    Returns
    -------
    dict
        Figure kind mapped to the written path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "importance": plot_importance(explanation),
        "summary": plot_summary(explanation),
    }
    strength = None
    if interactions is not None:
        strength = interaction_strength(interactions, explanation.feature_names)
        figures["interactions"] = plot_interaction_heatmap(strength)
    for feature in explanation.mean_abs().head(n_dependence).index:
        partner = None if strength is None else strongest_partner(strength, feature)
        figures[f"dependence_{feature}"] = plot_dependence(
            explanation, feature, partner
        )
    paths = {}
    for kind, fig in figures.items():
        path = out / f"{prefix}_{kind}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths[kind] = path
    return paths


__all__ = [
    "ShapExplanation",
    "explain_model",
    "interaction_strength",
    "permutation_shap",
    "plot_dependence",
    "plot_importance",
    "plot_interaction_heatmap",
    "plot_summary",
    "sample_rows",
    "save_shap_figures",
    "strongest_partner",
    "tree_shap",
    "tree_shap_interactions",
]
