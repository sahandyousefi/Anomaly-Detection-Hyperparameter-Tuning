r"""Local explanations of individual flagged transactions.

An analyst reviewing an alert needs to know which inputs pushed that one
transaction above the decision threshold :math:`\tau`. For a row :math:`x`
with SHAP values :math:`\phi_1(x), \dots, \phi_p(x)` the waterfall plot starts
at the expected output :math:`\phi_0`, adds the contributions in decreasing
order of :math:`|\phi_j(x)|`, and ends at :math:`f(x)`. Features outside the
top :math:`m` are pooled into one bar, so the chart always sums to the model
output.

For tree models the axis is the log-odds; the probability
:math:`\sigma(f(x))` is printed next to the final value. A large positive
contribution in log-odds can correspond to a small change in probability when
the base rate is low, which is why the log-odds scale is kept for the bars.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fraud_detection.explain.shap_analysis import ShapExplanation, explain_model
from fraud_detection.models.base import FraudModel

POSITIVE_COLOR = "#c44e52"
NEGATIVE_COLOR = "#4c72b0"


def select_flagged(
    scores: np.ndarray, threshold: float, max_rows: int | None = None
) -> np.ndarray:
    """Positions of rows at or above the threshold, highest score first.

    Parameters
    ----------
    scores
        Model scores.
    threshold
        Decision threshold :math:`\\tau`, chosen on the validation fold.
    max_rows
        Keep only the highest-scoring rows; ``None`` keeps all.

    Returns
    -------
    numpy.ndarray
        Integer positions into ``scores``.
    """
    scores = np.asarray(scores, dtype="float64")
    flagged = np.flatnonzero(scores >= threshold)
    # Stable sort keeps chronological order among tied scores
    order = flagged[np.argsort(-scores[flagged], kind="stable")]
    return order if max_rows is None else order[:max_rows]


def contribution_table(
    explanation: ShapExplanation, row: int, max_features: int = 10
) -> pd.DataFrame:
    """Top contributions of one row, with the remainder pooled.

    Parameters
    ----------
    explanation
        SHAP values containing the row.
    row
        Position of the row within ``explanation``.
    max_features
        Number of features listed individually.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``value`` and ``contribution``, ordered by
        decreasing absolute contribution. When features are pooled, the last
        row is ``"other features (k)"`` with a missing value. The
        contributions sum to ``f(x) - base_value``.
    """
    if max_features < 1:
        raise ValueError("max_features must be at least 1.")
    contributions = explanation.values[row]
    values = explanation.data.iloc[row].to_numpy(dtype="float64")
    order = np.argsort(-np.abs(contributions), kind="stable")
    top, rest = order[:max_features], order[max_features:]
    table = pd.DataFrame(
        {
            "feature": [explanation.feature_names[j] for j in top],
            "value": values[top],
            "contribution": contributions[top],
        }
    )
    if len(rest):
        pooled = pd.DataFrame(
            {
                "feature": [f"other features ({len(rest)})"],
                "value": [np.nan],
                "contribution": [float(contributions[rest].sum())],
            }
        )
        table = pd.concat([table, pooled], ignore_index=True)
    return table


def _label(feature: str, value: float) -> str:
    if np.isnan(value):
        return feature
    return f"{feature} = {value:.4g}"


def _final_annotation(explanation: ShapExplanation, output: float) -> str:
    if explanation.output == "log_odds":
        probability = 1.0 / (1.0 + np.exp(-output))
        return f"f(x) = {output:.3f} (p = {probability:.3f})"
    return f"f(x) = {output:.3f}"


def plot_waterfall(
    explanation: ShapExplanation,
    row: int,
    max_features: int = 10,
    title: str | None = None,
) -> Figure:
    """Waterfall chart from the expected output to the row's model output.

    Parameters
    ----------
    explanation
        SHAP values containing the row.
    row
        Position of the row within ``explanation``.
    max_features
        Number of features drawn individually; the rest form one bar.
    title
        Figure title, for example the transaction identifier.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the chart, largest contribution at the top.
    """
    table = contribution_table(explanation, row, max_features).iloc[::-1]
    contributions = table["contribution"].to_numpy()
    starts = explanation.base_value + np.concatenate(
        ([0.0], np.cumsum(contributions)[:-1])
    )
    colors = [POSITIVE_COLOR if c > 0 else NEGATIVE_COLOR for c in contributions]
    labels = [
        _label(f, v) for f, v in zip(table["feature"], table["value"], strict=True)
    ]
    fig = Figure(figsize=(7.5, 0.4 * len(table) + 1.6))
    ax = fig.add_subplot()
    positions = np.arange(len(table))
    ax.barh(positions, contributions, left=starts, color=colors)
    for position, start, width in zip(positions, starts, contributions, strict=True):
        ax.text(
            start + width,
            position,
            f" {width:+.3f} ",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize="small",
        )
    # Leave room for the value labels on both sides of the bars
    ends = np.concatenate((starts, starts + contributions))
    low, high = float(ends.min()), float(ends.max())
    pad = 0.18 * max(high - low, 1e-9)
    ax.set_xlim(low - pad, high + pad)
    output = float(explanation.base_value + contributions.sum())
    ax.axvline(explanation.base_value, color="grey", ls="--", lw=0.8)
    ax.axvline(output, color="black", lw=0.8)
    ax.set_yticks(positions, labels)
    space = "log-odds" if explanation.output == "log_odds" else "probability"
    ax.set_xlabel(
        f"Model output ({space}); E[f(X)] = {explanation.base_value:.3f}, "
        f"{_final_annotation(explanation, output)}"
    )
    ax.set_title(title or f"Local explanation, row {row}")
    return fig


def explain_flagged(
    model: FraudModel,
    X: pd.DataFrame,
    threshold: float,
    max_rows: int | None = 25,
    background: pd.DataFrame | None = None,
    seed: int = 0,
) -> ShapExplanation:
    """SHAP values of the highest-scoring flagged rows.

    Parameters
    ----------
    model
        Fitted model.
    X
        Rows to score, typically the test fold.
    threshold
        Decision threshold.
    max_rows
        Number of flagged rows explained, highest score first.
    background
        Reference rows for non-tree models.
    seed
        Random state of the permutation estimator.

    Returns
    -------
    ShapExplanation
        Explanation of the flagged rows, ordered by decreasing score. It is
        empty when nothing reaches the threshold.
    """
    positions = select_flagged(model.predict_scores(X), threshold, max_rows)
    if len(positions) == 0:
        return ShapExplanation(np.empty((0, X.shape[1])), 0.0, X.iloc[:0], "log_odds")
    return explain_model(model, X.iloc[positions], background, max_rows=None, seed=seed)


def save_waterfalls(
    explanation: ShapExplanation,
    out_dir: str | Path,
    labels: Sequence[str] | None = None,
    max_features: int = 10,
    prefix: str = "waterfall",
    dpi: int = 150,
) -> list[Path]:
    """Write one waterfall chart per explained row.

    Parameters
    ----------
    explanation
        Explanation of the rows to plot, e.g. from :func:`explain_flagged`.
    out_dir
        Output directory, created if missing.
    labels
        One title per row; defaults to the row index of ``explanation.data``.
    max_features
        Features drawn individually in each chart.
    prefix
        File name prefix.
    dpi
        Resolution of the saved images.

    Returns
    -------
    list of pathlib.Path
        Written files, in row order.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = [f"Transaction {index}" for index in explanation.data.index]
    if len(labels) != len(explanation.data):
        raise ValueError("labels must have one entry per explained row.")
    paths = []
    for row, label in enumerate(labels):
        fig = plot_waterfall(explanation, row, max_features, title=label)
        path = out / f"{prefix}_{row:03d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


__all__ = [
    "contribution_table",
    "explain_flagged",
    "plot_waterfall",
    "save_waterfalls",
    "select_flagged",
]
