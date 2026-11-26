r"""Permutation importance on a held-out fold, compared with SHAP rankings.

The two importance measures answer different questions.

* Mean absolute SHAP value :math:`I^{\text{SHAP}}_j` measures how much
  feature :math:`j` moves the model output. It needs no labels.
* Permutation importance measures how much ranking quality is lost when the
  link between feature :math:`j` and the label is broken (Breiman, 2001;
  Fisher et al., 2019). With :math:`\pi_r` the :math:`r`-th random permutation
  of the rows of column :math:`j`,

  .. math::

      I^{\text{perm}}_j = \text{PR-AUC}(y, f(X))
          - \frac{1}{R} \sum_{r=1}^{R} \text{PR-AUC}(y, f(X^{(j, \pi_r)})).

PR-AUC is used because it is the selection metric of the project and is
sensitive to the ranking of the rare positive class.

Disagreement between the two rankings is informative. A feature with large
SHAP values and near-zero permutation importance moves the score without
improving it on held-out data, which points to overfitting or to a feature
that is redundant with a correlated one. Permuting one of two correlated
features leaves the other available, so permutation importance understates
both, while the permuted rows can be implausible combinations that the model
never saw in training (Hooker et al., 2021). Neither measure is causal.

Agreement is summarised by Spearman's :math:`\rho` and Kendall's
:math:`\tau_b` between the two importance vectors and by the Jaccard overlap
of their top-:math:`k` sets.

References
----------
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but
many are useful: learning a variable's importance by studying an entire class
of prediction models simultaneously. Journal of Machine Learning Research,
20(177), 1-81.

Hooker, G., Mentch, L., and Zhou, S. (2021). Unrestricted permutation forces
extrapolation: variable importance requires at least one more model, or there
is no free variable importance. Statistics and Computing, 31(6), 82.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from fraud_detection.evaluation.metrics import pr_auc
from fraud_detection.models.base import FraudModel


def permutation_importance(
    model: FraudModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    features: Sequence[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop in PR-AUC when one column is permuted.

    Parameters
    ----------
    model
        Fitted model.
    X, y
        Held-out fold at the observed class ratio (validation or test). Using
        the training fold would measure memorisation instead of
        generalisation.
    n_repeats
        Permutations :math:`R` per feature.
    features
        Features to evaluate; defaults to all columns of ``X``.
    seed
        Random state of the permutations.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature, with columns ``importance`` (mean drop) and
        ``std`` (standard deviation over repeats), sorted by decreasing
        importance.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    features = list(features) if features is not None else list(X.columns)
    rng = np.random.default_rng(seed)
    baseline = pr_auc(y, model.predict_scores(X))
    rows = []
    for feature in features:
        original = X[feature].to_numpy()
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            permuted = X.copy()
            permuted[feature] = original[rng.permutation(len(original))]
            drops[r] = baseline - pr_auc(y, model.predict_scores(permuted))
        rows.append(
            {
                "feature": feature,
                "importance": drops.mean(),
                "std": drops.std(ddof=1) if n_repeats > 1 else 0.0,
            }
        )
    table = pd.DataFrame(rows).set_index("feature")
    return table.sort_values("importance", ascending=False)


@dataclass(frozen=True)
class RankAgreement:
    """Agreement between two importance rankings.

    Attributes
    ----------
    spearman
        Spearman rank correlation of the importance values.
    kendall
        Kendall's :math:`\\tau_b`, which handles tied importances.
    top_k_jaccard
        :math:`|A_k \\cap B_k| / |A_k \\cup B_k|` for the top-:math:`k` sets.
    k
        Size of the top sets.
    """

    spearman: float
    kendall: float
    top_k_jaccard: float
    k: int


def compare_rankings(
    shap_importance: pd.Series,
    permutation: pd.Series,
    k: int = 10,
) -> tuple[pd.DataFrame, RankAgreement]:
    """Side-by-side ranks and agreement statistics.

    Parameters
    ----------
    shap_importance
        Mean absolute SHAP value per feature, e.g.
        :meth:`ShapExplanation.mean_abs`.
    permutation
        Permutation importance per feature, e.g. the ``importance`` column of
        :func:`permutation_importance`.
    k
        Size of the top sets compared.

    Returns
    -------
    table : pandas.DataFrame
        One row per shared feature with both importances, their ranks
        (1 = most important) and the rank difference, sorted by SHAP rank.
    agreement : RankAgreement
        Correlation and overlap statistics.
    """
    shared = shap_importance.index.intersection(permutation.index)
    if len(shared) < 2:
        raise ValueError("At least two shared features are needed.")
    table = pd.DataFrame(
        {
            "shap": shap_importance.loc[shared].astype("float64"),
            "permutation": permutation.loc[shared].astype("float64"),
        }
    )
    table["shap_rank"] = table["shap"].rank(ascending=False, method="min")
    table["permutation_rank"] = table["permutation"].rank(ascending=False, method="min")
    table["rank_difference"] = table["permutation_rank"] - table["shap_rank"]
    table = table.sort_values("shap_rank")
    k = min(k, len(table))
    top_shap = set(table.nsmallest(k, "shap_rank").index)
    top_perm = set(table.nsmallest(k, "permutation_rank").index)
    agreement = RankAgreement(
        spearman=float(stats.spearmanr(table["shap"], table["permutation"])[0]),
        kendall=float(stats.kendalltau(table["shap"], table["permutation"])[0]),
        top_k_jaccard=len(top_shap & top_perm) / len(top_shap | top_perm),
        k=k,
    )
    return table, agreement


def plot_rank_comparison(table: pd.DataFrame, agreement: RankAgreement) -> Figure:
    """Scatter of SHAP rank against permutation rank, one point per feature.

    Parameters
    ----------
    table, agreement
        Output of :func:`compare_rankings`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure holding the plot. Points on the diagonal are ranked equally by
        both measures.
    """
    fig = Figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot()
    ax.scatter(table["shap_rank"], table["permutation_rank"], s=18, color="#4c72b0")
    for feature, row in table.iterrows():
        ax.annotate(
            str(feature),
            (row["shap_rank"], row["permutation_rank"]),
            fontsize="x-small",
            xytext=(3, 3),
            textcoords="offset points",
        )
    upper = float(max(table["shap_rank"].max(), table["permutation_rank"].max()))
    ax.plot([1, upper], [1, upper], color="grey", ls="--", lw=0.8)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Rank by mean |SHAP value|")
    ax.set_ylabel("Rank by permutation importance (PR-AUC drop)")
    ax.set_title(
        f"Spearman {agreement.spearman:.2f}, Kendall {agreement.kendall:.2f}, "
        f"top-{agreement.k} Jaccard {agreement.top_k_jaccard:.2f}"
    )
    return fig


__all__ = [
    "RankAgreement",
    "compare_rankings",
    "permutation_importance",
    "plot_rank_comparison",
]
