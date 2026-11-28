r"""Feature and score drift between a reference and a current period.

Two statistics are reported per variable.

Population stability index. With :math:`B` bins fixed on the reference
sample, reference shares :math:`e_b` and current shares :math:`a_b`,

.. math::

    \mathrm{PSI} = \sum_{b=1}^{B} (a_b - e_b) \ln \frac{a_b}{e_b}.

Bin edges are reference quantiles, so each bin holds about :math:`1/B` of
the reference rows. Variables with at most :math:`B` distinct reference
values (flags, counts, ordinal codes) are binned by value instead. Missing
values form their own bin, because a change in the share of first-seen
accounts is itself a shift. Empty bins are floored at a small
:math:`\varepsilon` before the logarithm. The usual reading is
:math:`\mathrm{PSI} < 0.1` stable, :math:`0.1` to :math:`0.25` moderate, and
above :math:`0.25` a significant shift; these are conventions, not tests.

Two-sample Kolmogorov-Smirnov statistic,
:math:`D = \sup_x |F_{ref}(x) - F_{cur}(x)|`, on the non-missing values, with
its asymptotic p-value. With PaySim-sized samples almost any difference is
significant, so :math:`D` itself is the more useful number; the p-value is
reported for small windows.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_BINS = 10
PSI_FLOOR = 1e-4
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25

SCORE_COLUMN = "score"


@dataclass(frozen=True)
class DriftResult:
    """Drift statistics of one variable.

    Attributes
    ----------
    psi
        Population stability index over the reference bins.
    ks_statistic, ks_pvalue
        Two-sample Kolmogorov-Smirnov statistic and p-value on non-missing
        values; NaN when either side has none.
    missing_reference, missing_current
        Share of missing values in each period.
    status
        ``"stable"``, ``"moderate"``, or ``"significant"`` by PSI.
    """

    psi: float
    ks_statistic: float
    ks_pvalue: float
    missing_reference: float
    missing_current: float
    status: str


def psi_status(psi: float) -> str:
    """Conventional reading of a PSI value."""
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _value_codes(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of each value in ``values``; ``len(values)`` if absent."""
    if values.size == 0:
        return np.zeros(x.shape, dtype="int64")
    idx = np.searchsorted(values, x)
    known = (idx < values.size) & (values[np.minimum(idx, values.size - 1)] == x)
    return np.where(known, idx, values.size)


def _bin_codes(
    reference: np.ndarray, current: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin indices of both samples and the number of bins.

    Discrete variables get one bin per reference value plus one for values
    unseen in the reference; continuous variables get quantile bins. Missing
    values always go to the last bin.
    """
    observed = reference[~np.isnan(reference)]
    values = np.unique(observed)
    if values.size <= n_bins:
        n_regular = values.size + 1

        def codes(x: np.ndarray) -> np.ndarray:
            return _value_codes(values, x)

    else:
        edges = np.unique(np.quantile(observed, np.linspace(0, 1, n_bins + 1)[1:-1]))
        n_regular = edges.size + 1

        def codes(x: np.ndarray) -> np.ndarray:
            return np.searchsorted(edges, x, side="right")

    def assign(x: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(x), n_regular, codes(np.nan_to_num(x)))

    return assign(reference), assign(current), n_regular + 1


def population_stability_index(
    reference: ArrayLike, current: ArrayLike, n_bins: int = DEFAULT_BINS
) -> float:
    """PSI of ``current`` against bins fixed on ``reference``.

    Parameters
    ----------
    reference, current
        Samples of one numeric variable; NaN marks a missing value.
    n_bins
        Number of quantile bins :math:`B` for continuous variables.

    Returns
    -------
    float
        Non-negative PSI; zero for identical distributions.
    """
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Both samples must be non-empty.")
    ref_codes, cur_codes, n_codes = _bin_codes(ref, cur, n_bins)
    expected = np.bincount(ref_codes, minlength=n_codes) / ref.size
    actual = np.bincount(cur_codes, minlength=n_codes) / cur.size
    used = (expected > 0) | (actual > 0)
    e = np.maximum(expected[used], PSI_FLOOR)
    a = np.maximum(actual[used], PSI_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_statistics(
    reference: ArrayLike, current: ArrayLike, n_bins: int = DEFAULT_BINS
) -> DriftResult:
    """PSI, KS statistic, and missing shares of one variable."""
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    psi = population_stability_index(ref, cur, n_bins)
    ref_obs, cur_obs = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref_obs.size and cur_obs.size:
        ks = stats.ks_2samp(ref_obs, cur_obs)
        ks_statistic, ks_pvalue = float(ks.statistic), float(ks.pvalue)
    else:
        ks_statistic = ks_pvalue = float("nan")
    return DriftResult(
        psi=psi,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        missing_reference=float(np.isnan(ref).mean()),
        missing_current=float(np.isnan(cur).mean()),
        status=psi_status(psi),
    )


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str] | None = None,
    n_bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """Drift statistics for every column, most shifted first.

    Parameters
    ----------
    reference, current
        Tables with the same numeric columns, e.g. model features of the
        training period and of a scoring window, optionally with a
        ``score`` column.
    columns
        Columns to compare; defaults to the columns of ``reference``.
    n_bins
        Quantile bins for continuous columns.

    Returns
    -------
    pandas.DataFrame
        One row per column, indexed by name, sorted by decreasing PSI.
    """
    columns = list(columns if columns is not None else reference.columns)
    missing = [c for c in columns if c not in current.columns]
    if missing:
        raise KeyError(f"Current table lacks columns: {missing}")
    rows = {
        column: drift_statistics(reference[column], current[column], n_bins).__dict__
        for column in columns
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "variable"
    return table.sort_values("psi", ascending=False)


def markdown_report(table: pd.DataFrame, title: str) -> str:
    """Render a drift table as a Markdown section."""
    counts = table["status"].value_counts()
    header = (
        f"# {title}\n\n"
        f"{len(table)} variables: {counts.get('significant', 0)} significant "
        f"(PSI >= {PSI_SIGNIFICANT}), {counts.get('moderate', 0)} moderate "
        f"(PSI >= {PSI_MODERATE}).\n\n"
    )
    lines = [
        "| Variable | PSI | KS D | KS p | Missing ref | Missing cur | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in table.iterrows():
        lines.append(
            f"| {name} | {row.psi:.4f} | {row.ks_statistic:.4f} | "
            f"{row.ks_pvalue:.3g} | {row.missing_reference:.3f} | "
            f"{row.missing_current:.3f} | {row.status} |"
        )
    return header + "\n".join(lines) + "\n"


def _scored_features(artifact_dir: Path, transactions: pd.DataFrame) -> pd.DataFrame:
    from fraud_detection.serving.artifact import load_artifact

    artifact = load_artifact(artifact_dir)
    features = artifact.build_features(transactions)
    features[SCORE_COLUMN] = artifact.model.predict_scores(features)
    return features


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options of the ``drift`` command."""
    parser.add_argument(
        "--artifact", type=Path, required=True, help="Model artifact directory."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Raw transactions of the reference period (CSV in PaySim layout).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Raw transactions of the period to monitor (CSV in PaySim layout).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/drift"), help="Report directory."
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)


def run_from_args(args: argparse.Namespace) -> int:
    """Execute the ``drift`` command.

    Each period is featurised on its own, so account aggregates only see
    history inside that file; pass files that include the account history
    needed for the features to be comparable.
    """
    reference = _scored_features(args.artifact, pd.read_csv(args.reference))
    current = _scored_features(args.artifact, pd.read_csv(args.current))
    table = drift_report(reference, current, n_bins=args.bins)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "drift.csv")
    title = f"Drift of {args.current.name} against {args.reference.name}"
    (args.output / "drift.md").write_text(markdown_report(table, title))
    score = table.loc[SCORE_COLUMN]
    summary = {
        "score_psi": float(score.psi),
        "score_ks": float(score.ks_statistic),
        "score_status": str(score.status),
        "significant_features": table.index[table["status"] == "significant"]
        .drop(SCORE_COLUMN, errors="ignore")
        .tolist(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "Score PSI %.4f (%s); %d features with significant drift",
        summary["score_psi"],
        summary["score_status"],
        len(summary["significant_features"]),
    )
    return 0


__all__ = [
    "DriftResult",
    "drift_report",
    "drift_statistics",
    "markdown_report",
    "population_stability_index",
    "psi_status",
]
