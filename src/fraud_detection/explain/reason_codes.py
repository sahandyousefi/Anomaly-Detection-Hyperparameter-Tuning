r"""Reason codes for fraud alerts derived from SHAP contributions.

Each alert carries up to :math:`k` short, stable reasons that an investigator
can read without access to the model. Reasons are built in three steps.

1. Features are grouped into reason groups :math:`G_1, \dots, G_R`, for
   example the three encodings of the hour of day, or all lag columns of the
   originator's history. Because SHAP values are additive, the contribution
   of a group is :math:`\psi_r(x) = \sum_{j \in G_r} \phi_j(x)`, which avoids
   splitting one concept across several weak reasons.
2. Groups are ranked by :math:`\psi_r(x)` and only groups with
   :math:`\psi_r(x) > \epsilon` are kept. Negative contributions lower the
   fraud score and are not reasons for an alert.
3. Each kept group is rendered with a fixed code and a sentence filled in
   from the transaction's own feature values.

Codes are stable across model versions, so alert volumes per reason can be
monitored over time. The sentences describe what the model responded to;
they are not a causal statement about why a transaction is fraudulent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.explain.shap_analysis import ShapExplanation

Renderer = Callable[[Mapping[str, float]], str]


@dataclass(frozen=True)
class ReasonGroup:
    """A set of features reported as one reason.

    Attributes
    ----------
    code
        Stable identifier, e.g. ``"R03"``.
    features
        Feature names whose SHAP values are summed.
    render
        Builds the sentence from the row's feature values (missing features
        are absent from the mapping).
    """

    code: str
    features: tuple[str, ...]
    render: Renderer


@dataclass(frozen=True)
class Reason:
    """One reason attached to an alert."""

    code: str
    description: str
    contribution: float


def _is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _amount(values: Mapping[str, float]) -> str:
    log_amount = values.get("log_amount")
    if _is_missing(log_amount):
        return "Transaction amount"
    return f"Transaction amount of {math.expm1(log_amount):,.0f}"


def _type(values: Mapping[str, float]) -> str:
    if values.get("is_transfer") == 1:
        return "Transaction type is TRANSFER"
    if values.get("is_cash_out") == 1:
        return "Transaction type is CASH_OUT"
    return "Transaction type is neither TRANSFER nor CASH_OUT"


def _hour(values: Mapping[str, float]) -> str:
    hour = values.get("hour_of_day")
    if _is_missing(hour):
        return "Time of day of the transaction"
    return f"Transaction at hour {int(hour):02d} of the day"


def _amount_ratio(values: Mapping[str, float]) -> str:
    ratio = values.get("amount_to_orig_prior_mean")
    if _is_missing(ratio):
        return "No earlier transactions to compare the amount with"
    return f"Amount is {ratio:.1f} times the originator's average earlier amount"


def _activity(role: str, label: str) -> Renderer:
    def render(values: Mapping[str, float]) -> str:
        count = values.get(f"{role}_prior_count")
        gap = values.get(f"{role}_steps_since_prev")
        if count == 0 or _is_missing(gap):
            return f"First observed transaction of the {label} account"
        return (
            f"{label.capitalize()} account has {int(count)} earlier "
            f"transactions, the last {gap:.0f} hours before"
        )

    return render


def _amount_profile(role: str, label: str) -> Renderer:
    def render(values: Mapping[str, float]) -> str:
        mean = values.get(f"{role}_prior_amount_mean")
        if _is_missing(mean):
            return f"No amount history for the {label} account"
        return f"Earlier amounts of the {label} account average {mean:,.0f}"

    return render


def _balance_residual(role: str, label: str) -> Renderer:
    def render(values: Mapping[str, float]) -> str:
        residual = values.get(f"error_balance_{role}")
        if _is_missing(residual):
            return f"{label.capitalize()} balance not recorded"
        if residual == 0:
            return f"{label.capitalize()} balance reconciles with the amount"
        return (
            f"{label.capitalize()} balance differs from the expected value "
            f"by {residual:,.0f}"
        )

    return render


def _dest_zero(values: Mapping[str, float]) -> str:
    if values.get("dest_balances_zero") == 1:
        return "Destination balances are zero before and after a credit"
    return "Destination balance pattern"


def _merchant(values: Mapping[str, float]) -> str:
    if values.get("is_merchant_dest") == 1:
        return "Destination is a merchant account"
    return "Destination is a customer account"


def _history(label: str) -> Renderer:
    def render(values: Mapping[str, float]) -> str:
        return f"Sequence of recent transactions of the {label} account"

    return render


def _reconstruction(values: Mapping[str, float]) -> str:
    error = values.get("ae_reconstruction_error")
    if _is_missing(error):
        return "Unusual combination of features"
    return (
        "Unusual combination of features relative to legitimate traffic "
        f"(reconstruction error {error:.3g})"
    )


def default_reason_groups() -> tuple[ReasonGroup, ...]:
    """Reason groups for the features built by :mod:`fraud_detection.features`.

    History lag columns are matched by pattern in :func:`group_features`, so
    they are not listed here.
    """
    return (
        ReasonGroup("R01", ("is_transfer", "is_cash_out"), _type),
        ReasonGroup("R02", ("log_amount", "amount_bin"), _amount),
        ReasonGroup("R03", ("amount_to_orig_prior_mean",), _amount_ratio),
        ReasonGroup(
            "R04",
            ("orig_prior_count", "orig_steps_since_prev"),
            _activity("orig", "originator"),
        ),
        ReasonGroup(
            "R05",
            ("orig_prior_amount_mean", "orig_prior_amount_std"),
            _amount_profile("orig", "originator"),
        ),
        ReasonGroup(
            "R06",
            ("dest_prior_count", "dest_steps_since_prev"),
            _activity("dest", "destination"),
        ),
        ReasonGroup(
            "R07",
            ("dest_prior_amount_mean", "dest_prior_amount_std"),
            _amount_profile("dest", "destination"),
        ),
        ReasonGroup(
            "R08", ("error_balance_orig",), _balance_residual("orig", "originator")
        ),
        ReasonGroup(
            "R09", ("error_balance_dest",), _balance_residual("dest", "destination")
        ),
        ReasonGroup("R10", ("dest_balances_zero",), _dest_zero),
        ReasonGroup("R11", ("hour_of_day", "hour_sin", "hour_cos"), _hour),
        ReasonGroup("R12", ("is_merchant_dest",), _merchant),
        ReasonGroup("R13", (), _history("originator")),
        ReasonGroup("R14", (), _history("destination")),
        ReasonGroup("R15", ("ae_reconstruction_error",), _reconstruction),
    )


_HISTORY_PATTERN = re.compile(r"^(orig|dest)_hist\d+_")
_HISTORY_CODES = {"orig": "R13", "dest": "R14"}
FALLBACK_CODE = "R99"


def group_features(
    feature_names: list[str], groups: tuple[ReasonGroup, ...]
) -> dict[str, list[str]]:
    """Assign every feature to the code of its reason group.

    Parameters
    ----------
    feature_names
        Model features.
    groups
        Reason groups, e.g. :func:`default_reason_groups`.

    Returns
    -------
    dict
        Code mapped to its features. Features of no group are each placed
        under their own key ``"R99:<feature>"``.
    """
    owner = {feature: group.code for group in groups for feature in group.features}
    assignment: dict[str, list[str]] = {}
    for feature in feature_names:
        match = _HISTORY_PATTERN.match(feature)
        if feature in owner:
            code = owner[feature]
        elif match:
            code = _HISTORY_CODES[match.group(1)]
        else:
            code = f"{FALLBACK_CODE}:{feature}"
        assignment.setdefault(code, []).append(feature)
    return assignment


def _render(
    code: str,
    features: list[str],
    values: Mapping[str, float],
    renderers: Mapping[str, Renderer],
) -> str:
    if code in renderers:
        return renderers[code](values)
    feature = features[0]
    return f"{feature} = {values[feature]:.4g}"


def reasons_for_row(
    explanation: ShapExplanation,
    row: int,
    max_reasons: int = 3,
    min_contribution: float = 0.0,
    groups: tuple[ReasonGroup, ...] | None = None,
) -> list[Reason]:
    """Top reasons that raised the fraud score of one row.

    Parameters
    ----------
    explanation
        SHAP values containing the row.
    row
        Position of the row within ``explanation``.
    max_reasons
        Maximum number of reasons :math:`k`.
    min_contribution
        Threshold :math:`\\epsilon` on the group contribution, in the
        explanation's output space.
    groups
        Reason groups; defaults to :func:`default_reason_groups`.

    Returns
    -------
    list of Reason
        Reasons ordered by decreasing contribution; may be shorter than
        ``max_reasons``.
    """
    groups = groups or default_reason_groups()
    renderers = {group.code: group.render for group in groups}
    names = explanation.feature_names
    contributions = pd.Series(explanation.values[row], index=names)
    values = {name: float(value) for name, value in explanation.data.iloc[row].items()}
    scored = []
    for code, features in group_features(names, groups).items():
        total = float(contributions[features].sum())
        if total > min_contribution:
            scored.append((total, code, features))
    scored.sort(key=lambda item: -item[0])
    return [
        Reason(
            code=code.split(":")[0],
            description=_render(code, features, values, renderers),
            contribution=total,
        )
        for total, code, features in scored[:max_reasons]
    ]


def reason_table(
    explanation: ShapExplanation,
    max_reasons: int = 3,
    min_contribution: float = 0.0,
    groups: tuple[ReasonGroup, ...] | None = None,
) -> pd.DataFrame:
    """Reason codes for every explained row.

    Parameters
    ----------
    explanation
        SHAP values of the alerted rows.
    max_reasons, min_contribution, groups
        Passed to :func:`reasons_for_row`.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``explanation.data``, with columns ``reason_<i>_code``,
        ``reason_<i>`` and ``reason_<i>_contribution`` for
        :math:`i = 1, \\dots, k`. Unused slots are missing.
    """
    columns = [
        f"reason_{i}{suffix}"
        for i in range(1, max_reasons + 1)
        for suffix in ("_code", "", "_contribution")
    ]
    records = []
    for row in range(len(explanation.data)):
        record: dict[str, object] = {}
        reasons = reasons_for_row(
            explanation, row, max_reasons, min_contribution, groups
        )
        for i in range(1, max_reasons + 1):
            reason = reasons[i - 1] if i <= len(reasons) else None
            record[f"reason_{i}_code"] = reason.code if reason else None
            record[f"reason_{i}"] = reason.description if reason else None
            record[f"reason_{i}_contribution"] = (
                reason.contribution if reason else np.nan
            )
        records.append(record)
    return pd.DataFrame(records, index=explanation.data.index, columns=columns)


__all__ = [
    "FALLBACK_CODE",
    "Reason",
    "ReasonGroup",
    "default_reason_groups",
    "group_features",
    "reason_table",
    "reasons_for_row",
]
