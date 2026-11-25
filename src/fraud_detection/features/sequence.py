r"""Point-in-time transaction histories of each account.

For a transaction :math:`t` of account :math:`u` at step :math:`\tau_t`, the
history of length :math:`L` is the sequence of the :math:`L` most recent
transactions of :math:`u` in strictly earlier steps,

.. math::

    H_L(t) = \big(x_{t_1}, \dots, x_{t_L}\big), \qquad
    \tau_{t_L} \le \dots \le \tau_{t_1} < \tau_t,

where :math:`t_1` is the latest of them. Each element carries the row-level
features in :data:`HISTORY_SOURCE_FEATURES` and its age
:math:`\tau_t - \tau_{t_k}` in steps. Transactions in the same step as
:math:`t` are excluded for the same reason as in
:func:`fraud_detection.features.build.prior_aggregates`: PaySim records no order
within an hour.

The histories are stored as flat columns ``<role>_hist<k>_<feature>``, with
:math:`k = 1` the most recent element, so they pass through the split and
resampling code like any other feature. Tree models can use them directly as
lag features, and :mod:`fraud_detection.models.sequence` reshapes them into
tensors. Elements beyond the available history are NaN.

Accounts are tracked separately in the originator role (prior transactions the
account sent) and the destination role (prior transactions it received). In
the public PaySim release most originator accounts occur only once, so
originator histories are mostly empty there; destination histories are longer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fraud_detection.features.build import ACCOUNT_ROLES

HISTORY_SOURCE_FEATURES: tuple[str, ...] = (
    "log_amount",
    "is_transfer",
    "is_cash_out",
    "is_merchant_dest",
    "hour_sin",
    "hour_cos",
)

HISTORY_ELEMENT_FEATURES: tuple[str, ...] = (*HISTORY_SOURCE_FEATURES, "steps_before")


def history_column(role: str, lag: int, feature: str) -> str:
    """Column name of one feature of one history element."""
    return f"{role}_hist{lag}_{feature}"


def history_columns(
    length: int, roles: Sequence[str] = tuple(ACCOUNT_ROLES)
) -> list[str]:
    """Names of the history columns, ordered by role, lag, then feature.

    Parameters
    ----------
    length
        History length :math:`L`; zero gives no columns.
    roles
        Account roles, keys of :data:`fraud_detection.features.build.ACCOUNT_ROLES`.

    Returns
    -------
    list of str
        Column names.
    """
    _check_roles(roles)
    return [
        history_column(role, lag, feature)
        for role in roles
        for lag in range(1, length + 1)
        for feature in HISTORY_ELEMENT_FEATURES
    ]


def _check_roles(roles: Sequence[str]) -> None:
    unknown = set(roles) - set(ACCOUNT_ROLES)
    if unknown:
        raise ValueError(
            f"Unknown account roles {sorted(unknown)}; expected {list(ACCOUNT_ROLES)}."
        )


def _sorted_blocks(
    keys: np.ndarray, steps: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort order and, per sorted row, the start of its account and of its step.

    Rows of one account and step are tied; sorting them by their feature
    values makes the history independent of the input row order.
    """
    codes = pd.factorize(keys)[0]
    order = np.lexsort((*values.T[::-1], steps, codes))
    n = len(order)
    positions = np.arange(n)
    codes_sorted, steps_sorted = codes[order], steps[order]
    new_account = np.r_[True, codes_sorted[1:] != codes_sorted[:-1]]
    new_step = new_account | np.r_[True, steps_sorted[1:] != steps_sorted[:-1]]
    account_start = np.maximum.accumulate(np.where(new_account, positions, 0))
    step_start = np.maximum.accumulate(np.where(new_step, positions, 0))
    return order, account_start, step_start


def account_history(
    df: pd.DataFrame,
    key: str,
    role: str,
    length: int,
    time_column: str = "step",
) -> pd.DataFrame:
    """History columns of one account role.

    Parameters
    ----------
    df
        Transactions holding ``key``, the time column, and
        :data:`HISTORY_SOURCE_FEATURES`. Row order does not matter.
    key
        Account column to group by.
    role
        Prefix of the output column names.
    length
        History length :math:`L`.
    time_column
        Integer time index.

    Returns
    -------
    pandas.DataFrame
        Aligned with ``df.index``, columns ``<role>_hist<k>_<feature>`` for
        :math:`k = 1, \\dots, L` and every feature in
        :data:`HISTORY_ELEMENT_FEATURES`.
    """
    if length < 1:
        raise ValueError("length must be at least 1.")
    missing = [c for c in HISTORY_SOURCE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Build the transaction features first; missing {missing}.")

    steps = df[time_column].to_numpy(dtype="int64")
    values = df[list(HISTORY_SOURCE_FEATURES)].to_numpy(dtype="float64")
    order, account_start, step_start = _sorted_blocks(df[key].to_numpy(), steps, values)
    values_sorted, steps_sorted = values[order], steps[order]

    columns: dict[str, np.ndarray] = {}
    for lag in range(1, length + 1):
        source = step_start - lag
        present = source >= account_start
        source = np.where(present, source, 0)
        element = np.where(present[:, None], values_sorted[source], np.nan)
        age = np.where(present, steps_sorted - steps_sorted[source], np.nan)
        for j, feature in enumerate(HISTORY_SOURCE_FEATURES):
            columns[history_column(role, lag, feature)] = element[:, j]
        columns[history_column(role, lag, "steps_before")] = age

    unsorted = np.empty_like(order)
    unsorted[order] = np.arange(len(order))
    return pd.DataFrame(
        {name: column[unsorted] for name, column in columns.items()}, index=df.index
    )


def add_history_features(
    df: pd.DataFrame,
    length: int,
    roles: Sequence[str] = tuple(ACCOUNT_ROLES),
    time_column: str = "step",
) -> pd.DataFrame:
    """Append the columns of :func:`history_columns` to a copy of ``df``.

    The histories use earlier steps only and no labels, so they can be built
    on the full chronological log before the split.
    """
    _check_roles(roles)
    if length == 0:
        return df.copy()
    parts = [
        account_history(df, ACCOUNT_ROLES[role], role, length, time_column)
        for role in roles
    ]
    return pd.concat([df, *parts], axis=1)
