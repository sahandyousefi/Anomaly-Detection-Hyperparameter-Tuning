"""Feature construction for PaySim transactions.

Transaction-level features depend only on the row itself, so they can be built
before the split without leaking information across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import FeaturesConfig
from fraud_detection.data.schema import FLAGGED_TRANSFER_THRESHOLD, MERCHANT_PREFIX

HOURS_PER_DAY = 24

# Fixed edges, so binning needs no fitting; 200,000 is the simulator's flag rule
AMOUNT_BIN_EDGES: tuple[float, ...] = (
    0.0,
    1e2,
    1e3,
    1e4,
    1e5,
    FLAGGED_TRANSFER_THRESHOLD,
    1e6,
    1e7,
    np.inf,
)

TRANSACTION_FEATURES: tuple[str, ...] = (
    "is_transfer",
    "is_cash_out",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "is_merchant_dest",
    "log_amount",
    "amount_bin",
)


def type_flags(types: pd.Series) -> pd.DataFrame:
    """One-hot flags for the two transaction types that carry fraud.

    Parameters
    ----------
    types
        Raw transaction type strings or a categorical with string categories.
        Flags must be built from the strings, not from label-encoded codes.

    Returns
    -------
    pandas.DataFrame
        ``is_transfer`` and ``is_cash_out`` as ``int8``.
    """
    as_text = types.astype(str)
    return pd.DataFrame(
        {
            "is_transfer": (as_text == "TRANSFER").astype("int8"),
            "is_cash_out": (as_text == "CASH_OUT").astype("int8"),
        },
        index=types.index,
    )


def hour_features(steps: pd.Series) -> pd.DataFrame:
    """Hour of day and its cyclic encoding.

    Step 1 is taken as hour 0 of the first simulated day. The sine and cosine
    terms place 23:00 next to 00:00, which an integer hour cannot express for
    linear or distance-based models.

    Parameters
    ----------
    steps
        Hourly step index starting at 1.

    Returns
    -------
    pandas.DataFrame
        ``hour_of_day`` (``int8``), ``hour_sin``, and ``hour_cos``.
    """
    hour = ((steps.astype("int64") - 1) % HOURS_PER_DAY).astype("int8")
    angle = 2 * np.pi * hour.to_numpy() / HOURS_PER_DAY
    return pd.DataFrame(
        {"hour_of_day": hour, "hour_sin": np.sin(angle), "hour_cos": np.cos(angle)},
        index=steps.index,
    )


def merchant_flag(destinations: pd.Series) -> pd.Series:
    """Flag destinations that are merchant accounts (prefix ``M``)."""
    flag = destinations.astype(str).str.startswith(MERCHANT_PREFIX)
    return flag.astype("int8").rename("is_merchant_dest")


def amount_features(amounts: pd.Series) -> pd.DataFrame:
    """Log-scaled amount and its ordinal bin.

    Parameters
    ----------
    amounts
        Non-negative transaction amounts.

    Returns
    -------
    pandas.DataFrame
        ``log_amount`` as ``log1p(amount)`` and ``amount_bin`` as the
        zero-based index of the interval in :data:`AMOUNT_BIN_EDGES`, with
        intervals closed on the left.
    """
    values = amounts.to_numpy(dtype="float64")
    bins = np.searchsorted(AMOUNT_BIN_EDGES, values, side="right") - 1
    return pd.DataFrame(
        {"log_amount": np.log1p(values), "amount_bin": bins.astype("int8")},
        index=amounts.index,
    )


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append row-level features to a PaySim transaction table.

    Parameters
    ----------
    df
        Transactions with at least ``type``, ``step``, ``nameDest``, and
        ``amount``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the columns in :data:`TRANSACTION_FEATURES` added.
    """
    features = pd.concat(
        [
            type_flags(df["type"]),
            hour_features(df["step"]),
            merchant_flag(df["nameDest"]),
            amount_features(df["amount"]),
        ],
        axis=1,
    )
    return pd.concat([df, features[list(TRANSACTION_FEATURES)]], axis=1)


ACCOUNT_ROLES: dict[str, str] = {"orig": "nameOrig", "dest": "nameDest"}

_AGGREGATE_SUFFIXES: tuple[str, ...] = (
    "prior_count",
    "prior_amount_mean",
    "prior_amount_std",
    "steps_since_prev",
)

ACCOUNT_FEATURES: tuple[str, ...] = (
    *(f"{role}_{suffix}" for role in ACCOUNT_ROLES for suffix in _AGGREGATE_SUFFIXES),
    "amount_to_orig_prior_mean",
)


def _reference_amount(
    keys: pd.Series, steps: pd.Series, amount: pd.Series
) -> pd.Series:
    """Smallest amount in each account's first step, broadcast to its rows.

    Sums of squares are taken around this value instead of zero, which avoids
    the cancellation that otherwise leaves a spurious standard deviation of
    order 1e-3 for accounts with large, repeated amounts. It depends only on
    the earliest step, so later rows cannot change it.
    """
    first_step = steps.groupby(keys.to_numpy()).transform("min")
    in_first = amount.where(steps == first_step)
    return in_first.groupby(keys.to_numpy()).transform("min")


def prior_aggregates(
    df: pd.DataFrame,
    key: str,
    prefix: str,
    time_column: str = "step",
    amount_column: str = "amount",
) -> pd.DataFrame:
    """Per-account statistics over transactions in strictly earlier steps.

    Rows that share a step with the current row are excluded, because PaySim
    gives no ordering within an hour. In particular the transfer and cash-out
    of a fraud event happen in the same step, so neither can see the other.

    Parameters
    ----------
    df
        Transactions. Row order does not matter.
    key
        Account column to group by.
    prefix
        Prefix of the output column names.
    time_column
        Integer time index.
    amount_column
        Column the amount statistics are computed on.

    Returns
    -------
    pandas.DataFrame
        Aligned with ``df.index``. ``<prefix>_prior_count`` is the number of
        earlier transactions of the account. The mean and population standard
        deviation of their amounts, and the number of steps since the latest
        earlier transaction, are NaN when the count is zero.
    """
    amount = df[amount_column].astype("float64")
    ref = _reference_amount(df[key], df[time_column], amount)
    shifted = (amount - ref).to_numpy()
    per_step = (
        pd.DataFrame(
            {
                "key": df[key].to_numpy(),
                "step": df[time_column].to_numpy(),
                "n": 1,
                "s": shifted,
                "ss": shifted**2,
            }
        )
        .groupby(["key", "step"], sort=True)
        .sum()
    )

    # Exclusive cumulative sums: totals over all earlier steps of the same key
    totals = per_step.groupby(level="key").cumsum() - per_step
    steps = per_step.index.get_level_values("step").to_series(index=per_step.index)
    prev_step = steps.groupby(level="key").shift(1)

    n = totals["n"].to_numpy()
    key_ref = ref.groupby(df[key].to_numpy()).first()
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted_mean = np.where(n > 0, totals["s"].to_numpy() / n, np.nan)
        var = np.where(n > 0, totals["ss"].to_numpy() / n - shifted_mean**2, np.nan)
    mean = shifted_mean + key_ref.reindex(per_step.index.get_level_values("key"))
    stats = pd.DataFrame(
        {
            f"{prefix}_prior_count": n.astype("int32"),
            f"{prefix}_prior_amount_mean": mean.to_numpy(),
            # Clip tiny negative variances caused by floating-point cancellation
            f"{prefix}_prior_amount_std": np.sqrt(np.clip(var, 0.0, None)),
            f"{prefix}_steps_since_prev": (steps - prev_step).to_numpy(),
        },
        index=per_step.index,
    )

    lookup = pd.MultiIndex.from_arrays(
        [df[key].to_numpy(), df[time_column].to_numpy()], names=["key", "step"]
    )
    result = stats.reindex(lookup)
    result.index = df.index
    return result


def add_account_features(df: pd.DataFrame, time_column: str = "step") -> pd.DataFrame:
    """Append point-in-time originator and destination aggregates.

    The aggregates use only transactions from earlier steps and never use
    labels, so they can be computed once on the full chronological log. A row
    in a later fold then sees the history that would have been available when
    it was scored.

    Parameters
    ----------
    df
        Transactions with ``nameOrig``, ``nameDest``, ``amount``, and the time
        column.
    time_column
        Integer time index.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the columns in :data:`ACCOUNT_FEATURES` added.
    """
    parts = [
        prior_aggregates(df, key, role, time_column=time_column)
        for role, key in ACCOUNT_ROLES.items()
    ]
    features = pd.concat(parts, axis=1)
    features["amount_to_orig_prior_mean"] = (
        df["amount"].astype("float64") / features["orig_prior_amount_mean"]
    ).replace([np.inf, -np.inf], np.nan)
    return pd.concat([df, features[list(ACCOUNT_FEATURES)]], axis=1)


BALANCE_FEATURES: tuple[str, ...] = (
    "error_balance_orig",
    "error_balance_dest",
    "dest_balances_zero",
)

# Transaction types in which the originator's balance increases
_CREDIT_TO_ORIG_TYPES: tuple[str, ...] = ("CASH_IN",)


def balance_features(df: pd.DataFrame) -> pd.DataFrame:
    r"""Reconciliation residuals of the originator and destination balances.

    For each side the residual is the recorded new balance minus the balance
    implied by the old balance and the amount:

    .. math::

        e_{orig} = b^{new}_{orig} - (b^{old}_{orig} - s \, a), \qquad
        e_{dest} = b^{new}_{dest} - (b^{old}_{dest} + s \, a),

    where :math:`a` is the amount and :math:`s = -1` for ``CASH_IN`` (money
    flows to the originator) and :math:`s = 1` otherwise. A consistent ledger
    gives zero on both sides. In PaySim the residuals separate fraud well,
    largely because destination balances of fraudulent transfers are often
    recorded as zero, so they partly describe the simulator rather than real
    fraud behaviour.

    Parameters
    ----------
    df
        Transactions with ``type``, ``amount``, ``nameDest``, and the four
        balance columns.

    Returns
    -------
    pandas.DataFrame
        ``error_balance_orig`` and ``error_balance_dest`` as ``float64``, and
        ``dest_balances_zero`` (``int8``), which marks a non-merchant
        destination whose old and new balances are both zero although a
        positive amount was received. Merchant balances are not recorded, so
        ``error_balance_dest`` is NaN for merchant destinations.
    """
    amount = df["amount"].to_numpy(dtype="float64")
    direction = np.where(
        df["type"].astype(str).isin(_CREDIT_TO_ORIG_TYPES).to_numpy(), -1.0, 1.0
    )
    old_orig = df["oldbalanceOrg"].to_numpy(dtype="float64")
    new_orig = df["newbalanceOrig"].to_numpy(dtype="float64")
    old_dest = df["oldbalanceDest"].to_numpy(dtype="float64")
    new_dest = df["newbalanceDest"].to_numpy(dtype="float64")
    is_merchant = merchant_flag(df["nameDest"]).to_numpy().astype(bool)

    error_dest = new_dest - (old_dest + direction * amount)
    dest_zero = (old_dest == 0) & (new_dest == 0) & (amount > 0) & ~is_merchant
    return pd.DataFrame(
        {
            "error_balance_orig": new_orig - (old_orig - direction * amount),
            "error_balance_dest": np.where(is_merchant, np.nan, error_dest),
            "dest_balances_zero": dest_zero.astype("int8"),
        },
        index=df.index,
    )


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the columns in :data:`BALANCE_FEATURES` to a copy of ``df``."""
    return pd.concat([df, balance_features(df)], axis=1)


def feature_columns(config: FeaturesConfig | None = None) -> list[str]:
    """Names of the model input columns produced by :func:`build_features`.

    Parameters
    ----------
    config
        Feature switches. Defaults to all groups enabled.

    Returns
    -------
    list of str
        Transaction, account, and (if enabled) balance and history feature
        names.
    """
    from fraud_detection.features.sequence import history_columns

    config = config or FeaturesConfig()
    columns = [*TRANSACTION_FEATURES, *ACCOUNT_FEATURES]
    if config.balance_features:
        columns.extend(BALANCE_FEATURES)
    columns.extend(history_columns(config.history_length, config.history_roles))
    return columns


def build_features(
    df: pd.DataFrame,
    config: FeaturesConfig | None = None,
    time_column: str = "step",
) -> pd.DataFrame:
    """Build every enabled feature group on the full chronological log.

    All groups are either row-level or point-in-time and none uses labels, so
    the table can be split into folds afterwards without leakage. Resampling
    must happen after this step and on the training fold only.

    Parameters
    ----------
    df
        Raw PaySim transactions.
    config
        Feature switches. Defaults to all groups enabled.
    time_column
        Integer time index.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the columns of :func:`feature_columns` appended.
    """
    # Imported here because the history module reuses ACCOUNT_ROLES from this one
    from fraud_detection.features.sequence import add_history_features

    config = config or FeaturesConfig()
    result = add_account_features(add_transaction_features(df), time_column)
    if config.balance_features:
        result = add_balance_features(result)
    if config.history_length > 0:
        result = add_history_features(
            result, config.history_length, config.history_roles, time_column
        )
    return result
