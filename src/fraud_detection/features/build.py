"""Feature construction for PaySim transactions.

Transaction-level features depend only on the row itself, so they can be built
before the split without leaking information across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
