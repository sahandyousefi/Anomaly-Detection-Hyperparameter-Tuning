"""Loading and validation of the PaySim transaction log."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fraud_detection.data.schema import (
    BALANCE_COLUMNS,
    LABEL_COLUMNS,
    PAYSIM_COLUMNS,
    TRANSACTION_TYPES,
)

TRANSACTION_TYPE_DTYPE = pd.CategoricalDtype(list(TRANSACTION_TYPES))


class SchemaError(ValueError):
    """Raised when a transaction table does not match the PaySim schema."""


def paysim_dtypes(money_dtype: str = "float64") -> dict[str, object]:
    """Return the column dtypes used when reading the PaySim CSV.

    Parameters
    ----------
    money_dtype
        Dtype of the amount and balance columns. ``float64`` is the default
        because ``float32`` keeps only about seven significant digits, which
        loses cents on the largest amounts and adds noise to the balance
        reconciliation features.

    Returns
    -------
    dict
        Mapping from column name to dtype, suitable for ``pandas.read_csv``.
    """
    dtypes: dict[str, object] = {
        "step": "int32",
        "type": TRANSACTION_TYPE_DTYPE,
        "amount": money_dtype,
        "nameOrig": "string",
        "nameDest": "string",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    dtypes.update(dict.fromkeys(BALANCE_COLUMNS, money_dtype))
    return dtypes


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}.")


def _require_no_nulls(df: pd.DataFrame, columns: Iterable[str]) -> None:
    counts = df[list(columns)].isna().sum()
    bad = counts[counts > 0]
    if not bad.empty:
        raise SchemaError(f"Null values found: {bad.to_dict()}.")


def _require_known_types(types: pd.Series) -> None:
    unknown = set(types.astype(str).unique()) - set(TRANSACTION_TYPES)
    if unknown:
        raise SchemaError(f"Unknown transaction types: {sorted(unknown)}.")


def _require_binary(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if not df[column].isin([0, 1]).all():
            raise SchemaError(f"Column {column!r} must contain only 0 and 1.")


def _require_non_negative(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if (df[column] < 0).any():
            raise SchemaError(f"Column {column!r} contains negative values.")


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Check that a table follows the PaySim schema and value domains.

    Parameters
    ----------
    df
        Transaction table with at least the PaySim columns.

    Returns
    -------
    pandas.DataFrame
        The input table, unchanged, so the call can be chained.

    Raises
    ------
    SchemaError
        If a column is missing, contains nulls, or holds values outside its
        domain (unknown transaction type, non-binary label, negative amount or
        balance, or a step below 1).
    """
    _require_columns(df, PAYSIM_COLUMNS)
    _require_no_nulls(df, PAYSIM_COLUMNS)
    _require_known_types(df["type"])
    _require_binary(df, LABEL_COLUMNS)
    _require_non_negative(df, ("amount", *BALANCE_COLUMNS))
    if (df["step"] < 1).any():
        raise SchemaError("Column 'step' must start at 1.")
    return df


def load_transactions(
    path: str | Path,
    *,
    nrows: int | None = None,
    money_dtype: str = "float64",
    validate: bool = True,
) -> pd.DataFrame:
    """Read the PaySim CSV with compact dtypes and optional validation.

    Parameters
    ----------
    path
        Location of the CSV file.
    nrows
        Read only the first ``nrows`` rows, for quick experiments.
    money_dtype
        Dtype of the amount and balance columns, see :func:`paysim_dtypes`.
    validate
        Run :func:`validate_transactions` on the result.

    Returns
    -------
    pandas.DataFrame
        Transactions with columns in PaySim order. ``type`` is categorical with
        a fixed category order, so codes are stable across files.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SchemaError
        If validation is enabled and fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Transaction file not found: {path}. See data/README.md for download "
            "instructions."
        )

    header = pd.read_csv(path, nrows=0).columns
    _require_columns(pd.DataFrame(columns=header), PAYSIM_COLUMNS)

    # Read type as text first so unknown values are reported instead of becoming NaN
    dtypes = paysim_dtypes(money_dtype)
    dtypes["type"] = "string"
    df = pd.read_csv(path, usecols=list(PAYSIM_COLUMNS), dtype=dtypes, nrows=nrows)
    df = df[list(PAYSIM_COLUMNS)]

    if validate:
        validate_transactions(df)
    df["type"] = df["type"].astype(TRANSACTION_TYPE_DTYPE)
    return df
