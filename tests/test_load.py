from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.load import (
    TRANSACTION_TYPE_DTYPE,
    SchemaError,
    load_transactions,
    paysim_dtypes,
    validate_transactions,
)
from fraud_detection.data.schema import PAYSIM_COLUMNS
from fraud_detection.data.synthetic import generate_paysim_like


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return generate_paysim_like(n_rows=2_000, fraud_rate=0.01, seed=3)


@pytest.fixture
def csv_path(frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "paysim.csv"
    frame.to_csv(path, index=False)
    return path


def test_round_trip_preserves_values(frame: pd.DataFrame, csv_path: Path) -> None:
    loaded = load_transactions(csv_path)
    assert tuple(loaded.columns) == PAYSIM_COLUMNS
    assert len(loaded) == len(frame)
    np.testing.assert_allclose(loaded["amount"], frame["amount"])
    assert (loaded["type"].astype(str) == frame["type"]).all()
    assert (loaded["nameOrig"].astype(str) == frame["nameOrig"]).all()


def test_compact_dtypes(csv_path: Path) -> None:
    loaded = load_transactions(csv_path)
    assert loaded["step"].dtype == np.int32
    assert loaded["isFraud"].dtype == np.int8
    assert loaded["type"].dtype == TRANSACTION_TYPE_DTYPE
    assert loaded["amount"].dtype == np.float64


def test_money_dtype_override(csv_path: Path) -> None:
    loaded = load_transactions(csv_path, money_dtype="float32")
    assert loaded["oldbalanceOrg"].dtype == np.float32
    assert paysim_dtypes("float32")["newbalanceDest"] == "float32"


def test_nrows_and_extra_columns_ignored(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "extra.csv"
    frame.assign(extra=1).iloc[:, ::-1].to_csv(path, index=False)
    loaded = load_transactions(path, nrows=100)
    assert len(loaded) == 100
    assert tuple(loaded.columns) == PAYSIM_COLUMNS


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_transactions(tmp_path / "absent.csv")


def test_missing_column_raises(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    frame.drop(columns="amount").to_csv(path, index=False)
    with pytest.raises(SchemaError, match="amount"):
        load_transactions(path)


def test_unknown_type_raises(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "unknown.csv"
    frame.assign(type=frame["type"].where(frame.index > 0, "REFUND")).to_csv(
        path, index=False
    )
    with pytest.raises(SchemaError, match="REFUND"):
        load_transactions(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("isFraud", 2, "only 0 and 1"),
        ("amount", -1.0, "negative"),
        ("newbalanceDest", -5.0, "negative"),
        ("step", 0, "start at 1"),
        ("amount", np.nan, "Null"),
    ],
)
def test_validation_rejects_bad_values(
    frame: pd.DataFrame, column: str, value: float, message: str
) -> None:
    bad = frame.copy()
    bad[column] = bad[column].astype("float64")
    bad.loc[0, column] = value
    with pytest.raises(SchemaError, match=message):
        validate_transactions(bad)


def test_validation_returns_input(frame: pd.DataFrame) -> None:
    assert validate_transactions(frame) is frame
