import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.schema import (
    FLAGGED_TRANSFER_THRESHOLD,
    FRAUD_TRANSACTION_TYPES,
    PAYSIM_COLUMNS,
    TRANSACTION_TYPES,
)
from fraud_detection.data.synthetic import generate_paysim_like


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_paysim_like(
        n_rows=20_000, fraud_rate=0.02, flag_probability=0.5, seed=7
    )


def test_schema(df: pd.DataFrame) -> None:
    assert tuple(df.columns) == PAYSIM_COLUMNS
    assert len(df) == 20_000
    assert set(df["type"]) <= set(TRANSACTION_TYPES)
    assert df[["isFraud", "isFlaggedFraud"]].isin([0, 1]).all().all()
    assert not df.isna().any().any()


def test_steps_sorted_and_in_range(df: pd.DataFrame) -> None:
    assert df["step"].is_monotonic_increasing
    assert df["step"].between(1, 743).all()


def test_seed_reproducibility() -> None:
    a = generate_paysim_like(n_rows=500, fraud_rate=0.05, seed=1)
    b = generate_paysim_like(n_rows=500, fraud_rate=0.05, seed=1)
    c = generate_paysim_like(n_rows=500, fraud_rate=0.05, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_fraud_rate(df: pd.DataFrame) -> None:
    assert df["isFraud"].sum() == 400


def test_fraud_only_in_transfer_and_cash_out(df: pd.DataFrame) -> None:
    assert set(df.loc[df["isFraud"] == 1, "type"]) == set(FRAUD_TRANSACTION_TYPES)


def test_fraud_transfers_empty_the_victim_account(df: pd.DataFrame) -> None:
    transfers = df[(df["isFraud"] == 1) & (df["type"] == "TRANSFER")]
    assert (transfers["newbalanceOrig"] == 0).all()
    np.testing.assert_allclose(transfers["amount"], transfers["oldbalanceOrg"])


def test_every_fraud_transfer_has_matching_cash_out(df: pd.DataFrame) -> None:
    fraud = df[df["isFraud"] == 1]
    transfers = fraud[fraud["type"] == "TRANSFER"][["step", "nameDest", "amount"]]
    cash_outs = fraud[fraud["type"] == "CASH_OUT"][["step", "nameOrig", "amount"]]
    merged = transfers.merge(
        cash_outs,
        left_on=["step", "nameDest", "amount"],
        right_on=["step", "nameOrig", "amount"],
    )
    assert len(merged) >= len(transfers)


def test_payments_go_to_merchants_with_zero_balances(df: pd.DataFrame) -> None:
    payments = df[df["type"] == "PAYMENT"]
    assert payments["nameDest"].str.startswith("M").all()
    merchants = df[df["nameDest"].str.startswith("M")]
    assert (merchants[["oldbalanceDest", "newbalanceDest"]] == 0).all().all()


def test_legitimate_outflows_reduce_origin_balance(df: pd.DataFrame) -> None:
    legit = df[(df["isFraud"] == 0) & (df["type"] != "CASH_IN")]
    expected = np.maximum(legit["oldbalanceOrg"] - legit["amount"], 0)
    np.testing.assert_allclose(legit["newbalanceOrig"], expected, atol=0.01)


def test_flagged_rows_are_large_fraudulent_transfers(df: pd.DataFrame) -> None:
    flagged = df[df["isFlaggedFraud"] == 1]
    assert len(flagged) > 0
    assert (flagged["isFraud"] == 1).all()
    assert (flagged["type"] == "TRANSFER").all()
    assert (flagged["amount"] > FLAGGED_TRANSFER_THRESHOLD).all()


def test_zero_fraud_rate() -> None:
    out = generate_paysim_like(n_rows=300, fraud_rate=0.0, seed=0)
    assert out["isFraud"].sum() == 0


def test_small_positive_rate_yields_one_event() -> None:
    out = generate_paysim_like(n_rows=100, fraud_rate=0.0001, seed=0)
    assert out["isFraud"].sum() == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rows": 1},
        {"fraud_rate": 1.0},
        {"fraud_rate": -0.1},
        {"n_steps": 0},
        {"flag_probability": 1.5},
    ],
)
def test_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        generate_paysim_like(**kwargs)
