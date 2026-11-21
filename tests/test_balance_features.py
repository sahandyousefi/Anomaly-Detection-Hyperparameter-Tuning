import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FeaturesConfig
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import (
    BALANCE_FEATURES,
    balance_features,
    build_features,
    feature_columns,
)


def _rows(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_consistent_ledger_gives_zero_residuals() -> None:
    df = _rows(
        type=["TRANSFER", "CASH_IN", "CASH_OUT"],
        amount=[100.0, 50.0, 30.0],
        nameDest=["C2", "C3", "C4"],
        oldbalanceOrg=[500.0, 10.0, 30.0],
        newbalanceOrig=[400.0, 60.0, 0.0],
        oldbalanceDest=[0.0, 80.0, 5.0],
        newbalanceDest=[100.0, 30.0, 35.0],
    )
    result = balance_features(df)
    np.testing.assert_allclose(result["error_balance_orig"], 0.0)
    np.testing.assert_allclose(result["error_balance_dest"], 0.0)
    assert result["dest_balances_zero"].tolist() == [0, 0, 0]


def test_residual_signs_and_zero_destination() -> None:
    # Account emptied into a destination whose balances were never updated
    df = _rows(
        type=["TRANSFER", "CASH_OUT"],
        amount=[1_000.0, 200.0],
        nameDest=["C7", "C8"],
        oldbalanceOrg=[1_000.0, 150.0],
        newbalanceOrig=[0.0, 0.0],
        oldbalanceDest=[0.0, 0.0],
        newbalanceDest=[0.0, 200.0],
    )
    result = balance_features(df)
    # The second originator overdrew by 50, which the ledger did not record
    np.testing.assert_allclose(result["error_balance_orig"], [0.0, 50.0])
    np.testing.assert_allclose(result["error_balance_dest"], [-1_000.0, 0.0])
    assert result["dest_balances_zero"].tolist() == [1, 0]


def test_merchant_destination_is_missing_not_zero() -> None:
    df = _rows(
        type=["PAYMENT"],
        amount=[25.0],
        nameDest=["M1"],
        oldbalanceOrg=[100.0],
        newbalanceOrig=[75.0],
        oldbalanceDest=[0.0],
        newbalanceDest=[0.0],
    )
    result = balance_features(df)
    assert np.isnan(result.loc[0, "error_balance_dest"])
    assert result.loc[0, "dest_balances_zero"] == 0
    assert result.loc[0, "error_balance_orig"] == 0.0


def test_synthetic_fraud_transfers_have_zero_destination() -> None:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, seed=3)
    result = balance_features(df)
    transfers = (df["isFraud"] == 1) & (df["type"] == "TRANSFER")
    assert result.loc[transfers, "dest_balances_zero"].eq(1).all()
    np.testing.assert_allclose(result.loc[transfers, "error_balance_orig"], 0.0)


@pytest.mark.parametrize("enabled", [True, False])
def test_ablation_flag_controls_balance_columns(enabled: bool) -> None:
    df = generate_paysim_like(n_rows=500, fraud_rate=0.02, seed=1)
    config = FeaturesConfig(balance_features=enabled)
    result = build_features(df, config)
    columns = feature_columns(config)
    assert list(result.columns) == [*df.columns, *columns]
    assert all((c in columns) == enabled for c in BALANCE_FEATURES)
    pd.testing.assert_frame_equal(result[df.columns], df)
