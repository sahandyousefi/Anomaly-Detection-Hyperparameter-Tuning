"""Every feature checked against values worked out by hand on a small ledger."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.build import build_features, feature_columns

NAN = np.nan
# Hour 1 on the daily circle
SIN_1, COS_1 = np.sin(np.pi / 12), np.cos(np.pi / 12)

# Account C1 sends four times, C2 twice; C3 receives twice, 24 steps apart
LEDGER = pd.DataFrame(
    {
        "step": [1, 1, 3, 25, 26, 26],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        "amount": [1_000.0, 1_000.0, 50.0, 400.0, 250_000.0, 100.0],
        "nameOrig": ["C1", "C2", "C1", "C1", "C2", "C1"],
        "oldbalanceOrg": [1_000.0, 1_500.0, 300.0, 250.0, 600_000.0, 650.0],
        "newbalanceOrig": [0.0, 500.0, 250.0, 650.0, 350_000.0, 600.0],
        "nameDest": ["C2", "C3", "M1", "C3", "C1", "C4"],
        "oldbalanceDest": [0.0, 200.0, 0.0, 1_200.0, 650.0, 0.0],
        "newbalanceDest": [0.0, 1_200.0, 0.0, 800.0, 250_650.0, 100.0],
        "isFraud": [1, 1, 0, 0, 0, 0],
        "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
    }
)

EXPECTED = pd.DataFrame(
    {
        "is_transfer": [1, 0, 0, 0, 1, 0],
        "is_cash_out": [0, 1, 0, 0, 0, 0],
        "hour_of_day": [0, 0, 2, 0, 1, 1],
        "hour_sin": [0.0, 0.0, 0.5, 0.0, SIN_1, SIN_1],
        "hour_cos": [1.0, 1.0, np.sqrt(3) / 2, 1.0, COS_1, COS_1],
        "is_merchant_dest": [0, 0, 1, 0, 0, 0],
        "log_amount": np.log1p([1_000.0, 1_000.0, 50.0, 400.0, 250_000.0, 100.0]),
        "amount_bin": [2, 2, 0, 1, 5, 1],
        "orig_prior_count": [0, 0, 1, 2, 1, 3],
        "orig_prior_amount_mean": [NAN, NAN, 1_000.0, 525.0, 1_000.0, 1_450.0 / 3],
        "orig_prior_amount_std": [NAN, NAN, 0.0, 475.0, 0.0, np.sqrt(4_155_000 / 27)],
        "orig_steps_since_prev": [NAN, NAN, 2.0, 22.0, 25.0, 1.0],
        "dest_prior_count": [0, 0, 0, 1, 0, 0],
        "dest_prior_amount_mean": [NAN, NAN, NAN, 1_000.0, NAN, NAN],
        "dest_prior_amount_std": [NAN, NAN, NAN, 0.0, NAN, NAN],
        "dest_steps_since_prev": [NAN, NAN, NAN, 24.0, NAN, NAN],
        "amount_to_orig_prior_mean": [NAN, NAN, 0.05, 400 / 525, 250.0, 300 / 1_450],
        "error_balance_orig": [0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        "error_balance_dest": [-1_000.0, 0.0, NAN, 0.0, 0.0, 0.0],
        "dest_balances_zero": [1, 0, 0, 0, 0, 0],
    }
)


def _check(result: pd.DataFrame) -> None:
    got = result[feature_columns()].astype("float64")
    expected = EXPECTED[feature_columns()].astype("float64")
    pd.testing.assert_frame_equal(got, expected, rtol=1e-6)


def test_expected_table_covers_every_feature() -> None:
    assert set(EXPECTED.columns) == set(feature_columns())


def test_features_match_hand_computation() -> None:
    _check(build_features(LEDGER))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_order_does_not_change_values(seed: int) -> None:
    shuffled = LEDGER.sample(frac=1.0, random_state=seed)
    _check(build_features(shuffled).loc[LEDGER.index])


def test_categorical_type_column_gives_same_values() -> None:
    ledger = LEDGER.astype({"type": "category", "nameDest": "string"})
    _check(build_features(ledger))
