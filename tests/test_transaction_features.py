import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.load import TRANSACTION_TYPE_DTYPE
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import (
    AMOUNT_BIN_EDGES,
    TRANSACTION_FEATURES,
    add_transaction_features,
    amount_features,
    hour_features,
    merchant_flag,
    type_flags,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return generate_paysim_like(n_rows=3_000, fraud_rate=0.01, seed=5)


@pytest.mark.parametrize("as_category", [False, True])
def test_type_flags_from_raw_strings(as_category: bool) -> None:
    types = pd.Series(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
    if as_category:
        types = types.astype(TRANSACTION_TYPE_DTYPE)
    flags = type_flags(types)
    assert flags["is_transfer"].tolist() == [1, 0, 0, 0, 0]
    assert flags["is_cash_out"].tolist() == [0, 1, 0, 0, 0]


def test_hour_features_wrap_daily() -> None:
    hours = hour_features(pd.Series([1, 24, 25, 48, 743]))
    assert hours["hour_of_day"].tolist() == [0, 23, 0, 23, 22]
    np.testing.assert_allclose(hours["hour_sin"] ** 2 + hours["hour_cos"] ** 2, 1.0)
    # 23:00 and 00:00 are neighbours on the circle
    gap = np.hypot(
        hours["hour_sin"][1] - hours["hour_sin"][0],
        hours["hour_cos"][1] - hours["hour_cos"][0],
    )
    assert gap == pytest.approx(2 * np.sin(np.pi / 24))


def test_merchant_flag() -> None:
    flags = merchant_flag(pd.Series(["M123", "C456", "M0"], dtype="string"))
    assert flags.name == "is_merchant_dest"
    assert flags.tolist() == [1, 0, 1]


def test_amount_bins_follow_edges() -> None:
    amounts = pd.Series([0.0, 99.99, 100.0, 199_999.0, 200_000.0, 5e7])
    result = amount_features(amounts)
    assert result["amount_bin"].tolist() == [0, 0, 1, 4, 5, 7]
    assert result["amount_bin"].max() < len(AMOUNT_BIN_EDGES) - 1
    np.testing.assert_allclose(result["log_amount"], np.log1p(amounts))


def test_add_transaction_features(frame: pd.DataFrame) -> None:
    result = add_transaction_features(frame)
    assert list(result.columns) == [*frame.columns, *TRANSACTION_FEATURES]
    assert result.index.equals(frame.index)
    assert not result[list(TRANSACTION_FEATURES)].isna().any().any()
    pd.testing.assert_frame_equal(result[frame.columns], frame)


def test_fraud_rows_are_transfer_or_cash_out(frame: pd.DataFrame) -> None:
    result = add_transaction_features(frame)
    fraud = result[result["isFraud"] == 1]
    assert ((fraud["is_transfer"] + fraud["is_cash_out"]) == 1).all()


def test_features_respect_non_default_index(frame: pd.DataFrame) -> None:
    shuffled = frame.sample(frac=1.0, random_state=0)
    result = add_transaction_features(shuffled)
    expected = (shuffled["type"] == "TRANSFER").astype("int8")
    assert (result["is_transfer"] == expected).all()
