import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ConfigError, FeaturesConfig
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import (
    add_transaction_features,
    build_features,
    feature_columns,
)
from fraud_detection.features.sequence import (
    HISTORY_ELEMENT_FEATURES,
    account_history,
    history_column,
    history_columns,
)


def _ledger() -> pd.DataFrame:
    rows = [
        # step, type, amount, nameOrig, nameDest
        (1, "PAYMENT", 100.0, "C1", "M1"),
        (2, "TRANSFER", 5_000.0, "C1", "C2"),
        (2, "CASH_OUT", 300.0, "C1", "C3"),
        (5, "TRANSFER", 7_000.0, "C1", "C2"),
        (5, "PAYMENT", 50.0, "C9", "M1"),
    ]
    df = pd.DataFrame(rows, columns=["step", "type", "amount", "nameOrig", "nameDest"])
    return add_transaction_features(df)


def test_column_names_and_order() -> None:
    names = history_columns(2, ["dest"])
    assert len(names) == 2 * len(HISTORY_ELEMENT_FEATURES)
    assert names[0] == "dest_hist1_log_amount"
    assert names[-1] == "dest_hist2_steps_before"
    assert history_columns(0) == []
    with pytest.raises(ValueError, match="roles"):
        history_columns(1, ["beneficiary"])


def test_history_uses_strictly_earlier_steps() -> None:
    hist = account_history(_ledger(), "nameOrig", "orig", length=3)
    amount1 = np.expm1(hist[history_column("orig", 1, "log_amount")])
    age1 = hist[history_column("orig", 1, "steps_before")]

    # Row 0 is the first transaction of C1
    assert hist.iloc[0].isna().all()
    # Both step-2 rows see only the step-1 payment, never each other
    np.testing.assert_allclose(amount1.iloc[1:3], [100.0, 100.0])
    np.testing.assert_allclose(age1.iloc[1:3], [1.0, 1.0])
    assert hist.loc[1:2, history_column("orig", 2, "log_amount")].isna().all()
    # The step-5 transfer sees the two step-2 rows first, then the payment
    lags = [
        np.expm1(hist.loc[3, history_column("orig", k, "log_amount")])
        for k in (1, 2, 3)
    ]
    np.testing.assert_allclose(sorted(lags[:2]), [300.0, 5_000.0])
    assert lags[2] == pytest.approx(100.0)
    assert hist.loc[3, history_column("orig", 3, "steps_before")] == 4
    assert hist.loc[3, history_column("orig", 3, "is_merchant_dest")] == 1


def test_destination_role_tracks_received_transactions() -> None:
    hist = account_history(_ledger(), "nameDest", "dest", length=1)
    # M1 received a payment at step 1 before the step-5 payment from C9
    assert hist.loc[4, history_column("dest", 1, "steps_before")] == 4
    assert hist.loc[3, history_column("dest", 1, "is_transfer")] == 1


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_paysim_like(
        n_rows=2_000, fraud_rate=0.02, n_steps=60, n_customers=120, seed=4
    )


CONFIG = FeaturesConfig(history_length=3)


def test_build_features_appends_history(raw: pd.DataFrame) -> None:
    table = build_features(raw, CONFIG)
    columns = feature_columns(CONFIG)
    assert set(history_columns(3)) <= set(columns)
    assert set(columns) <= set(table.columns)
    assert len(columns) == len(feature_columns()) + len(history_columns(3))
    # A populated lag implies every more recent lag is populated
    for role in ("orig", "dest"):
        ages = table[[history_column(role, k, "steps_before") for k in (1, 2, 3)]]
        filled = ages.notna().to_numpy()
        assert np.all(filled[:, 1:] <= filled[:, :-1])
        assert np.all(np.diff(ages.to_numpy(), axis=1)[filled[:, 1:]] >= 0)
        assert filled[:, 0].mean() > 0.2


def test_history_ignores_future_rows(raw: pd.DataFrame) -> None:
    full = build_features(raw, CONFIG)
    cutoff = 40
    truncated = build_features(raw[raw["step"] <= cutoff], CONFIG)
    columns = history_columns(3)
    pd.testing.assert_frame_equal(
        full.loc[truncated.index, columns], truncated[columns]
    )


def test_history_is_independent_of_row_order(raw: pd.DataFrame) -> None:
    shuffled = raw.sample(frac=1.0, random_state=0)
    columns = history_columns(3)
    pd.testing.assert_frame_equal(
        build_features(raw, CONFIG)[columns],
        build_features(shuffled, CONFIG).loc[raw.index, columns],
    )


def test_history_needs_transaction_features(raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="transaction features"):
        account_history(raw, "nameOrig", "orig", length=2)


def test_config_validates_history_settings() -> None:
    with pytest.raises(ConfigError, match="history_length"):
        FeaturesConfig(history_length=-1)
    with pytest.raises(ConfigError, match="history roles"):
        FeaturesConfig(history_roles=("beneficiary",))
