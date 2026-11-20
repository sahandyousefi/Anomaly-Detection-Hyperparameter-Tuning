import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import (
    ACCOUNT_FEATURES,
    add_account_features,
    prior_aggregates,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    # A small account pool gives many repeat transactions per account
    return generate_paysim_like(
        n_rows=1_500, fraud_rate=0.02, n_steps=60, n_customers=80, seed=9
    )


def _naive(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        past = df[(df[key] == row[key]) & (df["step"] < row["step"])]
        amounts = past["amount"].to_numpy()
        rows.append(
            {
                "count": len(past),
                "mean": amounts.mean() if len(amounts) else np.nan,
                "std": amounts.std() if len(amounts) else np.nan,
                "since": row["step"] - past["step"].max() if len(past) else np.nan,
            }
        )
    return pd.DataFrame(rows, index=df.index)


@pytest.mark.parametrize(
    ("key", "prefix"), [("nameOrig", "orig"), ("nameDest", "dest")]
)
def test_matches_naive_computation(frame: pd.DataFrame, key: str, prefix: str) -> None:
    sample = frame.sample(n=300, random_state=1)
    got = prior_aggregates(frame, key, prefix).loc[sample.index]
    expected = _naive(frame, key).loc[sample.index]
    np.testing.assert_array_equal(got[f"{prefix}_prior_count"], expected["count"])
    np.testing.assert_allclose(
        got[f"{prefix}_prior_amount_mean"], expected["mean"], rtol=1e-9
    )
    np.testing.assert_allclose(
        got[f"{prefix}_prior_amount_std"], expected["std"], rtol=1e-6, atol=1e-3
    )
    np.testing.assert_array_equal(got[f"{prefix}_steps_since_prev"], expected["since"])
    assert (got[f"{prefix}_prior_count"] > 0).mean() > 0.5


def test_same_step_rows_are_excluded() -> None:
    df = pd.DataFrame(
        {
            "step": [1, 2, 2, 3],
            "nameOrig": ["C1", "C1", "C1", "C1"],
            "nameDest": ["C9", "C9", "C8", "C9"],
            "amount": [10.0, 20.0, 40.0, 5.0],
        }
    )
    result = add_account_features(df)
    assert result["orig_prior_count"].tolist() == [0, 1, 1, 3]
    np.testing.assert_allclose(
        result["orig_prior_amount_mean"], [np.nan, 10.0, 10.0, 70.0 / 3]
    )
    np.testing.assert_allclose(result["orig_steps_since_prev"], [np.nan, 1.0, 1.0, 1.0])
    assert result["dest_prior_count"].tolist() == [0, 1, 0, 2]
    np.testing.assert_allclose(
        result["amount_to_orig_prior_mean"], [np.nan, 2.0, 4.0, 5.0 / (70.0 / 3)]
    )


def test_future_rows_do_not_change_past_features(frame: pd.DataFrame) -> None:
    cutoff = 30
    before = add_account_features(frame)
    altered = frame.copy()
    future = altered["step"] >= cutoff
    altered.loc[future, "amount"] = altered.loc[future, "amount"] * 7 + 1
    after = add_account_features(altered)
    past = ~future
    pd.testing.assert_frame_equal(
        before.loc[past, list(ACCOUNT_FEATURES)],
        after.loc[past, list(ACCOUNT_FEATURES)],
    )


def test_row_order_does_not_matter(frame: pd.DataFrame) -> None:
    shuffled = frame.sample(frac=1.0, random_state=4)
    expected = add_account_features(frame)[list(ACCOUNT_FEATURES)]
    got = add_account_features(shuffled)[list(ACCOUNT_FEATURES)].loc[frame.index]
    pd.testing.assert_frame_equal(got, expected)


def test_output_columns(frame: pd.DataFrame) -> None:
    result = add_account_features(frame)
    assert list(result.columns) == [*frame.columns, *ACCOUNT_FEATURES]
    assert result["orig_prior_count"].dtype == np.int32
    no_history = result["orig_prior_count"] == 0
    assert result.loc[no_history, "orig_prior_amount_mean"].isna().all()
    assert result.loc[~no_history, "orig_prior_amount_std"].notna().all()
