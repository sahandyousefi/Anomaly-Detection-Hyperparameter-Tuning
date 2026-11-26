import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ResamplingConfig, SplitConfig
from fraud_detection.data.resample import prepare_modeling_folds
from fraud_detection.data.split import time_split
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.explain.local_explanations import explain_flagged
from fraud_detection.explain.reason_codes import (
    default_reason_groups,
    group_features,
    reason_table,
    reasons_for_row,
)
from fraud_detection.explain.shap_analysis import ShapExplanation
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.features.sequence import history_columns
from fraud_detection.models.gbm import fit_xgboost


def _explanation(values: dict[str, float], shap: dict[str, float]) -> ShapExplanation:
    data = pd.DataFrame([values])
    contributions = np.array([[shap.get(name, 0.0) for name in data.columns]])
    return ShapExplanation(contributions, -5.0, data, "log_odds")


def test_codes_are_unique() -> None:
    codes = [group.code for group in default_reason_groups()]
    assert len(codes) == len(set(codes))


def test_every_built_feature_has_a_named_group() -> None:
    names = feature_columns() + history_columns(2)
    assignment = group_features(names, default_reason_groups())
    assert not any(code.startswith("R99") for code in assignment)
    assert sorted(f for features in assignment.values() for f in features) == sorted(
        names
    )
    assert assignment["R13"] == history_columns(2, ["orig"])
    assert assignment["R14"] == history_columns(2, ["dest"])


def test_group_contributions_are_summed() -> None:
    explanation = _explanation(
        {"hour_of_day": 3, "hour_sin": 0.7, "hour_cos": 0.7, "log_amount": 12.0},
        {"hour_of_day": 0.4, "hour_sin": 0.3, "hour_cos": 0.2, "log_amount": 0.8},
    )
    reasons = reasons_for_row(explanation, 0)
    assert [r.code for r in reasons] == ["R11", "R02"]
    assert reasons[0].contribution == pytest.approx(0.9)
    assert reasons[0].description == "Transaction at hour 03 of the day"
    assert "162,754" in reasons[1].description


def test_negative_and_small_contributions_are_dropped() -> None:
    explanation = _explanation(
        {"is_transfer": 1, "is_cash_out": 0, "dest_balances_zero": 1, "log_amount": 9},
        {"is_transfer": -1.0, "dest_balances_zero": 2.0, "log_amount": 0.05},
    )
    reasons = reasons_for_row(explanation, 0, min_contribution=0.1)
    assert [r.code for r in reasons] == ["R10"]
    assert reasons[0].description.startswith("Destination balances are zero")


def test_unknown_features_fall_back_to_their_value() -> None:
    explanation = _explanation({"custom_score": 2.5}, {"custom_score": 1.0})
    (reason,) = reasons_for_row(explanation, 0)
    assert reason.code == "R99"
    assert reason.description == "custom_score = 2.5"


def test_first_transaction_wording() -> None:
    explanation = _explanation(
        {"orig_prior_count": 0, "orig_steps_since_prev": np.nan},
        {"orig_prior_count": 0.5},
    )
    (reason,) = reasons_for_row(explanation, 0)
    assert reason.description == "First observed transaction of the originator account"


def test_reason_table_on_flagged_rows() -> None:
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.03, n_steps=80, seed=6)
    split = time_split(build_features(df), SplitConfig(0.6, 0.2, 0.2))
    folds = prepare_modeling_folds(
        split, feature_columns(), "isFraud", ResamplingConfig("none")
    )
    params = {"n_estimators": 150, "learning_rate": 0.3, "max_depth": 3, "n_jobs": 1}
    model = fit_xgboost(folds.train, folds.X_validation, folds.y_validation, params)
    scores = model.predict_scores(folds.X_test)
    threshold = float(np.quantile(scores, 0.99))
    explanation = explain_flagged(model, folds.X_test, threshold, max_rows=5)
    table = reason_table(explanation, max_reasons=2)
    assert list(table.index) == list(explanation.data.index)
    assert table["reason_1_code"].notna().all()
    contributions = table[["reason_1_contribution", "reason_2_contribution"]]
    assert (contributions.fillna(0) >= 0).all().all()
    assert (
        contributions["reason_1_contribution"]
        >= contributions["reason_2_contribution"].fillna(0)
    ).all()


def test_empty_reason_table_has_columns() -> None:
    explanation = ShapExplanation(
        np.empty((0, 1)), 0.0, pd.DataFrame({"a": []}), "log_odds"
    )
    table = reason_table(explanation, max_reasons=2)
    assert table.empty
    assert "reason_2_code" in table.columns
