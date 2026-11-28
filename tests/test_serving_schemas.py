import math

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError

from fraud_detection.data.schema import BALANCE_COLUMNS
from fraud_detection.serving.schemas import (
    MAX_TRANSACTIONS,
    ScoreRequest,
    Transaction,
    transactions_frame,
)


def _transaction(**overrides) -> dict:
    record = {
        "transaction_id": "t1",
        "step": 5,
        "type": "TRANSFER",
        "amount": 1_250.0,
        "nameOrig": "C100",
        "nameDest": "C200",
        "oldbalanceOrg": 1_250.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    record.update(overrides)
    return record


def test_valid_transaction_parses() -> None:
    transaction = Transaction.model_validate(_transaction())
    assert transaction.type == "TRANSFER"
    assert transaction.has_balances()


@pytest.mark.parametrize(
    "overrides",
    [
        {"step": 0},
        {"type": "WIRE"},
        {"amount": -1.0},
        {"amount": math.inf},
        {"nameOrig": "X100"},
        {"nameDest": ""},
        {"oldbalanceOrg": -5.0},
        {"newbalanceDest": math.nan},
        {"isFraud": 1},
    ],
)
def test_invalid_transaction_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Transaction.model_validate(_transaction(**overrides))


def test_balances_are_optional() -> None:
    record = _transaction()
    for column in BALANCE_COLUMNS:
        record.pop(column)
    transaction = Transaction.model_validate(record)
    assert not transaction.has_balances()


def test_request_needs_transactions() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({"transactions": []})


def test_request_size_limit() -> None:
    many = [_transaction(transaction_id=None)] * (MAX_TRANSACTIONS + 1)
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({"transactions": many})


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ScoreRequest.model_validate(
            {"transactions": [_transaction(), _transaction(amount=10.0)]}
        )


@pytest.mark.parametrize("max_reasons", [-1, 11])
def test_max_reasons_bounded(max_reasons: int) -> None:
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate(
            {"transactions": [_transaction()], "max_reasons": max_reasons}
        )


def test_frame_has_paysim_layout() -> None:
    request = ScoreRequest.model_validate(
        {
            "transactions": [
                _transaction(),
                _transaction(transaction_id="t2", oldbalanceDest=None),
            ]
        }
    )
    frame = transactions_frame(request.transactions)
    assert list(frame.columns) == [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        *BALANCE_COLUMNS,
    ]
    assert frame["oldbalanceDest"].dtype == "float64"
    assert math.isnan(frame.loc[1, "oldbalanceDest"])


def test_empty_frame_keeps_columns() -> None:
    frame = transactions_frame([])
    assert frame.empty
    assert "amount" in frame.columns
