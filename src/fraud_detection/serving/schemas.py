"""Request and response models of the scoring service.

Requests carry raw transactions in PaySim layout. The service rebuilds the
model features itself, so clients never compute features and a change of the
feature pipeline cannot be silently bypassed. Point-in-time account features
need the earlier transactions of each account, which a stateless service does
not hold; clients pass them in ``history`` (in production this would come
from a feature store). History rows are used only to build features and are
not scored.
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fraud_detection.data.schema import BALANCE_COLUMNS

MAX_TRANSACTIONS = 1_000
MAX_HISTORY = 20_000
MAX_REASONS = 10

RAW_COLUMNS: tuple[str, ...] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    *BALANCE_COLUMNS,
)

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

_ACCOUNT_PATTERN = r"^[CM][0-9A-Za-z_-]{1,63}$"


class Transaction(BaseModel):
    """One transaction in PaySim layout.

    Balance fields may be omitted only if the served model was trained
    without balance features; the service checks this against the schema.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(
        default=None, max_length=128, description="Client identifier, echoed back."
    )
    step: int = Field(ge=1, description="Hour index of the simulation, from 1.")
    type: TransactionType
    amount: float = Field(ge=0)
    nameOrig: str = Field(pattern=_ACCOUNT_PATTERN)
    nameDest: str = Field(pattern=_ACCOUNT_PATTERN)
    oldbalanceOrg: float | None = Field(default=None, ge=0)
    newbalanceOrig: float | None = Field(default=None, ge=0)
    oldbalanceDest: float | None = Field(default=None, ge=0)
    newbalanceDest: float | None = Field(default=None, ge=0)

    @field_validator(
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    )
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    def has_balances(self) -> bool:
        """Whether all four balance fields are present."""
        return all(getattr(self, column) is not None for column in BALANCE_COLUMNS)


class ScoreRequest(BaseModel):
    """Transactions to score and the account history needed for their features."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_TRANSACTIONS)
    history: list[Transaction] = Field(
        default_factory=list,
        max_length=MAX_HISTORY,
        description="Earlier transactions of the accounts involved; not scored.",
    )
    schema_version: str | None = Field(
        default=None,
        description="Feature schema the client expects; rejected if it differs.",
    )
    max_reasons: int = Field(default=3, ge=0, le=MAX_REASONS)

    @model_validator(mode="after")
    def _unique_ids(self) -> ScoreRequest:
        ids = [t.transaction_id for t in self.transactions if t.transaction_id]
        if len(ids) != len(set(ids)):
            raise ValueError("transaction_id values must be unique")
        return self


class ReasonCode(BaseModel):
    """One reason attached to an alert."""

    code: str
    description: str
    contribution: float = Field(description="Group SHAP value in log-odds.")


class TransactionScore(BaseModel):
    """Score, decision, and reasons for one transaction."""

    transaction_id: str
    score: float
    flagged: bool
    reasons: list[ReasonCode] = Field(
        default_factory=list, description="Populated for flagged transactions."
    )


class ScoreResponse(BaseModel):
    """Results in the order of the request's transactions."""

    model: str
    schema_version: str
    threshold: float | None = Field(description="Null when the model flags nothing.")
    results: list[TransactionScore]


class HealthResponse(BaseModel):
    """Service status and the identity of the loaded artifact."""

    status: Literal["ok"]
    model: str
    schema_version: str
    package_version: str
    synthetic_data: bool


class FeatureSchemaResponse(BaseModel):
    """Input contract of the loaded model."""

    schema_version: str
    raw_columns: list[str]
    features: list[dict[str, str]]


def transactions_frame(transactions: list[Transaction]) -> pd.DataFrame:
    """Raw transaction table in PaySim column layout.

    Missing balances become NaN; the identifier column is dropped.
    """
    records = [t.model_dump(exclude={"transaction_id"}) for t in transactions]
    frame = pd.DataFrame.from_records(records, columns=list(RAW_COLUMNS))
    for column in BALANCE_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame


__all__ = [
    "MAX_HISTORY",
    "MAX_REASONS",
    "MAX_TRANSACTIONS",
    "RAW_COLUMNS",
    "FeatureSchemaResponse",
    "HealthResponse",
    "ReasonCode",
    "ScoreRequest",
    "ScoreResponse",
    "Transaction",
    "TransactionScore",
    "transactions_frame",
]
