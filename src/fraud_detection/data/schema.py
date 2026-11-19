"""Column names and value domains of the PaySim transaction log."""

from __future__ import annotations

PAYSIM_COLUMNS: tuple[str, ...] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

TRANSACTION_TYPES: tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

FRAUD_TRANSACTION_TYPES: tuple[str, ...] = ("TRANSFER", "CASH_OUT")

CUSTOMER_PREFIX = "C"
MERCHANT_PREFIX = "M"

FLAGGED_TRANSFER_THRESHOLD = 200_000.0
