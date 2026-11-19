"""Synthetic transaction logs that follow the PaySim schema.

The generator is used by unit tests and CI, where the real dataset is not
available. It reproduces the schema and the main mechanics of the simulator
rather than its exact distributions:

* legitimate transactions follow the approximate type mix of the public release,
  with log-normal amounts that differ by type;
* payments go to merchants, whose balance columns are always zero;
* each fraud event is a ``TRANSFER`` that empties the victim's account into a
  mule account, followed by a ``CASH_OUT`` of the same amount by the mule in the
  same step;
* ``isFlaggedFraud`` marks a small subset of fraudulent transfers above
  200,000, mirroring its very low coverage in the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data.schema import (
    CUSTOMER_PREFIX,
    FLAGGED_TRANSFER_THRESHOLD,
    MERCHANT_PREFIX,
    PAYSIM_COLUMNS,
    TRANSACTION_TYPES,
)

# Approximate share of each type in the public PaySim release
_TYPE_PROBABILITIES = {
    "CASH_IN": 0.220,
    "CASH_OUT": 0.352,
    "DEBIT": 0.006,
    "PAYMENT": 0.338,
    "TRANSFER": 0.084,
}

# Log-normal (mu, sigma) of the amount for each legitimate transaction type
_AMOUNT_PARAMS = {
    "CASH_IN": (11.6, 1.0),
    "CASH_OUT": (11.7, 1.0),
    "DEBIT": (8.2, 1.0),
    "PAYMENT": (8.9, 1.1),
    "TRANSFER": (13.0, 1.2),
}

# PaySim splits larger fraudulent transfers into chunks of at most this amount
_MAX_FRAUD_TRANSFER = 10_000_000.0


def _account_ids(prefix: str, indices: np.ndarray) -> np.ndarray:
    return np.char.add(prefix, indices.astype(str))


def _legitimate_transactions(
    rng: np.random.Generator,
    n_rows: int,
    n_steps: int,
    n_customers: int,
    n_merchants: int,
) -> pd.DataFrame:
    probs = np.array([_TYPE_PROBABILITIES[t] for t in TRANSACTION_TYPES])
    types = rng.choice(np.array(TRANSACTION_TYPES), size=n_rows, p=probs / probs.sum())

    mu = np.array([_AMOUNT_PARAMS[t][0] for t in types])
    sigma = np.array([_AMOUNT_PARAMS[t][1] for t in types])
    amount = np.round(rng.lognormal(mu, sigma), 2)

    orig = _account_ids(CUSTOMER_PREFIX, rng.integers(0, n_customers, n_rows))
    is_payment = types == "PAYMENT"
    dest = np.where(
        is_payment,
        _account_ids(MERCHANT_PREFIX, rng.integers(0, n_merchants, n_rows)),
        _account_ids(CUSTOMER_PREFIX, rng.integers(0, n_customers, n_rows)),
    )

    old_orig = np.round(rng.lognormal(10.5, 1.8, n_rows), 2)
    is_cash_in = types == "CASH_IN"
    new_orig = np.where(
        is_cash_in, old_orig + amount, np.maximum(old_orig - amount, 0.0)
    )

    old_dest = np.where(is_payment, 0.0, np.round(rng.lognormal(12.0, 1.8, n_rows), 2))
    new_dest = np.where(
        is_payment,
        0.0,
        np.where(is_cash_in, np.maximum(old_dest - amount, 0.0), old_dest + amount),
    )

    return pd.DataFrame(
        {
            "step": rng.integers(1, n_steps + 1, n_rows),
            "type": types,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.round(new_orig, 2),
            "nameDest": dest,
            "oldbalanceDest": old_dest,
            "newbalanceDest": np.round(new_dest, 2),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
    )


def _fraud_transactions(
    rng: np.random.Generator,
    n_events: int,
    n_steps: int,
    n_customers: int,
    flag_probability: float,
) -> pd.DataFrame:
    steps = rng.integers(1, n_steps + 1, n_events)
    victims = _account_ids(CUSTOMER_PREFIX, rng.integers(0, n_customers, n_events))
    mules = _account_ids(CUSTOMER_PREFIX, rng.integers(0, n_customers, n_events))
    balance = np.round(
        np.minimum(rng.lognormal(12.5, 1.2, n_events), _MAX_FRAUD_TRANSFER), 2
    )
    mule_balance = np.round(rng.lognormal(10.0, 1.5, n_events), 2)

    flagged = (balance > FLAGGED_TRANSFER_THRESHOLD) & (
        rng.random(n_events) < flag_probability
    )
    zeros = np.zeros(n_events)

    # Destination balances of fraudulent transfers are zero in PaySim
    transfers = pd.DataFrame(
        {
            "step": steps,
            "type": "TRANSFER",
            "amount": balance,
            "nameOrig": victims,
            "oldbalanceOrg": balance,
            "newbalanceOrig": zeros,
            "nameDest": mules,
            "oldbalanceDest": zeros,
            "newbalanceDest": zeros,
            "isFraud": 1,
            "isFlaggedFraud": flagged.astype(int),
        }
    )
    cash_outs = pd.DataFrame(
        {
            "step": steps,
            "type": "CASH_OUT",
            "amount": balance,
            "nameOrig": mules,
            "oldbalanceOrg": mule_balance + balance,
            "newbalanceOrig": mule_balance,
            "nameDest": _account_ids(
                CUSTOMER_PREFIX, rng.integers(0, n_customers, n_events)
            ),
            "oldbalanceDest": zeros,
            "newbalanceDest": balance,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        }
    )
    return pd.concat([transfers, cash_outs], ignore_index=True)


def generate_paysim_like(
    n_rows: int = 10_000,
    fraud_rate: float = 0.0013,
    n_steps: int = 743,
    n_customers: int | None = None,
    n_merchants: int | None = None,
    flag_probability: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic transaction log with the PaySim schema.

    Parameters
    ----------
    n_rows
        Total number of rows, including fraudulent ones.
    fraud_rate
        Target share of fraudulent rows. Each fraud event contributes two rows,
        so the realized rate is rounded to an even number of rows. A positive
        rate always yields at least one event.
    n_steps
        Number of hourly steps covered by the log.
    n_customers
        Size of the customer account pool. Defaults to ``n_rows``, which gives
        mostly unique originators with some repeat activity.
    n_merchants
        Size of the merchant pool. Defaults to ``max(n_rows // 20, 1)``.
    flag_probability
        Probability that a fraudulent transfer above the 200,000 threshold is
        marked with ``isFlaggedFraud``.
    seed
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        Transactions sorted by ``step``, with columns in PaySim order.
    """
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2.")
    if not 0 <= fraud_rate < 1:
        raise ValueError("fraud_rate must be in [0, 1).")
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    if not 0 <= flag_probability <= 1:
        raise ValueError("flag_probability must be in [0, 1].")

    rng = np.random.default_rng(seed)
    n_customers = n_customers or n_rows
    n_merchants = n_merchants or max(n_rows // 20, 1)

    n_events = round(n_rows * fraud_rate / 2)
    if fraud_rate > 0:
        n_events = max(n_events, 1)
    n_events = min(n_events, (n_rows - 1) // 2)

    frames = [
        _legitimate_transactions(
            rng, n_rows - 2 * n_events, n_steps, n_customers, n_merchants
        )
    ]
    if n_events:
        frames.append(
            _fraud_transactions(rng, n_events, n_steps, n_customers, flag_probability)
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("step", kind="stable", ignore_index=True)
    return df.astype({"step": "int64", "isFraud": "int64", "isFlaggedFraud": "int64"})[
        list(PAYSIM_COLUMNS)
    ]
