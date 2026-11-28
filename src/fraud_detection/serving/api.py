"""HTTP scoring service.

Endpoints:

* ``GET /health``: status and identity of the loaded artifact;
* ``GET /schema``: the feature schema the model expects;
* ``POST /score``: score, alert decision, and reason codes per transaction.

The service loads one artifact written by the training run (see
:mod:`fraud_detection.serving.artifact`) from the directory in the
``FRAUD_DETECTION_ARTIFACT`` environment variable, or from the argument of
:func:`create_app`. Reason codes come from exact TreeSHAP values and are
computed only for flagged transactions, which keeps the cost of a request
proportional to its alert volume. Models without a TreeSHAP path return
flagged transactions without reasons.

Run with::

    FRAUD_DETECTION_ARTIFACT=artifacts/model/artifact \\
        uvicorn fraud_detection.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fraud_detection import __version__
from fraud_detection.data.schema import BALANCE_COLUMNS
from fraud_detection.explain.reason_codes import reasons_for_row
from fraud_detection.explain.shap_analysis import tree_shap
from fraud_detection.models.gbm import BoostedModel
from fraud_detection.serving.artifact import (
    ModelArtifact,
    SchemaMismatchError,
    load_artifact,
)
from fraud_detection.serving.schemas import (
    FeatureSchemaResponse,
    HealthResponse,
    ReasonCode,
    ScoreRequest,
    ScoreResponse,
    Transaction,
    TransactionScore,
    transactions_frame,
)

ARTIFACT_ENV = "FRAUD_DETECTION_ARTIFACT"


class SchemaVersionConflict(Exception):
    """Raised when a request was built for another feature schema."""


def _check_request(artifact: ModelArtifact, request: ScoreRequest) -> None:
    expected = artifact.schema.version
    if request.schema_version is not None and request.schema_version != expected:
        raise SchemaVersionConflict(
            f"Request targets schema {request.schema_version}; "
            f"the loaded model expects {expected}."
        )
    if set(BALANCE_COLUMNS) <= set(artifact.schema.raw_columns):
        rows = [*request.history, *request.transactions]
        if not all(t.has_balances() for t in rows):
            raise SchemaMismatchError(
                "The loaded model uses balance features; every transaction and "
                "history row needs all four balance fields."
            )


def _feature_rows(artifact: ModelArtifact, request: ScoreRequest) -> pd.DataFrame:
    """Model inputs of the scored transactions, built together with history.

    Scored transactions also act as history for later scored transactions of
    the same accounts, as rows of the training log do.
    """
    log = transactions_frame([*request.history, *request.transactions])
    features = artifact.build_features(log)
    return features.iloc[len(request.history) :]


def _reasons(
    artifact: ModelArtifact, X: pd.DataFrame, max_reasons: int
) -> list[list[ReasonCode]]:
    if max_reasons == 0 or X.empty or not isinstance(artifact.model, BoostedModel):
        return [[] for _ in range(len(X))]
    explanation = tree_shap(artifact.model, X)
    return [
        [
            ReasonCode(
                code=r.code, description=r.description, contribution=r.contribution
            )
            for r in reasons_for_row(explanation, row, max_reasons)
        ]
        for row in range(len(X))
    ]


def _identifier(transaction: Transaction, position: int) -> str:
    return transaction.transaction_id or str(position)


def score_request(artifact: ModelArtifact, request: ScoreRequest) -> ScoreResponse:
    """Score a validated request with a loaded artifact.

    Parameters
    ----------
    artifact
        Model, schema, and threshold.
    request
        Transactions to score, with optional account history.

    Returns
    -------
    ScoreResponse
        One result per transaction, in request order.

    Raises
    ------
    SchemaVersionConflict
        If the request names a schema version other than the artifact's.
    SchemaMismatchError
        If the transactions lack fields the feature schema requires.
    """
    _check_request(artifact, request)
    X = _feature_rows(artifact, request)
    scores = artifact.model.predict_scores(X)
    flagged = artifact.decide(scores).astype(bool)
    alert_rows = np.flatnonzero(flagged)
    reasons: list[list[ReasonCode]] = [[] for _ in range(len(X))]
    for row, row_reasons in zip(
        alert_rows,
        _reasons(artifact, X.iloc[alert_rows], request.max_reasons),
        strict=True,
    ):
        reasons[row] = row_reasons
    results = [
        TransactionScore(
            transaction_id=_identifier(transaction, i),
            score=float(scores[i]),
            flagged=bool(flagged[i]),
            reasons=reasons[i],
        )
        for i, transaction in enumerate(request.transactions)
    ]
    return ScoreResponse(
        model=str(artifact.metadata.get("model", "unknown")),
        schema_version=artifact.schema.version,
        threshold=None if math.isinf(artifact.threshold) else artifact.threshold,
        results=results,
    )


def _loaded(request: Request) -> ModelArtifact:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="No model artifact is loaded.")
    return artifact


def create_app(
    artifact: ModelArtifact | None = None, artifact_dir: Path | None = None
) -> FastAPI:
    """Build the scoring application.

    Parameters
    ----------
    artifact
        Artifact to serve directly, mainly for tests.
    artifact_dir
        Artifact directory to load at startup when ``artifact`` is not
        given. Defaults to the ``FRAUD_DETECTION_ARTIFACT`` variable.

    Returns
    -------
    fastapi.FastAPI
        Application with the ``/health``, ``/schema``, and ``/score`` routes.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if artifact is not None:
            app.state.artifact = artifact
        else:
            directory = artifact_dir or os.environ.get(ARTIFACT_ENV)
            app.state.artifact = load_artifact(Path(directory)) if directory else None
        yield

    app = FastAPI(
        title="PaySim fraud scoring",
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(SchemaMismatchError)
    async def _schema_mismatch(_: Request, exc: SchemaMismatchError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(SchemaVersionConflict)
    async def _version_conflict(_: Request, exc: SchemaVersionConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = _loaded(request)
        return HealthResponse(
            status="ok",
            model=str(loaded.metadata.get("model", "unknown")),
            schema_version=loaded.schema.version,
            package_version=__version__,
            synthetic_data=bool(loaded.metadata.get("synthetic_data", False)),
        )

    @app.get("/schema", response_model=FeatureSchemaResponse)
    def schema(request: Request) -> FeatureSchemaResponse:
        loaded = _loaded(request)
        return FeatureSchemaResponse(
            schema_version=loaded.schema.version,
            raw_columns=list(loaded.schema.raw_columns),
            features=[
                {"name": spec.name, "kind": spec.kind}
                for spec in loaded.schema.features
            ],
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        return score_request(_loaded(request), payload)

    return app


def __getattr__(name: str) -> FastAPI:
    # Built on first access so importing the module does not load a model
    if name == "app":
        return create_app()
    raise AttributeError(name)


__all__ = ["ARTIFACT_ENV", "SchemaVersionConflict", "create_app", "score_request"]
