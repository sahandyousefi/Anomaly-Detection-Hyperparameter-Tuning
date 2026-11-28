from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fraud_detection.config import (
    FeaturesConfig,
    ModelConfig,
    ResamplingConfig,
    load_config,
)
from fraud_detection.data.schema import BALANCE_COLUMNS
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.serving.api import ARTIFACT_ENV, create_app, score_request
from fraud_detection.serving.artifact import (
    FeatureSchema,
    ModelArtifact,
    artifact_metadata,
    save_artifact,
)
from fraud_detection.serving.schemas import MAX_TRANSACTIONS, ScoreRequest
from fraud_detection.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["isFraud", "isFlaggedFraud"]


class ConstantModel:
    """Scores every row the same, for decision and fallback checks."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


class PriorCountModel:
    """Returns the destination's prior transaction count as the score."""

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return X["dest_prior_count"].to_numpy(dtype="float64")


@pytest.fixture(scope="module")
def artifact() -> ModelArtifact:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config = replace(
        config,
        model=ModelConfig(
            name="xgboost",
            params={"n_estimators": 40, "max_depth": 3, "n_jobs": 1},
        ),
        resampling=ResamplingConfig(method="class_weight"),
    )
    df = generate_paysim_like(n_rows=4_000, fraud_rate=0.02, n_steps=100, seed=5)
    result = run_training(df, config)
    return ModelArtifact(
        model=result.model,
        schema=result.schema,
        threshold=result.threshold.threshold,
        metadata=artifact_metadata("xgboost", config.seed, synthetic=True),
    )


@pytest.fixture(scope="module")
def log() -> pd.DataFrame:
    return generate_paysim_like(n_rows=400, fraud_rate=0.05, n_steps=30, seed=9)


@pytest.fixture(scope="module")
def rows(log: pd.DataFrame) -> list[dict]:
    records = log.drop(columns=LABELS).to_dict(orient="records")
    for i, record in enumerate(records):
        record["transaction_id"] = f"tx-{i}"
    return records


@pytest.fixture
def client(artifact: ModelArtifact):
    with TestClient(create_app(artifact)) as test_client:
        yield test_client


def _constant_artifact(artifact: ModelArtifact, value: float) -> ModelArtifact:
    return replace(artifact, model=ConstantModel(value), threshold=0.5)


def test_health_reports_artifact(client: TestClient, artifact) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "xgboost"
    assert body["schema_version"] == artifact.schema.version
    assert body["synthetic_data"] is True


def test_schema_endpoint(client: TestClient, artifact) -> None:
    body = client.get("/schema").json()
    assert body["schema_version"] == artifact.schema.version
    assert [f["name"] for f in body["features"]] == artifact.schema.names
    assert set(BALANCE_COLUMNS) <= set(body["raw_columns"])


def test_scores_match_offline_pipeline(client: TestClient, artifact, log, rows):
    response = client.post(
        "/score", json={"history": rows[:-50], "transactions": rows[-50:]}
    )
    assert response.status_code == 200
    body = response.json()
    assert [r["transaction_id"] for r in body["results"]] == [
        f"tx-{i}" for i in range(len(rows) - 50, len(rows))
    ]
    table = build_features(log, artifact.schema.feature_config)
    expected = artifact.score(table.tail(50))
    served = np.array([r["score"] for r in body["results"]])
    np.testing.assert_allclose(served, expected, rtol=1e-6)
    flagged = np.array([r["flagged"] for r in body["results"]])
    np.testing.assert_array_equal(flagged, expected >= artifact.threshold)


def test_history_feeds_account_features(artifact, log, rows) -> None:
    served = replace(artifact, model=PriorCountModel())
    last = rows[-50:]
    with TestClient(create_app(served)) as client:
        with_history = client.post(
            "/score", json={"history": rows[:-50], "transactions": last}
        ).json()
        without = client.post("/score", json={"transactions": last}).json()
    table = build_features(log, artifact.schema.feature_config)
    expected = table["dest_prior_count"].tail(50).to_numpy(dtype="float64")
    served_counts = [r["score"] for r in with_history["results"]]
    np.testing.assert_array_equal(served_counts, expected)
    assert sum(r["score"] for r in without["results"]) < expected.sum()


def test_flagged_rows_carry_reasons(client: TestClient, rows) -> None:
    body = client.post(
        "/score",
        json={"history": rows[:-100], "transactions": rows[-100:], "max_reasons": 2},
    ).json()
    flagged = [r for r in body["results"] if r["flagged"]]
    assert flagged, "expected at least one alert on the synthetic sample"
    for result in body["results"]:
        if not result["flagged"]:
            assert result["reasons"] == []
    for result in flagged:
        assert 1 <= len(result["reasons"]) <= 2
        contributions = [reason["contribution"] for reason in result["reasons"]]
        assert contributions == sorted(contributions, reverse=True)
        assert all(c > 0 for c in contributions)
        assert all(reason["code"].startswith("R") for reason in result["reasons"])


def test_zero_reasons_requested(client: TestClient, rows) -> None:
    body = client.post(
        "/score", json={"transactions": rows[-100:], "max_reasons": 0}
    ).json()
    assert all(result["reasons"] == [] for result in body["results"])


def test_missing_ids_default_to_position(client: TestClient, rows) -> None:
    anonymous = [{k: v for k, v in r.items() if k != "transaction_id"} for r in rows]
    body = client.post("/score", json={"transactions": anonymous[:3]}).json()
    assert [r["transaction_id"] for r in body["results"]] == ["0", "1", "2"]


def test_schema_version_conflict(client: TestClient, rows) -> None:
    response = client.post(
        "/score", json={"transactions": rows[:2], "schema_version": "0" * 16}
    )
    assert response.status_code == 409
    assert "schema" in response.json()["detail"]


def test_matching_schema_version_accepted(client: TestClient, artifact, rows):
    response = client.post(
        "/score",
        json={"transactions": rows[:2], "schema_version": artifact.schema.version},
    )
    assert response.status_code == 200


def test_missing_balances_rejected(client: TestClient, rows) -> None:
    record = {k: v for k, v in rows[0].items() if k != "oldbalanceDest"}
    response = client.post("/score", json={"transactions": [record]})
    assert response.status_code == 422
    assert "balance" in response.json()["detail"]


def test_balances_optional_without_balance_features(artifact, rows) -> None:
    config = FeaturesConfig(balance_features=False)
    names = set(feature_columns(config))
    specs = tuple(s for s in artifact.schema.features if s.name in names)
    schema = FeatureSchema(features=specs, feature_config=config)
    served = ModelArtifact(ConstantModel(0.2), schema, threshold=0.5)
    record = {k: v for k, v in rows[0].items() if k not in BALANCE_COLUMNS}
    with TestClient(create_app(served)) as client:
        response = client.post("/score", json={"transactions": [record]})
    assert response.status_code == 200
    assert response.json()["results"][0]["flagged"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("type", "WIRE"), ("amount", -1.0), ("step", 0), ("nameOrig", "bad id")],
)
def test_invalid_fields_return_422(client: TestClient, rows, field, value) -> None:
    record = {**rows[0], field: value}
    response = client.post("/score", json={"transactions": [record]})
    assert response.status_code == 422


def test_unknown_field_rejected(client: TestClient, rows) -> None:
    response = client.post("/score", json={"transactions": [{**rows[0], "isFraud": 1}]})
    assert response.status_code == 422


def test_empty_and_oversized_batches_rejected(client: TestClient, rows) -> None:
    assert client.post("/score", json={"transactions": []}).status_code == 422
    oversized = [rows[0] | {"transaction_id": None}] * (MAX_TRANSACTIONS + 1)
    assert client.post("/score", json={"transactions": oversized}).status_code == 422


def test_non_tree_model_scores_without_reasons(artifact, rows) -> None:
    served = _constant_artifact(artifact, 0.9)
    response = score_request(
        served, ScoreRequest.model_validate({"transactions": rows[:5]})
    )
    assert all(r.flagged for r in response.results)
    assert all(r.reasons == [] for r in response.results)
    assert response.model == "xgboost"


def test_infinite_threshold_reported_as_null(artifact, rows) -> None:
    served = replace(artifact, threshold=float("inf"))
    with TestClient(create_app(served)) as client:
        body = client.post("/score", json={"transactions": rows[:3]}).json()
    assert body["threshold"] is None
    assert not any(r["flagged"] for r in body["results"])


def test_no_artifact_returns_503(monkeypatch, rows) -> None:
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/schema").status_code == 503
        response = client.post("/score", json={"transactions": rows[:1]})
    assert response.status_code == 503


def test_artifact_loaded_from_environment(artifact, tmp_path, monkeypatch) -> None:
    save_artifact(artifact, tmp_path / "artifact")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path / "artifact"))
    with TestClient(create_app()) as client:
        body = client.get("/health").json()
    assert body["schema_version"] == artifact.schema.version
