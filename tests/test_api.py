from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fraud_detection.config import ModelConfig, ResamplingConfig, load_config
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.serving.api import create_app
from fraud_detection.serving.artifact import ModelArtifact, artifact_metadata
from fraud_detection.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("isFraud", "isFlaggedFraud")


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


def test_health_and_score(artifact: ModelArtifact) -> None:
    log = generate_paysim_like(n_rows=300, fraud_rate=0.05, n_steps=30, seed=9)
    rows = log.drop(columns=list(LABELS)).to_dict(orient="records")
    with TestClient(create_app(artifact)) as client:
        health = client.get("/health").json()
        assert health["schema_version"] == artifact.schema.version
        assert health["synthetic_data"] is True

        response = client.post(
            "/score", json={"history": rows[:-20], "transactions": rows[-20:]}
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 20
    for result in body["results"]:
        assert 0.0 <= result["score"] <= 1.0
        assert result["reasons"] == [] or result["flagged"]
