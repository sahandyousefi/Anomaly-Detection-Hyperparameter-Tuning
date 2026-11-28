import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FeaturesConfig
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.features.build import build_features, feature_columns
from fraud_detection.serving.artifact import (
    MANIFEST_FILE,
    MODEL_FILE,
    SCHEMA_FILE,
    ArtifactError,
    FeatureSchema,
    FeatureSpec,
    ModelArtifact,
    SchemaMismatchError,
    load_artifact,
    required_raw_columns,
    save_artifact,
)


class AmountModel:
    """Scores rows by their log amount, enough to check the plumbing."""

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return X["log_amount"].to_numpy(dtype="float64") / 20.0


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return generate_paysim_like(n_rows=1_500, fraud_rate=0.02, n_steps=60, seed=2)


@pytest.fixture(scope="module")
def schema(transactions: pd.DataFrame) -> FeatureSchema:
    config = FeaturesConfig()
    table = build_features(transactions, config)
    return FeatureSchema.from_table(table, feature_columns(config), config)


@pytest.fixture
def artifact(schema: FeatureSchema) -> ModelArtifact:
    return ModelArtifact(
        model=AmountModel(),
        schema=schema,
        threshold=0.6,
        metadata={"model": "amount", "synthetic_data": True},
    )


def test_schema_infers_kinds(schema: FeatureSchema) -> None:
    kinds = {spec.name: spec.kind for spec in schema.features}
    assert kinds["is_transfer"] == "integer"
    assert kinds["orig_prior_count"] == "integer"
    assert kinds["log_amount"] == "float"
    assert kinds["orig_prior_amount_mean"] == "float"
    assert schema.names == feature_columns(FeaturesConfig())


def test_schema_round_trip_keeps_version(schema: FeatureSchema) -> None:
    restored = FeatureSchema.from_dict(json.loads(json.dumps(schema.to_dict())))
    assert restored == schema
    assert restored.version == schema.version


def test_schema_version_tracks_content(schema: FeatureSchema) -> None:
    reordered = replace(schema, features=schema.features[::-1])
    no_balance = replace(schema, feature_config=FeaturesConfig(balance_features=False))
    assert len({schema.version, reordered.version, no_balance.version}) == 3


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        FeatureSpec("amount", "string")


def test_raw_columns_follow_feature_switches() -> None:
    with_balances = required_raw_columns(FeaturesConfig(balance_features=True))
    without = required_raw_columns(FeaturesConfig(balance_features=False))
    assert "oldbalanceOrg" in with_balances
    assert "oldbalanceOrg" not in without
    assert {"step", "type", "amount", "nameOrig", "nameDest"} <= set(without)


def test_validate_orders_and_casts(schema: FeatureSchema, transactions) -> None:
    table = build_features(transactions, schema.feature_config)
    shuffled = table[[*schema.names[::-1], "isFraud"]]
    checked = schema.validate(shuffled)
    assert list(checked.columns) == schema.names
    assert checked["is_transfer"].dtype == "int64"
    assert checked["log_amount"].dtype == "float64"


def test_validate_reports_missing_column(schema: FeatureSchema, transactions) -> None:
    table = build_features(transactions, schema.feature_config)
    with pytest.raises(SchemaMismatchError, match="log_amount"):
        schema.validate(table.drop(columns="log_amount"))


@pytest.mark.parametrize("bad", [0.5, np.nan])
def test_validate_rejects_bad_integer(schema: FeatureSchema, transactions, bad) -> None:
    table = build_features(transactions.head(20), schema.feature_config)
    table["is_transfer"] = table["is_transfer"].astype("float64")
    table.loc[table.index[0], "is_transfer"] = bad
    with pytest.raises(SchemaMismatchError, match="is_transfer"):
        schema.validate(table)


def test_validate_rejects_text(schema: FeatureSchema, transactions) -> None:
    table = build_features(transactions.head(20), schema.feature_config)
    table["log_amount"] = "high"
    with pytest.raises(SchemaMismatchError, match="not numeric"):
        schema.validate(table)


def test_save_and_load_round_trip(
    artifact: ModelArtifact, transactions, tmp_path: Path
) -> None:
    save_artifact(artifact, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert manifest["schema_version"] == artifact.schema.version
    assert manifest["metadata"]["synthetic_data"] is True

    loaded = load_artifact(tmp_path)
    assert loaded.schema == artifact.schema
    assert loaded.threshold == pytest.approx(0.6)
    features = loaded.build_features(transactions)
    np.testing.assert_allclose(loaded.score(features), artifact.score(features))
    assert set(np.unique(loaded.decide(loaded.score(features)))) <= {0, 1}


def test_infinite_threshold_survives_json(artifact: ModelArtifact, tmp_path) -> None:
    save_artifact(replace(artifact, threshold=math.inf), tmp_path)
    assert json.loads((tmp_path / MANIFEST_FILE).read_text())["threshold"] is None
    assert math.isinf(load_artifact(tmp_path).threshold)


def test_tampered_model_rejected(artifact: ModelArtifact, tmp_path: Path) -> None:
    save_artifact(artifact, tmp_path)
    with (tmp_path / MODEL_FILE).open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ArtifactError, match="digest"):
        load_artifact(tmp_path)


def test_edited_schema_rejected(artifact: ModelArtifact, tmp_path: Path) -> None:
    save_artifact(artifact, tmp_path)
    raw = json.loads((tmp_path / SCHEMA_FILE).read_text())
    raw["features"] = raw["features"][:-1]
    (tmp_path / SCHEMA_FILE).write_text(json.dumps(raw))
    with pytest.raises(ArtifactError, match="Schema version"):
        load_artifact(tmp_path)


def test_unsupported_format_rejected(artifact: ModelArtifact, tmp_path: Path) -> None:
    save_artifact(artifact, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    manifest["format_version"] = 99
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="format"):
        load_artifact(tmp_path)


def test_missing_raw_columns_reported(artifact: ModelArtifact, transactions) -> None:
    with pytest.raises(SchemaMismatchError, match="nameDest"):
        artifact.build_features(transactions.drop(columns="nameDest"))
