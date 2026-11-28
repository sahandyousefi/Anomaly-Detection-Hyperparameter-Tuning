"""Self-contained model artifact with a versioned feature schema.

An artifact directory holds everything a scoring process needs:

* ``model.pkl``: the fitted model behind the common interface;
* ``feature_schema.json``: ordered model inputs with their value kind, the
  feature switches and time column they were built with, and the raw
  transaction columns required to rebuild them;
* ``manifest.json``: format version, schema version, SHA-256 digest of the
  pickled model, decision threshold, and provenance of the training run.

The schema version is a digest of the schema content, so any change to the
feature list, order, value kinds, or feature switches yields a new version.
A scoring service can therefore refuse a request built for a different
schema, and two artifacts share a version only if they expect identical
inputs. Loading checks the format version, the model digest, and the schema
version, so a truncated or edited artifact fails at load time instead of
producing silent mis-scores.

The model is stored with :mod:`pickle`. Load artifacts only from trusted
locations.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection import __version__
from fraud_detection.config import FeaturesConfig
from fraud_detection.data.schema import BALANCE_COLUMNS
from fraud_detection.evaluation.thresholds import apply_threshold
from fraud_detection.features.build import build_features
from fraud_detection.models.base import FraudModel

FORMAT_VERSION = 1
SCHEMA_VERSION_LENGTH = 16

MODEL_FILE = "model.pkl"
SCHEMA_FILE = "feature_schema.json"
MANIFEST_FILE = "manifest.json"

KIND_INTEGER = "integer"
KIND_FLOAT = "float"

# Columns read by build_features; balances only when balance features are on
_BASE_RAW_COLUMNS: tuple[str, ...] = ("step", "type", "amount", "nameOrig", "nameDest")


class ArtifactError(ValueError):
    """Raised when an artifact is malformed or inconsistent with its manifest."""


class SchemaMismatchError(ValueError):
    """Raised when an input table does not satisfy the feature schema."""


@dataclass(frozen=True)
class FeatureSpec:
    """One model input column.

    Attributes
    ----------
    name
        Column name.
    kind
        ``"integer"`` for flags, counts, and ordinal codes, which must be
        finite whole numbers, or ``"float"``, which may be missing.
    """

    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in (KIND_INTEGER, KIND_FLOAT):
            raise ValueError(f"Unknown feature kind {self.kind!r} for {self.name}.")


def _kind_of(dtype: Any) -> str:
    return KIND_INTEGER if pd.api.types.is_integer_dtype(dtype) else KIND_FLOAT


def required_raw_columns(
    feature_config: FeaturesConfig, time_column: str = "step"
) -> tuple[str, ...]:
    """Raw transaction columns needed to rebuild the model features."""
    columns = [time_column, *(c for c in _BASE_RAW_COLUMNS if c != "step")]
    if feature_config.balance_features:
        columns.extend(BALANCE_COLUMNS)
    return tuple(columns)


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered model inputs and the settings that produce them.

    Attributes
    ----------
    features
        Model input columns in training order.
    feature_config
        Feature switches used by :func:`build_features`.
    time_column
        Integer time index of the raw log.
    """

    features: tuple[FeatureSpec, ...]
    feature_config: FeaturesConfig = field(default_factory=FeaturesConfig)
    time_column: str = "step"

    @classmethod
    def from_table(
        cls,
        table: pd.DataFrame,
        names: Sequence[str],
        feature_config: FeaturesConfig,
        time_column: str = "step",
    ) -> FeatureSchema:
        """Infer value kinds from a feature table built for training."""
        missing = [name for name in names if name not in table.columns]
        if missing:
            raise SchemaMismatchError(f"Feature table lacks columns: {missing}")
        specs = tuple(FeatureSpec(name, _kind_of(table[name].dtype)) for name in names)
        return cls(specs, feature_config, time_column)

    @property
    def names(self) -> list[str]:
        """Feature names in model order."""
        return [spec.name for spec in self.features]

    @property
    def raw_columns(self) -> tuple[str, ...]:
        """Raw columns a request must provide to rebuild the features."""
        return required_raw_columns(self.feature_config, self.time_column)

    @property
    def version(self) -> str:
        """Content digest identifying this exact input contract."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:SCHEMA_VERSION_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation, without the derived version."""
        return {
            "features": [asdict(spec) for spec in self.features],
            "feature_config": {
                **asdict(self.feature_config),
                "history_roles": list(self.feature_config.history_roles),
            },
            "time_column": self.time_column,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSchema:
        """Inverse of :meth:`to_dict`."""
        config = dict(raw["feature_config"])
        config["history_roles"] = tuple(config["history_roles"])
        return cls(
            features=tuple(FeatureSpec(**spec) for spec in raw["features"]),
            feature_config=FeaturesConfig(**config),
            time_column=raw["time_column"],
        )

    def validate(self, table: pd.DataFrame) -> pd.DataFrame:
        """Select and check the model inputs of a feature table.

        Parameters
        ----------
        table
            Engineered features; extra columns are ignored.

        Returns
        -------
        pandas.DataFrame
            The schema columns in model order, as ``float64`` for float
            features and ``int64`` for integer features.

        Raises
        ------
        SchemaMismatchError
            If a column is missing, not numeric, or an integer feature holds
            missing, infinite, or fractional values.
        """
        missing = [name for name in self.names if name not in table.columns]
        if missing:
            raise SchemaMismatchError(f"Input is missing feature columns: {missing}")
        columns = {}
        for spec in self.features:
            column = table[spec.name]
            if not pd.api.types.is_numeric_dtype(column.dtype):
                raise SchemaMismatchError(f"Feature {spec.name} is not numeric.")
            values = column.to_numpy(dtype="float64")
            if spec.kind == KIND_INTEGER:
                whole = np.isfinite(values) & (values == np.round(values))
                if not whole.all():
                    raise SchemaMismatchError(
                        f"Integer feature {spec.name} has missing or fractional values."
                    )
                columns[spec.name] = values.astype("int64")
            else:
                columns[spec.name] = values
        return pd.DataFrame(columns, index=table.index)


@dataclass(frozen=True)
class ModelArtifact:
    """A fitted model, its input contract, and its decision threshold.

    Attributes
    ----------
    model
        Fitted model behind the common interface.
    schema
        Feature schema the model was trained on.
    threshold
        Validation-fold decision threshold; ``inf`` flags nothing.
    metadata
        Provenance of the training run (model name, package version, seed,
        data digest, whether the data were synthetic).
    """

    model: FraudModel
    schema: FeatureSchema
    threshold: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def build_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Engineer and validate model inputs from raw transactions.

        Point-in-time account features see only the rows passed in, so the
        frame must contain the earlier transactions of the accounts involved
        for those features to match training.
        """
        missing = [c for c in self.schema.raw_columns if c not in transactions.columns]
        if missing:
            raise SchemaMismatchError(f"Transactions lack raw columns: {missing}")
        table = build_features(
            transactions, self.schema.feature_config, self.schema.time_column
        )
        return self.schema.validate(table)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Fraud scores for a feature table checked against the schema."""
        return self.model.predict_scores(self.schema.validate(features))

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Binary alert decisions at the artifact's threshold."""
        return apply_threshold(scores, self.threshold)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_threshold(threshold: float) -> float | None:
    return None if math.isinf(threshold) else float(threshold)


def save_artifact(artifact: ModelArtifact, directory: Path) -> Path:
    """Write an artifact directory.

    Parameters
    ----------
    artifact
        Artifact to store.
    directory
        Target directory; created if missing. Existing artifact files are
        overwritten.

    Returns
    -------
    pathlib.Path
        Path of the written ``manifest.json``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILE
    with model_path.open("wb") as fh:
        pickle.dump(artifact.model, fh)
    (directory / SCHEMA_FILE).write_text(
        json.dumps(artifact.schema.to_dict(), indent=2) + "\n"
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "schema_version": artifact.schema.version,
        "model_sha256": _sha256(model_path),
        # JSON has no infinity; null means flag nothing
        "threshold": _json_threshold(artifact.threshold),
        "metadata": dict(artifact.metadata),
    }
    manifest_path = directory / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read and check the manifest of an artifact directory."""
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise ArtifactError(f"No {MANIFEST_FILE} in {directory}.")
    manifest = json.loads(path.read_text())
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ArtifactError(
            f"Unsupported artifact format {version}; expected {FORMAT_VERSION}."
        )
    return manifest


def load_artifact(directory: Path) -> ModelArtifact:
    """Load an artifact directory and verify it against its manifest.

    Raises
    ------
    ArtifactError
        If the manifest is missing or of another format, the model file does
        not match its recorded digest, or the schema file does not match the
        recorded schema version.
    """
    manifest = read_manifest(directory)
    model_path = directory / MODEL_FILE
    if _sha256(model_path) != manifest["model_sha256"]:
        raise ArtifactError(f"{model_path} does not match the manifest digest.")
    schema = FeatureSchema.from_dict(json.loads((directory / SCHEMA_FILE).read_text()))
    if schema.version != manifest["schema_version"]:
        raise ArtifactError(
            f"Schema version {schema.version} differs from the manifest "
            f"({manifest['schema_version']})."
        )
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    threshold = manifest["threshold"]
    return ModelArtifact(
        model=model,
        schema=schema,
        threshold=math.inf if threshold is None else float(threshold),
        metadata=manifest.get("metadata", {}),
    )


def artifact_metadata(
    model_name: str,
    seed: int,
    synthetic: bool,
    data_sha256: str | None = None,
) -> dict[str, Any]:
    """Provenance fields recorded in the manifest of a training run."""
    return {
        "model": model_name,
        "package_version": __version__,
        "seed": seed,
        "synthetic_data": synthetic,
        "data_sha256": data_sha256,
    }


__all__ = [
    "FORMAT_VERSION",
    "ArtifactError",
    "FeatureSchema",
    "FeatureSpec",
    "ModelArtifact",
    "SchemaMismatchError",
    "artifact_metadata",
    "load_artifact",
    "read_manifest",
    "required_raw_columns",
    "save_artifact",
]
