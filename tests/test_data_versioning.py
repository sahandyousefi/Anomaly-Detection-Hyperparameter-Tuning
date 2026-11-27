import json
from pathlib import Path

import pytest
import yaml

from fraud_detection import cli
from fraud_detection.data.synthetic import generate_paysim_like
from fraud_detection.data.versioning import (
    PAYSIM_ROWS,
    DataVersionError,
    data_fingerprint,
    describe_file,
    file_sha256,
    manifest_path,
    read_manifest,
    shape_warnings,
    verify_file,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "raw" / "paysim.csv"
    path.parent.mkdir(parents=True)
    generate_paysim_like(n_rows=500, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture
def config_path(tmp_path: Path, raw_file: Path) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"]["raw_data"] = str(raw_file)
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    return path


def test_file_sha256_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(path) == expected


def test_describe_file_counts_rows_and_columns(raw_file: Path) -> None:
    manifest = describe_file(raw_file)
    assert manifest.file_name == "paysim.csv"
    assert manifest.rows == 500
    assert manifest.columns == 11
    assert manifest.size_bytes == raw_file.stat().st_size
    assert shape_warnings(manifest) == [f"500 rows, expected {PAYSIM_ROWS}"]


def test_manifest_lives_in_tracked_data_directory(raw_file: Path) -> None:
    assert manifest_path(raw_file) == raw_file.parents[1] / "paysim.manifest.json"


def test_manifest_round_trip(raw_file: Path, tmp_path: Path) -> None:
    manifest = describe_file(raw_file)
    path = write_manifest(manifest, tmp_path / "m.json")
    assert read_manifest(path) == manifest
    assert verify_file(raw_file, path) == manifest


def test_modified_file_fails_verification(raw_file: Path, tmp_path: Path) -> None:
    path = write_manifest(describe_file(raw_file), tmp_path / "m.json")
    text = raw_file.read_text()
    raw_file.write_text(text.replace("TRANSFER", "CASH_OUT", 1))
    with pytest.raises(DataVersionError, match="sha256"):
        verify_file(raw_file, path)


def test_fingerprint_verifies_pinned_file(raw_file: Path) -> None:
    assert data_fingerprint(raw_file) == file_sha256(raw_file)
    write_manifest(describe_file(raw_file), manifest_path(raw_file))
    raw_file.write_text(raw_file.read_text() + raw_file.read_text().splitlines()[1])
    with pytest.raises(DataVersionError):
        data_fingerprint(raw_file)


def test_pin_and_verify_commands(config_path: Path, raw_file: Path) -> None:
    assert cli.main(["data", "pin", "--config", str(config_path)]) == 0
    pinned = json.loads(manifest_path(raw_file).read_text())
    assert pinned["sha256"] == file_sha256(raw_file)
    assert cli.main(["data", "verify", "--config", str(config_path)]) == 0

    with raw_file.open("a") as fh:
        fh.write(raw_file.read_text().splitlines()[1] + "\n")
    with pytest.raises(DataVersionError):
        cli.main(["data", "verify", "--config", str(config_path)])
    with pytest.raises(DataVersionError, match="--force"):
        cli.main(["data", "pin", "--config", str(config_path)])
    assert cli.main(["data", "pin", "--config", str(config_path), "--force"]) == 0


def test_training_on_real_file_records_digest(
    config_path: Path, raw_file: Path, tmp_path: Path
) -> None:
    raw = yaml.safe_load(config_path.read_text())
    raw["model"]["params"] = {"n_estimators": 20, "max_depth": 3, "n_jobs": 1}
    raw["resampling"] = {"method": "none", "sampling_strategy": 1.0}
    raw_file.unlink()
    log = generate_paysim_like(n_rows=6_000, fraud_rate=0.03, n_steps=120, seed=6)
    log.to_csv(raw_file, index=False)
    config_path.write_text(yaml.safe_dump(raw))

    out = tmp_path / "model"
    assert cli.main(["train", "--config", str(config_path), "--output", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data_sha256"] == file_sha256(raw_file)
    assert metrics["synthetic_data"] is False
