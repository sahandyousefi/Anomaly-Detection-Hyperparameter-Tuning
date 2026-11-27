"""Content-hash versioning of the raw PaySim file.

PaySim is a single, immutable public CSV, so a tool such as DVC, which adds a
remote cache and a pipeline graph, is not needed to identify the data. The
raw file stays untracked and its identity is pinned instead: a small JSON
manifest stored next to it in ``data/`` and committed to the repository
records the SHA-256 digest, the size in bytes, and the number of rows and
columns. Every run on the real file computes the digest again and stores it
with the results, so a reported metric can always be traced to the exact
bytes it was computed from.

Workflow::

    fraud-detection data pin      # once, after the first download
    fraud-detection data verify   # any later time, or on another machine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Shape of the public Kaggle release (ealaxi/paysim1), see data/README.md
PAYSIM_ROWS = 6_362_620
PAYSIM_COLUMNS_COUNT = 11

_CHUNK_BYTES = 1 << 20


class DataVersionError(ValueError):
    """Raised when a data file does not match its pinned manifest."""


@dataclass(frozen=True)
class DataManifest:
    """Identity of one raw data file.

    Attributes
    ----------
    file_name
        Base name of the file; the directory is not part of its identity.
    sha256
        Hex digest of the file contents.
    size_bytes
        File size.
    rows
        Data rows, excluding the header line.
    columns
        Fields in the header line.
    """

    file_name: str
    sha256: str
    size_bytes: int
    rows: int
    columns: int


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows_and_columns(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        header = fh.readline()
        rows = sum(1 for line in fh if line.strip())
    return rows, len(header.decode("utf-8").strip().split(","))


def describe_file(path: str | Path) -> DataManifest:
    """Compute the manifest of a CSV file.

    Parameters
    ----------
    path
        CSV file with a header line.

    Returns
    -------
    DataManifest
        Digest, size, and shape of the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. See data/README.md for download "
            "instructions."
        )
    rows, columns = _count_rows_and_columns(path)
    return DataManifest(
        file_name=path.name,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        rows=rows,
        columns=columns,
    )


def manifest_path(data_path: str | Path) -> Path:
    """Manifest location for a data file: ``data/<stem>.manifest.json``.

    The manifest sits in the parent of the raw directory, which is tracked,
    rather than next to the untracked raw file.
    """
    data_path = Path(data_path)
    return data_path.parent.parent / f"{data_path.stem}.manifest.json"


def write_manifest(manifest: DataManifest, path: str | Path) -> Path:
    """Write a manifest as indented JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return path


def read_manifest(path: str | Path) -> DataManifest:
    """Read a manifest written by :func:`write_manifest`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No data manifest at {path}. Run `fraud-detection data pin` after "
            "downloading the data."
        )
    return DataManifest(**json.loads(path.read_text()))


def shape_warnings(manifest: DataManifest) -> list[str]:
    """Differences from the published shape of the PaySim release."""
    messages = []
    if manifest.rows != PAYSIM_ROWS:
        messages.append(f"{manifest.rows} rows, expected {PAYSIM_ROWS}")
    if manifest.columns != PAYSIM_COLUMNS_COUNT:
        messages.append(f"{manifest.columns} columns, expected {PAYSIM_COLUMNS_COUNT}")
    return messages


def verify_file(data_path: str | Path, manifest_file: str | Path) -> DataManifest:
    """Check a data file against its pinned manifest.

    Parameters
    ----------
    data_path
        Raw data file.
    manifest_file
        Manifest written when the file was pinned.

    Returns
    -------
    DataManifest
        The manifest of the file, equal to the pinned one.

    Raises
    ------
    DataVersionError
        If size, shape, or digest differ from the pinned values.
    """
    expected = read_manifest(manifest_file)
    actual = describe_file(data_path)
    differences = [
        f"{field}: pinned {getattr(expected, field)}, found {getattr(actual, field)}"
        for field in ("size_bytes", "rows", "columns", "sha256")
        if getattr(expected, field) != getattr(actual, field)
    ]
    if differences:
        raise DataVersionError(
            f"{data_path} does not match {manifest_file}: " + "; ".join(differences)
        )
    return actual


def data_fingerprint(data_path: str | Path) -> str:
    """Digest of the data file, verified against its manifest when pinned.

    Raises
    ------
    DataVersionError
        If a manifest exists and the file does not match it.
    """
    pinned = manifest_path(data_path)
    if pinned.is_file():
        return verify_file(data_path, pinned).sha256
    logger.warning(
        "%s is not pinned; run `fraud-detection data pin` to record it", data_path
    )
    return file_sha256(data_path)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options of the ``data`` command."""
    parser.add_argument("action", choices=("pin", "verify"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration whose paths.raw_data is used (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With pin, replace an existing manifest that differs.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Execute ``data pin`` or ``data verify`` from parsed arguments."""
    from fraud_detection.config import load_config

    data_path = load_config(args.config).paths.raw_data
    target = manifest_path(data_path)
    if args.action == "verify":
        manifest = verify_file(data_path, target)
        logger.info("%s matches %s (sha256 %s)", data_path, target, manifest.sha256)
        return 0

    manifest = describe_file(data_path)
    for message in shape_warnings(manifest):
        logger.warning("%s differs from the public release: %s", data_path, message)
    if target.is_file() and read_manifest(target) != manifest and not args.force:
        raise DataVersionError(
            f"{target} pins a different file; pass --force to replace it."
        )
    write_manifest(manifest, target)
    logger.info("Pinned %s (sha256 %s) in %s", data_path, manifest.sha256, target)
    return 0
