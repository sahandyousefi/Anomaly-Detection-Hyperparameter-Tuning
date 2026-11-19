"""Fraud detection on the PaySim mobile money dataset."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fraud-detection")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
