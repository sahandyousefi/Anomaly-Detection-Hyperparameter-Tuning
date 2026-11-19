import importlib

import pytest

import fraud_detection

SUBPACKAGES = [
    "data",
    "features",
    "models",
    "tuning",
    "evaluation",
    "explain",
    "monitoring",
    "serving",
]


def test_version_is_string() -> None:
    assert isinstance(fraud_detection.__version__, str)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"fraud_detection.{name}")
    assert module.__doc__
