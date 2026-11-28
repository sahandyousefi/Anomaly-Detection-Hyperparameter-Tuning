import importlib.util
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
ARTIFACT_ENV = "FRAUD_DETECTION_ARTIFACT"


def _env(name: str) -> str:
    match = re.search(rf"{name}=(\S+)", DOCKERFILE)
    assert match, f"{name} is not set in the Dockerfile"
    return match.group(1)


def test_entrypoint_module_exists() -> None:
    match = re.search(r'"([\w.]+):(\w+)"', DOCKERFILE)
    assert match
    module, attribute = match.groups()
    assert importlib.util.find_spec(module) is not None
    assert attribute == "app"


def test_image_and_compose_agree_on_artifact_path() -> None:
    service = COMPOSE["services"]["scoring"]
    image_path = _env(ARTIFACT_ENV)
    assert service["environment"][ARTIFACT_ENV] == image_path
    mounts = [volume.split(":") for volume in service["volumes"]]
    assert [m[1:] for m in mounts if m[1] == image_path] == [[image_path, "ro"]]


def test_compose_mounts_training_output() -> None:
    sources = [v.split(":")[0] for v in COMPOSE["services"]["scoring"]["volumes"]]
    # make train writes its artifact to artifacts/model/artifact
    assert "./artifacts/model/artifact" in sources


def test_container_runs_as_non_root() -> None:
    users = re.findall(r"^USER (\S+)", DOCKERFILE, flags=re.MULTILINE)
    assert users and users[-1] not in ("root", "0")


def test_port_is_consistent() -> None:
    assert "EXPOSE 8000" in DOCKERFILE
    assert '"--port", "8000"' in DOCKERFILE
    assert "8000:8000" in COMPOSE["services"]["scoring"]["ports"]


def test_build_context_keeps_package_sources() -> None:
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    for needed in ("src", "pyproject.toml", "README.md", "LICENSE"):
        assert needed not in ignored
    assert "data" in ignored
    assert "artifacts" in ignored
