"""Seeding and environment capture for reproducible runs.

Every stochastic component of the package already receives ``config.seed``
explicitly (samplers, boosters, Isolation Forest, network initialisation).
:func:`seed_everything` additionally fixes the global generators that
third-party code may fall back on, and switches PyTorch to deterministic
kernels. Bitwise reproducibility is only expected for the same library
versions and hardware, which is why :func:`environment_info` is stored with
every run.
"""

from __future__ import annotations

import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version

import numpy as np

TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "optuna",
    "shap",
    "mlflow",
)

# Required by cuBLAS for deterministic matrix products on CUDA >= 10.2
_CUBLAS_WORKSPACE = ":4096:8"


def _seed_torch(seed: int, deterministic: bool) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # warn_only keeps operations without a deterministic kernel usable
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return True


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, bool]:
    """Seed Python, NumPy, and (if installed) PyTorch global generators.

    Parameters
    ----------
    seed
        Seed shared by all generators.
    deterministic
        Request deterministic PyTorch kernels and disable cuDNN autotuning.
        Slower on GPU; has no effect on CPU-only runs beyond the flag.

    Returns
    -------
    dict
        Which libraries were seeded.

    Notes
    -----
    ``PYTHONHASHSEED`` is exported for child processes (for example data
    loader workers); it cannot change string hashing of the running
    interpreter.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global state used by deps
    return {"python": True, "numpy": True, "torch": _seed_torch(seed, deterministic)}


def environment_info(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Interpreter, platform, and versions of the installed modeling libraries.

    Packages that are not installed are omitted.
    """
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            info[name] = version(name)
        except PackageNotFoundError:
            continue
    return info
