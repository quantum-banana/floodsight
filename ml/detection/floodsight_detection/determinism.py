"""Deterministic process controls, applied only by explicit run entrypoints."""

from __future__ import annotations

import os
import random
from typing import Any

from floodsight_detection.errors import DetectionInfrastructureError

_HASH_PROBE = "floodsight-detection-hash-seed-probe-v1"
_PINNED_HASH_PROBES = {20260831: -2887746748678338831}


def require_prestarted_hash_seed(seed: int) -> dict[str, Any]:
    """Prove the pinned hash seed was active when this interpreter started."""

    expected_probe = _PINNED_HASH_PROBES.get(seed)
    configured = os.environ.get("PYTHONHASHSEED")
    if configured != str(seed) or expected_probe is None or hash(_HASH_PROBE) != expected_probe:
        raise DetectionInfrastructureError(
            "PYTHONHASHSEED was not fixed to the frozen seed before Python started.",
            code="python_hash_seed_not_prestarted",
        )
    return {
        "python_hash_seed": configured,
        "python_hash_probe": expected_probe,
        "python_hash_seed_prestarted": True,
    }


def configure_determinism(seed: int, *, include_ml_libraries: bool) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    report: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_hash_seed_prestarted": (
            seed in _PINNED_HASH_PROBES
            and os.environ.get("PYTHONHASHSEED") == str(seed)
            and hash(_HASH_PROBE) == _PINNED_HASH_PROBES[seed]
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "numpy": False,
        "torch": False,
    }
    if not include_ml_libraries:
        return report
    # Imports remain behind the explicit smoke/training authorization gates.
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    report["numpy"] = True
    report["torch"] = True
    report["cuda"] = bool(torch.cuda.is_available())
    return report
