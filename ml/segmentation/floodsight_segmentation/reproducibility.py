"""Deterministic seeding and serializable random-number-generator state."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool,
    cudnn_benchmark: bool,
) -> None:
    """Seed Python, NumPy, and Torch and configure deterministic kernels."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if deterministic_algorithms and torch.cuda.is_available():
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
            )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.backends.cudnn.benchmark = cudnn_benchmark


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(*, data_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture every RNG used by transforms, sampling, and model operations."""

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_generator": data_generator.get_state() if data_generator is not None else None,
    }
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    data_generator: torch.Generator | None = None,
) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    # ``torch.load(..., map_location="cuda")`` moves every tensor in the
    # checkpoint, including RNG byte tensors, onto CUDA.  PyTorch's RNG restore
    # APIs require CPU ByteTensors even when they are restoring CUDA generators.
    # Normalize these bookkeeping tensors explicitly so direct-to-device
    # checkpoint loading remains exactly resumable.
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    generator_state = state.get("data_generator")
    if generator_state is not None and data_generator is not None:
        data_generator.set_state(generator_state.cpu())
