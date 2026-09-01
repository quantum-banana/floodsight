"""Explicit authorization boundary for all real-data model operations."""

from __future__ import annotations

from .errors import TrainingAuthorizationError


def require_training_authorization(allow_training: bool) -> None:
    """Require an unambiguous CLI/API opt-in before real data or weights are opened."""

    if allow_training is not True:
        raise TrainingAuthorizationError(
            "Real SegFormer train/validate operations are locked. Re-run with "
            "--allow-training only after the dataset, configuration, and human-review gates pass."
        )


def require_real_smoke_authorization(allow_real_smoke: bool) -> None:
    """Unlock only the bounded one-step real-manifest smoke path."""

    if allow_real_smoke is not True:
        raise TrainingAuthorizationError(
            "The bounded real-manifest smoke is locked. Re-run with --allow-real-smoke "
            "only after its exact config, manifests, model artifact, and human approval are frozen."
        )
