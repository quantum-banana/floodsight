"""Typed errors used by the segmentation training stack."""

from __future__ import annotations


class SegmentationError(RuntimeError):
    """Base class for expected, actionable segmentation errors."""


class ConfigurationError(SegmentationError):
    """Raised when a training configuration is incomplete or unsafe."""


class ManifestError(SegmentationError):
    """Raised when a frozen dataset manifest violates the loader contract."""


class TrainingAuthorizationError(SegmentationError):
    """Raised when a real-data operation was not explicitly authorized."""


class CheckpointError(SegmentationError):
    """Raised when a checkpoint is invalid or incompatible with the run."""


class DependencyError(SegmentationError):
    """Raised when the isolated ML environment is unavailable."""


class ArtifactError(SegmentationError):
    """Raised when locally staged model weights lack exact audited provenance."""
