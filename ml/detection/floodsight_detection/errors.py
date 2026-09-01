"""Typed, machine-readable failures for the detection training gate."""

from __future__ import annotations

from typing import Any


class DetectionInfrastructureError(RuntimeError):
    """A blocking validation, safety, dependency, or runtime failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or []

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}
