from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetToolError(Exception):
    """Expected user-facing failure without an automatic traceback."""

    message: str
    code: str = "dataset_error"
    details: list[dict[str, Any]] = field(default_factory=list)
    exit_code: int = 2

    def __str__(self) -> str:
        return self.message


class BlockingValidationError(DatasetToolError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "validation_failed",
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message, code, details or [], 3)
