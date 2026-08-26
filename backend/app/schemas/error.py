from typing import Any

from app.schemas.base import ContractModel


class ErrorDetail(ContractModel):
    code: str
    message: str
    details: list[Any]


class ErrorResponse(ContractModel):
    error: ErrorDetail

