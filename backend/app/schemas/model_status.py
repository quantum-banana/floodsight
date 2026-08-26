from enum import StrEnum

from app.schemas.base import ContractModel


class ModelState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelStatus(ContractModel):
    status: ModelState
    model: str | None


class ModelStatusResponse(ContractModel):
    segmentation: ModelStatus
    detection: ModelStatus
