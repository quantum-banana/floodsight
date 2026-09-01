from enum import StrEnum

from pydantic import Field

from app.schemas.base import ContractModel


class ModelState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    LOADING = "loading"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class InferenceState(StrEnum):
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    MODEL_LOADING = "MODEL_LOADING"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    SIMULATED_FALLBACK = "SIMULATED_FALLBACK"
    ERROR = "ERROR"


class ModelOperationalMode(StrEnum):
    REAL = "REAL"
    FALLBACK = "FALLBACK"
    SIMULATED = "SIMULATED"
    UNAVAILABLE = "UNAVAILABLE"


class ModelStatus(ContractModel):
    status: ModelState
    model: str | None
    mode: ModelOperationalMode = ModelOperationalMode.UNAVAILABLE
    version: str | None = None
    device: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    last_successful_inference_ms: int | None = Field(default=None, ge=0)
    provenance_mode: str | None = None
    message: str | None = None


class ModelStatusResponse(ContractModel):
    segmentation: ModelStatus
    detection: ModelStatus
    inference_state: InferenceState = InferenceState.MODEL_UNAVAILABLE
