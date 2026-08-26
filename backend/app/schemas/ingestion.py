from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from app.schemas.base import ContractModel
from app.schemas.live_result import DataOrigin, SourceMode


class MediaOrigin(StrEnum):
    USER_VIDEO_FILE = "USER_VIDEO_FILE"
    USER_WEBCAM = "USER_WEBCAM"


class IngestionSessionState(StrEnum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    EXPIRED = "EXPIRED"


class SessionCounters(ContractModel):
    frames_received: int = Field(default=0, ge=0)
    frames_accepted: int = Field(default=0, ge=0)
    frames_rejected: int = Field(default=0, ge=0)
    frames_out_of_order: int = Field(default=0, ge=0)
    protocol_errors: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)


class SessionLimits(ContractModel):
    recommended_capture_fps: float = Field(ge=1, le=10)
    jpeg_quality: float = Field(ge=0.5, le=0.95)
    max_frame_bytes: int = Field(ge=1)
    accepted_mime_types: list[str]


class IngestionSessionCreate(ContractModel):
    source_mode: Literal[SourceMode.VIDEO_FILE, SourceMode.WEBCAM]
    media_origin: MediaOrigin

    @model_validator(mode="after")
    def validate_provenance_pair(self) -> Self:
        valid_pair = (
            self.source_mode is SourceMode.VIDEO_FILE
            and self.media_origin is MediaOrigin.USER_VIDEO_FILE
        ) or (
            self.source_mode is SourceMode.WEBCAM
            and self.media_origin is MediaOrigin.USER_WEBCAM
        )
        if not valid_pair:
            raise ValueError("media_origin must match source_mode")
        return self


class IngestionSession(ContractModel):
    session_id: str = Field(min_length=20)
    source_mode: SourceMode
    media_origin: MediaOrigin
    state: IngestionSessionState
    created_at_ms: int = Field(ge=0)
    last_activity_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)
    counters: SessionCounters
    limits: SessionLimits
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC


class FrameMetadata(ContractModel):
    type: Literal["frame_metadata"] = "frame_metadata"
    frame_id: int = Field(ge=0)
    captured_at_ms: int = Field(ge=0)
    media_time_ms: int = Field(ge=0)
    source_mode: SourceMode
    media_origin: MediaOrigin
    mime_type: str = Field(min_length=1, max_length=100)
    byte_length: int = Field(gt=0)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)


class FrameQuality(ContractModel):
    mean_luminance: float = Field(ge=0, le=255)
    laplacian_variance: float = Field(ge=0)
    brightness_status: Literal["NORMAL", "DARK", "BRIGHT"]
    sharpness_status: Literal["NORMAL", "BLURRY"]
    warnings: list[str]
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC


class DecodedFrame(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(gt=0)


class FrameResult(ContractModel):
    type: Literal["frame_result"] = "frame_result"
    session_id: str
    frame_id: int | None
    accepted: bool
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    received_at_ms: int = Field(ge=0)
    processing_ms: float = Field(ge=0)
    byte_length: int = Field(ge=0)
    decoded_frame: DecodedFrame | None
    quality: FrameQuality | None
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC
