from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from app.inference.contracts import DetectorInferenceMode
from app.schemas.base import ContractModel
from app.schemas.live_result import (
    DataOrigin,
    DetectionCategory,
    LiveResult,
    MetricUnit,
    Route,
    Severity,
    SourceMode,
    Zone,
)
from app.schemas.model_status import InferenceState, ModelStatus

MAX_VIDEO_PRIORITY_OBSERVATIONS = 64
MAX_VIDEO_DETECTED_CLASSES = 64


class MediaOrigin(StrEnum):
    USER_VIDEO_FILE = "USER_VIDEO_FILE"
    USER_WEBCAM = "USER_WEBCAM"


class IngestionSessionState(StrEnum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    EXPIRED = "EXPIRED"


class SessionCounters(ContractModel):
    frames_received: int = Field(default=0, ge=0)
    frames_accepted: int = Field(default=0, ge=0)
    frames_rejected: int = Field(default=0, ge=0)
    frames_out_of_order: int = Field(default=0, ge=0)
    protocol_errors: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)
    inference_frames_submitted: int = Field(default=0, ge=0)
    inference_frames_dropped: int = Field(default=0, ge=0)
    intelligence_updates_sent: int = Field(default=0, ge=0)


class SessionLimits(ContractModel):
    recommended_capture_fps: float = Field(ge=1, le=10)
    jpeg_quality: float = Field(ge=0.5, le=0.95)
    max_frame_bytes: int = Field(ge=1)
    accepted_mime_types: list[str]


class IngestionSessionCreate(ContractModel):
    source_mode: Literal[SourceMode.VIDEO_FILE, SourceMode.WEBCAM]
    media_origin: MediaOrigin
    detector_mode: DetectorInferenceMode = DetectorInferenceMode.STANDARD

    @model_validator(mode="after")
    def validate_provenance_pair(self) -> Self:
        valid_pair = (
            self.source_mode is SourceMode.VIDEO_FILE
            and self.media_origin is MediaOrigin.USER_VIDEO_FILE
        ) or (
            self.source_mode is SourceMode.WEBCAM and self.media_origin is MediaOrigin.USER_WEBCAM
        )
        if not valid_pair:
            raise ValueError("media_origin must match source_mode")
        return self


class IngestionSession(ContractModel):
    session_id: str = Field(min_length=20)
    source_mode: SourceMode
    media_origin: MediaOrigin
    detector_mode: DetectorInferenceMode
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
    inference_state: InferenceState | None = None
    segmentation_status: ModelStatus | None = None
    detection_status: ModelStatus | None = None


class FrameIntelligence(ContractModel):
    type: Literal["frame_intelligence"] = "frame_intelligence"
    session_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    sequence: int = Field(ge=0)
    result: LiveResult


class AggregateMetricAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    NO_ANALYZED_FRAMES = "NO_ANALYZED_FRAMES"


class AggregateMetricAggregation(StrEnum):
    PEAK_SIMULTANEOUS_DIRECT_DETECTIONS = "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"
    PEAK_FRESH_SEGMENTATION = "PEAK_FRESH_SEGMENTATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AggregateMetric(ContractModel):
    value: float | None = Field(default=None, ge=0)
    unit: MetricUnit
    availability: AggregateMetricAvailability
    aggregation: AggregateMetricAggregation
    supporting_frame_count: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.availability is AggregateMetricAvailability.AVAILABLE and self.value is None:
            raise ValueError("available aggregate metrics require a value")
        if (
            self.availability is not AggregateMetricAvailability.AVAILABLE
            and self.value is not None
        ):
            raise ValueError("unavailable aggregate metrics cannot contain a value")
        return self


class VideoAnalysisStatistics(ContractModel):
    flooded_area_percent: AggregateMetric
    people_detected: AggregateMetric
    vehicles_detected: AggregateMetric
    blocked_road_cells: AggregateMetric
    damaged_buildings: AggregateMetric
    building_damage_coverage_percent: AggregateMetric


class DetectedClassFinding(ContractModel):
    label: str = Field(min_length=1)
    category: DetectionCategory
    peak_simultaneous_count: int = Field(ge=1)
    max_confidence: float = Field(ge=0, le=1)
    supporting_frame_count: int = Field(ge=1)
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC


class VideoPriorityObservation(ContractModel):
    zone: Zone
    source_frame_id: int = Field(ge=0)
    media_time_ms: int = Field(ge=0)
    supporting_update_count: int = Field(ge=1)
    segmentation_evidence_available: bool
    detection_evidence_available: bool
    building_damage_count_availability: AggregateMetricAvailability
    associated_route: Route | None = None
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC


class VideoAnalysisSummary(ContractModel):
    session_id: str = Field(min_length=20)
    generated_at_ms: int = Field(ge=0)
    frames_accepted: int = Field(ge=0)
    frames_analyzed: int = Field(ge=0)
    frames_dropped: int = Field(ge=0)
    first_analyzed_frame_id: int | None = Field(default=None, ge=0)
    last_analyzed_frame_id: int | None = Field(default=None, ge=0)
    first_media_time_ms: int | None = Field(default=None, ge=0)
    last_media_time_ms: int | None = Field(default=None, ge=0)
    statistics: VideoAnalysisStatistics
    detected_classes: list[DetectedClassFinding] = Field(
        max_length=MAX_VIDEO_DETECTED_CLASSES
    )
    detected_classes_truncated: bool = False
    priorities: list[VideoPriorityObservation] = Field(
        max_length=MAX_VIDEO_PRIORITY_OBSERVATIONS
    )
    priorities_truncated: bool = False
    highest_priority_zone_id: str | None
    incident_severity: Severity | None
    segmentation_status: ModelStatus
    detection_status: ModelStatus
    inference_state: InferenceState
    responsible_ai_statement: str = Field(min_length=1)
    data_origin: Literal[DataOrigin.DERIVED_ANALYTIC] = DataOrigin.DERIVED_ANALYTIC


class VideoAnalysisComplete(ContractModel):
    type: Literal["video_analysis_complete"] = "video_analysis_complete"
    session_id: str = Field(min_length=20)
    state: Literal[IngestionSessionState.COMPLETE] = IngestionSessionState.COMPLETE
    summary: VideoAnalysisSummary
    latest_result: LiveResult | None
