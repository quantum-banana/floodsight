from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.schemas.base import ContractModel
from app.schemas.model_status import ModelState

UnitInterval = Annotated[float, Field(ge=0, le=1)]
PositiveUnitInterval = Annotated[float, Field(gt=0, le=1)]
Percentage = Annotated[float, Field(ge=0, le=100)]
NonNegativeNumber = Annotated[float, Field(ge=0)]


class DataOrigin(StrEnum):
    REAL_ML_OUTPUT = "REAL_ML_OUTPUT"
    DERIVED_ANALYTIC = "DERIVED_ANALYTIC"
    GIS_EXTERNAL_DATA = "GIS_EXTERNAL_DATA"
    DEMO_SIMULATED = "DEMO_SIMULATED"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"


class SourceMode(StrEnum):
    VIDEO_FILE = "VIDEO_FILE"
    WEBCAM = "WEBCAM"
    DRONE_STREAM = "DRONE_STREAM"
    DEMO_REPLAY = "DEMO_REPLAY"


class ApiState(StrEnum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class SystemStatus(ContractModel):
    api: ApiState
    segmentation_model: ModelState
    detection_model: ModelState


class MetricUnit(StrEnum):
    COUNT = "count"
    PERCENT = "percent"


class Metric(ContractModel):
    value: NonNegativeNumber
    unit: MetricUnit
    confidence: UnitInterval | None
    data_origin: DataOrigin


class Statistics(ContractModel):
    flooded_area_percent: Metric
    people_detected: Metric
    vehicles_detected: Metric
    blocked_roads: Metric
    damaged_buildings: Metric


class Point(ContractModel):
    x: UnitInterval
    y: UnitInterval


class BoundingBox(ContractModel):
    x: UnitInterval
    y: UnitInterval
    width: PositiveUnitInterval
    height: PositiveUnitInterval


class DetectionCategory(StrEnum):
    PERSON = "PERSON"
    VEHICLE = "VEHICLE"
    OTHER = "OTHER"


class Detection(ContractModel):
    detection_id: str = Field(min_length=1)
    category: DetectionCategory
    label: str = Field(min_length=1)
    confidence: UnitInterval
    bbox: BoundingBox
    data_origin: DataOrigin


class SegmentationState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class SegmentationClass(ContractModel):
    label: str = Field(min_length=1)
    coverage_percent: Percentage
    confidence: UnitInterval
    data_origin: DataOrigin


class Segmentation(ContractModel):
    status: SegmentationState
    classes: list[SegmentationClass]


class RoadState(StrEnum):
    CLEAR = "CLEAR"
    FLOODED = "FLOODED"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class Road(ContractModel):
    road_id: str = Field(min_length=1)
    state: RoadState
    confidence: UnitInterval | None
    data_origin: DataOrigin


class Severity(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ZoneReason(ContractModel):
    label: str = Field(min_length=1)
    contribution: Percentage
    data_origin: DataOrigin


class Zone(ContractModel):
    zone_id: str = Field(min_length=1)
    severity: Severity
    priority_score: Percentage
    polygon: list[Point] = Field(min_length=3)
    reasons: list[ZoneReason]
    data_origin: DataOrigin


class EventSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class IncidentEvent(ContractModel):
    event_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    severity: EventSeverity
    message: str = Field(min_length=1)
    data_origin: DataOrigin


class RouteStatus(StrEnum):
    RECOMMENDED = "RECOMMENDED"
    UNAVAILABLE = "UNAVAILABLE"


class Route(ContractModel):
    route_id: str = Field(min_length=1)
    status: RouteStatus
    waypoints: list[Point]
    distance_m: NonNegativeNumber | None
    data_origin: DataOrigin


class LiveResult(ContractModel):
    incident_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    source_mode: SourceMode
    data_origin: DataOrigin
    system_status: SystemStatus
    statistics: Statistics
    detections: list[Detection]
    segmentation: Segmentation
    roads: list[Road]
    zones: list[Zone]
    events: list[IncidentEvent]
    route: Route | None
