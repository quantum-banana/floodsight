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
    SIMULATION = "SIMULATION"


class CoordinateSpace(StrEnum):
    NORMALIZED_IMAGE = "NORMALIZED_IMAGE"
    RELATIVE_TACTICAL = "RELATIVE_TACTICAL"


class StreamState(StrEnum):
    CONNECTING = "CONNECTING"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"


class ApiState(StrEnum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class SystemStatus(ContractModel):
    api: ApiState
    segmentation_model: ModelState
    detection_model: ModelState


class IncidentMetadata(ContractModel):
    incident_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location_label: str = Field(min_length=1)
    started_at_ms: int = Field(ge=0)
    coordinate_space: CoordinateSpace
    data_origin: DataOrigin


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
    SIMULATED = "simulated"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class SegmentationClass(ContractModel):
    label: str = Field(min_length=1)
    coverage_percent: Percentage
    confidence: UnitInterval
    data_origin: DataOrigin


class OverlayKind(StrEnum):
    FLOOD = "FLOOD"
    DAMAGED_BUILDING = "DAMAGED_BUILDING"


class OverlayRegion(ContractModel):
    overlay_id: str = Field(min_length=1)
    kind: OverlayKind
    label: str = Field(min_length=1)
    polygon: list[Point] = Field(min_length=3)
    confidence: UnitInterval
    data_origin: DataOrigin


class Segmentation(ContractModel):
    status: SegmentationState
    classes: list[SegmentationClass]
    regions: list[OverlayRegion] = Field(default_factory=list)


class RoadState(StrEnum):
    CLEAR = "CLEAR"
    FLOODED = "FLOODED"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class AccessStatus(StrEnum):
    ACCESSIBLE = "ACCESSIBLE"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    ISOLATED = "ISOLATED"
    UNKNOWN = "UNKNOWN"


class Road(ContractModel):
    road_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    state: RoadState
    access_status: AccessStatus
    geometry: list[Point] = Field(min_length=2)
    confidence: UnitInterval | None
    data_origin: DataOrigin


class Severity(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ZoneReason(ContractModel):
    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    contribution: Percentage
    data_origin: DataOrigin


class Zone(ContractModel):
    zone_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    rank: int = Field(ge=1)
    severity: Severity
    priority_score: Percentage
    confidence: UnitInterval
    polygon: list[Point] = Field(min_length=3)
    people_count: int = Field(ge=0)
    vehicle_count: int = Field(ge=0)
    flood_coverage_percent: Percentage
    building_damage_count: int = Field(ge=0)
    road_condition: RoadState
    access_status: AccessStatus
    primary_reason: str = Field(min_length=1)
    reasons: list[ZoneReason]
    updated_at_ms: int = Field(ge=0)
    data_origin: DataOrigin


class EventSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EventCategory(StrEnum):
    FLOOD = "FLOOD"
    DETECTION = "DETECTION"
    ACCESS = "ACCESS"
    PRIORITY = "PRIORITY"
    ROUTE = "ROUTE"
    SYSTEM = "SYSTEM"


class IncidentEvent(ContractModel):
    event_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    severity: EventSeverity
    category: EventCategory
    message: str = Field(min_length=1)
    data_origin: DataOrigin


class RouteStatus(StrEnum):
    RECOMMENDED = "RECOMMENDED"
    UNAVAILABLE = "UNAVAILABLE"


class Route(ContractModel):
    route_id: str = Field(min_length=1)
    status: RouteStatus
    target_zone_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    waypoints: list[Point]
    distance_m: NonNegativeNumber | None
    access_summary: str = Field(min_length=1)
    data_origin: DataOrigin


class LiveResult(ContractModel):
    incident_id: str = Field(min_length=1)
    incident: IncidentMetadata
    frame_id: int = Field(ge=0)
    snapshot_index: int = Field(ge=0)
    snapshot_count: int = Field(ge=1)
    timestamp_ms: int = Field(ge=0)
    source_mode: SourceMode
    coordinate_space: CoordinateSpace
    data_origin: DataOrigin
    stream_state: StreamState
    incident_severity: Severity
    highest_priority_zone_id: str | None
    system_status: SystemStatus
    statistics: Statistics
    detections: list[Detection]
    segmentation: Segmentation
    roads: list[Road]
    zones: list[Zone]
    events: list[IncidentEvent]
    route: Route | None
