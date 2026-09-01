from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import ContractModel
from app.schemas.model_status import InferenceState, ModelState, ModelStatus

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


class FeatureProvenance(StrEnum):
    SEGMENTATION = "SEGMENTATION"
    DETECTION = "DETECTION"
    DERIVED = "DERIVED"
    SIMULATED = "SIMULATED"


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
    inference_state: InferenceState | None = None
    segmentation_details: ModelStatus | None = None
    detection_details: ModelStatus | None = None


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


class SourceDimensions(ContractModel):
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)


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
    source_class: str | None = None
    source_class_id: int | None = Field(default=None, ge=0)
    source_confidence: UnitInterval | None = None
    detection_confidence: UnitInterval | None = None
    track_id: str | None = Field(default=None, min_length=1)
    track_confidence: UnitInterval | None = None
    persistence: int | None = Field(default=None, ge=1)
    observation_state: Literal["DETECTED", "TRACK_PERSISTED"] | None = None
    source_frame_id: int | None = Field(default=None, ge=0)
    model_id: str | None = None
    model_provenance: Literal["REAL_MODEL", "PRETRAINED_FALLBACK", "SIMULATED"] | None = None


class SegmentationState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    SIMULATED = "simulated"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class SegmentationClass(ContractModel):
    class_id: int | None = Field(default=None, ge=0, le=255)
    label: str = Field(min_length=1)
    coverage_percent: Percentage
    confidence: UnitInterval
    data_origin: DataOrigin
    color: list[int] | None = Field(default=None, min_length=3, max_length=3)


class SegmentationMask(ContractModel):
    encoding: Literal["PNG_BASE64"] = "PNG_BASE64"
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    data: str = Field(min_length=1)


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
    mask: SegmentationMask | None = None


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
    travel_cost: NonNegativeNumber | None = None
    enabled: bool | None = None
    uncertainty: UnitInterval | None = None


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


class EvidenceLevel(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class ZoneAlert(ContractModel):
    code: Literal["POTENTIAL_STRANDED_PERSON"] = "POTENTIAL_STRANDED_PERSON"
    title: str = "Potential stranded person"
    person_evidence: EvidenceLevel
    flood_exposure: EvidenceLevel
    primary_access: AccessStatus
    confidence: UnitInterval
    temporal_samples: int = Field(ge=1)
    reason_codes: list[str] = Field(min_length=2)
    data_origin: DataOrigin = DataOrigin.DERIVED_ANALYTIC


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
    grid_cells: list[str] = Field(default_factory=list)
    building_damage_coverage_percent: Percentage = 0
    pool_coverage_percent: Percentage = 0
    temporal_samples: int = Field(default=1, ge=1)
    stale: bool = False
    alerts: list[ZoneAlert] = Field(default_factory=list)


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
    code: str | None = None


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
    edge_ids: list[str] = Field(default_factory=list)
    route_cost: NonNegativeNumber | None = None
    changed_reason: str | None = None
    changed_reason_code: str | None = None
    previous_edge_ids: list[str] = Field(default_factory=list)


class SceneSummary(ContractModel):
    water_flood_coverage_percent: Percentage
    pool_coverage_percent: Percentage
    road_clear_coverage_percent: Percentage
    road_flooded_coverage_percent: Percentage
    road_blocked_coverage_percent: Percentage
    building_damage_coverage_percent: Percentage
    provenance: list[FeatureProvenance]
    data_origin: DataOrigin


class EvidenceFrames(ContractModel):
    segmentation_source_frame_id: int | None = Field(default=None, ge=0)
    detection_source_frame_id: int | None = Field(default=None, ge=0)
    segmentation_reused: bool = False
    detection_reused: bool = False


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
    route_alternatives: list[Route] = Field(default_factory=list)
    scene_summary: SceneSummary | None = None
    source_dimensions: SourceDimensions | None = None
    evidence_frames: EvidenceFrames | None = None
