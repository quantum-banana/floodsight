from pydantic import Field

from app.inference.contracts import EvidenceSource
from app.schemas.base import ContractModel
from app.schemas.live_result import AccessStatus, Point, RoadState


class GridCellEvidence(ContractModel):
    cell_id: str = Field(pattern=r"^[A-D][1-4]$")
    row: int = Field(ge=0, le=3)
    column: int = Field(ge=0, le=3)
    polygon: list[Point] = Field(min_length=4, max_length=4)
    flood_coverage_percent: float = Field(ge=0, le=100)
    pool_coverage_percent: float = Field(ge=0, le=100)
    person_confidences: list[float]
    vehicle_count: int = Field(ge=0)
    building_damage_coverage_percent: float = Field(ge=0, le=100)
    road_clear_coverage_percent: float = Field(ge=0, le=100)
    road_flooded_coverage_percent: float = Field(ge=0, le=100)
    road_blocked_coverage_percent: float = Field(ge=0, le=100)
    road_non_flooded_coverage_percent: float = Field(ge=0, le=100)
    road_state: RoadState
    access_status: AccessStatus
    confidence: float = Field(ge=0, le=1)
    risk_signal: float = Field(ge=0, le=100)
    sources: list[EvidenceSource]

    @property
    def people_count(self) -> int:
        return len(self.person_confidences)


class ZoneCandidate(ContractModel):
    candidate_id: str = Field(min_length=1)
    grid_cells: list[str] = Field(min_length=1)
    polygon: list[Point] = Field(min_length=4)
    timestamp_ms: int = Field(ge=0)
    person_confidences: list[float]
    vehicle_count: int = Field(ge=0)
    flood_coverage_percent: float = Field(ge=0, le=100)
    pool_coverage_percent: float = Field(ge=0, le=100)
    building_damage_coverage_percent: float = Field(ge=0, le=100)
    road_state: RoadState
    access_status: AccessStatus
    confidence: float = Field(ge=0, le=1)
    risk_signal: float = Field(ge=0, le=100)
    sources: list[EvidenceSource]


class OperationalZone(ContractModel):
    zone_id: str = Field(min_length=1)
    grid_cells: list[str] = Field(min_length=1)
    polygon: list[Point] = Field(min_length=4)
    timestamp_ms: int = Field(ge=0)
    people_count: int = Field(ge=0)
    max_person_confidence: float = Field(ge=0, le=1)
    vehicle_count: int = Field(ge=0)
    flood_coverage_percent: float = Field(ge=0, le=100)
    pool_coverage_percent: float = Field(ge=0, le=100)
    building_damage_coverage_percent: float = Field(ge=0, le=100)
    road_state: RoadState
    access_status: AccessStatus
    confidence: float = Field(ge=0, le=1)
    risk_signal: float = Field(ge=0, le=100)
    temporal_samples: int = Field(ge=1)
    stale: bool = False
    sources: list[EvidenceSource]


class RoadEdgeEvidence(ContractModel):
    edge_id: str = Field(min_length=1)
    start_node: str = Field(min_length=1)
    end_node: str = Field(min_length=1)
    geometry: list[Point] = Field(min_length=2, max_length=2)
    state: RoadState
    severity: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    travel_cost: float = Field(gt=0)
    enabled: bool
    vehicle_compatibility: str | None = None


class AccessibilityGraph(ContractModel):
    nodes: dict[str, Point]
    edges: list[RoadEdgeEvidence]
