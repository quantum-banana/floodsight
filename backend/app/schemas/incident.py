from app.schemas.base import ContractModel
from app.schemas.live_result import (
    CoordinateSpace,
    DataOrigin,
    IncidentMetadata,
    LiveResult,
    Severity,
    SourceMode,
    Statistics,
)


class IncidentSummary(ContractModel):
    incident_id: str
    title: str
    severity: Severity
    source_mode: SourceMode
    coordinate_space: CoordinateSpace
    snapshot_count: int
    data_origin: DataOrigin


class IncidentListResponse(ContractModel):
    incidents: list[IncidentSummary]
    data_origin: DataOrigin


class IncidentDetailResponse(ContractModel):
    incident: IncidentMetadata
    severity: Severity
    snapshot_count: int
    initial_snapshot: LiveResult
    latest_snapshot: LiveResult
    data_origin: DataOrigin


class IncidentReport(ContractModel):
    incident_id: str
    title: str
    generated_at_ms: int
    severity: Severity
    statistics: Statistics
    critical_zone_count: int
    highest_priority_zone_id: str
    highest_priority_zone_name: str
    explanation: str
    access_summary: str
    responsible_ai_statement: str
    data_origin: DataOrigin
