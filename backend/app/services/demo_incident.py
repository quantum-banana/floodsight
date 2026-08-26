from dataclasses import dataclass
from functools import lru_cache

from app.core.errors import AppError
from app.schemas.incident import (
    IncidentDetailResponse,
    IncidentListResponse,
    IncidentReport,
    IncidentSummary,
)
from app.schemas.live_result import (
    AccessStatus,
    ApiState,
    BoundingBox,
    CoordinateSpace,
    DataOrigin,
    Detection,
    DetectionCategory,
    EventCategory,
    EventSeverity,
    IncidentEvent,
    IncidentMetadata,
    LiveResult,
    Metric,
    MetricUnit,
    OverlayKind,
    OverlayRegion,
    Point,
    Road,
    RoadState,
    Route,
    RouteStatus,
    Segmentation,
    SegmentationClass,
    SegmentationState,
    Severity,
    SourceMode,
    Statistics,
    StreamState,
    SystemStatus,
    Zone,
    ZoneReason,
)
from app.schemas.model_status import ModelState

DEMO_INCIDENT_ID = "FS-001"
DEMO_TITLE = "Riverside Ward Flood Response"
STARTED_AT_MS = 1_725_960_100_000
SNAPSHOT_INTERVAL_MS = 10_000
ORIGIN = DataOrigin.DEMO_SIMULATED

INCIDENT = IncidentMetadata(
    incident_id=DEMO_INCIDENT_ID,
    title=DEMO_TITLE,
    location_label="Riverside Ward · Relative Sector",
    started_at_ms=STARTED_AT_MS,
    coordinate_space=CoordinateSpace.RELATIVE_TACTICAL,
    data_origin=ORIGIN,
)

SYSTEM_STATUS = SystemStatus(
    api=ApiState.OPERATIONAL,
    segmentation_model=ModelState.NOT_CONFIGURED,
    detection_model=ModelState.NOT_CONFIGURED,
)


def _point(x: float, y: float) -> Point:
    return Point(x=x, y=y)


def _polygon(points: tuple[tuple[float, float], ...]) -> list[Point]:
    return [_point(x, y) for x, y in points]


def _metric(value: float, unit: MetricUnit, confidence: float = 0.92) -> Metric:
    return Metric(value=value, unit=unit, confidence=confidence, data_origin=ORIGIN)


FLOOD_PERCENT = (18.0, 25.0, 31.0, 38.0, 42.0, 42.0)
PEOPLE_COUNTS = (0, 3, 3, 5, 6, 6)
VEHICLE_COUNTS = (1, 2, 2, 3, 4, 4)
BLOCKED_ROAD_COUNTS = (0, 0, 1, 1, 2, 2)
DAMAGED_BUILDING_COUNTS = (1, 1, 2, 4, 5, 5)
INCIDENT_SEVERITIES = (
    Severity.MODERATE,
    Severity.MODERATE,
    Severity.HIGH,
    Severity.CRITICAL,
    Severity.CRITICAL,
    Severity.CRITICAL,
)


def _statistics(index: int) -> Statistics:
    return Statistics(
        flooded_area_percent=_metric(FLOOD_PERCENT[index], MetricUnit.PERCENT),
        people_detected=_metric(PEOPLE_COUNTS[index], MetricUnit.COUNT),
        vehicles_detected=_metric(VEHICLE_COUNTS[index], MetricUnit.COUNT),
        blocked_roads=_metric(BLOCKED_ROAD_COUNTS[index], MetricUnit.COUNT),
        damaged_buildings=_metric(DAMAGED_BUILDING_COUNTS[index], MetricUnit.COUNT),
    )


PERSON_POSITIONS = (
    (0.60, 0.35),
    (0.64, 0.38),
    (0.68, 0.34),
    (0.62, 0.42),
    (0.73, 0.40),
    (0.34, 0.64),
)
VEHICLE_POSITIONS = (
    (0.20, 0.73),
    (0.45, 0.58),
    (0.77, 0.67),
    (0.38, 0.31),
)


def _detections(index: int) -> list[Detection]:
    detections: list[Detection] = []
    for offset, (x, y) in enumerate(PERSON_POSITIONS[: PEOPLE_COUNTS[index]], start=1):
        detections.append(
            Detection(
                detection_id=f"PERSON-{offset:02d}",
                category=DetectionCategory.PERSON,
                label="Simulated person",
                confidence=0.88 + (offset % 3) * 0.02,
                bbox=BoundingBox(x=x, y=y, width=0.025, height=0.045),
                data_origin=ORIGIN,
            )
        )
    for offset, (x, y) in enumerate(VEHICLE_POSITIONS[: VEHICLE_COUNTS[index]], start=1):
        detections.append(
            Detection(
                detection_id=f"VEHICLE-{offset:02d}",
                category=DetectionCategory.VEHICLE,
                label="Simulated vehicle",
                confidence=0.9,
                bbox=BoundingBox(x=x, y=y, width=0.055, height=0.035),
                data_origin=ORIGIN,
            )
        )
    return detections


FLOOD_POLYGONS = (
    ((0.00, 0.52), (0.21, 0.47), (0.42, 0.55), (0.58, 0.75), (0.38, 1.00), (0.00, 1.00)),
    ((0.00, 0.43), (0.23, 0.40), (0.46, 0.50), (0.64, 0.76), (0.42, 1.00), (0.00, 1.00)),
    ((0.00, 0.36), (0.25, 0.34), (0.53, 0.47), (0.70, 0.73), (0.48, 1.00), (0.00, 1.00)),
    ((0.00, 0.30), (0.28, 0.28), (0.58, 0.42), (0.76, 0.72), (0.54, 1.00), (0.00, 1.00)),
    ((0.00, 0.25), (0.31, 0.25), (0.62, 0.40), (0.80, 0.72), (0.58, 1.00), (0.00, 1.00)),
    ((0.00, 0.25), (0.31, 0.25), (0.62, 0.40), (0.80, 0.72), (0.58, 1.00), (0.00, 1.00)),
)

DAMAGE_POSITIONS = (
    (0.31, 0.40),
    (0.55, 0.28),
    (0.72, 0.47),
    (0.80, 0.32),
    (0.44, 0.72),
)


def _segmentation(index: int) -> Segmentation:
    regions = [
        OverlayRegion(
            overlay_id="FLOOD-01",
            kind=OverlayKind.FLOOD,
            label="Simulated flood extent",
            polygon=_polygon(FLOOD_POLYGONS[index]),
            confidence=0.94,
            data_origin=ORIGIN,
        )
    ]
    for offset, (x, y) in enumerate(
        DAMAGE_POSITIONS[: DAMAGED_BUILDING_COUNTS[index]],
        start=1,
    ):
        regions.append(
            OverlayRegion(
                overlay_id=f"DAMAGE-{offset:02d}",
                kind=OverlayKind.DAMAGED_BUILDING,
                label=f"Simulated damaged structure {offset}",
                polygon=_polygon(
                    (
                        (x, y),
                        (x + 0.055, y),
                        (x + 0.055, y + 0.055),
                        (x, y + 0.055),
                    )
                ),
                confidence=0.86 + (offset % 2) * 0.03,
                data_origin=ORIGIN,
            )
        )
    return Segmentation(
        status=SegmentationState.SIMULATED,
        classes=[
            SegmentationClass(
                label="Simulated flood extent",
                coverage_percent=FLOOD_PERCENT[index],
                confidence=0.94,
                data_origin=ORIGIN,
            )
        ],
        regions=regions,
    )


ROAD_GEOMETRY = {
    "R01": ((0.04, 0.84), (0.25, 0.70), (0.49, 0.60), (0.73, 0.55), (0.96, 0.41)),
    "R02": ((0.10, 0.18), (0.30, 0.31), (0.48, 0.45), (0.62, 0.61), (0.78, 0.86)),
    "R03": ((0.12, 0.55), (0.32, 0.52), (0.55, 0.50), (0.78, 0.48), (0.93, 0.51)),
    "R04": ((0.56, 0.10), (0.58, 0.28), (0.61, 0.46), (0.68, 0.66), (0.76, 0.88)),
    "R05": ((0.18, 0.92), (0.34, 0.78), (0.52, 0.68), (0.71, 0.61), (0.90, 0.62)),
}


def _roads(index: int) -> list[Road]:
    roads: list[Road] = []
    for road_id, geometry in ROAD_GEOMETRY.items():
        state = RoadState.CLEAR
        access = AccessStatus.ACCESSIBLE
        if road_id == "R04" and index >= 2 or road_id == "R03" and index >= 4:
            state = RoadState.BLOCKED
            access = AccessStatus.BLOCKED
        elif road_id in {"R02", "R05"} and index >= 1:
            state = RoadState.FLOODED
            access = AccessStatus.DEGRADED
        roads.append(
            Road(
                road_id=road_id,
                label=f"Relative road {road_id}",
                state=state,
                access_status=access,
                geometry=_polygon(geometry),
                confidence=0.91,
                data_origin=ORIGIN,
            )
        )
    return roads


ZONE_POLYGONS = {
    "ZONE-1": ((0.16, 0.54), (0.37, 0.50), (0.43, 0.69), (0.25, 0.78), (0.12, 0.68)),
    "ZONE-2": ((0.52, 0.22), (0.79, 0.20), (0.86, 0.46), (0.68, 0.58), (0.50, 0.44)),
    "ZONE-4": ((0.61, 0.61), (0.88, 0.58), (0.94, 0.83), (0.73, 0.92), (0.57, 0.80)),
}


@dataclass(frozen=True)
class ZoneState:
    zone_id: str
    rank: int
    score: float
    severity: Severity
    people: int
    vehicles: int
    flood: float
    damage: int
    road: RoadState
    access: AccessStatus
    primary_reason: str
    contributions: tuple[float, float, float, float]


ZONE_STATES = (
    (
        ZoneState(
            "ZONE-1",
            1,
            38,
            Severity.LOW,
            0,
            1,
            28,
            1,
            RoadState.CLEAR,
            AccessStatus.ACCESSIBLE,
            "Localized flood exposure",
            (4, 8, 8, 18),
        ),
        ZoneState(
            "ZONE-2",
            2,
            31,
            Severity.LOW,
            0,
            0,
            25,
            0,
            RoadState.CLEAR,
            AccessStatus.ACCESSIBLE,
            "Flood edge approaching residences",
            (2, 7, 7, 15),
        ),
    ),
    (
        ZoneState(
            "ZONE-2",
            1,
            58,
            Severity.MODERATE,
            3,
            0,
            44,
            0,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Three people near increasing floodwater",
            (24, 12, 4, 18),
        ),
        ZoneState(
            "ZONE-1",
            2,
            42,
            Severity.MODERATE,
            0,
            1,
            36,
            1,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Access beginning to degrade",
            (4, 12, 9, 17),
        ),
        ZoneState(
            "ZONE-4",
            3,
            30,
            Severity.LOW,
            0,
            1,
            22,
            0,
            RoadState.CLEAR,
            AccessStatus.ACCESSIBLE,
            "Monitor flood approach",
            (2, 6, 6, 16),
        ),
    ),
    (
        ZoneState(
            "ZONE-2",
            1,
            68,
            Severity.HIGH,
            3,
            0,
            52,
            1,
            RoadState.BLOCKED,
            AccessStatus.DEGRADED,
            "Primary access road R04 is blocked",
            (25, 20, 8, 15),
        ),
        ZoneState(
            "ZONE-1",
            2,
            48,
            Severity.MODERATE,
            0,
            1,
            43,
            1,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Flooded road limits access",
            (4, 14, 10, 20),
        ),
        ZoneState(
            "ZONE-4",
            3,
            40,
            Severity.MODERATE,
            0,
            1,
            31,
            0,
            RoadState.CLEAR,
            AccessStatus.ACCESSIBLE,
            "Flood exposure increasing",
            (2, 8, 8, 22),
        ),
    ),
    (
        ZoneState(
            "ZONE-2",
            1,
            92,
            Severity.CRITICAL,
            5,
            0,
            78,
            3,
            RoadState.BLOCKED,
            AccessStatus.ISOLATED,
            "People isolated by blocked access and structural damage",
            (35, 24, 18, 15),
        ),
        ZoneState(
            "ZONE-4",
            2,
            70,
            Severity.HIGH,
            0,
            2,
            58,
            1,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Damage and flood exposure are increasing",
            (18, 18, 17, 17),
        ),
        ZoneState(
            "ZONE-1",
            3,
            52,
            Severity.MODERATE,
            0,
            1,
            49,
            0,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Localized access degradation",
            (6, 14, 12, 20),
        ),
    ),
    (
        ZoneState(
            "ZONE-2",
            1,
            92,
            Severity.CRITICAL,
            4,
            1,
            81,
            3,
            RoadState.BLOCKED,
            AccessStatus.ISOLATED,
            "People isolated with primary ground access blocked",
            (35, 24, 18, 15),
        ),
        ZoneState(
            "ZONE-4",
            2,
            76,
            Severity.HIGH,
            2,
            2,
            63,
            2,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "People exposed near damaged structures",
            (26, 18, 17, 15),
        ),
        ZoneState(
            "ZONE-1",
            3,
            54,
            Severity.MODERATE,
            0,
            1,
            52,
            0,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Localized flooding with degraded access",
            (10, 14, 12, 18),
        ),
    ),
    (
        ZoneState(
            "ZONE-2",
            1,
            92,
            Severity.CRITICAL,
            4,
            1,
            81,
            3,
            RoadState.BLOCKED,
            AccessStatus.ISOLATED,
            "People isolated with primary ground access blocked",
            (35, 24, 18, 15),
        ),
        ZoneState(
            "ZONE-4",
            2,
            76,
            Severity.HIGH,
            2,
            2,
            63,
            2,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "People exposed near damaged structures",
            (26, 18, 17, 15),
        ),
        ZoneState(
            "ZONE-1",
            3,
            54,
            Severity.MODERATE,
            0,
            1,
            52,
            0,
            RoadState.FLOODED,
            AccessStatus.DEGRADED,
            "Localized flooding with degraded access",
            (10, 14, 12, 18),
        ),
    ),
)


def _zone_reasons(state: ZoneState) -> list[ZoneReason]:
    definitions = (
        ("HUMAN_RISK", "Human risk", f"{state.people} simulated people in the zone"),
        ("ACCESS", "Access and isolation", state.access.value.replace("_", " ").title()),
        ("STRUCTURE", "Structural damage", f"{state.damage} damaged structures"),
        ("FLOOD", "Flood severity", f"{state.flood:.0f}% simulated coverage"),
    )
    return [
        ZoneReason(
            code=code,
            label=label,
            description=description,
            contribution=contribution,
            data_origin=ORIGIN,
        )
        for (code, label, description), contribution in zip(
            definitions,
            state.contributions,
            strict=True,
        )
    ]


def _zones(index: int, timestamp_ms: int) -> list[Zone]:
    return [
        Zone(
            zone_id=state.zone_id,
            display_name=state.zone_id.replace("ZONE-", "Zone "),
            rank=state.rank,
            severity=state.severity,
            priority_score=state.score,
            confidence=0.93 if state.zone_id == "ZONE-2" else 0.89,
            polygon=_polygon(ZONE_POLYGONS[state.zone_id]),
            people_count=state.people,
            vehicle_count=state.vehicles,
            flood_coverage_percent=state.flood,
            building_damage_count=state.damage,
            road_condition=state.road,
            access_status=state.access,
            primary_reason=state.primary_reason,
            reasons=_zone_reasons(state),
            updated_at_ms=timestamp_ms,
            data_origin=ORIGIN,
        )
        for state in ZONE_STATES[index]
    ]


EVENT_DEFINITIONS = (
    (
        "EVENT-001",
        0,
        EventSeverity.INFO,
        EventCategory.FLOOD,
        "Localized flooding identified in the relative sensor view.",
    ),
    (
        "EVENT-002",
        1,
        EventSeverity.INFO,
        EventCategory.SYSTEM,
        "Zones 1 and 2 entered simulated monitoring.",
    ),
    (
        "EVENT-003",
        2,
        EventSeverity.WARNING,
        EventCategory.DETECTION,
        "A simulated cluster of three people persisted in Zone 2.",
    ),
    (
        "EVENT-004",
        3,
        EventSeverity.WARNING,
        EventCategory.FLOOD,
        "Flood coverage increased to 25 percent.",
    ),
    (
        "EVENT-005",
        4,
        EventSeverity.CRITICAL,
        EventCategory.ACCESS,
        "Road R04 changed to blocked; Zone 2 access degraded.",
    ),
    (
        "EVENT-006",
        5,
        EventSeverity.WARNING,
        EventCategory.PRIORITY,
        "Zone 2 rose to high rescue priority.",
    ),
    (
        "EVENT-007",
        6,
        EventSeverity.WARNING,
        EventCategory.DETECTION,
        "Structural damage reported near the Zone 2 cluster.",
    ),
    (
        "EVENT-008",
        7,
        EventSeverity.CRITICAL,
        EventCategory.PRIORITY,
        "Zone 2 escalated to critical priority.",
    ),
    (
        "EVENT-009",
        8,
        EventSeverity.WARNING,
        EventCategory.PRIORITY,
        "Rescue priorities reordered with Zone 2 ranked first.",
    ),
    (
        "EVENT-010",
        9,
        EventSeverity.INFO,
        EventCategory.ROUTE,
        "Relative alternate route identified to Zone 2.",
    ),
    (
        "EVENT-011",
        10,
        EventSeverity.INFO,
        EventCategory.SYSTEM,
        "Simulated conditions stabilised at critical severity.",
    ),
    (
        "EVENT-012",
        11,
        EventSeverity.INFO,
        EventCategory.SYSTEM,
        "Latest incident report state is ready.",
    ),
)
EVENT_COUNTS = (2, 4, 6, 8, 10, 12)


def _events(index: int) -> list[IncidentEvent]:
    return [
        IncidentEvent(
            event_id=event_id,
            timestamp_ms=STARTED_AT_MS + sequence * 5_000,
            severity=severity,
            category=category,
            message=message,
            data_origin=ORIGIN,
        )
        for event_id, sequence, severity, category, message in EVENT_DEFINITIONS[
            : EVENT_COUNTS[index]
        ]
    ]


def _route(index: int) -> Route | None:
    if index < 4:
        return None
    return Route(
        route_id="ROUTE-ALT-02",
        status=RouteStatus.RECOMMENDED,
        target_zone_id="ZONE-2",
        label="Relative alternate access corridor",
        waypoints=_polygon(
            ((0.08, 0.90), (0.22, 0.78), (0.39, 0.67), (0.49, 0.55), (0.54, 0.42), (0.64, 0.36))
        ),
        distance_m=None,
        access_summary="Relative route avoids simulated blockages on R03 and R04.",
        data_origin=ORIGIN,
    )


def _build_snapshot(index: int) -> LiveResult:
    timestamp_ms = STARTED_AT_MS + index * SNAPSHOT_INTERVAL_MS
    zones = _zones(index, timestamp_ms)
    return LiveResult(
        incident_id=DEMO_INCIDENT_ID,
        incident=INCIDENT.model_copy(deep=True),
        frame_id=index,
        snapshot_index=index,
        snapshot_count=len(ZONE_STATES),
        timestamp_ms=timestamp_ms,
        source_mode=SourceMode.SIMULATION,
        coordinate_space=CoordinateSpace.RELATIVE_TACTICAL,
        data_origin=ORIGIN,
        stream_state=StreamState.COMPLETE if index == len(ZONE_STATES) - 1 else StreamState.PLAYING,
        incident_severity=INCIDENT_SEVERITIES[index],
        highest_priority_zone_id=zones[0].zone_id if zones else None,
        system_status=SYSTEM_STATUS.model_copy(deep=True),
        statistics=_statistics(index),
        detections=_detections(index),
        segmentation=_segmentation(index),
        roads=_roads(index),
        zones=zones,
        events=_events(index),
        route=_route(index),
    )


@lru_cache
def _scenario() -> tuple[LiveResult, ...]:
    return tuple(_build_snapshot(index) for index in range(len(ZONE_STATES)))


def get_demo_snapshots(incident_id: str = DEMO_INCIDENT_ID) -> list[LiveResult]:
    _ensure_incident(incident_id)
    return [snapshot.model_copy(deep=True) for snapshot in _scenario()]


def list_demo_incidents() -> IncidentListResponse:
    latest = _scenario()[-1]
    return IncidentListResponse(
        incidents=[
            IncidentSummary(
                incident_id=DEMO_INCIDENT_ID,
                title=DEMO_TITLE,
                severity=latest.incident_severity,
                source_mode=SourceMode.SIMULATION,
                coordinate_space=CoordinateSpace.RELATIVE_TACTICAL,
                snapshot_count=len(_scenario()),
                data_origin=ORIGIN,
            )
        ],
        data_origin=ORIGIN,
    )


def get_demo_incident(incident_id: str) -> IncidentDetailResponse:
    _ensure_incident(incident_id)
    snapshots = _scenario()
    return IncidentDetailResponse(
        incident=INCIDENT.model_copy(deep=True),
        severity=snapshots[-1].incident_severity,
        snapshot_count=len(snapshots),
        initial_snapshot=snapshots[0].model_copy(deep=True),
        latest_snapshot=snapshots[-1].model_copy(deep=True),
        data_origin=ORIGIN,
    )


def get_demo_report(incident_id: str) -> IncidentReport:
    _ensure_incident(incident_id)
    latest = _scenario()[-1]
    highest = latest.zones[0]
    access_summary = latest.route.access_summary if latest.route else "No relative route available."
    return IncidentReport(
        incident_id=DEMO_INCIDENT_ID,
        title=DEMO_TITLE,
        generated_at_ms=latest.timestamp_ms,
        severity=latest.incident_severity,
        statistics=latest.statistics.model_copy(deep=True),
        critical_zone_count=sum(zone.severity is Severity.CRITICAL for zone in latest.zones),
        highest_priority_zone_id=highest.zone_id,
        highest_priority_zone_name=highest.display_name,
        explanation=highest.primary_reason,
        access_summary=access_summary,
        responsible_ai_statement=(
            "FloodSight is decision support. This simulated report requires human review "
            "before any operational action."
        ),
        data_origin=ORIGIN,
    )


def _ensure_incident(incident_id: str) -> None:
    if incident_id != DEMO_INCIDENT_ID:
        raise AppError(
            status_code=404,
            code="incident_not_found",
            message=f"Demo incident '{incident_id}' was not found.",
        )
