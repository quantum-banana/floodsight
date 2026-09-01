from pathlib import Path

import numpy as np

from app.inference.contracts import (
    DetectionProvenance,
    DetectionResult,
    EvidenceSource,
    ModelIdentity,
    NormalizedDetection,
    SegmentationClassStatistic,
    SegmentationProvenance,
    SegmentationResult,
    encode_mask,
)
from app.inference.coordinates import grid_cell_polygon
from app.inference.fusion import SceneFusionEngine
from app.inference.taxonomy import load_taxonomy
from app.intelligence.contracts import GridCellEvidence, OperationalZone, ZoneCandidate
from app.intelligence.priority import PriorityEngine
from app.intelligence.routing import AccessibilityEngine, RoutingEngine
from app.intelligence.temporal import TemporalZoneTracker
from app.intelligence.zones import RescueZoneEngine
from app.schemas.live_result import AccessStatus, BoundingBox, RoadState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = load_taxonomy(
    PROJECT_ROOT / "shared/taxonomy/segmentation-taxonomy-v2.yaml"
)
MODEL = ModelIdentity(model_id="fixture", architecture="fixture", version="v1")


def _segmentation(class_map: np.ndarray, timestamp_ms: int = 1_000) -> SegmentationResult:
    stats = []
    for item in TAXONOMY.classes:
        count = int(np.count_nonzero(class_map == item.class_id))
        stats.append(
            SegmentationClassStatistic(
                class_id=item.class_id,
                class_name=item.name,
                pixel_count=count,
                coverage_percent=count * 100 / class_map.size,
                mean_confidence=0.9 if count else 0,
            )
        )
    return SegmentationResult(
        frame_id=0,
        timestamp_ms=timestamp_ms,
        source_width=class_map.shape[1],
        source_height=class_map.shape[0],
        model=MODEL,
        taxonomy_version=TAXONOMY.version,
        mask=encode_mask(class_map),
        class_statistics=stats,
        inference_latency_ms=1,
        device="cpu",
        provenance_mode=SegmentationProvenance.REAL_MODEL,
    )


def _detection(x: float, y: float, confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(
        frame_id=0,
        timestamp_ms=1_000,
        source_width=16,
        source_height=16,
        model=MODEL,
        taxonomy_version="detection-taxonomy-v1",
        detections=[
            NormalizedDetection(
                detection_id="person-1",
                application_class="person",
                source_class="person",
                source_class_id=0,
                confidence=confidence,
                bbox=BoundingBox(x=x, y=y, width=0.05, height=0.05),
            )
        ],
        inference_latency_ms=1,
        device="cpu",
        provenance_mode=DetectionProvenance.REAL_MODEL,
    )


def _scene(class_map: np.ndarray, detection: DetectionResult | None = None):
    return SceneFusionEngine(TAXONOMY).fuse(
        frame_id=0,
        timestamp_ms=1_000,
        source_width=16,
        source_height=16,
        segmentation=_segmentation(class_map),
        detection=detection,
    )


def test_pool_is_not_flood_and_person_assigns_to_four_by_four_cell() -> None:
    class_map = np.full((16, 16), 15, dtype=np.uint8)
    scene = _scene(class_map, _detection(0.28, 0.28))
    cells = RescueZoneEngine(TAXONOMY).cells(scene)
    by_id = {item.cell_id: item for item in cells}

    assert scene.pool_class_id == 15
    assert 15 not in scene.flood_class_ids
    assert by_id["B2"].people_count == 1
    assert by_id["B2"].pool_coverage_percent == 100
    assert by_id["B2"].flood_coverage_percent == 0


def test_adjacent_high_risk_cells_merge_deterministically() -> None:
    class_map = np.zeros((16, 16), dtype=np.uint8)
    class_map[4:8, 4:12] = 1
    _, candidates = RescueZoneEngine(TAXONOMY, risk_threshold=5).build(_scene(class_map))

    merged = next(item for item in candidates if {"B2", "B3"} <= set(item.grid_cells))
    assert merged.candidate_id.startswith("GRID-B2-B3")
    assert merged.grid_cells == sorted(merged.grid_cells)


def _candidate(
    timestamp_ms: int,
    *,
    cells: list[str] | None = None,
    people: list[float] | None = None,
    flood: float = 20,
) -> ZoneCandidate:
    selected = cells or ["B2"]
    return ZoneCandidate(
        candidate_id="GRID-" + "-".join(selected),
        grid_cells=selected,
        polygon=grid_cell_polygon(selected[0]),
        timestamp_ms=timestamp_ms,
        person_confidences=people or [],
        vehicle_count=0,
        flood_coverage_percent=flood,
        pool_coverage_percent=0,
        building_damage_coverage_percent=0,
        road_state=RoadState.UNKNOWN,
        access_status=AccessStatus.UNKNOWN,
        confidence=0.7,
        risk_signal=25,
        sources=[EvidenceSource.SEGMENTATION],
    )


def test_temporal_ids_stabilize_and_strong_person_escalates_immediately() -> None:
    tracker = TemporalZoneTracker()
    assert tracker.update([_candidate(1_000)], 1_000) == []
    stable = tracker.update([_candidate(1_500, cells=["B2", "B3"])], 1_500)
    assert stable[0].zone_id == "ZONE-01"

    urgent_tracker = TemporalZoneTracker(urgent_person_confidence=0.85)
    urgent = urgent_tracker.update([_candidate(2_000, people=[0.96])], 2_000)
    assert urgent[0].people_count == 1
    assert urgent[0].temporal_samples == 1


def test_priority_keeps_urgency_separate_from_confidence_and_emits_reasons() -> None:
    zone = OperationalZone(
        zone_id="ZONE-01",
        grid_cells=["B2"],
        polygon=grid_cell_polygon("B2"),
        timestamp_ms=2_000,
        people_count=4,
        max_person_confidence=0.55,
        vehicle_count=0,
        flood_coverage_percent=80,
        pool_coverage_percent=0,
        building_damage_coverage_percent=18,
        road_state=RoadState.BLOCKED,
        access_status=AccessStatus.ISOLATED,
        confidence=0.42,
        risk_signal=90,
        temporal_samples=2,
        sources=[EvidenceSource.DERIVED],
    )
    result = PriorityEngine().prioritize([zone])[0]

    assert result.priority_score >= 80
    assert result.confidence == 0.42
    codes = {reason.code for reason in result.reasons}
    assert {"PERSON_EVIDENCE", "PRIMARY_ACCESS_BLOCKED", "HIGH_FLOOD_EXPOSURE"} <= codes


def _cell(cell_id: str, state: RoadState = RoadState.CLEAR) -> GridCellEvidence:
    row = ord(cell_id[0]) - ord("A")
    column = int(cell_id[1:]) - 1
    access = AccessStatus.BLOCKED if state is RoadState.BLOCKED else AccessStatus.ACCESSIBLE
    return GridCellEvidence(
        cell_id=cell_id,
        row=row,
        column=column,
        polygon=grid_cell_polygon(cell_id),
        flood_coverage_percent=0,
        pool_coverage_percent=0,
        person_confidences=[],
        vehicle_count=0,
        building_damage_coverage_percent=0,
        road_clear_coverage_percent=10,
        road_flooded_coverage_percent=0,
        road_blocked_coverage_percent=10 if state is RoadState.BLOCKED else 0,
        road_non_flooded_coverage_percent=0,
        road_state=state,
        access_status=access,
        confidence=0.9,
        risk_signal=0,
        sources=[EvidenceSource.DERIVED],
    )


def test_route_changes_when_primary_access_becomes_blocked() -> None:
    engine = AccessibilityEngine()
    router = RoutingEngine(base_node="D1")
    cells = [_cell(f"{row}{column}") for row in "ABCD" for column in range(1, 5)]
    clear_graph = engine.build(cells)
    first, _ = router.route(clear_graph, zone_id="ZONE-01", target_cells=["A1"])

    blocked_cells = [
        _cell(cell.cell_id, RoadState.BLOCKED if cell.cell_id == "C1" else RoadState.CLEAR)
        for cell in cells
    ]
    blocked_graph = engine.build(blocked_cells)
    second, _ = router.route(blocked_graph, zone_id="ZONE-01", target_cells=["A1"])

    assert first.edge_ids != second.edge_ids
    assert second.changed_reason is not None
    assert second.distance_m is None
