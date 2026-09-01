from collections import deque

import numpy as np

from app.inference.contracts import FusedScene, NormalizedDetection, decode_mask
from app.inference.coordinates import bbox_bottom_center, grid_cell_for_point, grid_cell_polygon
from app.inference.taxonomy import Taxonomy
from app.intelligence.contracts import GridCellEvidence, PersonSpatialEvidence, ZoneCandidate
from app.schemas.live_result import AccessStatus, Point, RoadState

DAMAGE_CLASSES = {
    "building_minor_damage",
    "building_major_damage",
    "building_destroyed",
}
ROAD_CLASSES = {
    "road_clear",
    "road_flooded",
    "road_blocked",
    "road_non_flooded",
}
VEHICLE_CLASSES = {"car", "van", "truck", "bus", "bicycle", "motorcycle", "tricycle"}


class RescueZoneEngine:
    def __init__(self, taxonomy: Taxonomy, *, risk_threshold: float = 18.0) -> None:
        self.taxonomy = taxonomy
        self.risk_threshold = risk_threshold

    def cells(self, scene: FusedScene) -> list[GridCellEvidence]:
        class_map = decode_mask(scene.semantic_mask) if scene.semantic_mask is not None else None
        height = (
            scene.semantic_mask.height if scene.semantic_mask is not None else scene.source_height
        )
        width = scene.semantic_mask.width if scene.semantic_mask is not None else scene.source_width
        confidences = {item.class_name: item.confidence for item in scene.semantic_evidence}
        detections: dict[str, list[object]] = {
            f"{chr(ord('A') + row)}{column + 1}": [] for row in range(4) for column in range(4)
        }
        for detection in scene.detections:
            anchor = (
                bbox_bottom_center(detection.bbox)
                if detection.application_class == "person"
                else Point(
                    x=detection.bbox.x + detection.bbox.width / 2,
                    y=detection.bbox.y + detection.bbox.height / 2,
                )
            )
            detections[grid_cell_for_point(anchor)].append(detection)

        output: list[GridCellEvidence] = []
        for row in range(4):
            for column in range(4):
                cell_id = f"{chr(ord('A') + row)}{column + 1}"
                y1, y2 = row * height // 4, (row + 1) * height // 4
                x1, x2 = column * width // 4, (column + 1) * width // 4
                region = class_map[y1:y2, x1:x2] if class_map is not None else None
                total = max(1, (y2 - y1) * (x2 - x1))

                flood = _coverage(
                    region,
                    total,
                    self.taxonomy,
                    {
                        item.name
                        for item in self.taxonomy.classes
                        if item.class_id in scene.flood_class_ids
                    },
                )
                pool = _coverage(region, total, self.taxonomy, {"pool"})
                damage = _coverage(region, total, self.taxonomy, DAMAGE_CLASSES)
                clear = _coverage(region, total, self.taxonomy, {"road_clear"})
                flooded = _coverage(region, total, self.taxonomy, {"road_flooded"})
                blocked = _coverage(region, total, self.taxonomy, {"road_blocked"})
                non_flooded = _coverage(region, total, self.taxonomy, {"road_non_flooded"})
                road_state, access = _road_state(clear, flooded, blocked, non_flooded)
                cell_detections = detections[cell_id]
                person_evidence = [
                    _person_spatial_evidence(
                        item,
                        class_map=class_map,
                        width=width,
                        height=height,
                        taxonomy=self.taxonomy,
                        flood_class_ids=set(scene.flood_class_ids),
                    )
                    for item in cell_detections
                    if item.application_class == "person"
                ]
                people = [item.confidence for item in person_evidence]
                vehicles = sum(
                    item.application_class in VEHICLE_CLASSES for item in cell_detections
                )
                semantic_confidences = [
                    confidences.get(name, 0.0)
                    for name in (
                        "water",
                        "road_flooded",
                        "building_flooded",
                        "road_blocked",
                        "building_major_damage",
                        "building_destroyed",
                    )
                    if confidences.get(name, 0.0) > 0
                ]
                evidence_confidences = semantic_confidences + people
                confidence = (
                    sum(evidence_confidences) / len(evidence_confidences)
                    if evidence_confidences
                    else 0.0
                )
                risk = min(
                    100.0,
                    min(40.0, len(people) * 20.0)
                    + min(25.0, flood * 0.5)
                    + min(20.0, damage * 0.8)
                    + (15.0 if road_state is RoadState.BLOCKED else 7.0 if flooded else 0.0),
                )
                output.append(
                    GridCellEvidence(
                        cell_id=cell_id,
                        row=row,
                        column=column,
                        polygon=grid_cell_polygon(cell_id),
                        flood_coverage_percent=round(flood, 4),
                        pool_coverage_percent=round(pool, 4),
                        person_confidences=people,
                        person_evidence=person_evidence,
                        vehicle_count=vehicles,
                        building_damage_coverage_percent=round(damage, 4),
                        road_clear_coverage_percent=round(clear, 4),
                        road_flooded_coverage_percent=round(flooded, 4),
                        road_blocked_coverage_percent=round(blocked, 4),
                        road_non_flooded_coverage_percent=round(non_flooded, 4),
                        road_state=road_state,
                        access_status=access,
                        confidence=round(confidence, 6),
                        risk_signal=round(risk, 4),
                        sources=scene.provenance,
                    )
                )
        return output

    def build(self, scene: FusedScene) -> tuple[list[GridCellEvidence], list[ZoneCandidate]]:
        cells = self.cells(scene)
        return cells, self.candidates(
            cells,
            scene.timestamp_ms,
            person_observation_fresh=not scene.detection_reused,
        )

    def candidates(
        self,
        cells: list[GridCellEvidence],
        timestamp_ms: int,
        *,
        person_observation_fresh: bool = True,
    ) -> list[ZoneCandidate]:
        selected = {
            cell.cell_id: cell
            for cell in cells
            if cell.risk_signal >= self.risk_threshold or cell.people_count > 0
        }
        candidates: list[ZoneCandidate] = []
        visited: set[str] = set()
        for cell_id in sorted(selected):
            if cell_id in visited:
                continue
            component: list[GridCellEvidence] = []
            queue = deque([cell_id])
            visited.add(cell_id)
            while queue:
                current = queue.popleft()
                component.append(selected[current])
                for neighbor in _neighbors(current):
                    if neighbor in selected and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            candidates.append(
                _merge_component(
                    component,
                    timestamp_ms,
                    person_observation_fresh=person_observation_fresh,
                )
            )
        return candidates


def _person_spatial_evidence(
    detection: NormalizedDetection,
    *,
    class_map: object | None,
    width: int,
    height: int,
    taxonomy: Taxonomy,
    flood_class_ids: set[int],
) -> PersonSpatialEvidence:
    anchor = bbox_bottom_center(detection.bbox)
    center_x = min(width - 1, max(0, int(anchor.x * width)))
    center_y = min(height - 1, max(0, int(anchor.y * height)))
    radius_x = max(1, round(detection.bbox.width * width * 0.75))
    radius_y = max(1, round(detection.bbox.height * height * 0.35))
    x1, x2 = max(0, center_x - radius_x), min(width, center_x + radius_x + 1)
    y1, y2 = max(0, center_y - radius_y), min(height, center_y + radius_y + 1)
    region = class_map[y1:y2, x1:x2] if class_map is not None else None
    total = max(1, (y2 - y1) * (x2 - x1))
    flood_names = {item.name for item in taxonomy.classes if item.class_id in flood_class_ids}
    return PersonSpatialEvidence(
        detection_id=detection.detection_id,
        confidence=detection.confidence,
        bottom_center=anchor,
        local_flood_coverage_percent=round(_coverage(region, total, taxonomy, flood_names), 4),
        local_damage_coverage_percent=round(_coverage(region, total, taxonomy, DAMAGE_CLASSES), 4),
    )


def _road_state(
    clear: float, flooded: float, blocked: float, non_flooded: float
) -> tuple[RoadState, AccessStatus]:
    road_total = clear + flooded + blocked + non_flooded
    if blocked >= 0.5 and blocked / max(road_total, 0.001) >= 0.12:
        return RoadState.BLOCKED, AccessStatus.BLOCKED
    if flooded >= 0.5 and flooded / max(road_total, 0.001) >= 0.12:
        return RoadState.FLOODED, AccessStatus.DEGRADED
    if clear >= 0.5:
        return RoadState.CLEAR, AccessStatus.ACCESSIBLE
    # Non-flooded is deliberately not promoted to operationally clear.
    return RoadState.UNKNOWN, AccessStatus.UNKNOWN


def _coverage(
    region: object | None,
    total: int,
    taxonomy: Taxonomy,
    names: set[str],
) -> float:
    ids = [item.class_id for item in taxonomy.classes if item.name in names]
    if region is None or not ids:
        return 0.0
    return float(np.isin(region, ids).sum()) * 100 / total


def _neighbors(cell_id: str) -> list[str]:
    row = ord(cell_id[0]) - ord("A")
    column = int(cell_id[1:]) - 1
    return [
        f"{chr(ord('A') + next_row)}{next_column + 1}"
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
        if 0 <= next_row < 4 and 0 <= next_column < 4
    ]


def _merge_component(
    cells: list[GridCellEvidence],
    timestamp_ms: int,
    *,
    person_observation_fresh: bool,
) -> ZoneCandidate:
    ordered = sorted(cells, key=lambda item: item.cell_id)
    rows = [item.row for item in ordered]
    columns = [item.column for item in ordered]
    x1, x2 = min(columns) / 4, (max(columns) + 1) / 4
    y1, y2 = min(rows) / 4, (max(rows) + 1) / 4
    people = [confidence for cell in ordered for confidence in cell.person_confidences]
    person_evidence = [item for cell in ordered for item in cell.person_evidence]
    state = max((cell.road_state for cell in ordered), key=_road_priority)
    access = max((cell.access_status for cell in ordered), key=_access_priority)
    sources = list(dict.fromkeys(source for cell in ordered for source in cell.sources))
    return ZoneCandidate(
        candidate_id="GRID-" + "-".join(item.cell_id for item in ordered),
        grid_cells=[item.cell_id for item in ordered],
        polygon=[Point(x=x1, y=y1), Point(x=x2, y=y1), Point(x=x2, y=y2), Point(x=x1, y=y2)],
        timestamp_ms=timestamp_ms,
        person_confidences=people,
        person_evidence=person_evidence,
        person_observation_fresh=person_observation_fresh,
        vehicle_count=sum(item.vehicle_count for item in ordered),
        flood_coverage_percent=round(
            sum(item.flood_coverage_percent for item in ordered) / len(ordered), 4
        ),
        pool_coverage_percent=round(
            sum(item.pool_coverage_percent for item in ordered) / len(ordered), 4
        ),
        building_damage_coverage_percent=round(
            sum(item.building_damage_coverage_percent for item in ordered) / len(ordered), 4
        ),
        road_state=state,
        access_status=access,
        confidence=round(sum(item.confidence for item in ordered) / len(ordered), 6),
        risk_signal=round(max(item.risk_signal for item in ordered), 4),
        sources=sources,
    )


def _road_priority(state: RoadState) -> int:
    return {
        RoadState.CLEAR: 0,
        RoadState.UNKNOWN: 1,
        RoadState.FLOODED: 2,
        RoadState.BLOCKED: 3,
    }[state]


def _access_priority(status: AccessStatus) -> int:
    return {
        AccessStatus.ACCESSIBLE: 0,
        AccessStatus.UNKNOWN: 1,
        AccessStatus.DEGRADED: 2,
        AccessStatus.BLOCKED: 3,
        AccessStatus.ISOLATED: 4,
    }[status]
