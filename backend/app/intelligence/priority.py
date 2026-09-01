from app.inference.contracts import EvidenceSource
from app.intelligence.contracts import OperationalZone
from app.schemas.live_result import DataOrigin, Severity, Zone, ZoneReason


class PriorityEngine:
    """Score urgency explainably while reporting confidence separately."""

    def prioritize(self, zones: list[OperationalZone]) -> list[Zone]:
        scored = [self._score(item) for item in zones]
        scored.sort(key=lambda item: (-item.priority_score, item.zone_id))
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(scored, 1)]

    def _score(self, zone: OperationalZone) -> Zone:
        human = min(35.0, zone.people_count * 7.0)
        if zone.people_count and zone.max_person_confidence >= 0.85:
            human = max(20.0, human)
        access = {
            "ACCESSIBLE": 0.0,
            "UNKNOWN": 8.0,
            "DEGRADED": 12.0,
            "BLOCKED": 20.0,
            "ISOLATED": 25.0,
        }[zone.access_status.value]
        structure = min(20.0, zone.building_damage_coverage_percent)
        flood = min(15.0, zone.flood_coverage_percent * 0.3)
        other = min(5.0, zone.vehicle_count * 1.0)
        score = round(min(100.0, human + access + structure + flood + other), 2)
        severity = (
            Severity.CRITICAL
            if score >= 80
            else Severity.HIGH
            if score >= 60
            else Severity.MODERATE
            if score >= 40
            else Severity.LOW
        )
        origin = (
            DataOrigin.DEMO_SIMULATED
            if zone.sources and set(zone.sources) == {EvidenceSource.SIMULATED}
            else DataOrigin.DERIVED_ANALYTIC
        )
        definitions = [
            (
                "PERSON_EVIDENCE" if zone.people_count else "NO_PERSON_EVIDENCE",
                "Human exposure",
                f"{zone.people_count} person observations; strongest confidence "
                f"{zone.max_person_confidence:.0%}",
                human,
            ),
            (
                "PRIMARY_ACCESS_BLOCKED"
                if zone.access_status.value in {"BLOCKED", "ISOLATED"}
                else "ACCESS_DEGRADED"
                if zone.access_status.value == "DEGRADED"
                else "ACCESS_UNCERTAIN"
                if zone.access_status.value == "UNKNOWN"
                else "ACCESS_AVAILABLE",
                "Access and isolation",
                f"Relative image-space access is {zone.access_status.value.lower()}",
                access,
            ),
            (
                "SEVERE_BUILDING_DAMAGE"
                if zone.building_damage_coverage_percent >= 10
                else "STRUCTURAL_DAMAGE_EVIDENCE",
                "Structural damage",
                f"{zone.building_damage_coverage_percent:.1f}% semantic damage coverage; "
                "not an inferred building count",
                structure,
            ),
            (
                "HIGH_FLOOD_EXPOSURE" if zone.flood_coverage_percent >= 40 else "FLOOD_EXPOSURE",
                "Flood exposure",
                f"{zone.flood_coverage_percent:.1f}% water/flood-class coverage; pool excluded",
                flood,
            ),
        ]
        reasons = [
            ZoneReason(
                code=code,
                label=label,
                description=description,
                contribution=round(contribution, 2),
                data_origin=origin,
            )
            for code, label, description, contribution in definitions
            if contribution > 0 or code == "NO_PERSON_EVIDENCE"
        ]
        primary = max(reasons, key=lambda item: item.contribution).description
        return Zone(
            zone_id=zone.zone_id,
            display_name=zone.zone_id.replace("ZONE-", "Zone "),
            rank=1,
            severity=severity,
            priority_score=score,
            confidence=round(zone.confidence, 4),
            polygon=zone.polygon,
            people_count=zone.people_count,
            vehicle_count=zone.vehicle_count,
            flood_coverage_percent=zone.flood_coverage_percent,
            building_damage_count=0,
            road_condition=zone.road_state,
            access_status=zone.access_status,
            primary_reason=primary,
            reasons=reasons,
            updated_at_ms=zone.timestamp_ms,
            data_origin=origin,
            grid_cells=zone.grid_cells,
            building_damage_coverage_percent=zone.building_damage_coverage_percent,
            pool_coverage_percent=zone.pool_coverage_percent,
            temporal_samples=zone.temporal_samples,
            stale=zone.stale,
        )
