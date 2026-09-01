from app.inference.contracts import EvidenceSource
from app.intelligence.contracts import OperationalZone
from app.schemas.live_result import (
    DataOrigin,
    EvidenceLevel,
    Severity,
    Zone,
    ZoneAlert,
    ZoneReason,
)


class PriorityEngine:
    """Score urgency explainably while reporting confidence separately."""

    def prioritize(self, zones: list[OperationalZone]) -> list[Zone]:
        scored = [self._score(item) for item in zones]
        scored.sort(key=lambda item: (-item.priority_score, item.zone_id))
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(scored, 1)]

    def _score(self, zone: OperationalZone) -> Zone:
        origin = (
            DataOrigin.DEMO_SIMULATED
            if zone.sources and set(zone.sources) == {EvidenceSource.SIMULATED}
            else DataOrigin.DERIVED_ANALYTIC
        )
        alert = _potential_stranded_person(zone, origin)
        human = min(35.0, zone.people_count * 7.0)
        if zone.people_count and zone.max_person_confidence >= 0.85:
            human = max(20.0, human)
        if alert is not None:
            human = 35.0
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
        rescue_coupling = 10.0 if alert is not None else 0.0
        score = round(min(100.0, human + access + structure + flood + other + rescue_coupling), 2)
        severity = (
            Severity.CRITICAL
            if score >= 80
            else Severity.HIGH
            if score >= 60
            else Severity.MODERATE
            if score >= 40
            else Severity.LOW
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
                "NEAR_DAMAGED_STRUCTURE"
                if alert is not None and zone.building_damage_coverage_percent >= 10
                else "SEVERE_BUILDING_DAMAGE"
                if zone.building_damage_coverage_percent >= 10
                else "STRUCTURAL_DAMAGE_EVIDENCE",
                "Structural damage",
                f"{zone.building_damage_coverage_percent:.1f}% semantic damage coverage; "
                "not an inferred building count",
                structure,
            ),
            (
                "PERSON_IN_HIGH_FLOOD_ZONE"
                if alert is not None
                and max(
                    zone.flood_coverage_percent,
                    zone.person_local_flood_coverage_percent,
                )
                >= 40
                else "HIGH_FLOOD_EXPOSURE"
                if zone.flood_coverage_percent >= 40
                else "FLOOD_EXPOSURE",
                "Flood exposure",
                f"{zone.flood_coverage_percent:.1f}% water/flood-class coverage; pool excluded",
                flood,
            ),
            (
                "POTENTIAL_STRANDED_PERSON",
                "Combined person and access risk",
                "Potential stranded-person evidence requires trained-personnel review",
                rescue_coupling,
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
            alerts=[alert] if alert is not None else [],
        )


def _potential_stranded_person(zone: OperationalZone, origin: DataOrigin) -> ZoneAlert | None:
    if zone.people_count == 0:
        return None
    flood_exposure = max(
        zone.flood_coverage_percent,
        zone.person_local_flood_coverage_percent,
    )
    damage_exposure = max(
        zone.building_damage_coverage_percent,
        zone.person_local_damage_coverage_percent,
    )
    access_risk = zone.access_status.value in {"BLOCKED", "ISOLATED"}
    risk_context = flood_exposure >= 20 or access_risk or damage_exposure >= 10
    person_samples = zone.person_evidence_samples or zone.temporal_samples
    persistent = person_samples >= 2
    rapid_high_confidence = zone.max_person_confidence >= 0.85 and (
        flood_exposure >= 40 or access_risk or damage_exposure >= 20
    )
    if not risk_context or not (persistent or rapid_high_confidence):
        return None

    reason_codes = ["POTENTIAL_STRANDED_PERSON", "PERSON_EVIDENCE"]
    if flood_exposure >= 40:
        reason_codes.append("PERSON_IN_HIGH_FLOOD_ZONE")
    if access_risk:
        reason_codes.append("PRIMARY_ACCESS_BLOCKED")
    if damage_exposure >= 10:
        reason_codes.append("NEAR_DAMAGED_STRUCTURE")
    context_confidence = max(
        min(1.0, flood_exposure / 100),
        1.0 if access_risk else 0.0,
        min(1.0, damage_exposure / 100),
    )
    return ZoneAlert(
        person_evidence=_person_evidence_level(zone.max_person_confidence),
        flood_exposure=_flood_evidence_level(flood_exposure),
        primary_access=zone.access_status,
        confidence=round(
            min(1.0, zone.max_person_confidence * 0.65 + context_confidence * 0.35), 4
        ),
        temporal_samples=max(1, person_samples),
        reason_codes=reason_codes,
        data_origin=origin,
    )


def _person_evidence_level(confidence: float) -> EvidenceLevel:
    if confidence >= 0.85:
        return EvidenceLevel.HIGH
    if confidence >= 0.60:
        return EvidenceLevel.MODERATE
    return EvidenceLevel.LOW


def _flood_evidence_level(coverage_percent: float) -> EvidenceLevel:
    if coverage_percent >= 40:
        return EvidenceLevel.HIGH
    if coverage_percent >= 10:
        return EvidenceLevel.MODERATE
    return EvidenceLevel.LOW
