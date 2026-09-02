from collections import deque
from dataclasses import dataclass, field
from statistics import median

from app.intelligence.contracts import GridCellEvidence, OperationalZone, ZoneCandidate
from app.schemas.live_result import AccessStatus, RoadState


@dataclass(slots=True)
class _Track:
    zone_id: str
    history: deque[ZoneCandidate] = field(default_factory=deque)
    last_seen_ms: int = 0
    confirmed: bool = False


@dataclass(slots=True)
class _RoadTrack:
    history: deque[tuple[int, RoadState]] = field(default_factory=deque)
    stable_state: RoadState = RoadState.UNKNOWN
    last_observation_ms: int | None = None


class TemporalZoneTracker:
    def __init__(
        self,
        *,
        window_ms: int = 1_500,
        track_ttl_ms: int = 2_000,
        urgent_person_confidence: float = 0.85,
    ) -> None:
        self.window_ms = window_ms
        self.track_ttl_ms = track_ttl_ms
        self.urgent_person_confidence = urgent_person_confidence
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def update(self, candidates: list[ZoneCandidate], timestamp_ms: int) -> list[OperationalZone]:
        unmatched = set(self._tracks)
        for candidate in sorted(candidates, key=lambda item: item.candidate_id):
            track = self._best_track(candidate, unmatched)
            if track is None:
                zone_id = f"ZONE-{self._next_id:02d}"
                self._next_id += 1
                track = _Track(zone_id=zone_id)
                self._tracks[zone_id] = track
            unmatched.discard(track.zone_id)
            track.history.append(candidate)
            track.last_seen_ms = timestamp_ms
            self._trim(track, timestamp_ms)
            strong_person = max(candidate.person_confidences, default=0) >= (
                self.urgent_person_confidence
            )
            track.confirmed = (
                track.confirmed
                or strong_person
                or candidate.risk_signal >= 40
                or (len(track.history) >= 2)
            )

        expired = [
            zone_id
            for zone_id, track in self._tracks.items()
            if timestamp_ms - track.last_seen_ms > self.track_ttl_ms
        ]
        for zone_id in expired:
            self._tracks.pop(zone_id, None)

        output = [
            self._smooth(track, timestamp_ms)
            for track in self._tracks.values()
            if track.confirmed and track.history
        ]
        return sorted(output, key=lambda item: item.zone_id)

    def _best_track(self, candidate: ZoneCandidate, available: set[str]) -> _Track | None:
        candidate_cells = set(candidate.grid_cells)
        scored = []
        for zone_id in available:
            track = self._tracks[zone_id]
            if not track.history:
                continue
            previous = set(track.history[-1].grid_cells)
            overlap = len(candidate_cells & previous) / len(candidate_cells | previous)
            if overlap >= 0.25:
                scored.append((overlap, zone_id, track))
        return max(scored, default=(0.0, "", None), key=lambda item: (item[0], item[1]))[2]

    def _trim(self, track: _Track, timestamp_ms: int) -> None:
        while track.history and timestamp_ms - track.history[0].timestamp_ms > self.window_ms:
            track.history.popleft()

    def _smooth(self, track: _Track, timestamp_ms: int) -> OperationalZone:
        history = list(track.history)
        latest = history[-1]
        strong_current = max(latest.person_confidences, default=0) >= self.urgent_person_confidence
        people = (
            len(latest.person_confidences)
            if strong_current
            else int(median(len(item.person_confidences) for item in history) + 0.5)
        )
        max_confidence = max(
            (value for item in history for value in item.person_confidences), default=0
        )
        road = max((item.road_state for item in history), key=_road_priority)
        access = max((item.access_status for item in history), key=_access_priority)
        if people > 0 and road is RoadState.BLOCKED and latest.flood_coverage_percent >= 20:
            access = AccessStatus.ISOLATED
        stale = track.last_seen_ms != timestamp_ms
        return OperationalZone(
            zone_id=track.zone_id,
            grid_cells=latest.grid_cells,
            polygon=latest.polygon,
            timestamp_ms=timestamp_ms,
            people_count=people,
            max_person_confidence=max_confidence,
            person_evidence_samples=sum(
                bool(item.person_confidences) and item.person_observation_fresh for item in history
            ),
            person_local_flood_coverage_percent=round(
                float(
                    median(
                        max(
                            (
                                evidence.local_flood_coverage_percent
                                for evidence in item.person_evidence
                            ),
                            default=0.0,
                        )
                        for item in history
                        if item.person_confidences
                    )
                )
                if any(item.person_confidences for item in history)
                else 0.0,
                4,
            ),
            person_local_damage_coverage_percent=round(
                float(
                    median(
                        max(
                            (
                                evidence.local_damage_coverage_percent
                                for evidence in item.person_evidence
                            ),
                            default=0.0,
                        )
                        for item in history
                        if item.person_confidences
                    )
                )
                if any(item.person_confidences for item in history)
                else 0.0,
                4,
            ),
            vehicle_count=round(median(item.vehicle_count for item in history)),
            flood_coverage_percent=round(
                float(median(item.flood_coverage_percent for item in history)), 4
            ),
            pool_coverage_percent=round(
                float(median(item.pool_coverage_percent for item in history)), 4
            ),
            building_damage_coverage_percent=round(
                float(median(item.building_damage_coverage_percent for item in history)), 4
            ),
            road_state=road,
            access_status=access,
            confidence=max(0.0, latest.confidence * (0.8 if stale else 1.0)),
            risk_signal=latest.risk_signal,
            temporal_samples=len(history),
            stale=stale,
            # Smoothed values summarize the whole retained window, so provenance
            # must summarize that same window rather than only the latest candidate.
            sources=list(
                dict.fromkeys(source for item in history for source in item.sources)
            ),
        )


class TemporalRoadTracker:
    """Apply short persistence to semantic road states before route decisions."""

    def __init__(self, *, window_ms: int = 1_500) -> None:
        self.window_ms = window_ms
        self._tracks: dict[str, _RoadTrack] = {}

    def update(
        self,
        cells: list[GridCellEvidence],
        timestamp_ms: int,
        *,
        fresh_observation: bool = True,
    ) -> list[GridCellEvidence]:
        output: list[GridCellEvidence] = []
        for cell in cells:
            track = self._tracks.setdefault(cell.cell_id, _RoadTrack())
            if fresh_observation:
                track.history.append((timestamp_ms, cell.road_state))
                track.last_observation_ms = timestamp_ms
                while track.history and timestamp_ms - track.history[0][0] > self.window_ms:
                    track.history.popleft()
                track.stable_state = self._stabilize(track, cell)
            elif (
                track.last_observation_ms is None
                or timestamp_ms - track.last_observation_ms > self.window_ms
            ):
                track.stable_state = RoadState.UNKNOWN
            stable = track.stable_state
            output.append(
                cell.model_copy(
                    update={
                        "road_state": stable,
                        "access_status": _access_for_road(stable),
                    }
                )
            )
        return output

    @staticmethod
    def _stabilize(track: _RoadTrack, latest: GridCellEvidence) -> RoadState:
        current = latest.road_state
        strong_hazard = latest.confidence >= 0.95 and (
            (current is RoadState.BLOCKED and latest.road_blocked_coverage_percent >= 20)
            or (current is RoadState.FLOODED and latest.road_flooded_coverage_percent >= 30)
        )
        if strong_hazard:
            return current
        recent = [state for _, state in track.history]
        if len(recent) >= 2 and recent[-1] is recent[-2]:
            return current
        return track.stable_state


def _access_for_road(state: RoadState) -> AccessStatus:
    return {
        RoadState.CLEAR: AccessStatus.ACCESSIBLE,
        RoadState.FLOODED: AccessStatus.DEGRADED,
        RoadState.BLOCKED: AccessStatus.BLOCKED,
        RoadState.UNKNOWN: AccessStatus.UNKNOWN,
    }[state]


def _road_priority(state: RoadState) -> int:
    return {RoadState.CLEAR: 0, RoadState.UNKNOWN: 1, RoadState.FLOODED: 2, RoadState.BLOCKED: 3}[
        state
    ]


def _access_priority(status: AccessStatus) -> int:
    return {
        AccessStatus.ACCESSIBLE: 0,
        AccessStatus.UNKNOWN: 1,
        AccessStatus.DEGRADED: 2,
        AccessStatus.BLOCKED: 3,
        AccessStatus.ISOLATED: 4,
    }[status]
