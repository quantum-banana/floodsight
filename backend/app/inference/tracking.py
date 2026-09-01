import math
from dataclasses import dataclass

from app.inference.contracts import (
    DetectionObservationState,
    DetectionResult,
    NormalizedDetection,
)
from app.inference.yolo import VEHICLE_APPLICATION_CLASSES
from app.schemas.live_result import BoundingBox

TRACKED_APPLICATION_CLASSES = frozenset({"person", *VEHICLE_APPLICATION_CLASSES})


@dataclass(slots=True)
class _ObjectTrack:
    track_id: str
    application_family: str
    latest: NormalizedDetection
    first_seen_ms: int
    last_seen_ms: int
    hits: int
    track_confidence: float


class TemporalDetectionTracker:
    """Stabilize model detections without inventing an unobserved initial object."""

    def __init__(self, *, track_ttl_ms: int = 2_000) -> None:
        if track_ttl_ms <= 0:
            raise ValueError("track TTL must be positive")
        self.track_ttl_ms = track_ttl_ms
        self._tracks: dict[str, _ObjectTrack] = {}
        self._next_track_id = 1
        self._last_template: DetectionResult | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._last_template = None
        self._next_track_id = 1

    def update(
        self,
        result: DetectionResult | None,
        *,
        frame_id: int,
        timestamp_ms: int,
        fresh_observation: bool = True,
    ) -> DetectionResult | None:
        if result is not None:
            self._last_template = result
        template = result or self._last_template
        if template is None:
            return None

        observations = (
            [
                item
                for item in result.detections
                if item.application_class in TRACKED_APPLICATION_CLASSES
            ]
            if result is not None and fresh_observation
            else []
        )
        assignments = self._assign(observations)
        matched_tracks: set[str] = set()
        output: list[NormalizedDetection] = []

        for detection_index, track_id in assignments.items():
            detection = observations[detection_index]
            track = self._tracks[track_id]
            track.hits += 1
            track.last_seen_ms = timestamp_ms
            track.track_confidence = min(
                0.99,
                0.60 * track.track_confidence
                + 0.40 * detection.confidence
                + 0.04 * min(track.hits - 1, 3),
            )
            track.latest = detection
            matched_tracks.add(track_id)
            output.append(
                self._observed_detection(
                    detection,
                    track=track,
                    frame_id=frame_id,
                )
            )

        assigned_detection_indexes = set(assignments)
        for index, detection in enumerate(observations):
            if index in assigned_detection_indexes:
                continue
            track = self._start_track(detection, timestamp_ms)
            matched_tracks.add(track.track_id)
            output.append(
                self._observed_detection(
                    detection,
                    track=track,
                    frame_id=frame_id,
                )
            )

        expired: list[str] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_tracks:
                continue
            elapsed_ms = timestamp_ms - track.last_seen_ms
            if elapsed_ms >= self.track_ttl_ms:
                expired.append(track_id)
                continue
            established = track.hits >= 2 or track.track_confidence >= 0.70
            if not established:
                continue
            decay = max(0.0, 1.0 - elapsed_ms / self.track_ttl_ms)
            persisted_confidence = track.track_confidence * decay
            if persisted_confidence <= 0:
                continue
            latest = track.latest
            output.append(
                latest.model_copy(
                    update={
                        "detection_id": f"{frame_id}-{track_id}-persisted",
                        "track_id": track_id,
                        "track_confidence": round(persisted_confidence, 6),
                        "persistence": track.hits,
                        "observation_state": DetectionObservationState.TRACK_PERSISTED,
                        "source_frame_id": latest.source_frame_id,
                    }
                )
            )
        for track_id in expired:
            self._tracks.pop(track_id, None)

        output.sort(key=lambda item: (item.track_id or "", item.detection_id))
        return template.model_copy(
            update={
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "detections": output,
            }
        )

    def _assign(self, observations: list[NormalizedDetection]) -> dict[int, str]:
        candidates: list[tuple[float, int, str]] = []
        for index, detection in enumerate(observations):
            family = _application_family(detection.application_class)
            for track_id, track in self._tracks.items():
                if track.application_family != family:
                    continue
                iou = _bbox_iou(detection.bbox, track.latest.bbox)
                distance = _center_distance(detection.bbox, track.latest.bbox)
                distance_limit = max(
                    0.04,
                    0.75 * min(_bbox_diagonal(detection.bbox), _bbox_diagonal(track.latest.bbox)),
                )
                if iou < 0.15 and distance > distance_limit:
                    continue
                distance_score = max(0.0, 1.0 - distance / distance_limit)
                candidates.append((2.0 * iou + distance_score, index, track_id))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        assigned_detections: set[int] = set()
        assigned_tracks: set[str] = set()
        assignments: dict[int, str] = {}
        for _, detection_index, track_id in candidates:
            if detection_index in assigned_detections or track_id in assigned_tracks:
                continue
            assignments[detection_index] = track_id
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)
        return assignments

    def _start_track(
        self,
        detection: NormalizedDetection,
        timestamp_ms: int,
    ) -> _ObjectTrack:
        track_id = f"track-{self._next_track_id:06d}"
        self._next_track_id += 1
        track = _ObjectTrack(
            track_id=track_id,
            application_family=_application_family(detection.application_class),
            latest=detection,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
            hits=1,
            track_confidence=detection.confidence,
        )
        self._tracks[track_id] = track
        return track

    @staticmethod
    def _observed_detection(
        detection: NormalizedDetection,
        *,
        track: _ObjectTrack,
        frame_id: int,
    ) -> NormalizedDetection:
        return detection.model_copy(
            update={
                "track_id": track.track_id,
                "track_confidence": round(track.track_confidence, 6),
                "persistence": track.hits,
                "observation_state": DetectionObservationState.DETECTED,
                "source_frame_id": frame_id,
            }
        )


def _application_family(application_class: str) -> str:
    return "vehicle" if application_class in VEHICLE_APPLICATION_CLASSES else application_class


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    left_x2, left_y2 = left.x + left.width, left.y + left.height
    right_x2, right_y2 = right.x + right.width, right.y + right.height
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(left: BoundingBox, right: BoundingBox) -> float:
    left_center = (left.x + left.width / 2, left.y + left.height / 2)
    right_center = (right.x + right.width / 2, right.y + right.height / 2)
    return math.dist(left_center, right_center)


def _bbox_diagonal(box: BoundingBox) -> float:
    return math.hypot(box.width, box.height)
