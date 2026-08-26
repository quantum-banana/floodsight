import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.ingestion import (
    IngestionSession,
    IngestionSessionCreate,
    IngestionSessionState,
    SessionCounters,
    SessionLimits,
)


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    request: IngestionSessionCreate
    created_at_ms: int
    last_activity_at_ms: int
    expires_at_ms: int
    state: IngestionSessionState = IngestionSessionState.READY
    counters: SessionCounters = field(default_factory=SessionCounters)
    last_frame_id: int = -1


class IngestionSessionManager:
    """Bounded process-local session metadata; encoded and decoded frames are never retained."""

    def __init__(
        self,
        *,
        settings: Settings,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._sessions: dict[str, SessionRecord] = {}

    def _now_ms(self) -> int:
        return int(self._clock() * 1_000)

    def _expiry_ms(self, now_ms: int) -> int:
        return now_ms + self.settings.ingest_session_ttl_seconds * 1_000

    def _prune_expired(self) -> None:
        now_ms = self._now_ms()
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if record.expires_at_ms <= now_ms
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def create(self, request: IngestionSessionCreate) -> IngestionSession:
        self._prune_expired()
        if len(self._sessions) >= self.settings.ingest_max_sessions:
            raise AppError(
                status_code=503,
                code="ingestion_capacity_reached",
                message=(
                    "The ingestion session limit has been reached; "
                    "stop an idle session and retry."
                ),
            )
        now_ms = self._now_ms()
        session_id = secrets.token_urlsafe(24)
        record = SessionRecord(
            session_id=session_id,
            request=request,
            created_at_ms=now_ms,
            last_activity_at_ms=now_ms,
            expires_at_ms=self._expiry_ms(now_ms),
        )
        self._sessions[session_id] = record
        return self.to_contract(record)

    def get_record(self, session_id: str) -> SessionRecord:
        self._prune_expired()
        record = self._sessions.get(session_id)
        if record is None:
            raise AppError(
                status_code=404,
                code="ingestion_session_not_found",
                message=f"Ingestion session '{session_id}' was not found or has expired.",
            )
        return record

    def get(self, session_id: str) -> IngestionSession:
        return self.to_contract(self.get_record(session_id))

    def delete(self, session_id: str) -> None:
        self._prune_expired()
        self._sessions.pop(session_id, None)

    def touch(self, record: SessionRecord, state: IngestionSessionState | None = None) -> None:
        now_ms = self._now_ms()
        record.last_activity_at_ms = now_ms
        record.expires_at_ms = self._expiry_ms(now_ms)
        if state is not None:
            record.state = state

    def to_contract(self, record: SessionRecord) -> IngestionSession:
        return IngestionSession(
            session_id=record.session_id,
            source_mode=record.request.source_mode,
            media_origin=record.request.media_origin,
            state=record.state,
            created_at_ms=record.created_at_ms,
            last_activity_at_ms=record.last_activity_at_ms,
            expires_at_ms=record.expires_at_ms,
            counters=record.counters,
            limits=SessionLimits(
                recommended_capture_fps=self.settings.ingest_capture_fps,
                jpeg_quality=self.settings.ingest_jpeg_quality,
                max_frame_bytes=self.settings.ingest_max_frame_bytes,
                accepted_mime_types=["image/jpeg"],
            ),
        )
