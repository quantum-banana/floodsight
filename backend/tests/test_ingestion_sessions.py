import pytest
from httpx import AsyncClient

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.ingestion import (
    IngestionSession,
    IngestionSessionCreate,
    MediaOrigin,
)
from app.schemas.live_result import SourceMode
from app.services.ingestion_sessions import IngestionSessionManager

pytestmark = pytest.mark.anyio


async def test_create_video_file_session(client: AsyncClient) -> None:
    response = await client.post(
        "/api/ingest/sessions",
        json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
    )

    assert response.status_code == 201
    session = IngestionSession.model_validate(response.json())
    assert session.source_mode is SourceMode.VIDEO_FILE
    assert session.media_origin is MediaOrigin.USER_VIDEO_FILE
    assert session.state.value == "READY"
    assert session.data_origin.value == "DERIVED_ANALYTIC"
    assert len(session.session_id) >= 20


async def test_create_webcam_session(client: AsyncClient) -> None:
    response = await client.post(
        "/api/ingest/sessions",
        json={"source_mode": "WEBCAM", "media_origin": "USER_WEBCAM"},
    )

    assert response.status_code == 201
    session = IngestionSession.model_validate(response.json())
    assert session.source_mode is SourceMode.WEBCAM
    assert session.media_origin is MediaOrigin.USER_WEBCAM


@pytest.mark.parametrize(
    "payload",
    [
        {"source_mode": "DRONE_STREAM", "media_origin": "USER_VIDEO_FILE"},
        {"source_mode": "VIDEO_FILE", "media_origin": "USER_WEBCAM"},
        {"source_mode": "INVALID", "media_origin": "USER_VIDEO_FILE"},
    ],
)
async def test_invalid_source_or_provenance_is_structured(
    client: AsyncClient,
    payload: dict[str, str],
) -> None:
    response = await client.post("/api/ingest/sessions", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_session_detail_and_idempotent_deletion(client: AsyncClient) -> None:
    created = await client.post(
        "/api/ingest/sessions",
        json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
    )
    session_id = created.json()["session_id"]

    detail = await client.get(f"/api/ingest/sessions/{session_id}")
    first_delete = await client.delete(f"/api/ingest/sessions/{session_id}")
    second_delete = await client.delete(f"/api/ingest/sessions/{session_id}")

    assert detail.status_code == 200
    assert detail.json()["session_id"] == session_id
    assert first_delete.status_code == 204
    assert second_delete.status_code == 204


async def test_unknown_session_uses_structured_error(client: AsyncClient) -> None:
    response = await client.get("/api/ingest/sessions/not-a-real-session")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "ingestion_session_not_found",
            "message": (
                "Ingestion session 'not-a-real-session' was not found or has expired."
            ),
            "details": [],
        }
    }


def test_session_manager_is_bounded_and_expiring() -> None:
    now = 100.0
    settings = Settings(ingest_max_sessions=1, ingest_session_ttl_seconds=30)
    manager = IngestionSessionManager(settings=settings, clock=lambda: now)
    request = IngestionSessionCreate(
        source_mode=SourceMode.WEBCAM,
        media_origin=MediaOrigin.USER_WEBCAM,
    )
    manager.create(request)

    with pytest.raises(AppError, match="session limit"):
        manager.create(request)

    now = 131.0
    replacement = manager.create(request)
    assert replacement.source_mode is SourceMode.WEBCAM
