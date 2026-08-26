from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.schemas.incident import IncidentDetailResponse, IncidentListResponse, IncidentReport
from app.schemas.live_result import LiveResult
from app.services.demo_incident import (
    get_demo_incident,
    get_demo_report,
    list_demo_incidents,
)
from app.services.demo_result import get_demo_live_result
from app.services.demo_stream import stream_demo_snapshots

router = APIRouter(prefix="/demo", tags=["demo"])
ws_router = APIRouter(tags=["demo-stream"])
logger = get_logger(__name__)


@router.get(
    "/live-result",
    response_model=LiveResult,
    summary="Return the explicitly simulated Phase 0 incident example",
)
async def demo_live_result() -> LiveResult:
    return get_demo_live_result()


@router.get(
    "/incidents",
    response_model=IncidentListResponse,
    summary="List deterministic demo incidents",
)
async def demo_incidents() -> IncidentListResponse:
    return list_demo_incidents()


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentDetailResponse,
    summary="Get deterministic demo incident details",
)
async def demo_incident(incident_id: str) -> IncidentDetailResponse:
    return get_demo_incident(incident_id)


@router.get(
    "/incidents/{incident_id}/report",
    response_model=IncidentReport,
    summary="Generate a report from the final deterministic snapshot",
)
async def demo_incident_report(incident_id: str) -> IncidentReport:
    return get_demo_report(incident_id)


@ws_router.websocket("/ws/demo/incidents/{incident_id}/live")
async def demo_incident_stream(
    websocket: WebSocket,
    incident_id: str,
    start_index: Annotated[int, Query(ge=0)] = 0,
    interval_ms: Annotated[int | None, Query(ge=0, le=60_000)] = None,
    loop: bool = False,
) -> None:
    try:
        get_demo_incident(incident_id)
    except AppError as exc:
        await websocket.accept()
        await websocket.send_json(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": [],
                }
            }
        )
        await websocket.close(code=4_404, reason="Demo incident not found")
        return

    settings = get_settings()
    delay_ms = interval_ms if interval_ms is not None else settings.demo_stream_interval_ms
    await websocket.accept()

    try:
        async for snapshot in stream_demo_snapshots(
            incident_id,
            start_index=start_index,
            interval_ms=delay_ms,
            loop=loop,
        ):
            await websocket.send_json(snapshot.model_dump(mode="json"))
    except WebSocketDisconnect:
        logger.debug("Demo WebSocket client disconnected", extra={"incident_id": incident_id})
    finally:
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close(code=1_000, reason="Deterministic demo stream complete")
