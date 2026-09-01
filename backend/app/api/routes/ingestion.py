import asyncio
import json
import time

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.core.errors import AppError
from app.core.logging import get_logger
from app.ingestion.protocol import process_frame_with_packet
from app.schemas.incident import IncidentReport
from app.schemas.ingestion import (
    FrameIntelligence,
    FrameMetadata,
    FrameResult,
    IngestionSession,
    IngestionSessionCreate,
    IngestionSessionState,
)
from app.schemas.live_result import DataOrigin
from app.services.incident_reporting import build_incident_report
from app.services.inference_coordinator import InferenceCoordinator
from app.services.ingestion_sessions import IngestionSessionManager, SessionRecord

router = APIRouter(prefix="/ingest/sessions", tags=["ingestion"])
ws_router = APIRouter(tags=["ingestion-stream"])
logger = get_logger(__name__)


def _manager(connection: Request | WebSocket) -> IngestionSessionManager:
    return connection.app.state.ingestion_manager


def _coordinator(connection: Request | WebSocket) -> InferenceCoordinator:
    return connection.app.state.inference_coordinator


@router.post(
    "",
    response_model=IngestionSession,
    status_code=status.HTTP_201_CREATED,
    summary="Create a bounded media-frame ingestion session",
)
async def create_ingestion_session(
    request: Request,
    payload: IngestionSessionCreate,
) -> IngestionSession:
    return _manager(request).create(payload)


@router.get(
    "/{session_id}",
    response_model=IngestionSession,
    summary="Get ingestion session state and counters",
)
async def get_ingestion_session(request: Request, session_id: str) -> IngestionSession:
    return _manager(request).get(session_id)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop and forget an ingestion session",
)
async def delete_ingestion_session(request: Request, session_id: str) -> Response:
    _coordinator(request).close(session_id)
    _manager(request).delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _protocol_result(
    record: SessionRecord,
    *,
    code: str,
    message: str,
    frame_id: int | None = None,
) -> FrameResult:
    record.counters.protocol_errors += 1
    return FrameResult(
        session_id=record.session_id,
        frame_id=frame_id,
        accepted=False,
        code=code,
        message=message,
        received_at_ms=int(time.time() * 1_000),
        processing_ms=0,
        byte_length=0,
        decoded_frame=None,
        quality=None,
        data_origin=DataOrigin.DERIVED_ANALYTIC,
    )


async def _send_result(websocket: WebSocket, result: FrameResult) -> None:
    await websocket.send_json(result.model_dump(mode="json"))


@router.get(
    "/{session_id}/intelligence",
    response_model=FrameIntelligence,
    summary="Get the latest backend-computed frame intelligence",
)
async def get_latest_intelligence(request: Request, session_id: str) -> FrameIntelligence:
    _manager(request).get(session_id)
    latest = _coordinator(request).latest(session_id)
    if latest is None:
        raise AppError(
            status_code=404,
            code="intelligence_unavailable",
            message="No frame intelligence is available for this session yet.",
        )
    return latest


@router.get(
    "/{session_id}/report",
    response_model=IncidentReport,
    summary="Generate a report from the latest backend intelligence state",
)
async def get_live_incident_report(request: Request, session_id: str) -> IncidentReport:
    latest = await get_latest_intelligence(request, session_id)
    return build_incident_report(latest.result)


@ws_router.websocket("/ws/ingest/sessions/{session_id}/frames")
async def ingest_frames(websocket: WebSocket, session_id: str) -> None:
    manager = _manager(websocket)
    try:
        record = manager.get_record(session_id)
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
        await websocket.close(code=4_404, reason="Ingestion session not found")
        return

    await websocket.accept()
    manager.touch(record, IngestionSessionState.ACTIVE)
    send_lock = asyncio.Lock()

    async def send_payload(payload: FrameResult | FrameIntelligence) -> None:
        async with send_lock:
            await websocket.send_json(payload.model_dump(mode="json"))

    try:
        while True:
            metadata_message = await websocket.receive()
            if metadata_message["type"] == "websocket.disconnect":
                break
            metadata_text = metadata_message.get("text")
            if metadata_text is None:
                await send_payload(
                    _protocol_result(
                        record,
                        code="metadata_required",
                        message="Send one frame_metadata JSON message before each binary frame.",
                    ),
                )
                continue

            try:
                metadata = FrameMetadata.model_validate(json.loads(metadata_text))
            except (json.JSONDecodeError, ValidationError):
                await send_payload(
                    _protocol_result(
                        record,
                        code="invalid_metadata",
                        message="Frame metadata did not match the ingestion contract.",
                    ),
                )
                continue

            frame_message = await websocket.receive()
            if frame_message["type"] == "websocket.disconnect":
                break
            payload = frame_message.get("bytes")
            if payload is None:
                await send_payload(
                    _protocol_result(
                        record,
                        code="binary_frame_required",
                        message="A binary JPEG frame must immediately follow frame metadata.",
                        frame_id=metadata.frame_id,
                    ),
                )
                continue

            result, packet = process_frame_with_packet(record, metadata, payload, manager.settings)
            model_status = websocket.app.state.inference_pipeline.status()
            result = result.model_copy(
                update={
                    "inference_state": model_status.inference_state,
                    "segmentation_status": model_status.segmentation,
                    "detection_status": model_status.detection,
                }
            )
            manager.touch(record, IngestionSessionState.ACTIVE)
            await send_payload(result)
            if result.accepted and packet is not None:
                pipeline = websocket.app.state.inference_pipeline
                if pipeline.should_process(metadata.frame_id):
                    record.counters.inference_frames_submitted += 1

                    async def publish(message: FrameIntelligence) -> None:
                        record.counters.intelligence_updates_sent += 1
                        await send_payload(message)

                    if not _coordinator(websocket).submit(
                        session_id=session_id,
                        frame=packet.bgr,
                        metadata=metadata,
                        callback=publish,
                    ):
                        record.counters.inference_frames_dropped += 1
    except WebSocketDisconnect:
        logger.debug(
            "Ingestion WebSocket client disconnected",
            extra={"session_id": session_id},
        )
    finally:
        _coordinator(websocket).disconnect(session_id)
        try:
            active_record = manager.get_record(session_id)
            if active_record is record:
                manager.touch(record, IngestionSessionState.IDLE)
        except AppError:
            pass
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close(code=1_000, reason="Ingestion stream stopped")
