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
    VideoAnalysisComplete,
)
from app.schemas.live_result import DataOrigin
from app.services.incident_reporting import build_incident_report, build_video_analysis_report
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
    manager = _manager(request)
    try:
        record = manager.get_record(session_id)
    except AppError as exc:
        if exc.code == "ingestion_session_not_found":
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        raise
    if record.state is IngestionSessionState.FINALIZING:
        raise AppError(
            status_code=409,
            code="analysis_finalizing",
            message="The video analysis is finalizing; wait for completion before deleting it.",
        )
    _coordinator(request).close(session_id)
    manager.delete(session_id)
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


def _analysis_closed_result(
    record: SessionRecord,
    *,
    frame_id: int | None = None,
) -> FrameResult:
    """Reject post-finalization intake without recording a malformed-frame error."""
    return FrameResult(
        session_id=record.session_id,
        frame_id=frame_id,
        accepted=False,
        code="analysis_closed",
        message="Video analysis is finalizing or complete; no more frames are accepted.",
        received_at_ms=int(time.time() * 1_000),
        processing_ms=0,
        byte_length=0,
        decoded_frame=None,
        quality=None,
        data_origin=DataOrigin.DERIVED_ANALYTIC,
    )


def _analysis_is_closed(record: SessionRecord) -> bool:
    return record.state in {
        IngestionSessionState.FINALIZING,
        IngestionSessionState.COMPLETE,
    }


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
    summary="Generate a report from live or completed backend intelligence",
)
async def get_live_incident_report(request: Request, session_id: str) -> IncidentReport:
    record = _manager(request).get_record(session_id)
    if record.state is IngestionSessionState.FINALIZING:
        raise AppError(
            status_code=409,
            code="analysis_finalizing",
            message="The whole-video analysis is still finalizing; retry when it is complete.",
        )
    completed = _coordinator(request).completed(session_id)
    if completed is not None:
        return build_video_analysis_report(completed)
    latest = await get_latest_intelligence(request, session_id)
    return build_incident_report(latest.result)


@router.post(
    "/{session_id}/complete",
    response_model=VideoAnalysisComplete,
    summary="Finalize a video session and return whole-video findings",
)
async def complete_video_analysis(
    request: Request,
    session_id: str,
) -> VideoAnalysisComplete:
    manager = _manager(request)
    record = manager.get_record(session_id)
    if record.request.source_mode.value != "VIDEO_FILE":
        raise AppError(
            status_code=409,
            code="video_completion_not_supported",
            message="Explicit whole-video completion is only available for video-file sessions.",
        )
    manager.touch(record, IngestionSessionState.FINALIZING)
    try:
        completed = await _coordinator(request).finalize(
            session_id,
            frames_accepted=record.counters.frames_accepted,
            frames_dropped=record.counters.inference_frames_dropped,
        )
    except asyncio.CancelledError:
        manager.touch(record, IngestionSessionState.IDLE)
        raise
    except Exception:
        manager.touch(record, IngestionSessionState.IDLE)
        raise
    manager.touch(record, IngestionSessionState.COMPLETE)
    return completed


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
    if _analysis_is_closed(record):
        await _send_result(websocket, _analysis_closed_result(record))
        await websocket.close(code=1_008, reason="Video analysis is closed")
        return
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
            if _analysis_is_closed(record):
                await send_payload(_analysis_closed_result(record))
                break
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

            analysis_closed = False
            async with send_lock:
                if _analysis_is_closed(record):
                    analysis_closed = True
                    result = _analysis_closed_result(record, frame_id=metadata.frame_id)
                else:
                    result, packet = process_frame_with_packet(
                        record, metadata, payload, manager.settings
                    )
                    model_status = websocket.app.state.inference_pipeline.status()
                    result = result.model_copy(
                        update={
                            "inference_state": model_status.inference_state,
                            "segmentation_status": model_status.segmentation,
                            "detection_status": model_status.detection,
                        }
                    )
                    manager.touch(record, IngestionSessionState.ACTIVE)
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
                                detector_mode=record.request.detector_mode,
                            ):
                                record.counters.inference_frames_dropped += 1
                await websocket.send_json(result.model_dump(mode="json"))
            if analysis_closed:
                break
    except WebSocketDisconnect:
        logger.debug(
            "Ingestion WebSocket client disconnected",
            extra={"session_id": session_id},
        )
    finally:
        _coordinator(websocket).disconnect(session_id)
        try:
            active_record = manager.get_record(session_id)
            if active_record is record and record.state not in (
                IngestionSessionState.FINALIZING,
                IngestionSessionState.COMPLETE,
            ):
                manager.touch(record, IngestionSessionState.IDLE)
        except AppError:
            pass
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close(code=1_000, reason="Ingestion stream stopped")
