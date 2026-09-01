import time

from app.core.config import Settings
from app.ingestion.decoder import FrameDecodeError, decode_jpeg
from app.ingestion.frame_packet import FramePacket
from app.ingestion.quality import assess_frame_quality
from app.schemas.ingestion import DecodedFrame, FrameMetadata, FrameQuality, FrameResult
from app.services.ingestion_sessions import SessionRecord


def _result(
    record: SessionRecord,
    metadata: FrameMetadata,
    *,
    accepted: bool,
    code: str,
    message: str,
    started_at: float,
    byte_length: int,
    decoded_frame: DecodedFrame | None = None,
    quality: FrameQuality | None = None,
) -> FrameResult:
    return FrameResult(
        session_id=record.session_id,
        frame_id=metadata.frame_id,
        accepted=accepted,
        code=code,
        message=message,
        received_at_ms=int(time.time() * 1_000),
        processing_ms=round((time.perf_counter() - started_at) * 1_000, 3),
        byte_length=byte_length,
        decoded_frame=decoded_frame,
        quality=quality,
    )


def process_frame_with_packet(
    record: SessionRecord,
    metadata: FrameMetadata,
    payload: bytes,
    settings: Settings,
) -> tuple[FrameResult, FramePacket | None]:
    """Validate, decode, and assess one transient JPEG without retaining frame data."""
    started_at = time.perf_counter()
    record.counters.frames_received += 1
    record.counters.bytes_received += len(payload)

    def reject(code: str, message: str) -> FrameResult:
        record.counters.frames_rejected += 1
        return _result(
            record,
            metadata,
            accepted=False,
            code=code,
            message=message,
            started_at=started_at,
            byte_length=len(payload),
        )

    if metadata.frame_id <= record.last_frame_id:
        record.counters.frames_out_of_order += 1
        return reject(
            "frame_out_of_order", "Frame IDs must increase monotonically per session."
        ), None
    record.last_frame_id = metadata.frame_id

    if (
        metadata.source_mode != record.request.source_mode
        or metadata.media_origin != record.request.media_origin
    ):
        return reject(
            "source_mismatch", "Frame provenance does not match the session source."
        ), None
    if metadata.mime_type.lower() != "image/jpeg":
        return reject(
            "unsupported_media_type", "Only image/jpeg frame payloads are accepted."
        ), None
    if metadata.byte_length > settings.ingest_max_frame_bytes:
        return reject("frame_too_large", "Frame metadata exceeds the configured byte limit."), None
    if len(payload) > settings.ingest_max_frame_bytes:
        return reject("frame_too_large", "Frame payload exceeds the configured byte limit."), None
    if metadata.byte_length != len(payload):
        return reject(
            "byte_length_mismatch", "Frame metadata byte length does not match payload."
        ), None

    try:
        packet = decode_jpeg(record.session_id, metadata, payload)
    except FrameDecodeError as exc:
        return reject("decode_failed", str(exc)), None

    height, width, channels = packet.bgr.shape
    decoded_frame = DecodedFrame(width=width, height=height, channels=channels)
    if metadata.width != width or metadata.height != height:
        return reject(
            "dimension_mismatch", "Client dimensions do not match the decoded frame."
        ), None

    quality = assess_frame_quality(
        packet,
        dark_threshold=settings.ingest_dark_luminance_threshold,
        bright_threshold=settings.ingest_bright_luminance_threshold,
        blur_threshold=settings.ingest_blur_variance_threshold,
    )
    record.counters.frames_accepted += 1
    message = (
        "Frame decoded with quality warnings."
        if quality.warnings
        else "Frame decoded and accepted."
    )
    return _result(
        record,
        metadata,
        accepted=True,
        code="accepted_with_warnings" if quality.warnings else "accepted",
        message=message,
        started_at=started_at,
        byte_length=len(payload),
        decoded_frame=decoded_frame,
        quality=quality,
    ), packet


def process_frame(
    record: SessionRecord,
    metadata: FrameMetadata,
    payload: bytes,
    settings: Settings,
) -> FrameResult:
    result, _ = process_frame_with_packet(record, metadata, payload, settings)
    return result
