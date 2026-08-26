from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.ingestion import FrameResult, IngestionSession


def _jpeg(width: int = 96, height: int = 54, value: int = 110) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[:, ::4] = 255 - value
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert success
    return encoded.tobytes()


def _metadata(
    payload: bytes,
    *,
    frame_id: int = 0,
    width: int = 96,
    height: int = 54,
    mime_type: str = "image/jpeg",
    byte_length: int | None = None,
) -> dict[str, object]:
    return {
        "type": "frame_metadata",
        "frame_id": frame_id,
        "captured_at_ms": 1_725_000_000_000 + frame_id,
        "media_time_ms": frame_id * 250,
        "source_mode": "VIDEO_FILE",
        "media_origin": "USER_VIDEO_FILE",
        "mime_type": mime_type,
        "byte_length": len(payload) if byte_length is None else byte_length,
        "width": width,
        "height": height,
    }


@pytest.fixture
def sync_client() -> Iterator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def _create_session(client: TestClient) -> str:
    response = client.post(
        "/api/ingest/sessions",
        json={"source_mode": "VIDEO_FILE", "media_origin": "USER_VIDEO_FILE"},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def _send_pair(websocket: object, metadata: dict[str, object], payload: bytes) -> FrameResult:
    websocket.send_json(metadata)
    websocket.send_bytes(payload)
    return FrameResult.model_validate(websocket.receive_json())


def test_valid_jpeg_decodes_dimensions_quality_and_provenance(sync_client: TestClient) -> None:
    session_id = _create_session(sync_client)
    payload = _jpeg()

    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        result = _send_pair(websocket, _metadata(payload), payload)

    assert result.accepted is True
    assert result.code in {"accepted", "accepted_with_warnings"}
    assert result.decoded_frame is not None
    assert (result.decoded_frame.width, result.decoded_frame.height) == (96, 54)
    assert result.quality is not None
    assert result.quality.mean_luminance >= 0
    assert result.quality.laplacian_variance >= 0
    assert result.quality.data_origin.value == "DERIVED_ANALYTIC"
    assert result.data_origin.value == "DERIVED_ANALYTIC"


def test_duplicate_and_out_of_order_frame_ids_are_rejected(sync_client: TestClient) -> None:
    session_id = _create_session(sync_client)
    payload = _jpeg()

    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        accepted = _send_pair(websocket, _metadata(payload, frame_id=5), payload)
        duplicate = _send_pair(websocket, _metadata(payload, frame_id=5), payload)
        out_of_order = _send_pair(websocket, _metadata(payload, frame_id=4), payload)

    assert accepted.accepted is True
    assert duplicate.code == "frame_out_of_order"
    assert out_of_order.code == "frame_out_of_order"


def test_binary_frame_without_metadata_is_a_protocol_error(sync_client: TestClient) -> None:
    session_id = _create_session(sync_client)
    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        websocket.send_bytes(_jpeg())
        result = FrameResult.model_validate(websocket.receive_json())

    assert result.accepted is False
    assert result.frame_id is None
    assert result.code == "metadata_required"


def test_metadata_without_binary_frame_is_a_protocol_error(sync_client: TestClient) -> None:
    session_id = _create_session(sync_client)
    payload = _jpeg()
    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        websocket.send_json(_metadata(payload))
        websocket.send_text("not binary")
        result = FrameResult.model_validate(websocket.receive_json())

    assert result.accepted is False
    assert result.frame_id == 0
    assert result.code == "binary_frame_required"


@pytest.mark.parametrize(
    ("metadata_changes", "payload_factory", "expected_code"),
    [
        ({"byte_length": 1}, _jpeg, "byte_length_mismatch"),
        ({"mime_type": "image/png"}, _jpeg, "unsupported_media_type"),
        ({"byte_length": 2_000_001}, _jpeg, "frame_too_large"),
        ({}, lambda: b"not-a-jpeg", "decode_failed"),
        ({"width": 95}, _jpeg, "dimension_mismatch"),
    ],
)
def test_invalid_frames_are_rejected_with_specific_codes(
    sync_client: TestClient,
    metadata_changes: dict[str, object],
    payload_factory: object,
    expected_code: str,
) -> None:
    session_id = _create_session(sync_client)
    payload = payload_factory()
    metadata = _metadata(payload)
    metadata.update(metadata_changes)

    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        result = _send_pair(websocket, metadata, payload)

    assert result.accepted is False
    assert result.code == expected_code
    assert result.quality is None


def test_session_counters_and_quiet_disconnect(sync_client: TestClient) -> None:
    session_id = _create_session(sync_client)
    payload = _jpeg()

    with sync_client.websocket_connect(
        f"/ws/ingest/sessions/{session_id}/frames"
    ) as websocket:
        _send_pair(websocket, _metadata(payload, frame_id=3), payload)
        _send_pair(websocket, _metadata(payload, frame_id=3), payload)
        websocket.send_bytes(payload)
        FrameResult.model_validate(websocket.receive_json())

    session = IngestionSession.model_validate(
        sync_client.get(f"/api/ingest/sessions/{session_id}").json()
    )
    assert session.state.value == "IDLE"
    assert session.counters.frames_received == 2
    assert session.counters.frames_accepted == 1
    assert session.counters.frames_rejected == 1
    assert session.counters.frames_out_of_order == 1
    assert session.counters.protocol_errors == 1
    assert session.counters.bytes_received == len(payload) * 2


def test_unknown_websocket_session_returns_structured_error(sync_client: TestClient) -> None:
    with sync_client.websocket_connect(
        "/ws/ingest/sessions/not-a-real-session/frames"
    ) as websocket:
        payload = websocket.receive_json()

    assert payload["error"]["code"] == "ingestion_session_not_found"
