# FloodSight API

The FastAPI backend preserves the Phase 0/1 health, model-status, deterministic incident REST, report, and demo WebSocket behavior. Phase 2 adds bounded, expiring media-frame sessions and transient OpenCV JPEG decode with basic image-quality metrics. It contains no model inference.

## Run

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Configuration uses `FLOODSIGHT_` variables from the process, root `.env`, or `backend/.env`.

## Phase 2 routes

- `POST /api/ingest/sessions`
- `GET /api/ingest/sessions/{session_id}`
- `DELETE /api/ingest/sessions/{session_id}`
- `WS /ws/ingest/sessions/{session_id}/frames`

Create request:

```json
{"source_mode":"VIDEO_FILE","media_origin":"USER_VIDEO_FILE"}
```

`WEBCAM` must pair with `USER_WEBCAM`. IDs are generated with `secrets.token_urlsafe`; session capacity and TTL are configurable. Expiry is enforced on session operations. DELETE is idempotent.

The WebSocket accepts one strict `frame_metadata` text message followed by one binary JPEG. It responds with one `frame_result` acknowledgement. The acknowledgement declares acceptance/rejection, decoded BGR dimensions, processing latency, mean grayscale luminance, Laplacian variance, quality warnings, and `DERIVED_ANALYTIC` provenance. The implementation never retains a `FramePacket`, encoded payload, or decoded NumPy array after processing.

The backend accepts valid dark/bright/blurry images but reports warnings. Invalid sequence, MIME, size, byte length, JPEG decode, source provenance, decoded dimensions, and monotonic ID violations are rejected explicitly.

## Preserved routes

- `GET /health`
- `GET /api/models/status`
- `GET /api/demo/live-result`
- `GET /api/demo/incidents`
- `GET /api/demo/incidents/{incident_id}`
- `GET /api/demo/incidents/{incident_id}/report`
- `WS /ws/demo/incidents/{incident_id}/live`
- `GET /docs`

Every incident value remains `DEMO_SIMULATED`; segmentation and detection remain `not_configured`.

## Tests

```powershell
..\.venv\Scripts\python.exe -m ruff check app tests
..\.venv\Scripts\python.exe -m pytest tests
```

Tests generate small JPEGs with OpenCV/NumPy and cover REST lifecycle, capacity/expiry, strict WebSocket order, valid decode, dimensions, quality, counters, rejection cases, quiet disconnect, provenance, shared JSON Schemas, and all preserved endpoints.
