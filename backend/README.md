# FloodSight API

The FastAPI backend preserves the Phase 0 readiness endpoints and adds the Phase 1 deterministic incident service. It contains no video, ML, GIS, or remote-service dependencies.

## Run locally

After root setup:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Configuration uses `FLOODSIGHT_` variables from the process, root `.env`, or `backend/.env`. `FLOODSIGHT_DEMO_STREAM_INTERVAL_MS` controls the default replay delay.

## Routes

- `GET /health`
- `GET /api/models/status`
- `GET /api/demo/live-result`
- `GET /api/demo/incidents`
- `GET /api/demo/incidents/{incident_id}`
- `GET /api/demo/incidents/{incident_id}/report`
- `WS /ws/demo/incidents/{incident_id}/live`
- `GET /docs`

The WebSocket supports `start_index`, `interval_ms`, and `loop`. It emits a valid first snapshot immediately, streams fixed snapshots in order, and closes after the final snapshot unless looping is requested. Unknown incident REST responses use the existing structured error contract; the WebSocket sends the same error shape before closing with code 4404.

Every Phase 1 incident value is `DEMO_SIMULATED`. Model status remains `not_configured`, and the segmentation state is `simulated`, not model-ready.

## Tests

```powershell
..\.venv\Scripts\python.exe -m ruff check app tests
..\.venv\Scripts\python.exe -m pytest tests
```

The suite covers preserved endpoints, incident list/detail/report, structured not-found behavior, deterministic WebSocket order, stable IDs, provenance, and JSON Schema validation for every scenario snapshot.
