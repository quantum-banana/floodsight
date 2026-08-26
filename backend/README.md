# FloodSight API

This lightweight FastAPI service is the Phase 0 backend foundation. It exposes service readiness, honest model configuration state, and a shared simulated incident example. It contains no ML or video-processing dependencies.

## Run locally

After running the root setup script:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Configuration uses `FLOODSIGHT_` environment variables from the process, the root `.env`, or `backend/.env`. See the root `.env.example`.

## Endpoints

- `GET /health`
- `GET /api/models/status`
- `GET /api/demo/live-result`
- `GET /docs`

The demo endpoint always identifies its Phase 0 payload as `DEMO_SIMULATED`.

## Tests

```powershell
..\.venv\Scripts\python.exe -m pytest tests
```

