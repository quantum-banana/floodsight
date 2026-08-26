# FloodSight

**From Drone Pixels to Rescue Decisions**

FloodSight is a flood-response decision-intelligence platform for emergency command-centre personnel. Phase 1 provides a complete, judge-facing command-centre experience driven by one deterministic simulated incident. It demonstrates the product flow from disaster evidence to rescue priorities, explanations, relative access intelligence, events, and an incident report.

All operational values in this phase are visibly and structurally labelled `DEMO_SIMULATED`. No video processing, real model inference, GIS, or autonomous dispatch is implemented.

## Phase 1 architecture

```text
FastAPI deterministic scenario service
  ├─ REST: incident metadata, latest state, and report
  ├─ WebSocket: six ordered, schema-valid snapshots
  └─ Pydantic live-result contract
                  │
                  ├─ shared JSON Schema + example
                  │
                  ▼
React + strict TypeScript command centre
  ├─ scalable SVG observation overlays
  ├─ ranked rescue zones and explanation drawer
  ├─ relative tactical map and route
  ├─ bounded incident event timeline
  ├─ current-state report
  └─ Phase 0 diagnostics at /system
```

The live-result contract is synchronized across:

- `shared/schemas/live-result.schema.json` — language-neutral JSON Schema;
- `shared/examples/live-result.sample.json` — minimal compatible example;
- `backend/app/schemas/live_result.py` — strict Pydantic models;
- `frontend/src/types/liveResult.ts` — strict TypeScript interfaces.

Normalized coordinates remain in the repository's established `0–1` range. The UI maps them into responsive SVG view boxes and never presents them as geographic coordinates or real-world distance.

## Deterministic incident

Phase 1 contains one reproducible six-snapshot scenario:

- ID: `FS-001`
- title: `Riverside Ward Flood Response`
- source mode: `SIMULATION`
- coordinate space: `RELATIVE_TACTICAL`
- provenance: `DEMO_SIMULATED`

The stable final snapshot has CRITICAL incident severity, 42% simulated flood coverage, 6 people, 4 vehicles, 2 blocked roads, and 5 damaged buildings. Its supplied rescue ranking is Zone 2 at 92, Zone 4 at 76, and Zone 1 at 54. These values and score contributions are fixture data, not outputs from a real priority engine.

## Prerequisites

- Python 3.11 or newer;
- Node.js 20.19+, 22.12+, or newer and npm;
- Git;
- PowerShell 5.1+ on Windows, or a POSIX-compatible shell on Linux/macOS.

Docker, GPU tooling, model weights, map tiles, and runtime internet access are not required.

## Setup

Windows PowerShell, from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env.local
.\scripts\setup.ps1
```

Linux/macOS:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
chmod +x scripts/*.sh
./scripts/setup.sh
```

## Start Phase 1

Run both services:

```powershell
.\scripts\dev.ps1
```

or:

```bash
./scripts/dev.sh
```

Development URLs:

- command centre: `http://127.0.0.1:5173/`
- system diagnostics: `http://127.0.0.1:5173/system`
- backend health: `http://127.0.0.1:8000/health`
- API documentation: `http://127.0.0.1:8000/docs`
- demo WebSocket: `ws://127.0.0.1:8000/ws/demo/incidents/FS-001/live`

The command centre loads the first snapshot through REST and connects to the deterministic WebSocket stream. Use Start, Pause, Resume, and Reset in the replay bar. The report and diagnostics actions are in the application header. The interface never substitutes local incident values if the backend is unavailable.

To start each service separately on Windows:

```powershell
Push-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
Pop-Location

Push-Location frontend
npm.cmd run dev -- --host 127.0.0.1
Pop-Location
```

## Demo API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service readiness |
| GET | `/api/models/status` | Honest `not_configured` model state |
| GET | `/api/demo/live-result` | Preserved Phase 0 contract preview |
| GET | `/api/demo/incidents` | Deterministic incident list |
| GET | `/api/demo/incidents/FS-001` | Incident metadata plus initial/latest snapshots |
| GET | `/api/demo/incidents/FS-001/report` | Report-ready final state |
| WS | `/ws/demo/incidents/FS-001/live` | Ordered snapshot replay |

The WebSocket accepts optional `start_index`, `interval_ms`, and `loop` query parameters. The default interval comes from `FLOODSIGHT_DEMO_STREAM_INTERVAL_MS`.

## Verification

Run the complete suite:

```powershell
.\scripts\check.ps1
```

```bash
./scripts/check.sh
```

Equivalent Windows commands:

```powershell
.\.venv\Scripts\python.exe -m ruff check backend\app backend\tests
.\.venv\Scripts\python.exe -m pytest backend\tests
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run test
npm.cmd --prefix frontend run build
```

Backend tests validate every deterministic snapshot against JSON Schema, provenance, stable zone/event IDs, REST errors, and WebSocket order. Frontend tests cover dashboard states, ranking changes, zone details, layers, reports, diagnostics, and a deterministic WebSocket mock.

## Environment configuration

Backend variables use the `FLOODSIGHT_` prefix. The root `.env.example` documents host, port, logging, CORS origins, and demo interval. The application reads the root `.env` and `backend/.env` when present.

Frontend variables are documented in `frontend/.env.example`:

- `VITE_API_BASE_URL` — REST API base URL;
- `VITE_WS_BASE_URL` — WebSocket base URL;
- `VITE_DEV_PROXY_TARGET` — Vite REST/WebSocket proxy target.

Use `frontend/.env.local` for machine-specific values and never commit it.

## Truth and provenance labels

- `DEMO_SIMULATED` — synthetic Phase 1 scenario data;
- `REAL_ML_OUTPUT` — reserved for a direct configured model output;
- `DERIVED_ANALYTIC` — reserved for a declared deterministic calculation;
- `GIS_EXTERNAL_DATA` — reserved for external geographic data;
- `HUMAN_VERIFIED` — reserved for an authorized human confirmation.

Phase 1 only emits `DEMO_SIMULATED`. Segmentation is explicitly `simulated`; SegFormer and YOLO remain `not_configured`.

## Implemented and out of scope

Phase 1 implements the responsive command-centre UI, deterministic REST/WebSocket replay, simulated SVG overlays, supplied ranked zones and explanations, relative route view, bounded events, report/copy/print actions, explicit connection states, and preserved system diagnostics.

Still unimplemented:

- video upload, decoding, webcam, or actual drone streaming;
- OpenCV ingestion or temporal scene fusion;
- datasets, training, SegFormer, YOLO, PyTorch, or real inference;
- real rescue-zone generation or priority calculation;
- real GIS, map distances, travel times, or route optimization;
- autonomous dispatch or an LLM assistant.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository foundation and frontend/backend connectivity | Complete |
| 1 | Complete command-centre UI with deterministic simulated data | Complete |
| 2 | Unified video-file and webcam ingestion | Planned |
| 3 | Dataset validation, inspection, and taxonomy mapping | Planned |
| 4 | SegFormer segmentation training and evaluation | Planned |
| 5 | YOLO aerial detection training and evaluation | Planned |
| 6 | Real-time model inference integration | Planned |
| 7 | Rescue-zone generation and temporal stability | Planned |
| 8 | Explainable deterministic priority engine | Planned |
| 9 | Tactical intelligence, routing, events, and reports | Planned |
| 10 | Demo hardening, tuning, fallback, and reliability | Planned |

## Troubleshooting

### PowerShell blocks npm or project scripts

Use `Set-ExecutionPolicy -Scope Process Bypass` for the current shell. Repository scripts invoke `npm.cmd`, avoiding the commonly blocked `npm.ps1` shim.

### The command centre says the backend is offline

Open `http://127.0.0.1:8000/health`, verify `VITE_API_BASE_URL`, and confirm the frontend origin is present in `FLOODSIGHT_CORS_ORIGINS`. Restart both services after environment changes.

### The stream remains reconnecting or disconnected

Confirm `VITE_WS_BASE_URL` uses `ws://` for HTTP development and `wss://` for HTTPS. If the frontend host or backend port changed, update both the WebSocket URL and Vite proxy target. A reverse proxy must support WebSocket upgrade requests on `/ws`.

### Browser reports CORS or origin failures

Keep the exact scheme, host, and port in `FLOODSIGHT_CORS_ORIGINS`; `localhost` and `127.0.0.1` are different origins. REST CORS configuration does not repair a blocked or misrouted WebSocket upgrade, so inspect the WebSocket request separately.

### Shared-schema validation fails

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests\test_shared_schema.py -vv
```

Keep the JSON Schema, Pydantic models, TypeScript interfaces, examples, and scenario snapshots aligned.
