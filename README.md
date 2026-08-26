# FloodSight

**From Drone Pixels to Rescue Decisions**

FloodSight is a flood-response decision-intelligence platform designed to turn live or recorded drone observations into an explainable, priority-aware rescue and resource-allocation picture for emergency command centres.

Phase 0 provides the runnable monorepo foundation and an honest development-status interface. It does **not** yet perform machine-learning inference, video analysis, rescue-zone generation, prioritisation, or routing.

## Architecture overview

```text
React + TypeScript + Tailwind
        │
        │ REST (Phase 0)
        ▼
FastAPI application
        │
        ├── health and model readiness
        └── explicitly simulated shared incident example

Future phases:
drone/video → SegFormer + YOLO → temporal scene fusion
            → rescue zones → explainable priorities → routing/dashboard
```

The live-result contract is defined in three synchronized forms:

- `shared/schemas/live-result.schema.json` — language-neutral JSON Schema;
- `backend/app/schemas/live_result.py` — Pydantic API models;
- `frontend/src/types/liveResult.ts` — TypeScript interfaces.

## Repository layout

```text
floodsight/
├── frontend/                 React/Vite status application
├── backend/                  FastAPI service and tests
├── shared/
│   ├── schemas/              Cross-runtime JSON contracts
│   └── examples/             Contract examples
├── ml/
│   ├── segmentation/         Reserved for later SegFormer work
│   └── detection/            Reserved for later YOLO work
├── models/
│   ├── segmentation/         Local checkpoints (ignored)
│   └── detection/            Local checkpoints (ignored)
├── datasets/                 Local datasets (ignored)
├── demo/
│   ├── videos/               Local demo media (ignored)
│   └── replay/               Replay fixtures in later phases
├── docs/                     Product context and documentation
├── scripts/                  Setup, development, and check helpers
└── tests/                    Cross-project test area
```

## Prerequisites

- Python 3.11 or newer;
- Node.js 20.19+, 22.12+, or newer and npm;
- Git for normal source-control workflows;
- PowerShell 5.1+ on Windows, or a POSIX-compatible shell on Linux/macOS.

Docker and GPU tooling are not required for Phase 0.

## Windows PowerShell setup

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env.local
.\scripts\setup.ps1
```

The setup script creates `.venv`, installs the backend with development dependencies, and runs `npm install` in `frontend`.

## Linux/macOS setup

From the repository root:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
chmod +x scripts/*.sh
./scripts/setup.sh
```

## Start development servers

Run both services:

```powershell
# Windows PowerShell
.\scripts\dev.ps1
```

```bash
# Linux/macOS
./scripts/dev.sh
```

Development URLs:

- frontend: `http://127.0.0.1:5173`;
- backend health: `http://127.0.0.1:8000/health`;
- interactive API documentation: `http://127.0.0.1:8000/docs`.

To run the services separately on Windows:

```powershell
Push-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
Pop-Location

Push-Location frontend
npm.cmd run dev -- --host 127.0.0.1
Pop-Location
```

## Run all checks

```powershell
# Windows PowerShell
.\scripts\check.ps1
```

```bash
# Linux/macOS
./scripts/check.sh
```

Equivalent individual commands are:

```powershell
.\.venv\Scripts\python.exe -m ruff check backend\app backend\tests
.\.venv\Scripts\python.exe -m pytest backend\tests
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run test
npm.cmd --prefix frontend run build
```

The backend test suite includes JSON Schema validation for the shared sample.

## Environment configuration

Backend settings use the `FLOODSIGHT_` prefix. See `.env.example` for host, port, log level, environment name, and allowed CORS origins. The application reads the root `.env` and `backend/.env` when present.

The frontend reads `VITE_API_BASE_URL`. Use `frontend/.env.local` for machine-specific settings; never commit it. Vite's development proxy can also use `VITE_DEV_PROXY_TARGET`.

## Current implementation status

Phase 0 includes:

- a typed FastAPI application with health, honest model-status, and simulated-demo endpoints;
- a shared live-result schema and example;
- a responsive development-status frontend with loading, offline, and retry states;
- frontend/backend tests, linting, and production build configuration;
- cross-platform setup, development, and verification scripts.

Not yet implemented:

- dataset ingestion or label harmonisation;
- SegFormer or YOLO loading, training, or inference;
- video or webcam ingestion;
- temporal fusion, rescue zones, priority calculations, routing, GIS, reports, or the complete command-centre dashboard.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository foundation and frontend/backend connectivity | Implemented |
| 1 | Polished command-centre UI using visibly simulated data | Planned |
| 2 | Unified video-file and webcam ingestion | Planned |
| 3 | Dataset validation, inspection, and taxonomy mapping | Planned |
| 4 | SegFormer segmentation training and evaluation | Planned |
| 5 | YOLO aerial detection training and evaluation | Planned |
| 6 | Real-time model inference integration | Planned |
| 7 | Rescue-zone generation and temporal stability | Planned |
| 8 | Explainable deterministic priority engine | Planned |
| 9 | Tactical intelligence, routing, events, and reports | Planned |
| 10 | Demo hardening, tuning, fallback, and reliability | Planned |

## Truth labels

Every operational value must communicate its origin:

- `REAL_ML_OUTPUT` — a direct output from a configured and executed ML model;
- `DERIVED_ANALYTIC` — deterministically calculated from declared inputs;
- `GIS_EXTERNAL_DATA` — supplied by a geographic or other external data source;
- `DEMO_SIMULATED` — synthetic data used only for development or a clearly labelled demonstration;
- `HUMAN_VERIFIED` — reviewed and explicitly confirmed by an authorized human.

Phase 0's incident preview is `DEMO_SIMULATED`. Model statuses are `not_configured`; no endpoint represents the sample as inference.

## Troubleshooting

### PowerShell blocks npm or project scripts

Use `Set-ExecutionPolicy -Scope Process Bypass` for the current PowerShell process. The repository's PowerShell scripts invoke `npm.cmd`, which also avoids the common blocked `npm.ps1` shim.

### `python` is not found on Windows

Install Python 3.11+ and enable its PATH option, or use the Python launcher to create the environment manually:

```powershell
py -3.11 -m venv .venv
```

Then rerun `scripts/setup.ps1`.

### The frontend reports that the backend is offline

Confirm `http://127.0.0.1:8000/health` opens, verify `VITE_API_BASE_URL` in `frontend/.env.local`, and ensure the frontend origin appears in `FLOODSIGHT_CORS_ORIGINS`. Restart both development servers after changing environment files.

### Port 5173 or 8000 is already in use

Stop the conflicting process or start the service manually on another port. If the backend port changes, update `VITE_API_BASE_URL` to match.

### Dependency installation fails

Check network access, then upgrade packaging tools and retry:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
npm.cmd cache verify
.\scripts\setup.ps1
```

Do not add ML packages as a workaround; Phase 0 intentionally has a lightweight dependency set.

### Shared sample validation fails

Run `.\.venv\Scripts\python.exe -m pytest backend\tests\test_shared_schema.py -vv`. Keep JSON Schema, Pydantic, TypeScript, and the sample field names aligned when changing the contract.

