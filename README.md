# FloodSight

**From Drone Pixels to Rescue Decisions**

FloodSight is a flood-response decision-intelligence platform. It turns drone or local-video observations into traceable semantic evidence, detections, rescue zones, explainable priority rankings, relative access routes, events, and incident reports. It supports trained emergency personnel; it does not autonomously make life-critical decisions.

This branch contains the parallel application-integration path. It preserves the deterministic FS-001 simulation and the isolated Phase 3 dataset/training work while connecting application-ready model adapters to the same video-file and webcam pipeline.

## Application pipeline

```text
video file / webcam
        ↓ bounded JPEG capture (browser)
FastAPI ingestion + quality acknowledgement
        ↓ latest-frame inference worker (bounded, no raw-frame persistence)
SegFormer adapter ─┐
YOLO adapter ──────┼→ semantic/detection fusion
                   ↓
          deterministic 4×4 evidence grid
                   ↓
     rescue-zone generation + temporal tracking
                   ↓
       explainable 0–100 urgency ranking
                   ↓
       relative accessibility graph + routing
                   ↓
REST/WebSocket backend state → command-centre UI/report
```

The backend is the source of truth for zones, scores, routes, events, and reports. The frontend renders those contracts and never recalculates operational analytics. Frame acknowledgements and intelligence updates are separate WebSocket messages, so slow inference cannot create an unbounded capture queue.

## Model integration and honest fallback

The model registry is [configs/models/registry.json](configs/models/registry.json). Artifact locations come only from environment variables and are never returned to the UI:

- `FLOODSIGHT_SEGMENTATION_CHECKPOINT`: H100-produced FloodSight SegFormer-B2 checkpoint or compatible local Hugging Face directory;
- `FLOODSIGHT_DETECTION_CHECKPOINT`: final FloodSight/VisDrone YOLO checkpoint slot, disabled until a verified artifact is supplied;
- `FLOODSIGHT_DETECTION_FALLBACK_CHECKPOINT`: explicitly labelled pretrained COCO fallback, not the final VisDrone model.

Missing or incompatible artifacts do not silently produce detections. The API reports `MODEL_UNAVAILABLE`, `DEGRADED`, `REAL`, `FALLBACK`, or `SIMULATED` model state. PyTorch, Transformers, and Ultralytics are optional inference dependencies; the API and FS-001 simulation remain runnable without them.

The segmentation adapter preserves the frozen FloodSight taxonomy in `shared/taxonomy/segmentation-taxonomy-v2.yaml`. `pool` is rendered distinctly and is excluded from flood evidence. Road states remain distinct (`CLEAR`, `FLOODED`, `BLOCKED`, `UNKNOWN`). Unknown final-model detection labels are errors; the pretrained fallback retains its original source class while mapping only documented application categories.

## Provenance

Contracts and UI distinguish:

- `REAL_ML_OUTPUT`: direct segmentation or detection model output;
- `DERIVED_ANALYTIC`: fusion, grid evidence, zones, temporal state, scoring, routing, quality metrics, and reports derived from supplied evidence;
- `GIS_EXTERNAL_DATA`: external geographic evidence when genuinely connected;
- `DEMO_SIMULATED`: the explicit FS-001 deterministic replay or an explicitly configured simulated adapter;
- `HUMAN_VERIFIED`: a value confirmed through a human workflow.

Confidence and urgency are separate. Confidence describes evidence reliability; urgency describes operational priority and is never discounted merely because confidence is lower. Semantic building-damage coverage is not presented as a building-instance count. Relative routes do not claim metres, travel time, or GIS accuracy.

## Setup

Prerequisites: Python 3.11+, Node.js 20.19+/22.12+, npm, and a modern browser. CUDA-capable inference additionally requires a compatible NVIDIA driver and inference packages.

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env.local
.\scripts\setup.ps1

# Install only when local model inference is required:
.\.venv\Scripts\python -m pip install -e ".\backend[inference]"

.\scripts\dev.ps1
```

Linux/macOS:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
chmod +x scripts/*.sh
./scripts/setup.sh

# Install only when local model inference is required:
./.venv/bin/python -m pip install -e './backend[inference]'

./scripts/dev.sh
```

Open the command centre at `http://127.0.0.1:5173/`, diagnostics at `http://127.0.0.1:5173/system`, and API documentation at `http://127.0.0.1:8000/docs`.

## Live API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/models/status` | Sanitized model mode, version, device, latency, and availability |
| `POST` | `/api/ingest/sessions` | Create a video-file or webcam session |
| `GET` | `/api/ingest/sessions/{session_id}` | Read bounded counters and session state |
| `DELETE` | `/api/ingest/sessions/{session_id}` | Stop and forget a session |
| `GET` | `/api/ingest/sessions/{session_id}/intelligence` | Latest backend-computed intelligence |
| `GET` | `/api/ingest/sessions/{session_id}/report` | Backend-generated report from the latest intelligence |
| `WS` | `/ws/ingest/sessions/{session_id}/frames` | Frame metadata/binary input, `frame_result` acknowledgements, and ordered `frame_intelligence` updates |

The server accepts one metadata message followed by one JPEG binary message. The browser waits only for the lightweight `frame_result` acknowledgement before capturing another frame. The backend inference coordinator independently keeps at most the latest pending frame per session.

## Decision intelligence

- Segmentation and detection results are normalized to original image coordinates before fusion.
- Evidence is assigned deterministically to a 4×4 grid (`A1` through `D4`); adjacent candidate cells merge into zones.
- Temporal tracking uses rolling evidence, stable zone IDs, bounded history, and TTL expiry. High-confidence person evidence can escalate immediately rather than waiting for smoothing.
- Priority reasons expose their individual contributions. Pool pixels do not increase flood urgency.
- Accessibility edges are enabled, degraded, uncertain, or excluded when blocked. Routing uses relative image-space cost and emits an alternative when one exists.
- Route changes and significant evidence changes become backend events.

## Simulation and datasets

FS-001 remains a six-snapshot, visibly `DEMO_SIMULATED` replay and does not depend on model artifacts. Actual media mode never substitutes FS-001 analytics while inference is loading or unavailable.

Dataset tooling remains isolated in `.venv-datasets`. Set `FLOODSIGHT_DATA_ROOT` and `FLOODSIGHT_DATA_CACHE` to external locations and read `docs/DATASETS.md`, `docs/TAXONOMY.md`, and `docs/DATASET_SERVER_RUNBOOK.md`. Synthetic fixtures establish code readiness, not real-data verification. The parallel integration does not change training code, mappings, taxonomies, datasets, or checkpoints.

## Verification

```powershell
.\scripts\check.ps1
git diff --check
```

Targeted checks are also available:

```powershell
& .\.venv\Scripts\python -m pytest backend\tests -q
& .\.venv\Scripts\python -m ruff check backend\app backend\tests
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
```

Inference adapter tests use injectable stub runtimes and do not require a GPU or checkpoint. A real model is only considered operational when its configured artifact and runtime load successfully.
