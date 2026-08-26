# FloodSight

**From Drone Pixels to Rescue Decisions**

FloodSight is a flood-response decision-intelligence prototype. Phase 2 adds real browser video-file and webcam ingestion to the verified Phase 1 command centre. The browser previews the chosen source, samples JPEG frames through one shared canvas pipeline, and sends only those frames to FastAPI for OpenCV decode and basic image-quality checks.

This phase does **not** perform machine-learning inference. Incident statistics, detections, flood masks, zones, routes, events, and reports remain the deterministic `DEMO_SIMULATED` Phase 1 scenario. On actual media, those analytics are visibly separated from the video and no simulated overlays are drawn over it.

## Architecture

```text
Local video file ─┐
                  ├─ HTMLVideoElement → shared canvas → bounded JPEG frames
Webcam stream ────┘                         │
                                           │ metadata JSON, then binary JPEG
                                           ▼
FastAPI ingestion session → OpenCV BGR decode → luminance/blur quality result
        │                                      (DERIVED_ANALYTIC)
        └─ bounded, expiring counters only; no raw frame persistence

FastAPI deterministic incident REST/WebSocket → Phase 1 command-centre analytics
                                                (DEMO_SIMULATED)
```

The ingestion contract is synchronized across:

- `shared/schemas/ingest-session.schema.json` and `frame-result.schema.json`;
- `shared/examples/ingest-session.sample.json`, `frame-metadata.sample.json`, and `frame-result.sample.json`;
- `backend/app/schemas/ingestion.py` strict Pydantic models;
- `frontend/src/types/ingestion.ts` strict TypeScript interfaces.

The established live-result contract and all Phase 0/1 routes remain compatible.

## Prerequisites and setup

- Python 3.11 or newer;
- Node.js 20.19+, 22.12+, or newer and npm;
- a modern browser with Canvas, Blob, and WebSocket support;
- camera permission for webcam mode.

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env.local
.\scripts\setup.ps1
.\scripts\dev.ps1
```

Linux/macOS:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/dev.sh
```

Open:

- command centre: `http://127.0.0.1:5173/`;
- diagnostics: `http://127.0.0.1:5173/system`;
- API docs: `http://127.0.0.1:8000/docs`.

## Using media inputs

The command-centre source selector provides:

- **Simulation** — the verified deterministic six-snapshot replay;
- **Video file** — choose a browser-playable local MP4/WebM and use Play, Pause/Resume, or Stop/Reset;
- **Webcam** — start a video-only camera request, then Pause/Resume or Stop.

The original file is represented by a browser object URL and is never uploaded as a whole. Webcam requests use `audio: false`; no microphone is requested. Source switching and unmounting revoke object URLs, stop every camera track, close the frame WebSocket, and delete the server session.

Capture defaults are 4 FPS, JPEG quality `0.75`, and a 1280×720 bound while retaining aspect ratio. At most one frame awaits acknowledgement. Capture opportunities encountered during that wait are counted as client drops instead of creating an unbounded queue.

## Ingestion API and protocol

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/ingest/sessions` | Create a `VIDEO_FILE` or `WEBCAM` session |
| GET | `/api/ingest/sessions/{session_id}` | Read state, limits, and counters |
| DELETE | `/api/ingest/sessions/{session_id}` | Idempotently stop and forget a session |
| WS | `/ws/ingest/sessions/{session_id}/frames` | Send frame metadata/binary pairs and receive results |

For every frame, the client sends exactly:

1. one `frame_metadata` JSON text message;
2. one binary JPEG message;
3. waits for one `frame_result` JSON acknowledgement before sending another frame.

The backend rejects unsupported MIME types, oversized or mismatched payloads, invalid JPEGs, mismatched dimensions/provenance, duplicate or out-of-order IDs, and invalid message order. Valid low-quality frames are accepted with explicit dark/bright/blurry warnings. Only session metadata and counters live in process memory; encoded bytes, decoded arrays, and original media are not stored.

## Provenance

- `USER_VIDEO_FILE` / `USER_WEBCAM`: where actual media came from;
- `DERIVED_ANALYTIC`: decoded dimensions, byte counts, timing, luminance, and Laplacian blur variance;
- `DEMO_SIMULATED`: all incident analysis currently shown in the command centre;
- `REAL_ML_OUTPUT`, `GIS_EXTERNAL_DATA`, and `HUMAN_VERIFIED`: reserved and not emitted by this phase.

Segmentation and detection remain `not_configured`. No PyTorch, model checkpoint, dataset, training code, inference pipeline, real GIS, rescue scoring, or autonomous dispatch was added.

## Configuration

Backend `FLOODSIGHT_` settings include session TTL/capacity, maximum frame bytes, recommended FPS/JPEG quality, and quality thresholds. Frontend `VITE_` settings include REST/WebSocket URLs, file-size limit, capture rate, JPEG quality, and capture bounds. See both `.env.example` files.

## Verification

```powershell
.\scripts\check.ps1
git diff --check
```

or:

```bash
./scripts/check.sh
git diff --check
```

The suites programmatically generate JPEG fixtures and mock camera, media element, object URL, canvas, timer, and WebSocket behavior. They require neither committed video fixtures nor a physical webcam.

## Troubleshooting

### Camera permission is denied

Camera access requires a secure context (`https://` or localhost). Allow camera access for the site, close applications already using the camera, and retry. Stop returns all acquired tracks to the browser.

### A selected video will not play

Browser support depends on its container and codecs. Try a standard H.264/AAC MP4 or VP8/VP9 WebM. The file extension alone does not guarantee codec support. No server-side transcoding occurs.

### Ingestion is offline or acknowledgements time out

Verify `/health`, `VITE_API_BASE_URL`, and `VITE_WS_BASE_URL`. For local development, use matching `http`/`ws` or `https`/`wss` schemes. Reverse proxies must upgrade `/ws`. The UI stops queuing after one unacknowledged frame and exposes the error in command-centre and diagnostics views.

### CORS fails

Add the exact frontend scheme, host, and port to `FLOODSIGHT_CORS_ORIGINS`. `localhost` and `127.0.0.1` are different origins.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository foundation and frontend/backend connectivity | Complete |
| 1 | Command-centre UI with deterministic simulated data | Complete |
| 2 | Unified video-file and webcam frame ingestion | Implemented, pending freeze |
| 3 | Dataset validation, inspection, and taxonomy mapping | Planned |
| 4–10 | Model training, inference, rescue intelligence, and hardening | Planned |
