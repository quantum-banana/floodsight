# FloodSight command centre

The React/Vite frontend is the Phase 1 command-centre product UI. The main route renders the deterministic incident; `/system` and `/diagnostics` preserve the Phase 0 readiness view.

## Run

```powershell
Copy-Item .env.example .env.local
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/`. The backend must be available at the URLs configured by `VITE_API_BASE_URL` and `VITE_WS_BASE_URL`.

## Controls and states

- Start restarts from snapshot one.
- Pause closes the current stream while retaining the last valid backend state.
- Resume continues from the next snapshot.
- Reset restores the initial state and starts a new replay.
- Retry re-fetches incident metadata after an offline or disconnected state.
- Incident report builds copy/print content from the current snapshot.
- Diagnostics opens the preserved Phase 0 system-status view.

The interface explicitly presents loading, connecting, replaying, paused, reconnecting, complete, malformed, disconnected, and backend-offline states. It never uses local fallback incident values.

## Architecture

Feature folders separate command-centre composition, incident identity/overview, observation overlays, rescue priorities, tactical map, events, reports, and diagnostics. `useDemoIncident` owns REST bootstrap and WebSocket lifecycle; `demoStream.ts` wraps the native WebSocket; `validation.ts` rejects unsafe messages; shared types live in `types/liveResult.ts`.

Observation and map visuals are local responsive SVG/CSS renderings of normalized backend geometry. There are no external map tiles, CDN assets, remote fonts, stock footage, or real-world coordinates.

## Checks

```powershell
npm.cmd run lint
npm.cmd run test
npm.cmd run build
```

Tests use typed Phase 1 fixtures and a deterministic native-WebSocket mock; they do not depend on uncontrolled replay timers.

## Troubleshooting

- REST offline: verify the backend health URL and `VITE_API_BASE_URL`.
- Reconnecting WebSocket: verify `VITE_WS_BASE_URL`, protocol (`ws`/`wss`), backend port, and proxy upgrade support for `/ws`.
- CORS failure: add the exact frontend scheme/host/port to backend `FLOODSIGHT_CORS_ORIGINS`.
- Environment changes: restart Vite; values are read at startup.

All incident content remains `DEMO_SIMULATED`. Video ingestion and ML are intentionally outside Phase 1.
