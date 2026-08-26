# FloodSight command centre

The React/Vite frontend preserves the deterministic Phase 1 command centre and adds Phase 2 browser-local video file and webcam sources. `/system` and `/diagnostics` include sanitized ingestion metrics alongside the established API/model view.

## Run

```powershell
Copy-Item .env.example .env.local
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

## Media architecture

- `useMediaSource` owns file validation, object URLs, webcam permission, media controls, track cleanup, and source switching.
- `useFrameIngestion` owns session/WebSocket lifecycle, scheduling, one-in-flight backpressure, acknowledgements, counters, and diagnostics.
- `frameCapture.ts` is the single video-to-canvas-to-JPEG path used by both actual source types.
- `ingestionSocket.ts` enforces metadata-before-binary sending and validates returned acknowledgements.

The local source renders as a plain `<video>` without simulated masks or boxes. The original file/stream is never sent wholesale. Only sampled JPEGs and required metadata leave the browser. All incident panels remain `DEMO_SIMULATED` and display a warning that they are not derived from current video.

Camera constraints request `audio: false`. Source changes, stop, and unmount stop tracks as applicable; object URLs are revoked; sessions and WebSockets are cleaned up. Webcam video is muted in preview and captured frames are not mirrored.

## Checks

```powershell
npm.cmd run lint
npm.cmd run test
npm.cmd run build
```

Tests mock object URLs, media elements, `getUserMedia`, video-frame callbacks, canvas capture, REST sessions, WebSockets, acknowledgements, and cleanup. No physical camera or binary video fixture is needed for the automated suite.

## Troubleshooting

- REST offline: verify backend `/health` and `VITE_API_BASE_URL`.
- Frame socket offline: verify `VITE_WS_BASE_URL`, scheme, port, and proxy upgrade support for `/ws`.
- Camera denial: use localhost/HTTPS, allow site permission, and close competing camera applications.
- Unsupported file: try an H.264 MP4 or VP8/VP9 WebM supported by the current browser.
- Environment changes: restart Vite because variables are read at startup.

No ML model is configured. Quality measurements are `DERIVED_ANALYTIC`; incident analysis remains `DEMO_SIMULATED`.
