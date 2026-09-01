# Parallel application integration status

Date: 2026-09-01

Branch: `phase-4-parallel-integration`

Base Phase 3 commit: `f272a31d9d340bcb2a9b2ab16379f524397ffb03`

## Scope boundary

This branch integrates application-facing inference and rescue intelligence. It does not change the H100 training implementations, training configurations, dataset mappings, taxonomy definitions, source datasets, or checkpoint contents. The VM handoff state was preserved in a separate baseline commit before integration work began.

## Implemented

- stable serializable segmentation, detection, semantic-fusion, intelligence, routing, and WebSocket contracts;
- sanitized model registry with environment-resolved external artifacts and optional SHA-256 verification;
- lazy SegFormer-B2 adapter compatible with `floodsight-segformer-checkpoint-v3` and a local Hugging Face directory format;
- lazy Ultralytics YOLO adapter with explicit final-model and pretrained-fallback label handling;
- deterministic original-image coordinate normalization and class-coloured segmentation overlay;
- semantic/detection fusion with `pool` explicitly excluded from flood evidence;
- deterministic 4×4 rescue evidence grid and four-neighbour zone merging;
- rolling temporal smoothing, stable zone IDs, TTL expiry, and immediate escalation for strong person evidence;
- explainable urgency scoring with confidence reported separately;
- relative accessibility graph, blocked-edge exclusion, uncertainty penalties, primary/alternative routes, and route-change events;
- bounded per-session latest-frame inference coordination;
- FastAPI model status, latest intelligence, backend report, and ordered `frame_intelligence` WebSocket output;
- command-centre rendering for live masks, detections, zones, routes, events, class legend, mask opacity, model mode/device/latency, and backend reports;
- preserved FS-001 `DEMO_SIMULATED` replay with actual-media analytics kept strictly separate.

## Artifact-gated status

- Segmentation integration slot: enabled, taxonomy `segmentation-taxonomy-v2`, external path `FLOODSIGHT_SEGMENTATION_CHECKPOINT`. The registry records the supplied epoch-1 mIoU reference `0.436409`; it does not claim this is a final model.
- Final detection slot: present but disabled until a verified FloodSight/VisDrone checkpoint is supplied through `FLOODSIGHT_DETECTION_CHECKPOINT`.
- Pretrained detection fallback: enabled as an explicit `PRETRAINED_FALLBACK` slot through `FLOODSIGHT_DETECTION_FALLBACK_CHECKPOINT`; it is never labelled as the final detector.
- With no artifact configured, the API and UI report `MODEL_UNAVAILABLE`. No output is silently mocked.

## Operational semantics

- `REAL_ML_OUTPUT` is reserved for direct adapter output.
- Fusion, zones, temporal state, priority, accessibility, routing, and reports are `DERIVED_ANALYTIC` unless all contributing adapters are explicitly simulated.
- `pool` remains a separate semantic class and contributes zero flood urgency.
- `road_non_flooded` is not silently promoted to operationally `CLEAR`.
- semantic damage coverage is not converted into a damaged-building count.
- routing is relative image-space decision support. Metres, travel time, GIS position, and real-world traversability are not claimed.
- trained emergency personnel retain verification and response authority.

## Checkpoint handoff

1. Copy the approved checkpoint outside the repository.
2. Set the applicable environment variable in `.env` or the launch environment.
3. Optionally pin its SHA-256 in `configs/models/registry.json` after the artifact is frozen.
4. Install `backend[inference]` in the application environment.
5. Start the backend and verify `/api/models/status` reports the expected model ID, version, provenance mode, device, and `ready` state.
6. Run an actual video session and verify direct output provenance, latency, overlay alignment, zone stability, and report contents before any demonstration.

## Verification gates

- backend Ruff and pytest;
- shared JSON Schema validation through backend contract tests;
- frontend ESLint, Vitest, TypeScript, and Vite production build;
- dataset-package tests in the isolated dataset environment;
- segmentation/detection training-package tests only in their documented environments. Windows-only dependency gaps are reported, not bypassed.
