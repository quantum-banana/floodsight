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
- explicit per-session `STANDARD`, operator-selected `AERIAL`, and bounded
  `AERIAL_HIGH_RECALL` detector modes; aerial mode
  fuses a 640-pixel full-frame pass with deterministic 2x2 overlapping 1280-pixel tile passes,
  maps every box to original-frame coordinates, and applies person/vehicle-family NMS;
- high-recall aerial mode retains that multiscale evidence and adds deterministic 3x3 tiles at
  1280 pixels with 25% overlap, a benchmark-selected 0.50 person floor, and optional one-crop
  REAL SegFormer-guided reinspection that can only admit a YOLO-confirmed supported class; its
  session contract recommends a stable 1 FPS command-centre capture cadence;
- high-recall per-session person/vehicle tracks use class-family-aware IoU/centre-distance association,
  explicit detection/track confidence and real-confirmation persistence, bounded decay, and TTL
  expiry; no track exists until a real model detection seeds it;
- deterministic original-image coordinate normalization and class-coloured segmentation overlay;
- semantic/detection fusion with `pool` explicitly excluded from flood evidence;
- deterministic 4×4 rescue evidence grid and four-neighbour zone merging;
- rolling temporal smoothing, stable zone IDs, TTL expiry, and immediate escalation for strong person evidence;
- person bottom-centre/local semantic fusion and explicit `POTENTIAL_STRANDED_PERSON` alerts with evidence levels, temporal samples, and review-oriented reason codes;
- explainable urgency scoring with confidence reported separately;
- temporally stabilized semantic road states (`CLEAR`, `FLOODED`, `BLOCKED`, or UI-labelled `UNCERTAIN`);
- relative accessibility graph, blocked/unsafe-flooded edge exclusion, uncertainty penalties, primary/alternative routes, and coded route-change events;
- bounded per-session latest-frame inference coordination;
- FastAPI model status, latest intelligence, backend report, and ordered `frame_intelligence` WebSocket output;
- command-centre rendering for live masks, detections, zones, routes, events, class legend, mask opacity, model mode/device/latency, and backend reports;
- preserved FS-001 `DEMO_SIMULATED` replay with actual-media analytics kept strictly separate.

## Artifact-gated status

- Segmentation final slot: enabled, taxonomy `segmentation-taxonomy-v2`, external path `FLOODSIGHT_SEGMENTATION_CHECKPOINT`. The hackathon release identity is the epoch-33 production handoff (mIoU `0.6531609738189703`); the artifact remains outside Git.
- Final detection slot: enabled for the verified epoch-12 VisDrone-fine-tuned YOLO11l handoff supplied externally through `FLOODSIGHT_DETECTION_CHECKPOINT`. Its registry provenance is `REAL_MODEL`.
- Pretrained detection fallback: the official Ultralytics YOLO11l COCO-pretrained asset from release `v8.4.0`, SHA-256 `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f`, is pinned in the registry and supplied externally through `FLOODSIGHT_DETECTION_FALLBACK_CHECKPOINT`. It is always labelled `PRETRAINED_FALLBACK`, never final or VisDrone fine-tuned.
- Detector mode is selected explicitly when an ingestion session is created. `STANDARD` is the
  efficient full-frame path. `AERIAL` adds full-frame-plus-tile fusion for small drone-view
  objects; it does not alter the COCO source labels, the application detection taxonomy, or the
  `PRETRAINED_FALLBACK` provenance.
- `AERIAL_HIGH_RECALL` is the final bounded COCO-fallback accuracy path. It preserves
  `source_class`, `source_class_id`, and `source_confidence`; cell phone and all other unsupported
  COCO classes remain excluded. It is not a VisDrone-accuracy claim and is not a substitute for
  the pending fine-tuned detector.
- With no artifact configured, the API and UI report `MODEL_UNAVAILABLE`. No output is silently mocked.

## Operational semantics

- `REAL_ML_OUTPUT` is reserved for direct adapter output.
- Direct boxes are labelled `DETECTED`. A decayed box from an established object track is labelled
  `TRACK_PERSISTED` and `DERIVED_ANALYTIC`, retains the last real source frame/class/confidence,
  and expires after the configured TTL.
- Fusion, zones, temporal state, priority, accessibility, routing, and reports are `DERIVED_ANALYTIC` unless all contributing adapters are explicitly simulated.
- `pool` remains a separate semantic class and contributes zero flood urgency.
- `road_non_flooded` is not silently promoted to operationally `CLEAR`.
- road transitions normally require short persistence; very strong blocked/flooded evidence may take effect immediately.
- an unsafe edge on the previously recommended path produces `ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE`, exposes the prior edge IDs, and recomputes through the existing image-space graph.
- `POTENTIAL_STRANDED_PERSON` is a model-driven potential-risk signal, not a definitive stranded-person claim; trained personnel must verify it.
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
