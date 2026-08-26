# FloodSight Engineering Instructions

- Read `docs/FLOODSIGHT_MASTER_CONTEXT.md` before changing architecture, scope, ML contracts, or product positioning.
- FloodSight is a flood-response decision-intelligence platform that turns drone observations into explainable rescue and resource-allocation guidance. Do not reduce it to a detector, segmentation demo, or video viewer.
- The frozen stack is React, TypeScript, Vite, Tailwind CSS, Python, FastAPI, REST, WebSockets, SegFormer, YOLO, temporal scene fusion, rescue-zone generation, explainable priority scoring, routing, and a command-centre dashboard.
- Implement in the documented Phase 0 through Phase 10 order. Keep the repository runnable and verifiable at the end of every phase.
- Preserve human control: FloodSight supports trained emergency personnel; it does not autonomously make life-critical decisions.
- Label provenance honestly. Distinguish `REAL_ML_OUTPUT`, `DERIVED_ANALYTIC`, `GIS_EXTERNAL_DATA`, `DEMO_SIMULATED`, and `HUMAN_VERIFIED` values in contracts and UI.
- Never silently mock core ML output. Any simulated value must be visibly and structurally labelled `DEMO_SIMULATED`.
- Do not invent, infer, or merge dataset labels without evidence. Preserve distinctions such as flooded road versus blocked road, and document every taxonomy mapping.
- Keep datasets outside the repository under an explicit `FLOODSIGHT_DATA_ROOT`. Treat `raw/` as immutable and never commit source data, archives, processed outputs, reports, inspections, locks, caches, checkpoints, or runs.
- Keep product, segmentation-training, and detection-training taxonomies distinct. Unknown source labels are blocking errors; never map them silently to background.
- Phase 3 mappings are candidates until complete real source inventories and human visual review are recorded. Synthetic fixtures establish CODE READY, never DATA VERIFIED.
- Keep dataset tooling in its isolated lightweight environment. Do not add PyTorch, torchvision, Transformers, Ultralytics, TensorFlow, training, or inference to Phase 3.
- Prefer lightweight dependencies and phase-scoped implementation. Do not pull later-phase work into an earlier phase without an explicit request.
- Run all relevant tests, lint checks, schema validation, and builds before declaring work complete. Repair failures rather than bypassing checks.
- Do not claim that a model, data source, analytic, or workflow is implemented until it is genuinely connected and verified.
