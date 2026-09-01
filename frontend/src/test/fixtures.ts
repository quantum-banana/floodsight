import type { IncidentDetailResponse, SystemSnapshot } from "../types/api";
import type { LiveResult, Metric, Zone } from "../types/liveResult";

const origin = "DEMO_SIMULATED" as const;
const pointSquare = (x: number, y: number) => [
  { x, y },
  { x: x + 0.18, y },
  { x: x + 0.18, y: y + 0.18 },
  { x, y: y + 0.18 },
];
const metric = (value: number, unit: Metric["unit"]): Metric => ({
  value,
  unit,
  confidence: null,
  data_origin: origin,
});

const makeZone = (
  zoneId: string,
  displayName: string,
  rank: number,
  priorityScore: number,
  severity: Zone["severity"],
  x: number,
  y: number,
): Zone => ({
  zone_id: zoneId,
  display_name: displayName,
  rank,
  severity,
  priority_score: priorityScore,
  confidence: 0.91,
  polygon: pointSquare(x, y),
  people_count: zoneId === "ZONE-2" ? 4 : 1,
  vehicle_count: zoneId === "ZONE-4" ? 2 : 1,
  flood_coverage_percent: zoneId === "ZONE-2" ? 78 : 46,
  building_damage_count: zoneId === "ZONE-4" ? 3 : 1,
  road_condition: zoneId === "ZONE-2" ? "BLOCKED" : "FLOODED",
  access_status: zoneId === "ZONE-2" ? "ISOLATED" : "DEGRADED",
  primary_reason: `${displayName} combines simulated flood exposure and access constraints.`,
  reasons: [
    {
      code: "FLOOD_EXPOSURE",
      label: "Flood exposure",
      description: "Supplied deterministic contribution for the demonstration.",
      contribution: priorityScore,
      data_origin: origin,
    },
  ],
  updated_at_ms: 1_725_000_005_000,
  data_origin: origin,
});

const zone1 = makeZone("ZONE-1", "Zone 1", 3, 54, "MODERATE", 0.08, 0.24);
const zone2 = makeZone("ZONE-2", "Zone 2", 1, 92, "CRITICAL", 0.4, 0.36);
const zone4 = makeZone("ZONE-4", "Zone 4", 2, 76, "HIGH", 0.7, 0.54);

export const commandSnapshot: LiveResult = {
  incident_id: "FS-001",
  incident: {
    incident_id: "FS-001",
    title: "Riverside Ward Flood Response",
    location_label: "Riverside Ward · Demonstration Grid",
    started_at_ms: 1_725_000_000_000,
    coordinate_space: "RELATIVE_TACTICAL",
    data_origin: origin,
  },
  frame_id: 5,
  snapshot_index: 5,
  snapshot_count: 6,
  timestamp_ms: 1_725_000_005_000,
  source_mode: "SIMULATION",
  coordinate_space: "RELATIVE_TACTICAL",
  data_origin: origin,
  stream_state: "COMPLETE",
  incident_severity: "CRITICAL",
  highest_priority_zone_id: "ZONE-2",
  system_status: {
    api: "operational",
    segmentation_model: "not_configured",
    detection_model: "not_configured",
  },
  statistics: {
    flooded_area_percent: metric(42, "percent"),
    people_detected: metric(6, "count"),
    vehicles_detected: metric(4, "count"),
    blocked_roads: metric(2, "count"),
    damaged_buildings: metric(5, "count"),
  },
  detections: [
    {
      detection_id: "DET-P-001",
      category: "PERSON",
      label: "Simulated person",
      confidence: 0.91,
      bbox: { x: 0.48, y: 0.42, width: 0.035, height: 0.075 },
      data_origin: origin,
    },
    {
      detection_id: "DET-V-001",
      category: "VEHICLE",
      label: "Simulated vehicle",
      confidence: 0.88,
      bbox: { x: 0.73, y: 0.62, width: 0.07, height: 0.045 },
      data_origin: origin,
    },
  ],
  segmentation: {
    status: "simulated",
    classes: [
      { label: "flood_water", coverage_percent: 42, confidence: 0.9, data_origin: origin },
    ],
    regions: [
      {
        overlay_id: "OV-FLOOD-001",
        kind: "FLOOD",
        label: "Simulated flood water",
        polygon: [
          { x: 0.02, y: 0.5 }, { x: 0.35, y: 0.35 }, { x: 0.64, y: 0.49 },
          { x: 0.96, y: 0.37 }, { x: 0.96, y: 0.78 }, { x: 0.08, y: 0.82 },
        ],
        confidence: 0.9,
        data_origin: origin,
      },
      {
        overlay_id: "OV-DAMAGE-001",
        kind: "DAMAGED_BUILDING",
        label: "Simulated damaged building",
        polygon: pointSquare(0.72, 0.2),
        confidence: 0.84,
        data_origin: origin,
      },
    ],
  },
  roads: [
    {
      road_id: "R-1",
      label: "North access",
      state: "BLOCKED",
      access_status: "BLOCKED",
      geometry: [{ x: 0.05, y: 0.7 }, { x: 0.48, y: 0.46 }, { x: 0.92, y: 0.2 }],
      confidence: null,
      data_origin: origin,
    },
    {
      road_id: "R-2",
      label: "South access",
      state: "FLOODED",
      access_status: "DEGRADED",
      geometry: [{ x: 0.08, y: 0.9 }, { x: 0.5, y: 0.68 }, { x: 0.9, y: 0.74 }],
      confidence: null,
      data_origin: origin,
    },
  ],
  zones: [zone1, zone4, zone2],
  events: [
    {
      event_id: "EVT-006",
      timestamp_ms: 1_725_000_005_000,
      severity: "CRITICAL",
      category: "PRIORITY",
      message: "Zone 2 elevated to the highest rescue priority.",
      data_origin: origin,
    },
    {
      event_id: "EVT-005",
      timestamp_ms: 1_725_000_004_000,
      severity: "WARNING",
      category: "ACCESS",
      message: "North access is blocked in the deterministic scenario.",
      data_origin: origin,
    },
  ],
  route: {
    route_id: "ROUTE-001",
    status: "RECOMMENDED",
    target_zone_id: "ZONE-2",
    label: "Relative access route to Zone 2",
    waypoints: [{ x: 0.07, y: 0.86 }, { x: 0.3, y: 0.72 }, { x: 0.48, y: 0.5 }],
    distance_m: null,
    access_summary: "Relative route avoids the simulated blocked north access.",
    data_origin: origin,
  },
};

export const reorderedSnapshot: LiveResult = {
  ...commandSnapshot,
  frame_id: 4,
  snapshot_index: 4,
  stream_state: "PLAYING",
  highest_priority_zone_id: "ZONE-4",
  zones: [
    { ...zone1, rank: 2, priority_score: 62 },
    { ...zone2, rank: 3, priority_score: 58 },
    { ...zone4, rank: 1, priority_score: 84 },
  ],
};

const readyModel = (model: string, mode: "REAL" | "FALLBACK") => ({
  status: "ready" as const,
  model,
  mode,
  version: "integration-test",
  device: "cuda:0",
  latency_ms: 12.5,
  last_successful_inference_ms: 1_725_000_005_000,
  provenance_mode: mode === "REAL" ? "LOCAL_CHECKPOINT" : "PRETRAINED_FALLBACK",
  message: null,
});

export const liveSnapshot: LiveResult = {
  ...commandSnapshot,
  incident_id: "LIVE-frame-session",
  incident: {
    incident_id: "LIVE-frame-session",
    title: "Live Frame Intelligence",
    location_label: "Normalized image-space assessment",
    started_at_ms: 1_725_000_005_000,
    coordinate_space: "NORMALIZED_IMAGE",
    data_origin: "DERIVED_ANALYTIC",
  },
  source_mode: "VIDEO_FILE",
  coordinate_space: "NORMALIZED_IMAGE",
  data_origin: "DERIVED_ANALYTIC",
  stream_state: "PLAYING",
  source_dimensions: { width: 1280, height: 720 },
  evidence_frames: {
    segmentation_source_frame_id: 5,
    detection_source_frame_id: 5,
    segmentation_reused: false,
    detection_reused: false,
  },
  statistics: {
    ...commandSnapshot.statistics,
    flooded_area_percent: {
      ...commandSnapshot.statistics.flooded_area_percent,
      value: 38,
      data_origin: "DERIVED_ANALYTIC",
    },
  },
  system_status: {
    api: "operational",
    segmentation_model: "ready",
    detection_model: "ready",
    inference_state: "LIVE",
    segmentation_details: readyModel("segformer-b2-floodsight", "REAL"),
    detection_details: readyModel("yolo11-pretrained-fallback", "FALLBACK"),
  },
  detections: commandSnapshot.detections.map((detection) => ({
    ...detection,
    data_origin: "REAL_ML_OUTPUT",
    source_class: detection.category === "PERSON" ? "person" : "car",
    model_id: "yolo11-pretrained-fallback",
  })),
  segmentation: {
    status: "ready",
    classes: [
      { class_id: 1, label: "water", coverage_percent: 38, confidence: 0.9, data_origin: "REAL_ML_OUTPUT", color: [14, 165, 233] },
      { class_id: 9, label: "pool", coverage_percent: 4, confidence: 0.88, data_origin: "REAL_ML_OUTPUT", color: [168, 85, 247] },
    ],
    regions: [],
    mask: { encoding: "PNG_BASE64", width: 2, height: 2, data: "iVBORw0KGgo=" },
  },
  roads: commandSnapshot.roads.map((road) => ({ ...road, data_origin: "DERIVED_ANALYTIC" })),
  zones: commandSnapshot.zones.map((zone) => ({
    ...zone,
    data_origin: "DERIVED_ANALYTIC",
    grid_cells: ["B2"],
    building_damage_coverage_percent: 8,
    pool_coverage_percent: 0,
    temporal_samples: 3,
    stale: false,
  })),
  events: commandSnapshot.events.map((event) => ({ ...event, data_origin: "DERIVED_ANALYTIC" })),
  route: commandSnapshot.route && { ...commandSnapshot.route, data_origin: "DERIVED_ANALYTIC", edge_ids: ["E-B2-B3"], route_cost: 2.4 },
  route_alternatives: [],
  scene_summary: {
    water_flood_coverage_percent: 38,
    pool_coverage_percent: 4,
    road_clear_coverage_percent: 12,
    road_flooded_coverage_percent: 6,
    road_blocked_coverage_percent: 2,
    building_damage_coverage_percent: 8,
    provenance: ["SEGMENTATION", "DETECTION", "DERIVED"],
    data_origin: "DERIVED_ANALYTIC",
  },
};

export const incidentDetail: IncidentDetailResponse = {
  incident: commandSnapshot.incident,
  severity: "CRITICAL",
  snapshot_count: 6,
  initial_snapshot: { ...commandSnapshot, frame_id: 0, snapshot_index: 0, stream_state: "CONNECTING" },
  latest_snapshot: commandSnapshot,
  data_origin: origin,
};

export const diagnosticsSnapshot: SystemSnapshot = {
  health: { status: "ok", service: "floodsight-api", version: "0.1.0" },
  models: {
    segmentation: {
      status: "not_configured",
      model: null,
      mode: "UNAVAILABLE",
      version: null,
      device: null,
      latency_ms: null,
      last_successful_inference_ms: null,
      provenance_mode: null,
      message: "No segmentation model loaded.",
    },
    detection: {
      status: "not_configured",
      model: null,
      mode: "UNAVAILABLE",
      version: null,
      device: null,
      latency_ms: null,
      last_successful_inference_ms: null,
      provenance_mode: null,
      message: "No detection model loaded.",
    },
    inference_state: "MODEL_UNAVAILABLE",
  },
  sample: commandSnapshot,
};
