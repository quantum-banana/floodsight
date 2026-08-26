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
    segmentation: { status: "not_configured", model: null },
    detection: { status: "not_configured", model: null },
  },
  sample: commandSnapshot,
};
