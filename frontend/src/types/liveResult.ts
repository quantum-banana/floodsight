export type DataOrigin =
  | "REAL_ML_OUTPUT"
  | "DERIVED_ANALYTIC"
  | "GIS_EXTERNAL_DATA"
  | "DEMO_SIMULATED"
  | "HUMAN_VERIFIED";

export type SourceMode =
  | "VIDEO_FILE"
  | "WEBCAM"
  | "DRONE_STREAM"
  | "DEMO_REPLAY"
  | "SIMULATION";
export type CoordinateSpace = "NORMALIZED_IMAGE" | "RELATIVE_TACTICAL";
export type ModelState = "not_configured" | "loading" | "ready" | "unavailable" | "error";
export type InferenceState =
  | "CONNECTING"
  | "LIVE"
  | "DEGRADED"
  | "MODEL_LOADING"
  | "MODEL_UNAVAILABLE"
  | "SIMULATED_FALLBACK"
  | "ERROR";
export type ModelOperationalMode = "REAL" | "FALLBACK" | "SIMULATED" | "UNAVAILABLE";
export type StreamState = "CONNECTING" | "PLAYING" | "PAUSED" | "COMPLETE";
export type Severity = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export type AccessStatus = "ACCESSIBLE" | "DEGRADED" | "BLOCKED" | "ISOLATED" | "UNKNOWN";
export type RoadState = "CLEAR" | "FLOODED" | "BLOCKED" | "UNKNOWN";

export interface SystemStatus {
  api: "operational" | "degraded" | "offline";
  segmentation_model: ModelState;
  detection_model: ModelState;
  inference_state?: InferenceState | null;
  segmentation_details?: ModelStatus | null;
  detection_details?: ModelStatus | null;
}

export interface ModelStatus {
  status: ModelState;
  model: string | null;
  mode: ModelOperationalMode;
  version: string | null;
  device: string | null;
  latency_ms: number | null;
  last_successful_inference_ms: number | null;
  provenance_mode: string | null;
  message: string | null;
}

export interface IncidentMetadata {
  incident_id: string;
  title: string;
  location_label: string;
  started_at_ms: number;
  coordinate_space: CoordinateSpace;
  data_origin: DataOrigin;
}

export interface Metric {
  value: number;
  unit: "count" | "percent";
  confidence: number | null;
  data_origin: DataOrigin;
}

export interface Statistics {
  flooded_area_percent: Metric;
  people_detected: Metric;
  vehicles_detected: Metric;
  blocked_roads: Metric;
  damaged_buildings: Metric;
}

export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox extends Point {
  width: number;
  height: number;
}

export interface Detection {
  detection_id: string;
  category: "PERSON" | "VEHICLE" | "OTHER";
  label: string;
  confidence: number;
  bbox: BoundingBox;
  data_origin: DataOrigin;
  source_class?: string | null;
  source_class_id?: number | null;
  model_id?: string | null;
  model_provenance?: "REAL_MODEL" | "PRETRAINED_FALLBACK" | "SIMULATED" | null;
}

export interface SegmentationClass {
  class_id?: number | null;
  label: string;
  coverage_percent: number;
  confidence: number;
  data_origin: DataOrigin;
  color?: [number, number, number] | null;
}

export interface OverlayRegion {
  overlay_id: string;
  kind: "FLOOD" | "DAMAGED_BUILDING";
  label: string;
  polygon: Point[];
  confidence: number;
  data_origin: DataOrigin;
}

export interface Segmentation {
  status: "not_configured" | "simulated" | "processing" | "ready" | "error";
  classes: SegmentationClass[];
  regions: OverlayRegion[];
  mask?: {
    encoding: "PNG_BASE64";
    width: number;
    height: number;
    data: string;
  } | null;
}

export interface Road {
  road_id: string;
  label: string;
  state: RoadState;
  access_status: AccessStatus;
  geometry: Point[];
  confidence: number | null;
  data_origin: DataOrigin;
  travel_cost?: number | null;
  enabled?: boolean | null;
  uncertainty?: number | null;
}

export interface ZoneReason {
  code: string;
  label: string;
  description: string;
  contribution: number;
  data_origin: DataOrigin;
}

export interface ZoneAlert {
  code: "POTENTIAL_STRANDED_PERSON";
  title: string;
  person_evidence: "LOW" | "MODERATE" | "HIGH";
  flood_exposure: "LOW" | "MODERATE" | "HIGH";
  primary_access: AccessStatus;
  confidence: number;
  temporal_samples: number;
  reason_codes: string[];
  data_origin: DataOrigin;
}

export interface Zone {
  zone_id: string;
  display_name: string;
  rank: number;
  severity: Severity;
  priority_score: number;
  confidence: number;
  polygon: Point[];
  people_count: number;
  vehicle_count: number;
  flood_coverage_percent: number;
  building_damage_count: number;
  road_condition: RoadState;
  access_status: AccessStatus;
  primary_reason: string;
  reasons: ZoneReason[];
  updated_at_ms: number;
  data_origin: DataOrigin;
  grid_cells?: string[];
  building_damage_coverage_percent?: number;
  pool_coverage_percent?: number;
  temporal_samples?: number;
  stale?: boolean;
  alerts?: ZoneAlert[];
}

export interface IncidentEvent {
  event_id: string;
  timestamp_ms: number;
  severity: "INFO" | "WARNING" | "CRITICAL";
  category: "FLOOD" | "DETECTION" | "ACCESS" | "PRIORITY" | "ROUTE" | "SYSTEM";
  message: string;
  data_origin: DataOrigin;
  code?: string | null;
}

export interface Route {
  route_id: string;
  status: "RECOMMENDED" | "UNAVAILABLE";
  target_zone_id: string;
  label: string;
  waypoints: Point[];
  distance_m: number | null;
  access_summary: string;
  data_origin: DataOrigin;
  edge_ids?: string[];
  route_cost?: number | null;
  changed_reason?: string | null;
  changed_reason_code?: string | null;
  previous_edge_ids?: string[];
}

export interface SceneSummary {
  water_flood_coverage_percent: number;
  pool_coverage_percent: number;
  road_clear_coverage_percent: number;
  road_flooded_coverage_percent: number;
  road_blocked_coverage_percent: number;
  building_damage_coverage_percent: number;
  provenance: Array<"SEGMENTATION" | "DETECTION" | "DERIVED" | "SIMULATED">;
  data_origin: DataOrigin;
}

export interface LiveResult {
  incident_id: string;
  incident: IncidentMetadata;
  frame_id: number;
  snapshot_index: number;
  snapshot_count: number;
  timestamp_ms: number;
  source_mode: SourceMode;
  coordinate_space: CoordinateSpace;
  data_origin: DataOrigin;
  stream_state: StreamState;
  incident_severity: Severity;
  highest_priority_zone_id: string | null;
  system_status: SystemStatus;
  statistics: Statistics;
  detections: Detection[];
  segmentation: Segmentation;
  roads: Road[];
  zones: Zone[];
  events: IncidentEvent[];
  route: Route | null;
  route_alternatives?: Route[];
  scene_summary?: SceneSummary | null;
  source_dimensions?: { width: number; height: number } | null;
  evidence_frames?: {
    segmentation_source_frame_id: number | null;
    detection_source_frame_id: number | null;
    segmentation_reused: boolean;
    detection_reused: boolean;
  } | null;
}
