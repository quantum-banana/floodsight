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
export type ModelState = "not_configured" | "loading" | "ready" | "error";
export type StreamState = "CONNECTING" | "PLAYING" | "PAUSED" | "COMPLETE";
export type Severity = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export type AccessStatus = "ACCESSIBLE" | "DEGRADED" | "BLOCKED" | "ISOLATED" | "UNKNOWN";
export type RoadState = "CLEAR" | "FLOODED" | "BLOCKED" | "UNKNOWN";

export interface SystemStatus {
  api: "operational" | "degraded" | "offline";
  segmentation_model: ModelState;
  detection_model: ModelState;
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
}

export interface SegmentationClass {
  label: string;
  coverage_percent: number;
  confidence: number;
  data_origin: DataOrigin;
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
}

export interface Road {
  road_id: string;
  label: string;
  state: RoadState;
  access_status: AccessStatus;
  geometry: Point[];
  confidence: number | null;
  data_origin: DataOrigin;
}

export interface ZoneReason {
  code: string;
  label: string;
  description: string;
  contribution: number;
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
}

export interface IncidentEvent {
  event_id: string;
  timestamp_ms: number;
  severity: "INFO" | "WARNING" | "CRITICAL";
  category: "FLOOD" | "DETECTION" | "ACCESS" | "PRIORITY" | "ROUTE" | "SYSTEM";
  message: string;
  data_origin: DataOrigin;
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
}

