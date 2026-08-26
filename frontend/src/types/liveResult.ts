export type DataOrigin =
  | "REAL_ML_OUTPUT"
  | "DERIVED_ANALYTIC"
  | "GIS_EXTERNAL_DATA"
  | "DEMO_SIMULATED"
  | "HUMAN_VERIFIED";

export type SourceMode = "VIDEO_FILE" | "WEBCAM" | "DRONE_STREAM" | "DEMO_REPLAY";
export type ModelState = "not_configured" | "loading" | "ready" | "error";

export interface SystemStatus {
  api: "operational" | "degraded" | "offline";
  segmentation_model: ModelState;
  detection_model: ModelState;
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

export interface Segmentation {
  status: "not_configured" | "processing" | "ready" | "error";
  classes: SegmentationClass[];
}

export interface Road {
  road_id: string;
  state: "CLEAR" | "FLOODED" | "BLOCKED" | "UNKNOWN";
  confidence: number | null;
  data_origin: DataOrigin;
}

export interface ZoneReason {
  label: string;
  contribution: number;
  data_origin: DataOrigin;
}

export interface Zone {
  zone_id: string;
  severity: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  priority_score: number;
  polygon: Point[];
  reasons: ZoneReason[];
  data_origin: DataOrigin;
}

export interface IncidentEvent {
  event_id: string;
  timestamp_ms: number;
  severity: "INFO" | "WARNING" | "CRITICAL";
  message: string;
  data_origin: DataOrigin;
}

export interface Route {
  route_id: string;
  status: "RECOMMENDED" | "UNAVAILABLE";
  waypoints: Point[];
  distance_m: number | null;
  data_origin: DataOrigin;
}

export interface LiveResult {
  incident_id: string;
  frame_id: number;
  timestamp_ms: number;
  source_mode: SourceMode;
  data_origin: DataOrigin;
  system_status: SystemStatus;
  statistics: Statistics;
  detections: Detection[];
  segmentation: Segmentation;
  roads: Road[];
  zones: Zone[];
  events: IncidentEvent[];
  route: Route | null;
}

