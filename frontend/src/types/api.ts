import type {
  CoordinateSpace,
  DataOrigin,
  IncidentMetadata,
  LiveResult,
  ModelState,
  Severity,
  SourceMode,
  Statistics,
} from "./liveResult";

export interface HealthResponse {
  status: "ok";
  service: "floodsight-api";
  version: string;
}

export interface ModelStatus {
  status: ModelState;
  model: string | null;
}

export interface ModelStatusResponse {
  segmentation: ModelStatus;
  detection: ModelStatus;
}

export interface SystemSnapshot {
  health: HealthResponse;
  models: ModelStatusResponse;
  sample: LiveResult;
}

export interface IncidentSummary {
  incident_id: string;
  title: string;
  severity: Severity;
  source_mode: SourceMode;
  coordinate_space: CoordinateSpace;
  snapshot_count: number;
  data_origin: DataOrigin;
}

export interface IncidentListResponse {
  incidents: IncidentSummary[];
  data_origin: DataOrigin;
}

export interface IncidentDetailResponse {
  incident: IncidentMetadata;
  severity: Severity;
  snapshot_count: number;
  initial_snapshot: LiveResult;
  latest_snapshot: LiveResult;
  data_origin: DataOrigin;
}

export interface IncidentReport {
  incident_id: string;
  title: string;
  generated_at_ms: number;
  severity: Severity;
  statistics: Statistics;
  critical_zone_count: number;
  highest_priority_zone_id: string;
  highest_priority_zone_name: string;
  explanation: string;
  access_summary: string;
  responsible_ai_statement: string;
  data_origin: DataOrigin;
}
