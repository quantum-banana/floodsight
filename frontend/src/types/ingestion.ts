import type {
  DataOrigin,
  InferenceState,
  LiveResult,
  ModelStatus,
  Route,
  Severity,
  Zone,
} from "./liveResult";

export type ActualSourceMode = "VIDEO_FILE" | "WEBCAM";
export type MediaOrigin = "USER_VIDEO_FILE" | "USER_WEBCAM";
export type DetectorInferenceMode = "STANDARD" | "AERIAL" | "AERIAL_HIGH_RECALL";
export type IngestionSessionState =
  | "READY"
  | "ACTIVE"
  | "IDLE"
  | "FINALIZING"
  | "COMPLETE"
  | "EXPIRED";

export interface SessionCounters {
  frames_received: number;
  frames_accepted: number;
  frames_rejected: number;
  frames_out_of_order: number;
  protocol_errors: number;
  bytes_received: number;
  inference_frames_submitted?: number;
  inference_frames_dropped?: number;
  intelligence_updates_sent?: number;
}

export interface SessionLimits {
  recommended_capture_fps: number;
  jpeg_quality: number;
  max_frame_bytes: number;
  accepted_mime_types: string[];
}

export interface IngestionSession {
  session_id: string;
  source_mode: ActualSourceMode;
  media_origin: MediaOrigin;
  detector_mode: DetectorInferenceMode;
  state: IngestionSessionState;
  created_at_ms: number;
  last_activity_at_ms: number;
  expires_at_ms: number;
  counters: SessionCounters;
  limits: SessionLimits;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface FrameIntelligence {
  type: "frame_intelligence";
  session_id: string;
  frame_id: number;
  sequence: number;
  result: LiveResult;
}

export interface FrameMetadata {
  type: "frame_metadata";
  frame_id: number;
  captured_at_ms: number;
  media_time_ms: number;
  source_mode: ActualSourceMode;
  media_origin: MediaOrigin;
  mime_type: "image/jpeg";
  byte_length: number;
  width: number;
  height: number;
}

export interface FrameQuality {
  mean_luminance: number;
  laplacian_variance: number;
  brightness_status: "NORMAL" | "DARK" | "BRIGHT";
  sharpness_status: "NORMAL" | "BLURRY";
  warnings: string[];
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface FrameResult {
  type: "frame_result";
  session_id: string;
  frame_id: number | null;
  accepted: boolean;
  code: string;
  message: string;
  received_at_ms: number;
  processing_ms: number;
  byte_length: number;
  decoded_frame: { width: number; height: number; channels: number } | null;
  quality: FrameQuality | null;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
  inference_state?: InferenceState | null;
  segmentation_status?: ModelStatus | null;
  detection_status?: ModelStatus | null;
}

export type AggregateMetricAvailability =
  | "AVAILABLE"
  | "MODEL_UNAVAILABLE"
  | "NOT_SUPPORTED"
  | "NO_ANALYZED_FRAMES";

export type AggregateMetricAggregation =
  | "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"
  | "PEAK_FRESH_SEGMENTATION"
  | "NOT_APPLICABLE";

export interface AggregateMetric {
  value: number | null;
  unit: "count" | "percent";
  availability: AggregateMetricAvailability;
  aggregation: AggregateMetricAggregation;
  supporting_frame_count: number;
  confidence: number | null;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface VideoAnalysisStatistics {
  flooded_area_percent: AggregateMetric;
  people_detected: AggregateMetric;
  vehicles_detected: AggregateMetric;
  blocked_road_cells: AggregateMetric;
  damaged_buildings: AggregateMetric;
  building_damage_coverage_percent: AggregateMetric;
}

export interface DetectedClassFinding {
  label: string;
  category: "PERSON" | "VEHICLE" | "OTHER";
  peak_simultaneous_count: number;
  max_confidence: number;
  supporting_frame_count: number;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface VideoPriorityObservation {
  zone: Zone;
  source_frame_id: number;
  media_time_ms: number;
  supporting_update_count: number;
  segmentation_evidence_available: boolean;
  detection_evidence_available: boolean;
  building_damage_count_availability: AggregateMetricAvailability;
  associated_route: Route | null;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface VideoAnalysisSummary {
  session_id: string;
  generated_at_ms: number;
  frames_accepted: number;
  frames_analyzed: number;
  frames_dropped: number;
  first_analyzed_frame_id: number | null;
  last_analyzed_frame_id: number | null;
  first_media_time_ms: number | null;
  last_media_time_ms: number | null;
  statistics: VideoAnalysisStatistics;
  detected_classes: DetectedClassFinding[];
  detected_classes_truncated: boolean;
  priorities: VideoPriorityObservation[];
  priorities_truncated: boolean;
  highest_priority_zone_id: string | null;
  incident_severity: Severity | null;
  segmentation_status: ModelStatus;
  detection_status: ModelStatus;
  inference_state: InferenceState;
  responsible_ai_statement: string;
  data_origin: Extract<DataOrigin, "DERIVED_ANALYTIC">;
}

export interface VideoAnalysisComplete {
  type: "video_analysis_complete";
  session_id: string;
  state: Extract<IngestionSessionState, "COMPLETE">;
  summary: VideoAnalysisSummary;
  latest_result: LiveResult | null;
}

export type IngestionConnectionState =
  | "idle"
  | "creating_session"
  | "connecting"
  | "streaming"
  | "paused"
  | "stopped"
  | "offline"
  | "malformed";

export interface IngestionMetrics {
  sessionId: string | null;
  sessionState: IngestionSessionState | null;
  sourceMode: ActualSourceMode | null;
  mediaOrigin: MediaOrigin | null;
  detectorMode: DetectorInferenceMode | null;
  connectionState: IngestionConnectionState;
  requestedFps: number;
  measuredFps: number;
  capturedFrames: number;
  acknowledgedFrames: number;
  rejectedFrames: number;
  clientDroppedFrames: number;
  latestFrameId: number | null;
  latestDimensions: string | null;
  latestBlurScore: number | null;
  latestLuminance: number | null;
  latestProcessingMs: number | null;
  latestQualityState: string | null;
  lastError: string | null;
  modelStatus: string;
  analysisStatus: InferenceState | "AWAITING_FRAME";
}
