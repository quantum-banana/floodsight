import type {
  DataOrigin,
  InferenceState,
  LiveResult,
  ModelStatus,
} from "./liveResult";

export type ActualSourceMode = "VIDEO_FILE" | "WEBCAM";
export type MediaOrigin = "USER_VIDEO_FILE" | "USER_WEBCAM";
export type IngestionSessionState = "READY" | "ACTIVE" | "IDLE" | "EXPIRED";

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
