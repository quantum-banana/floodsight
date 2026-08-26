import type { LiveResult, ModelState } from "./liveResult";

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

