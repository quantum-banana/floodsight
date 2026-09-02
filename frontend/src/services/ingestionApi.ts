import { API_BASE_URL, API_TIMEOUT_MS } from "../config/environment";
import type { IncidentReport } from "../types/api";
import type {
  ActualSourceMode,
  DetectorInferenceMode,
  IngestionSession,
  MediaOrigin,
  VideoAnalysisComplete,
} from "../types/ingestion";
import { ApiError } from "./api";

const VIDEO_COMPLETION_TIMEOUT_MS = 60_000;

async function ingestionRequest<T>(
  path: string,
  init: RequestInit,
  timeoutMs = API_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
      },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new ApiError(`FloodSight ingestion API returned ${response.status}.`, response.status);
    }
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("FloodSight ingestion API did not respond in time.");
    }
    throw new ApiError("Unable to reach the FloodSight ingestion API.");
  } finally {
    window.clearTimeout(timeout);
  }
}

export function createIngestionSession(
  sourceMode: ActualSourceMode,
  mediaOrigin: MediaOrigin,
  detectorMode: DetectorInferenceMode,
): Promise<IngestionSession> {
  return ingestionRequest<IngestionSession>("/api/ingest/sessions", {
    method: "POST",
    body: JSON.stringify({
      source_mode: sourceMode,
      media_origin: mediaOrigin,
      detector_mode: detectorMode,
    }),
  });
}

export function getIngestionSession(sessionId: string): Promise<IngestionSession> {
  return ingestionRequest<IngestionSession>(
    `/api/ingest/sessions/${encodeURIComponent(sessionId)}`,
    { method: "GET" },
  );
}

export function deleteIngestionSession(sessionId: string): Promise<void> {
  return ingestionRequest<void>(
    `/api/ingest/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE", keepalive: true },
  );
}

export function completeIngestionSession(sessionId: string): Promise<VideoAnalysisComplete> {
  return ingestionRequest<VideoAnalysisComplete>(
    `/api/ingest/sessions/${encodeURIComponent(sessionId)}/complete`,
    { method: "POST" },
    VIDEO_COMPLETION_TIMEOUT_MS,
  );
}

export function getLiveIncidentReport(sessionId: string): Promise<IncidentReport> {
  return ingestionRequest<IncidentReport>(
    `/api/ingest/sessions/${encodeURIComponent(sessionId)}/report`,
    { method: "GET" },
  );
}
