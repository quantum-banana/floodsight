import { API_BASE_URL, API_TIMEOUT_MS } from "../config/environment";
import type { ActualSourceMode, IngestionSession, MediaOrigin } from "../types/ingestion";
import { ApiError } from "./api";

async function ingestionRequest<T>(path: string, init: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);
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
): Promise<IngestionSession> {
  return ingestionRequest<IngestionSession>("/api/ingest/sessions", {
    method: "POST",
    body: JSON.stringify({ source_mode: sourceMode, media_origin: mediaOrigin }),
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
