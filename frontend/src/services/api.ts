import { API_BASE_URL, API_TIMEOUT_MS } from "../config/environment";
import type {
  HealthResponse,
  IncidentDetailResponse,
  IncidentListResponse,
  IncidentReport,
  ModelStatusResponse,
  SystemSnapshot,
} from "../types/api";
import type { LiveResult } from "../types/liveResult";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new ApiError(`FloodSight API returned ${response.status}.`, response.status);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("FloodSight API did not respond in time.");
    }
    throw new ApiError("Unable to reach the FloodSight API.");
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function getSystemSnapshot(): Promise<SystemSnapshot> {
  const [health, models, sample] = await Promise.all([
    request<HealthResponse>("/health"),
    request<ModelStatusResponse>("/api/models/status"),
    request<LiveResult>("/api/demo/live-result"),
  ]);

  return { health, models, sample };
}

export function getDemoIncidents(): Promise<IncidentListResponse> {
  return request<IncidentListResponse>("/api/demo/incidents");
}

export function getDemoIncident(incidentId = "FS-001"): Promise<IncidentDetailResponse> {
  return request<IncidentDetailResponse>(`/api/demo/incidents/${encodeURIComponent(incidentId)}`);
}

export function getDemoIncidentReport(incidentId = "FS-001"): Promise<IncidentReport> {
  return request<IncidentReport>(
    `/api/demo/incidents/${encodeURIComponent(incidentId)}/report`,
  );
}

