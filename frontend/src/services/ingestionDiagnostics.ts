import type { IngestionMetrics } from "../types/ingestion";

const STORAGE_KEY = "floodsight.ingestion.diagnostics.v1";

export function saveIngestionDiagnostics(metrics: IngestionMetrics): void {
  try {
    const sanitized = {
      ...metrics,
      sessionId: metrics.sessionId ? `${metrics.sessionId.slice(0, 8)}…` : null,
    };
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(sanitized));
  } catch {
    // Diagnostics persistence is best-effort and contains no frame bytes or secrets.
  }
}

export function loadIngestionDiagnostics(): IngestionMetrics | null {
  try {
    const value = window.sessionStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as IngestionMetrics) : null;
  } catch {
    return null;
  }
}
