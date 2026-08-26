import type { DataOrigin, Severity } from "../types/liveResult";

export function formatTimestamp(timestampMs: number): string {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }).format(new Date(timestampMs));
}

export function formatOrigin(origin: DataOrigin): string {
  return origin;
}

export function severityLabel(severity: Severity): string {
  return severity.charAt(0) + severity.slice(1).toLowerCase();
}

export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}
