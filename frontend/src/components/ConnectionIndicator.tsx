import type { ConnectionState } from "../hooks/useDemoIncident";

interface ConnectionIndicatorProps {
  state: ConnectionState;
  label?: string;
}

const labels: Record<ConnectionState, string> = {
  loading: "Loading",
  connecting: "Connecting",
  connected: "Replaying",
  paused: "Paused",
  reconnecting: "Reconnecting",
  complete: "Replay complete",
  offline: "Backend offline",
  malformed: "Invalid message",
  disconnected: "Disconnected",
};

export function ConnectionIndicator({ state, label }: ConnectionIndicatorProps) {
  const healthy = state === "connected" || state === "complete";
  const waiting = state === "connecting" || state === "reconnecting" || state === "loading";
  const tone = healthy ? "emerald" : waiting || state === "paused" ? "amber" : "rose";
  const toneClasses = {
    emerald: "border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-300",
    amber: "border-amber-300/20 bg-amber-300/[0.08] text-amber-200",
    rose: "border-rose-400/20 bg-rose-400/[0.08] text-rose-300",
  }[tone];
  const dotClasses = {
    emerald: "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.65)]",
    amber: "bg-amber-300",
    rose: "bg-rose-400",
  }[tone];

  return (
    <span
      role="status"
      aria-label={`Stream connection: ${label ?? labels[state]}`}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[0.66rem] font-semibold tracking-[0.12em] uppercase ${toneClasses}`}
    >
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full ${dotClasses} ${waiting ? "animate-pulse" : ""}`}
      />
      {label ?? labels[state]}
    </span>
  );
}
