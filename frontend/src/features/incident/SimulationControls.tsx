import { Icon } from "../../components/Icon";
import type { ConnectionState } from "../../hooks/useDemoIncident";

interface SimulationControlsProps {
  state: ConnectionState;
  snapshotIndex: number;
  snapshotCount: number;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onReset: () => void;
}

export function SimulationControls({
  state,
  snapshotIndex,
  snapshotCount,
  onStart,
  onPause,
  onResume,
  onReset,
}: SimulationControlsProps) {
  const canPause = state === "connected" || state === "connecting";
  const canResume = state === "paused" || state === "complete";

  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Simulation controls">
      <span className="mr-1 font-mono text-[0.66rem] text-slate-500">
        STEP {snapshotIndex + 1}/{snapshotCount}
      </span>
      <button type="button" onClick={onStart} className="command-control" aria-label="Start simulation">
        <Icon name="play" /> Start
      </button>
      {canPause ? (
        <button type="button" onClick={onPause} className="command-control" aria-label="Pause simulation">
          <Icon name="pause" /> Pause
        </button>
      ) : (
        <button type="button" onClick={onResume} disabled={!canResume} className="command-control" aria-label="Resume simulation">
          <Icon name="play" /> Resume
        </button>
      )}
      <button type="button" onClick={onReset} className="command-control" aria-label="Reset simulation">
        <Icon name="reset" /> Reset
      </button>
    </div>
  );
}
