import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { LiveResult } from "../../types/liveResult";
import { formatTimestamp } from "../../utils/format";
import { LayerControls } from "../tactical-map/LayerControls";
import type { LayerKey, LayerState } from "../tactical-map/layers";
import { OverlayRenderer } from "./OverlayRenderer";

interface ObservationPanelProps {
  snapshot: LiveResult;
  layers: LayerState;
  selectedZoneId: string | null;
  onToggleLayer: (layer: LayerKey) => void;
  onSelectZone: (zoneId: string) => void;
}

export function ObservationPanel({
  snapshot,
  layers,
  selectedZoneId,
  onToggleLayer,
  onSelectZone,
}: ObservationPanelProps) {
  return (
    <section className="command-panel min-w-0 overflow-hidden" aria-labelledby="observation-heading">
      <div className="panel-heading flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="status-pulse" aria-hidden="true" />
            <h2 id="observation-heading" className="panel-title">Simulated sensor view</h2>
          </div>
          <p className="panel-subtitle">Normalized scene evidence · Snapshot {snapshot.snapshot_index + 1}</p>
        </div>
        <div className="flex items-center gap-2"><span className="font-mono text-[0.65rem] text-slate-500">{formatTimestamp(snapshot.timestamp_ms)} UTC</span><OriginBadge origin={snapshot.data_origin} compact /></div>
      </div>

      <div className="border-b border-white/[0.06] px-4 py-2.5"><LayerControls layers={layers} onToggle={onToggleLayer} compact /></div>

      <div className="sensor-scene relative aspect-[16/8.7] min-h-64 overflow-hidden bg-[#08141a]">
        <OverlayRenderer snapshot={snapshot} layers={layers} selectedZoneId={selectedZoneId} onSelectZone={onSelectZone} />
        <div aria-hidden="true" className="sensor-scan-line" />
        <div className="pointer-events-none absolute right-3 bottom-3 flex items-center gap-2 rounded-lg border border-white/[0.08] bg-[#071016]/80 px-2.5 py-1.5 font-mono text-[0.6rem] tracking-wider text-slate-500 uppercase backdrop-blur">
          <Icon name="eye" className="h-3 w-3 text-cyan-300" /> Synthetic geometry only
        </div>
      </div>
    </section>
  );
}
