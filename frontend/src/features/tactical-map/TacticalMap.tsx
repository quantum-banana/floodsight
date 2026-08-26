import { Icon } from "../../components/Icon";
import { OriginBadge } from "../../components/OriginBadge";
import type { LiveResult } from "../../types/liveResult";
import { toSvgPath, toSvgPoints } from "../../utils/geometry";
import { LayerControls } from "./LayerControls";
import type { LayerKey, LayerState } from "./layers";

interface TacticalMapProps {
  snapshot: LiveResult;
  layers: LayerState;
  selectedZoneId: string | null;
  onToggleLayer: (layer: LayerKey) => void;
  onSelectZone: (zoneId: string) => void;
}

const roadStroke = (state: string) =>
  state === "BLOCKED" ? "#fb7185" : state === "FLOODED" ? "#38bdf8" : "#34d399";

const severityStroke = (severity: string) =>
  severity === "CRITICAL" ? "#fb4f64" : severity === "HIGH" ? "#fb923c" : severity === "MODERATE" ? "#facc15" : "#34d399";

export function TacticalMap({
  snapshot,
  layers,
  selectedZoneId,
  onToggleLayer,
  onSelectZone,
}: TacticalMapProps) {
  return (
    <section id="tactical-map" className="command-panel min-w-0 overflow-hidden" aria-labelledby="tactical-heading">
      <div className="panel-heading flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2"><Icon name="map" className="h-4 w-4 text-cyan-300" /><h2 id="tactical-heading" className="panel-title">Relative tactical view</h2></div>
          <p className="panel-subtitle">No geographic scale, distance, or travel time</p>
        </div>
        <OriginBadge origin={snapshot.data_origin} compact />
      </div>
      <div className="border-b border-white/[0.06] px-4 py-2.5"><LayerControls layers={layers} onToggle={onToggleLayer} compact /></div>

      <div className="relative aspect-[16/9] min-h-64 overflow-hidden bg-[#08131a]">
        <div aria-hidden="true" className="tactical-grid absolute inset-0" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" role="img" aria-label="Relative tactical map">
          <title>Relative tactical route and rescue zones</title>
          {layers.flood && snapshot.segmentation.regions.filter((region) => region.kind === "FLOOD").map((region) => (
            <polygon key={region.overlay_id} points={toSvgPoints(region.polygon)} fill="#0284c7" fillOpacity="0.18" stroke="#38bdf8" strokeOpacity="0.45" strokeWidth="0.4" />
          ))}
          {layers.roads && snapshot.roads.map((road) => (
            <g key={road.road_id}>
              <path d={toSvgPath(road.geometry)} fill="none" stroke="#020617" strokeWidth="3.2" strokeLinecap="round" />
              <path d={toSvgPath(road.geometry)} fill="none" stroke={roadStroke(road.state)} strokeWidth="1.15" strokeLinecap="round" strokeDasharray={road.state === "BLOCKED" ? "2 1.2" : undefined} />
            </g>
          ))}
          {layers.zones && snapshot.zones.map((zone) => {
            const selected = zone.zone_id === selectedZoneId;
            const color = severityStroke(zone.severity);
            const anchor = zone.polygon[0] ?? { x: 0, y: 0 };
            return (
              <g key={zone.zone_id} className="cursor-pointer" onClick={() => onSelectZone(zone.zone_id)}>
                <polygon points={toSvgPoints(zone.polygon)} fill={color} fillOpacity={selected ? 0.3 : 0.12} stroke={color} strokeWidth={selected ? 1.25 : 0.7} />
                <circle cx={anchor.x * 100 + 2} cy={anchor.y * 100 + 2} r="3.4" fill="#071016" stroke={color} strokeWidth="0.55" />
                <text x={anchor.x * 100 + 2} y={anchor.y * 100 + 2.8} fill={color} textAnchor="middle" fontSize="2.5" fontWeight="800">{zone.rank}</text>
              </g>
            );
          })}
          {layers.buildings && snapshot.segmentation.regions.filter((region) => region.kind === "DAMAGED_BUILDING").map((region) => (
            <polygon key={region.overlay_id} points={toSvgPoints(region.polygon)} fill="#fb923c" fillOpacity="0.55" stroke="#fed7aa" strokeWidth="0.4" />
          ))}
          {(layers.people || layers.vehicles) && snapshot.detections.map((detection) => {
            if (detection.category === "PERSON" && !layers.people) return null;
            if (detection.category === "VEHICLE" && !layers.vehicles) return null;
            const person = detection.category === "PERSON";
            const x = (detection.bbox.x + detection.bbox.width / 2) * 100;
            const y = (detection.bbox.y + detection.bbox.height / 2) * 100;
            return person ? <circle key={detection.detection_id} cx={x} cy={y} r="1.25" fill="#f8fafc" stroke="#071016" strokeWidth="0.45" /> : <rect key={detection.detection_id} x={x - 1.4} y={y - 1} width="2.8" height="2" rx="0.35" fill="#67e8f9" stroke="#071016" strokeWidth="0.4" />;
          })}
          {layers.route && snapshot.route && (
            <g>
              <path d={toSvgPath(snapshot.route.waypoints)} fill="none" stroke="#f8fafc" strokeOpacity="0.35" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d={toSvgPath(snapshot.route.waypoints)} fill="none" stroke="#67e8f9" strokeWidth="0.9" strokeDasharray="2 1.5" strokeLinecap="round" strokeLinejoin="round" className="route-path" />
            </g>
          )}
          <g transform="translate(7 86)"><circle r="3.5" fill="#071016" stroke="#67e8f9" strokeWidth="0.6" /><path d="M-1.7 0h3.4M0-1.7v3.4" stroke="#67e8f9" strokeWidth="0.7" /><text x="5" y="1" fill="#a5f3fc" fontSize="2.5" fontWeight="700">RESCUE BASE</text></g>
        </svg>
        {!snapshot.route && <div className="absolute right-3 bottom-3 rounded-lg border border-white/[0.07] bg-[#071016]/85 px-3 py-2 text-[0.65rem] text-slate-500">Route pending in this snapshot</div>}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-white/[0.06] px-4 py-3 text-[0.62rem] font-medium tracking-wide text-slate-500 uppercase">
        <Legend color="bg-sky-400" label="Flood" /><Legend color="bg-rose-400" label="Critical" /><Legend color="bg-orange-400" label="High" /><Legend color="bg-yellow-400" label="Moderate" /><Legend color="bg-emerald-400" label="Accessible" /><Legend color="bg-cyan-200" label="Relative route" />
      </div>
    </section>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span className="inline-flex items-center gap-1.5"><span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${color}`} />{label}</span>;
}
