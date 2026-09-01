import type { LiveResult } from "../../types/liveResult";
import { toSvgPath, toSvgPoints } from "../../utils/geometry";
import type { LayerState } from "../tactical-map/layers";

interface OverlayRendererProps {
  snapshot: LiveResult;
  layers: LayerState;
  selectedZoneId: string | null;
  onSelectZone: (zoneId: string) => void;
  showBase?: boolean;
  segmentationOpacity?: number;
  simulated?: boolean;
}

const roadColor = (state: string) => {
  if (state === "BLOCKED") return "#fb7185";
  if (state === "FLOODED") return "#38bdf8";
  if (state === "UNKNOWN") return "#fbbf24";
  return "#34d399";
};

const zoneColor = (severity: string) => {
  if (severity === "CRITICAL") return "#fb4f64";
  if (severity === "HIGH") return "#fb923c";
  if (severity === "MODERATE") return "#facc15";
  return "#34d399";
};

export function OverlayRenderer({
  snapshot,
  layers,
  selectedZoneId,
  onSelectZone,
  showBase = true,
  segmentationOpacity = 0.42,
  simulated = true,
}: OverlayRendererProps) {
  const mask = snapshot.segmentation.mask;
  const dimensions = snapshot.source_dimensions;
  const viewportHeight = !showBase && dimensions
    ? 100 * dimensions.height / dimensions.width
    : 100;
  return (
    <svg
      viewBox={`0 0 100 ${viewportHeight}`}
      preserveAspectRatio={showBase ? "none" : "xMidYMid meet"}
      className="absolute inset-0 h-full w-full"
      role="img"
      aria-label={`Normalized ${simulated ? "simulated" : "model-derived"} flood observation with tactical overlays`}
    >
      <title>{simulated ? "Simulated" : "Model-derived"} sensor overlay for {snapshot.incident.title}</title>
      <defs>
        <pattern id="blocked-hatch" width="3" height="3" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="3" stroke="#fecdd3" strokeWidth="0.8" />
        </pattern>
        <filter id="zone-glow"><feGaussianBlur stdDeviation="0.65" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>

      <g transform={`scale(1 ${viewportHeight / 100})`}>

      {showBase && <g className="scene-base" opacity="0.72">
        <path d="M0 0H100V100H0Z" fill="#0b1820" />
        <path d="M0 18 22 8l19 14 17-11 20 10 22-8v34L80 56 58 48 34 60 12 48 0 53Z" fill="#10262b" />
        <g fill="#193139" stroke="#274650" strokeWidth="0.25">
          <rect x="12" y="18" width="9" height="7" rx="1" /><rect x="25" y="15" width="8" height="10" rx="1" />
          <rect x="38" y="26" width="10" height="7" rx="1" /><rect x="69" y="10" width="11" height="8" rx="1" />
          <rect x="82" y="23" width="8" height="10" rx="1" /><rect x="15" y="38" width="10" height="7" rx="1" />
          <rect x="30" y="36" width="9" height="8" rx="1" /><rect x="75" y="43" width="10" height="8" rx="1" />
          <rect x="12" y="64" width="8" height="8" rx="1" /><rect x="83" y="70" width="9" height="9" rx="1" />
        </g>
      </g>}

      {layers.flood && mask && (
        <image
          href={`data:image/png;base64,${mask.data}`}
          x="0"
          y="0"
          width="100"
          height="100"
          preserveAspectRatio="none"
          opacity={segmentationOpacity}
        />
      )}

      {layers.flood && (
        <g>
          {snapshot.segmentation.regions.filter((region) => region.kind === "FLOOD").map((region) => (
            <polygon key={region.overlay_id} points={toSvgPoints(region.polygon)} fill="#0891b2" fillOpacity="0.3" stroke="#38bdf8" strokeOpacity="0.75" strokeWidth="0.45" />
          ))}
        </g>
      )}

      {layers.roads && (
        <g fill="none">
          {snapshot.roads.map((road) => (
            <g key={road.road_id}>
              <path d={toSvgPath(road.geometry)} stroke="#020617" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />
              <path d={toSvgPath(road.geometry)} stroke={roadColor(road.state)} strokeWidth={road.state === "BLOCKED" ? 1.25 : 0.9} strokeDasharray={road.state === "BLOCKED" ? "2 1.3" : undefined} strokeLinecap="round" strokeLinejoin="round" />
              <text x={road.geometry[1]?.x ? road.geometry[1].x * 100 : 0} y={road.geometry[1]?.y ? road.geometry[1].y * 100 - 1.3 : 0} fill="#cbd5e1" fontSize="2.2" paintOrder="stroke" stroke="#071016" strokeWidth="0.7">{road.road_id}</text>
            </g>
          ))}
        </g>
      )}

      {layers.buildings && snapshot.segmentation.regions.filter((region) => region.kind === "DAMAGED_BUILDING").map((region) => (
        <polygon key={region.overlay_id} points={toSvgPoints(region.polygon)} fill="#fb923c" fillOpacity="0.35" stroke="#fdba74" strokeWidth="0.55" />
      ))}

      {(layers.people || layers.vehicles) && snapshot.detections.map((detection) => {
        if (detection.category === "PERSON" && !layers.people) return null;
        if (detection.category === "VEHICLE" && !layers.vehicles) return null;
        const person = detection.category === "PERSON";
        const color = person ? "#f8fafc" : "#67e8f9";
        return (
          <g key={detection.detection_id}>
            <rect x={detection.bbox.x * 100} y={detection.bbox.y * 100} width={detection.bbox.width * 100} height={detection.bbox.height * 100} fill="none" stroke={color} strokeWidth="0.55" />
            <rect x={detection.bbox.x * 100} y={detection.bbox.y * 100 - 3.1} width={person ? 7.2 : 8.8} height="3.1" fill={color} />
            <text x={detection.bbox.x * 100 + 0.6} y={detection.bbox.y * 100 - 0.8} fill="#071016" fontSize="1.7" fontWeight="700">{person ? "PERSON" : "VEHICLE"} {Math.round(detection.confidence * 100)}</text>
          </g>
        );
      })}

      {layers.zones && snapshot.zones.map((zone) => {
        const selected = selectedZoneId === zone.zone_id;
        const color = zoneColor(zone.severity);
        const anchor = zone.polygon[0];
        return (
          <g key={zone.zone_id} className="cursor-pointer" role="button" tabIndex={0} aria-label={`View ${zone.display_name}`} onClick={() => onSelectZone(zone.zone_id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelectZone(zone.zone_id); }}>
            <polygon points={toSvgPoints(zone.polygon)} fill={color} fillOpacity={selected ? 0.22 : 0.1} stroke={color} strokeWidth={selected ? 1.1 : 0.65} strokeDasharray={selected ? undefined : "2 1"} filter={selected ? "url(#zone-glow)" : undefined} />
            <rect x={anchor.x * 100} y={anchor.y * 100 - 5} width="14" height="4.4" rx="1" fill="#071016" fillOpacity="0.88" stroke={color} strokeWidth="0.35" />
            <text x={anchor.x * 100 + 1} y={anchor.y * 100 - 2} fill={color} fontSize="2.2" fontWeight="700">{zone.display_name.toUpperCase()} · {zone.priority_score}</text>
          </g>
        );
      })}
      </g>
    </svg>
  );
}
