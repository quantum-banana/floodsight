import { Icon } from "../../components/Icon";
import { LAYER_LABELS, type LayerKey, type LayerState } from "./layers";

interface LayerControlsProps {
  layers: LayerState;
  onToggle: (layer: LayerKey) => void;
  compact?: boolean;
}

export function LayerControls({ layers, onToggle, compact = false }: LayerControlsProps) {
  return (
    <fieldset className="flex flex-wrap gap-1.5" aria-label="Map layers">
      <legend className="sr-only">Visible tactical layers</legend>
      {(Object.keys(LAYER_LABELS) as LayerKey[]).map((layer) => (
        <button
          key={layer}
          type="button"
          aria-pressed={layers[layer]}
          onClick={() => onToggle(layer)}
          className={`inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[0.66rem] font-medium transition focus-visible:outline-2 focus-visible:outline-cyan-300 ${
            layers[layer]
              ? "border-cyan-300/20 bg-cyan-300/[0.08] text-cyan-200"
              : "border-white/[0.06] bg-white/[0.02] text-slate-600"
          } ${compact && layer === "route" ? "hidden sm:inline-flex" : ""}`}
        >
          <span className={`grid h-3 w-3 place-items-center rounded-sm border ${layers[layer] ? "border-cyan-300/40 bg-cyan-300/20" : "border-slate-700"}`}>
            {layers[layer] && <Icon name="check" className="h-2.5 w-2.5" />}
          </span>
          {LAYER_LABELS[layer]}
        </button>
      ))}
    </fieldset>
  );
}
