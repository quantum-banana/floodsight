import { Icon } from "../../components/Icon";
import { LAYER_LABELS, type LayerKey, type LayerState } from "./layers";

interface LayerControlsProps {
  layers: LayerState;
  onToggle: (layer: LayerKey) => void;
  compact?: boolean;
}

const primaryLayers: LayerKey[] = ["flood", "people", "vehicles", "zones", "route"];
const secondaryLayers: LayerKey[] = ["roads", "buildings"];
const layerIcons: Record<LayerKey, "water" | "road" | "people" | "vehicle" | "building" | "focus" | "route"> = {
  flood: "water",
  roads: "road",
  people: "people",
  vehicles: "vehicle",
  buildings: "building",
  zones: "focus",
  route: "route",
};

export function LayerControls({ layers, onToggle }: LayerControlsProps) {
  const renderButton = (layer: LayerKey) => (
    <button
      key={layer}
      type="button"
      aria-label={LAYER_LABELS[layer]}
      aria-pressed={layers[layer]}
      onClick={() => onToggle(layer)}
      className={`layer-tool ${layers[layer] ? "layer-tool-active" : ""}`}
      title={LAYER_LABELS[layer]}
    >
      <Icon name={layerIcons[layer]} />
      <span>{LAYER_LABELS[layer]}</span>
    </button>
  );

  return (
    <fieldset className="layer-tools" aria-label="Map layers">
      <legend className="sr-only">Visible tactical layers</legend>
      <span className="layer-tools-label">Layers</span>
      {primaryLayers.map(renderButton)}
      <details className="layer-more">
        <summary aria-label="More layers" title="More layers">•••</summary>
        <div>{secondaryLayers.map(renderButton)}</div>
      </details>
    </fieldset>
  );
}
