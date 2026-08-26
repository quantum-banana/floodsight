export type LayerKey = "flood" | "roads" | "people" | "vehicles" | "buildings" | "zones" | "route";
export type LayerState = Record<LayerKey, boolean>;

export const DEFAULT_LAYERS: LayerState = {
  flood: true,
  roads: true,
  people: true,
  vehicles: true,
  buildings: true,
  zones: true,
  route: true,
};

export const LAYER_LABELS: Record<LayerKey, string> = {
  flood: "Flood",
  roads: "Roads",
  people: "People",
  vehicles: "Vehicles",
  buildings: "Buildings",
  zones: "Rescue zones",
  route: "Route",
};
