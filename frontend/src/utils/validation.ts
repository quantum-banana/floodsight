import type { LiveResult } from "../types/liveResult";

type UnknownRecord = Record<string, unknown>;

const origins = [
  "REAL_ML_OUTPUT",
  "DERIVED_ANALYTIC",
  "GIS_EXTERNAL_DATA",
  "DEMO_SIMULATED",
  "HUMAN_VERIFIED",
] as const;
const severities = ["LOW", "MODERATE", "HIGH", "CRITICAL"] as const;
const accessStates = ["ACCESSIBLE", "DEGRADED", "BLOCKED", "ISOLATED", "UNKNOWN"] as const;
const roadStates = ["CLEAR", "FLOODED", "BLOCKED", "UNKNOWN"] as const;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isString = (value: unknown): value is string => typeof value === "string";
const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const isNullableNumber = (value: unknown): value is number | null =>
  value === null || isNumber(value);
const isOneOf = <T extends readonly string[]>(value: unknown, choices: T): value is T[number] =>
  isString(value) && choices.includes(value);
const isOrigin = (value: unknown) => isOneOf(value, origins);
const isOriginRecord = (value: UnknownRecord) => isOrigin(value.data_origin);

const isPoint = (value: unknown): value is UnknownRecord & { x: number; y: number } =>
  isRecord(value) && isNumber(value.x) && isNumber(value.y);
const isPointList = (value: unknown) => Array.isArray(value) && value.every(isPoint);
const isBoundingBox = (value: unknown) =>
  isPoint(value) && isNumber(value.width) && isNumber(value.height);

const isMetric = (value: unknown) =>
  isRecord(value) &&
  isNumber(value.value) &&
  isOneOf(value.unit, ["count", "percent"] as const) &&
  isNullableNumber(value.confidence) &&
  isOriginRecord(value);

const isStatistics = (value: unknown) =>
  isRecord(value) &&
  isMetric(value.flooded_area_percent) &&
  isMetric(value.people_detected) &&
  isMetric(value.vehicles_detected) &&
  isMetric(value.blocked_roads) &&
  isMetric(value.damaged_buildings);

const isIncident = (value: unknown) =>
  isRecord(value) &&
  isString(value.incident_id) &&
  isString(value.title) &&
  isString(value.location_label) &&
  isNumber(value.started_at_ms) &&
  isOneOf(value.coordinate_space, ["NORMALIZED_IMAGE", "RELATIVE_TACTICAL"] as const) &&
  isOriginRecord(value);

const isSystemStatus = (value: unknown) =>
  isRecord(value) &&
  isOneOf(value.api, ["operational", "degraded", "offline"] as const) &&
  isOneOf(value.segmentation_model, ["not_configured", "loading", "ready", "error"] as const) &&
  isOneOf(value.detection_model, ["not_configured", "loading", "ready", "error"] as const);

const isDetection = (value: unknown) =>
  isRecord(value) &&
  isString(value.detection_id) &&
  isOneOf(value.category, ["PERSON", "VEHICLE", "OTHER"] as const) &&
  isString(value.label) &&
  isNumber(value.confidence) &&
  isBoundingBox(value.bbox) &&
  isOriginRecord(value);

const isSegmentationClass = (value: unknown) =>
  isRecord(value) &&
  isString(value.label) &&
  isNumber(value.coverage_percent) &&
  isNumber(value.confidence) &&
  isOriginRecord(value);

const isOverlayRegion = (value: unknown) =>
  isRecord(value) &&
  isString(value.overlay_id) &&
  isOneOf(value.kind, ["FLOOD", "DAMAGED_BUILDING"] as const) &&
  isString(value.label) &&
  isPointList(value.polygon) &&
  isNumber(value.confidence) &&
  isOriginRecord(value);

const isSegmentation = (value: unknown) =>
  isRecord(value) &&
  isOneOf(
    value.status,
    ["not_configured", "simulated", "processing", "ready", "error"] as const,
  ) &&
  Array.isArray(value.classes) &&
  value.classes.every(isSegmentationClass) &&
  Array.isArray(value.regions) &&
  value.regions.every(isOverlayRegion);

const isRoad = (value: unknown) =>
  isRecord(value) &&
  isString(value.road_id) &&
  isString(value.label) &&
  isOneOf(value.state, roadStates) &&
  isOneOf(value.access_status, accessStates) &&
  isPointList(value.geometry) &&
  isNullableNumber(value.confidence) &&
  isOriginRecord(value);

const isZoneReason = (value: unknown) =>
  isRecord(value) &&
  isString(value.code) &&
  isString(value.label) &&
  isString(value.description) &&
  isNumber(value.contribution) &&
  isOriginRecord(value);

const isZone = (value: unknown) =>
  isRecord(value) &&
  isString(value.zone_id) &&
  isString(value.display_name) &&
  isNumber(value.rank) &&
  isOneOf(value.severity, severities) &&
  isNumber(value.priority_score) &&
  isNumber(value.confidence) &&
  isPointList(value.polygon) &&
  isNumber(value.people_count) &&
  isNumber(value.vehicle_count) &&
  isNumber(value.flood_coverage_percent) &&
  isNumber(value.building_damage_count) &&
  isOneOf(value.road_condition, roadStates) &&
  isOneOf(value.access_status, accessStates) &&
  isString(value.primary_reason) &&
  Array.isArray(value.reasons) &&
  value.reasons.every(isZoneReason) &&
  isNumber(value.updated_at_ms) &&
  isOriginRecord(value);

const isIncidentEvent = (value: unknown) =>
  isRecord(value) &&
  isString(value.event_id) &&
  isNumber(value.timestamp_ms) &&
  isOneOf(value.severity, ["INFO", "WARNING", "CRITICAL"] as const) &&
  isOneOf(
    value.category,
    ["FLOOD", "DETECTION", "ACCESS", "PRIORITY", "ROUTE", "SYSTEM"] as const,
  ) &&
  isString(value.message) &&
  isOriginRecord(value);

const isRoute = (value: unknown) =>
  isRecord(value) &&
  isString(value.route_id) &&
  isOneOf(value.status, ["RECOMMENDED", "UNAVAILABLE"] as const) &&
  isString(value.target_zone_id) &&
  isString(value.label) &&
  isPointList(value.waypoints) &&
  isNullableNumber(value.distance_m) &&
  isString(value.access_summary) &&
  isOriginRecord(value);

export function parseLiveResult(value: unknown): LiveResult | null {
  if (!isRecord(value) || value.data_origin !== "DEMO_SIMULATED") return null;
  const valid =
    isString(value.incident_id) &&
    isIncident(value.incident) &&
    isNumber(value.frame_id) &&
    isNumber(value.snapshot_index) &&
    isNumber(value.snapshot_count) &&
    isNumber(value.timestamp_ms) &&
    isOneOf(
      value.source_mode,
      ["VIDEO_FILE", "WEBCAM", "DRONE_STREAM", "DEMO_REPLAY", "SIMULATION"] as const,
    ) &&
    isOneOf(value.coordinate_space, ["NORMALIZED_IMAGE", "RELATIVE_TACTICAL"] as const) &&
    isOneOf(value.stream_state, ["CONNECTING", "PLAYING", "PAUSED", "COMPLETE"] as const) &&
    isOneOf(value.incident_severity, severities) &&
    (value.highest_priority_zone_id === null || isString(value.highest_priority_zone_id)) &&
    isSystemStatus(value.system_status) &&
    isStatistics(value.statistics) &&
    Array.isArray(value.detections) &&
    value.detections.every(isDetection) &&
    isSegmentation(value.segmentation) &&
    Array.isArray(value.roads) &&
    value.roads.every(isRoad) &&
    Array.isArray(value.zones) &&
    value.zones.every(isZone) &&
    Array.isArray(value.events) &&
    value.events.every(isIncidentEvent) &&
    (value.route === null || isRoute(value.route));

  return valid ? (value as unknown as LiveResult) : null;
}
