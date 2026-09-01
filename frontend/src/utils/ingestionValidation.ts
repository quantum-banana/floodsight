import type { FrameIntelligence, FrameResult } from "../types/ingestion";
import { parseLiveResult } from "./validation";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

export function parseFrameResult(value: unknown): FrameResult | null {
  if (!isRecord(value)) return null;
  const decoded = value.decoded_frame;
  const quality = value.quality;
  const validDecoded =
    decoded === null ||
    (isRecord(decoded) &&
      isNumber(decoded.width) &&
      isNumber(decoded.height) &&
      isNumber(decoded.channels));
  const validQuality =
    quality === null ||
    (isRecord(quality) &&
      isNumber(quality.mean_luminance) &&
      isNumber(quality.laplacian_variance) &&
      ["NORMAL", "DARK", "BRIGHT"].includes(String(quality.brightness_status)) &&
      ["NORMAL", "BLURRY"].includes(String(quality.sharpness_status)) &&
      Array.isArray(quality.warnings) &&
      quality.warnings.every((warning) => typeof warning === "string") &&
      quality.data_origin === "DERIVED_ANALYTIC");

  const valid =
    value.type === "frame_result" &&
    typeof value.session_id === "string" &&
    (value.frame_id === null || isNumber(value.frame_id)) &&
    typeof value.accepted === "boolean" &&
    typeof value.code === "string" &&
    typeof value.message === "string" &&
    isNumber(value.received_at_ms) &&
    isNumber(value.processing_ms) &&
    isNumber(value.byte_length) &&
    validDecoded &&
    validQuality &&
    value.data_origin === "DERIVED_ANALYTIC";
  return valid ? (value as unknown as FrameResult) : null;
}

export function parseFrameIntelligence(value: unknown): FrameIntelligence | null {
  if (!isRecord(value)) return null;
  const result = parseLiveResult(value.result);
  const valid =
    value.type === "frame_intelligence" &&
    typeof value.session_id === "string" &&
    isNumber(value.frame_id) &&
    isNumber(value.sequence) &&
    result !== null &&
    result.frame_id === value.frame_id;
  return valid ? ({ ...value, result } as FrameIntelligence) : null;
}
