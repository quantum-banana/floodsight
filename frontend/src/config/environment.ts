const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = configuredBaseUrl
  ? configuredBaseUrl.replace(/\/+$/, "")
  : "";

export const API_TIMEOUT_MS = 8_000;

const readNumber = (value: string | undefined, fallback: number) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const INGEST_CAPTURE_FPS = Math.min(
  10,
  Math.max(1, readNumber(import.meta.env.VITE_INGEST_CAPTURE_FPS, 4)),
);
export const INGEST_JPEG_QUALITY = Math.min(
  0.95,
  Math.max(0.5, readNumber(import.meta.env.VITE_INGEST_JPEG_QUALITY, 0.75)),
);
export const INGEST_MAX_WIDTH = Math.max(
  320,
  readNumber(import.meta.env.VITE_INGEST_MAX_WIDTH, 1280),
);
export const INGEST_MAX_HEIGHT = Math.max(
  240,
  readNumber(import.meta.env.VITE_INGEST_MAX_HEIGHT, 720),
);
export const VIDEO_FILE_MAX_BYTES = Math.max(
  1,
  readNumber(import.meta.env.VITE_VIDEO_FILE_MAX_MB, 250),
) * 1024 * 1024;
export const INGEST_ACK_TIMEOUT_MS = 6_000;

const configuredWebSocketUrl = import.meta.env.VITE_WS_BASE_URL?.trim();

export const WS_BASE_URL = configuredWebSocketUrl
  ? configuredWebSocketUrl.replace(/\/+$/, "")
  : API_BASE_URL
    ? API_BASE_URL.replace(/^http/, "ws")
    : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
