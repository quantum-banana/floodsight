const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = configuredBaseUrl
  ? configuredBaseUrl.replace(/\/+$/, "")
  : "";

export const API_TIMEOUT_MS = 8_000;

const configuredWebSocketUrl = import.meta.env.VITE_WS_BASE_URL?.trim();

export const WS_BASE_URL = configuredWebSocketUrl
  ? configuredWebSocketUrl.replace(/\/+$/, "")
  : API_BASE_URL
    ? API_BASE_URL.replace(/^http/, "ws")
    : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
