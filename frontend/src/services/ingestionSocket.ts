import { WS_BASE_URL } from "../config/environment";
import type { FrameIntelligence, FrameMetadata, FrameResult } from "../types/ingestion";
import {
  parseFrameIntelligence,
  parseFrameResult,
} from "../utils/ingestionValidation";

export interface IngestionSocketHandlers {
  onOpen: () => void;
  onResult: (result: FrameResult) => void;
  onIntelligence: (message: FrameIntelligence) => void;
  onMalformedMessage: () => void;
  onError: () => void;
  onClose: (event: CloseEvent) => void;
}

export interface IngestionSocketConnection {
  sendFrame: (metadata: FrameMetadata, payload: ArrayBuffer) => boolean;
  close: () => void;
  isOpen: () => boolean;
}

export function openIngestionSocket(
  sessionId: string,
  handlers: IngestionSocketHandlers,
): IngestionSocketConnection {
  const socket = new WebSocket(
    `${WS_BASE_URL}/ws/ingest/sessions/${encodeURIComponent(sessionId)}/frames`,
  );
  socket.binaryType = "arraybuffer";
  socket.addEventListener("open", handlers.onOpen);
  socket.addEventListener("error", handlers.onError);
  socket.addEventListener("close", handlers.onClose);
  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") {
      handlers.onMalformedMessage();
      return;
    }
    try {
      const payload: unknown = JSON.parse(event.data);
      const result = parseFrameResult(payload);
      if (result) {
        handlers.onResult(result);
        return;
      }
      const intelligence = parseFrameIntelligence(payload);
      if (intelligence) {
        handlers.onIntelligence(intelligence);
        return;
      }
      handlers.onMalformedMessage();
    } catch {
      handlers.onMalformedMessage();
    }
  });

  return {
    sendFrame: (metadata, payload) => {
      if (socket.readyState !== WebSocket.OPEN) return false;
      socket.send(JSON.stringify(metadata));
      socket.send(payload);
      return true;
    },
    close: () => {
      if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
        socket.close(1_000, "Media source stopped");
      }
    },
    isOpen: () => socket.readyState === WebSocket.OPEN,
  };
}
