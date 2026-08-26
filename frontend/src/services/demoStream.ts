import { WS_BASE_URL } from "../config/environment";
import type { LiveResult } from "../types/liveResult";
import { parseLiveResult } from "../utils/validation";

export interface DemoStreamHandlers {
  onOpen: () => void;
  onMessage: (snapshot: LiveResult) => void;
  onMalformedMessage: () => void;
  onClose: (event: CloseEvent) => void;
  onError: () => void;
}

export interface DemoStreamConnection {
  close: () => void;
}

export function openDemoStream(
  incidentId: string,
  startIndex: number,
  handlers: DemoStreamHandlers,
): DemoStreamConnection {
  const query = new URLSearchParams({
    start_index: startIndex.toString(),
    loop: "false",
  });
  const socket = new WebSocket(
    `${WS_BASE_URL}/ws/demo/incidents/${encodeURIComponent(incidentId)}/live?${query}`,
  );

  socket.addEventListener("open", handlers.onOpen);
  socket.addEventListener("error", handlers.onError);
  socket.addEventListener("close", handlers.onClose);
  socket.addEventListener("message", (event) => {
    try {
      const snapshot = parseLiveResult(JSON.parse(String(event.data)));
      if (!snapshot) {
        handlers.onMalformedMessage();
        return;
      }
      handlers.onMessage(snapshot);
    } catch {
      handlers.onMalformedMessage();
    }
  });

  return {
    close: () => {
      if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
        socket.close(1_000, "Simulation control changed");
      }
    },
  };
}
