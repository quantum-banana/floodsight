import { afterEach, describe, expect, it, vi } from "vitest";

import { openDemoStream } from "../services/demoStream";
import { commandSnapshot } from "./fixtures";

type Listener = (event: Event | MessageEvent | CloseEvent) => void;

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static latest: FakeWebSocket | null = null;

  readonly url: string;
  readyState = FakeWebSocket.OPEN;
  close = vi.fn(() => {
    this.readyState = FakeWebSocket.CLOSED;
  });
  private readonly listeners = new Map<string, Listener[]>();

  constructor(url: string | URL) {
    this.url = url.toString();
    FakeWebSocket.latest = this;
  }

  addEventListener(type: string, listener: EventListener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener as Listener);
    this.listeners.set(type, listeners);
  }

  emit(type: string, event: Event | MessageEvent | CloseEvent) {
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }
}

const originalWebSocket = globalThis.WebSocket;

describe("deterministic demo WebSocket client", () => {
  afterEach(() => {
    globalThis.WebSocket = originalWebSocket;
    FakeWebSocket.latest = null;
  });

  it("parses valid snapshots in order and rejects malformed messages safely", () => {
    globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
    const onMessage = vi.fn();
    const onMalformedMessage = vi.fn();
    const connection = openDemoStream("FS-001", 2, {
      onOpen: vi.fn(),
      onMessage,
      onMalformedMessage,
      onClose: vi.fn(),
      onError: vi.fn(),
    });
    const socket = FakeWebSocket.latest;
    expect(socket?.url).toContain("/ws/demo/incidents/FS-001/live?start_index=2&loop=false");

    socket?.emit("message", new MessageEvent("message", { data: JSON.stringify(commandSnapshot) }));
    socket?.emit(
      "message",
      new MessageEvent("message", {
        data: JSON.stringify({
          ...commandSnapshot,
          zones: [{ zone_id: "BROKEN-ZONE" }],
        }),
      }),
    );
    socket?.emit("message", new MessageEvent("message", { data: "not-json" }));

    expect(onMessage).toHaveBeenCalledWith(commandSnapshot);
    expect(onMalformedMessage).toHaveBeenCalledTimes(2);
    connection.close();
    expect(socket?.close).toHaveBeenCalledWith(1_000, "Simulation control changed");
  });
});
