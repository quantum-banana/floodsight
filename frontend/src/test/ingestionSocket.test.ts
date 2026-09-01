import { afterEach, describe, expect, it, vi } from "vitest";

import { openIngestionSocket } from "../services/ingestionSocket";
import type { FrameMetadata } from "../types/ingestion";
import { liveSnapshot } from "./fixtures";

type Listener = (event: Event | MessageEvent | CloseEvent) => void;

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static latest: FakeWebSocket | null = null;
  readonly url: string;
  readyState = FakeWebSocket.OPEN;
  binaryType = "blob";
  send = vi.fn();
  close = vi.fn();
  private readonly listeners = new Map<string, Listener[]>();

  constructor(url: string | URL) {
    this.url = url.toString();
    FakeWebSocket.latest = this;
  }

  addEventListener(type: string, listener: EventListener) {
    const list = this.listeners.get(type) ?? [];
    list.push(listener as Listener);
    this.listeners.set(type, list);
  }

  emit(type: string, event: Event | MessageEvent | CloseEvent) {
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }
}

const originalWebSocket = globalThis.WebSocket;

describe("frame WebSocket client", () => {
  afterEach(() => {
    globalThis.WebSocket = originalWebSocket;
    FakeWebSocket.latest = null;
  });

  it("sends metadata then binary and validates acknowledgements", () => {
    globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
    const onResult = vi.fn();
    const onIntelligence = vi.fn();
    const onMalformedMessage = vi.fn();
    const connection = openIngestionSocket("session-123", {
      onOpen: vi.fn(),
      onResult,
      onIntelligence,
      onMalformedMessage,
      onError: vi.fn(),
      onClose: vi.fn(),
    });
    const metadata: FrameMetadata = {
      type: "frame_metadata",
      frame_id: 0,
      captured_at_ms: 100,
      media_time_ms: 25,
      source_mode: "VIDEO_FILE",
      media_origin: "USER_VIDEO_FILE",
      mime_type: "image/jpeg",
      byte_length: 3,
      width: 2,
      height: 2,
    };
    const bytes = new ArrayBuffer(3);

    expect(connection.sendFrame(metadata, bytes)).toBe(true);
    expect(FakeWebSocket.latest?.send).toHaveBeenNthCalledWith(1, JSON.stringify(metadata));
    expect(FakeWebSocket.latest?.send).toHaveBeenNthCalledWith(2, bytes);
    FakeWebSocket.latest?.emit("message", new MessageEvent("message", {
      data: JSON.stringify({
        type: "frame_result",
        session_id: "session-123",
        frame_id: 0,
        accepted: true,
        code: "accepted",
        message: "ok",
        received_at_ms: 101,
        processing_ms: 1,
        byte_length: 3,
        decoded_frame: { width: 2, height: 2, channels: 3 },
        quality: null,
        data_origin: "DERIVED_ANALYTIC",
      }),
    }));
    FakeWebSocket.latest?.emit("message", new MessageEvent("message", {
      data: JSON.stringify({
        type: "frame_intelligence",
        session_id: "session-123",
        frame_id: liveSnapshot.frame_id,
        sequence: 1,
        result: liveSnapshot,
      }),
    }));
    FakeWebSocket.latest?.emit("message", new MessageEvent("message", { data: "broken" }));

    expect(onResult).toHaveBeenCalledOnce();
    expect(onIntelligence).toHaveBeenCalledWith(expect.objectContaining({ sequence: 1 }));
    expect(onMalformedMessage).toHaveBeenCalledOnce();
    expect(FakeWebSocket.latest?.url).toContain("/ws/ingest/sessions/session-123/frames");
    connection.close();
    expect(FakeWebSocket.latest?.close).toHaveBeenCalledWith(1_000, "Media source stopped");
  });
});
