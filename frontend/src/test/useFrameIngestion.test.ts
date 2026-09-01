import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useFrameIngestion } from "../hooks/useFrameIngestion";
import { captureJpeg } from "../services/frameCapture";
import { createIngestionSession, deleteIngestionSession } from "../services/ingestionApi";
import {
  openIngestionSocket,
  type IngestionSocketHandlers,
} from "../services/ingestionSocket";
import { liveSnapshot } from "./fixtures";

vi.mock("../services/ingestionApi", () => ({
  createIngestionSession: vi.fn(),
  deleteIngestionSession: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../services/ingestionSocket", () => ({
  openIngestionSocket: vi.fn(),
}));
vi.mock("../services/frameCapture", () => ({
  captureJpeg: vi.fn(),
}));

const session = {
  session_id: "frame-session-1234567890",
  source_mode: "VIDEO_FILE" as const,
  media_origin: "USER_VIDEO_FILE" as const,
  state: "READY" as const,
  created_at_ms: 1,
  last_activity_at_ms: 1,
  expires_at_ms: 901_000,
  counters: {
    frames_received: 0,
    frames_accepted: 0,
    frames_rejected: 0,
    frames_out_of_order: 0,
    protocol_errors: 0,
    bytes_received: 0,
  },
  limits: {
    recommended_capture_fps: 4,
    jpeg_quality: 0.75,
    max_frame_bytes: 2_000_000,
    accepted_mime_types: ["image/jpeg"],
  },
  data_origin: "DERIVED_ANALYTIC" as const,
};

function videoFixture() {
  const video = document.createElement("video");
  let callback: VideoFrameRequestCallback | null = null;
  Object.defineProperties(video, {
    paused: { configurable: true, value: false },
    ended: { configurable: true, value: false },
    currentTime: { configurable: true, value: 1.25, writable: true },
    videoWidth: { configurable: true, value: 640 },
    videoHeight: { configurable: true, value: 360 },
    requestVideoFrameCallback: {
      configurable: true,
      value: vi.fn((next: VideoFrameRequestCallback) => {
        callback = next;
        return 7;
      }),
    },
    cancelVideoFrameCallback: { configurable: true, value: vi.fn() },
  });
  return { video, getCallback: () => callback };
}

describe("frame ingestion lifecycle", () => {
  let handlers: IngestionSocketHandlers | null;
  let sendFrame: ReturnType<typeof vi.fn>;
  let close: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    handlers = null;
    sendFrame = vi.fn(() => true);
    close = vi.fn();
    vi.mocked(createIngestionSession).mockResolvedValue(session);
    vi.mocked(deleteIngestionSession).mockResolvedValue(undefined);
    vi.mocked(captureJpeg).mockResolvedValue({
      blob: { arrayBuffer: vi.fn().mockResolvedValue(new ArrayBuffer(8)) } as unknown as Blob,
      width: 640,
      height: 360,
    });
    vi.mocked(openIngestionSocket).mockImplementation((_sessionId, nextHandlers) => {
      handlers = nextHandlers;
      return { sendFrame, close, isOpen: () => true };
    });
  });

  it("creates a session, connects, captures metadata, acknowledges, and drops under backpressure", async () => {
    const { video, getCallback } = videoFixture();
    const { result } = renderHook(() =>
      useFrameIngestion({
        videoElement: video,
        sourceMode: "VIDEO_FILE",
        mediaOrigin: "USER_VIDEO_FILE",
        sourceReady: true,
        captureActive: true,
        sourceGeneration: 1,
      }),
    );

    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledWith(session.session_id, expect.any(Object)));
    act(() => handlers?.onOpen());
    await waitFor(() => expect(getCallback()).not.toBeNull());
    act(() => getCallback()?.(300, {} as VideoFrameCallbackMetadata));
    await waitFor(() => expect(sendFrame).toHaveBeenCalledOnce());

    const [metadata, bytes] = sendFrame.mock.calls[0];
    expect(metadata).toMatchObject({
      type: "frame_metadata",
      frame_id: 0,
      media_time_ms: 1250,
      source_mode: "VIDEO_FILE",
      media_origin: "USER_VIDEO_FILE",
      mime_type: "image/jpeg",
      byte_length: 8,
      width: 640,
      height: 360,
    });
    expect(bytes).toBeInstanceOf(ArrayBuffer);

    act(() => getCallback()?.(600, {} as VideoFrameCallbackMetadata));
    await waitFor(() => expect(result.current.metrics.clientDroppedFrames).toBeGreaterThan(0));
    act(() => handlers?.onResult({
      type: "frame_result",
      session_id: session.session_id,
      frame_id: 0,
      accepted: true,
      code: "accepted",
      message: "Frame decoded and accepted.",
      received_at_ms: 2,
      processing_ms: 2.5,
      byte_length: 8,
      decoded_frame: { width: 640, height: 360, channels: 3 },
      quality: {
        mean_luminance: 110,
        laplacian_variance: 90,
        brightness_status: "NORMAL",
        sharpness_status: "NORMAL",
        warnings: [],
        data_origin: "DERIVED_ANALYTIC",
      },
      data_origin: "DERIVED_ANALYTIC",
    }));

    expect(result.current.metrics.acknowledgedFrames).toBe(1);
    expect(result.current.metrics.latestDimensions).toBe("640×360");
    expect(result.current.metrics.latestProcessingMs).toBe(2.5);
    act(() => handlers?.onIntelligence({
      type: "frame_intelligence",
      session_id: session.session_id,
      frame_id: liveSnapshot.frame_id,
      sequence: 2,
      result: liveSnapshot,
    }));
    expect(result.current.intelligence).toEqual(liveSnapshot);
    expect(result.current.metrics.analysisStatus).toBe("LIVE");
    act(() => handlers?.onIntelligence({
      type: "frame_intelligence",
      session_id: session.session_id,
      frame_id: 1,
      sequence: 1,
      result: { ...liveSnapshot, frame_id: 1 },
    }));
    expect(result.current.intelligence?.frame_id).toBe(liveSnapshot.frame_id);
    act(() => handlers?.onMalformedMessage());
    expect(result.current.metrics.connectionState).toBe("malformed");
    expect(result.current.metrics.lastError).toMatch(/malformed acknowledgement/i);
  });

  it("pauses without deleting, resumes, then cleans up session and socket on stop", async () => {
    const { video } = videoFixture();
    const base = {
      videoElement: video,
      sourceMode: "VIDEO_FILE" as const,
      mediaOrigin: "USER_VIDEO_FILE" as const,
      sourceReady: true,
      captureActive: true,
      sourceGeneration: 4,
    };
    const { result, rerender } = renderHook(
      ({ active, ready }) => useFrameIngestion({ ...base, captureActive: active, sourceReady: ready }),
      { initialProps: { active: true, ready: true } },
    );
    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledOnce());
    act(() => handlers?.onOpen());
    rerender({ active: false, ready: true });
    await waitFor(() => expect(result.current.metrics.connectionState).toBe("paused"));
    expect(deleteIngestionSession).not.toHaveBeenCalled();

    rerender({ active: true, ready: true });
    await waitFor(() => expect(result.current.metrics.connectionState).toBe("streaming"));
    rerender({ active: false, ready: false });
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    expect(deleteIngestionSession).toHaveBeenCalledWith(session.session_id);
  });

  it("reports a backend-offline state when session creation fails", async () => {
    vi.mocked(createIngestionSession).mockRejectedValue(new Error("Backend unavailable"));
    const { result } = renderHook(() =>
      useFrameIngestion({
        videoElement: videoFixture().video,
        sourceMode: "WEBCAM",
        mediaOrigin: "USER_WEBCAM",
        sourceReady: true,
        captureActive: true,
        sourceGeneration: 8,
      }),
    );

    await waitFor(() => expect(result.current.metrics.connectionState).toBe("offline"));
    expect(result.current.metrics.lastError).toBe("Backend unavailable");
  });
});
