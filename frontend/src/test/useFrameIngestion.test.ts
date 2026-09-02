import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useFrameIngestion } from "../hooks/useFrameIngestion";
import { captureJpeg } from "../services/frameCapture";
import {
  completeIngestionSession,
  createIngestionSession,
  deleteIngestionSession,
} from "../services/ingestionApi";
import {
  openIngestionSocket,
  type IngestionSocketHandlers,
} from "../services/ingestionSocket";
import type { AggregateMetric, VideoAnalysisComplete } from "../types/ingestion";
import { liveSnapshot } from "./fixtures";

vi.mock("../services/ingestionApi", () => ({
  completeIngestionSession: vi.fn(),
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
  detector_mode: "AERIAL_HIGH_RECALL" as const,
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

const aggregateMetric = (
  value: number,
  unit: AggregateMetric["unit"],
  aggregation: AggregateMetric["aggregation"],
): AggregateMetric => ({
  value,
  unit,
  availability: "AVAILABLE",
  aggregation,
  supporting_frame_count: 2,
  confidence: 0.9,
  data_origin: "DERIVED_ANALYTIC",
});

const finalSnapshot = { ...liveSnapshot, frame_id: 9 };
const completedAnalysis: VideoAnalysisComplete = {
  type: "video_analysis_complete",
  session_id: session.session_id,
  state: "COMPLETE",
  summary: {
    session_id: session.session_id,
    generated_at_ms: 1_725_000_010_000,
    frames_accepted: 10,
    frames_analyzed: 4,
    frames_dropped: 1,
    first_analyzed_frame_id: 0,
    last_analyzed_frame_id: 9,
    first_media_time_ms: 0,
    last_media_time_ms: 9_000,
    statistics: {
      flooded_area_percent: aggregateMetric(38, "percent", "PEAK_FRESH_SEGMENTATION"),
      people_detected: aggregateMetric(4, "count", "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"),
      vehicles_detected: aggregateMetric(2, "count", "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"),
      blocked_road_cells: aggregateMetric(2, "count", "PEAK_FRESH_SEGMENTATION"),
      damaged_buildings: {
        value: null,
        unit: "count",
        availability: "NOT_SUPPORTED",
        aggregation: "NOT_APPLICABLE",
        supporting_frame_count: 0,
        confidence: null,
        data_origin: "DERIVED_ANALYTIC",
      },
      building_damage_coverage_percent: aggregateMetric(8, "percent", "PEAK_FRESH_SEGMENTATION"),
    },
    detected_classes: [],
    detected_classes_truncated: false,
    priorities: [{
      zone: liveSnapshot.zones[0],
      source_frame_id: 5,
      media_time_ms: 5_000,
      supporting_update_count: 2,
      segmentation_evidence_available: true,
      detection_evidence_available: true,
      building_damage_count_availability: "NOT_SUPPORTED",
      associated_route: null,
      data_origin: "DERIVED_ANALYTIC",
    }],
    priorities_truncated: false,
    highest_priority_zone_id: liveSnapshot.zones[0].zone_id,
    incident_severity: liveSnapshot.zones[0].severity,
    segmentation_status: liveSnapshot.system_status.segmentation_details!,
    detection_status: liveSnapshot.system_status.detection_details!,
    inference_state: "LIVE",
    responsible_ai_statement: "Human verification is required.",
    data_origin: "DERIVED_ANALYTIC",
  },
  latest_result: finalSnapshot,
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
    vi.mocked(completeIngestionSession).mockResolvedValue(completedAnalysis);
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
        detectorMode: "AERIAL_HIGH_RECALL",
        sourceReady: true,
        captureActive: true,
        sourceComplete: false,
        sourceGeneration: 1,
      }),
    );

    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledWith(session.session_id, expect.any(Object)));
    expect(createIngestionSession).toHaveBeenCalledWith(
      "VIDEO_FILE",
      "USER_VIDEO_FILE",
      "AERIAL_HIGH_RECALL",
    );
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
      detectorMode: "AERIAL_HIGH_RECALL" as const,
      sourceReady: true,
      captureActive: true,
      sourceComplete: false,
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
        detectorMode: "AERIAL_HIGH_RECALL",
        sourceReady: true,
        captureActive: true,
        sourceComplete: false,
        sourceGeneration: 8,
      }),
    );

    await waitFor(() => expect(result.current.metrics.connectionState).toBe("offline"));
    expect(result.current.metrics.lastError).toBe("Backend unavailable");
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("finalizes once after acknowledgements, accepts late intelligence, and resets for a new run", async () => {
    let resolveCompletion: ((value: VideoAnalysisComplete) => void) | null = null;
    vi.mocked(completeIngestionSession).mockImplementation(() => new Promise((resolve) => {
      resolveCompletion = resolve;
    }));
    const base = {
      videoElement: videoFixture().video,
      sourceMode: "VIDEO_FILE" as const,
      mediaOrigin: "USER_VIDEO_FILE" as const,
      detectorMode: "AERIAL_HIGH_RECALL" as const,
      sourceReady: true,
      captureActive: false,
    };
    const { result, rerender } = renderHook(
      ({ complete, generation }) => useFrameIngestion({
        ...base,
        sourceComplete: complete,
        sourceGeneration: generation,
      }),
      { initialProps: { complete: false, generation: 1 } },
    );
    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledOnce());
    act(() => handlers?.onOpen());

    rerender({ complete: true, generation: 1 });
    await waitFor(() => expect(result.current.completionState).toBe("FINALIZING"));
    expect(completeIngestionSession).toHaveBeenCalledOnce();
    expect(completeIngestionSession).toHaveBeenCalledWith(session.session_id);

    const lateSnapshot = { ...liveSnapshot, frame_id: 8 };
    act(() => handlers?.onIntelligence({
      type: "frame_intelligence",
      session_id: session.session_id,
      frame_id: 8,
      sequence: 3,
      result: lateSnapshot,
    }));
    expect(result.current.intelligence?.frame_id).toBe(8);

    await act(async () => resolveCompletion?.(completedAnalysis));
    await waitFor(() => expect(result.current.completionState).toBe("COMPLETE"));
    expect(result.current.completion).toEqual(completedAnalysis);
    expect(result.current.intelligence?.frame_id).toBe(9);
    expect(result.current.metrics.sessionState).toBe("COMPLETE");
    expect(completeIngestionSession).toHaveBeenCalledOnce();

    rerender({ complete: false, generation: 2 });
    await waitFor(() => expect(result.current.completionState).toBe("IDLE"));
    expect(result.current.completion).toBeNull();
  });

  it("does not send a capture that finishes encoding after video completion", async () => {
    let resolveBuffer: ((value: ArrayBuffer) => void) | null = null;
    vi.mocked(captureJpeg).mockResolvedValue({
      blob: {
        arrayBuffer: vi.fn(() => new Promise((resolve) => {
          resolveBuffer = resolve;
        })),
      } as unknown as Blob,
      width: 640,
      height: 360,
    });
    const { video, getCallback } = videoFixture();
    const base = {
      videoElement: video,
      sourceMode: "VIDEO_FILE" as const,
      mediaOrigin: "USER_VIDEO_FILE" as const,
      detectorMode: "AERIAL_HIGH_RECALL" as const,
      sourceReady: true,
      sourceGeneration: 6,
    };
    const { rerender } = renderHook(
      ({ active, complete }) => useFrameIngestion({
        ...base,
        captureActive: active,
        sourceComplete: complete,
      }),
      { initialProps: { active: true, complete: false } },
    );
    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledOnce());
    act(() => handlers?.onOpen());
    await waitFor(() => expect(getCallback()).not.toBeNull());
    act(() => getCallback()?.(300, {} as VideoFrameCallbackMetadata));
    await waitFor(() => expect(captureJpeg).toHaveBeenCalledOnce());

    rerender({ active: false, complete: true });
    await waitFor(() => expect(completeIngestionSession).toHaveBeenCalledOnce());
    await act(async () => resolveBuffer?.(new ArrayBuffer(8)));

    expect(sendFrame).not.toHaveBeenCalled();
  });

  it("retries a failed completion request on the same ingestion session", async () => {
    vi.mocked(completeIngestionSession)
      .mockRejectedValueOnce(new Error("Completion request timed out"))
      .mockResolvedValueOnce(completedAnalysis);
    const video = videoFixture().video;
    const { result } = renderHook(() =>
      useFrameIngestion({
        videoElement: video,
        sourceMode: "VIDEO_FILE",
        mediaOrigin: "USER_VIDEO_FILE",
        detectorMode: "AERIAL_HIGH_RECALL",
        sourceReady: true,
        captureActive: false,
        sourceComplete: true,
        sourceGeneration: 7,
      }),
    );

    await waitFor(() => expect(result.current.completionState).toBe("ERROR"));
    expect(completeIngestionSession).toHaveBeenCalledTimes(1);
    expect(completeIngestionSession).toHaveBeenLastCalledWith(session.session_id);
    expect(createIngestionSession).toHaveBeenCalledTimes(1);
    expect(result.current.metrics.sessionId).toBe(session.session_id);

    act(() => result.current.retry());

    await waitFor(() => expect(result.current.completionState).toBe("COMPLETE"));
    expect(completeIngestionSession).toHaveBeenCalledTimes(2);
    expect(completeIngestionSession).toHaveBeenLastCalledWith(session.session_id);
    expect(createIngestionSession).toHaveBeenCalledTimes(1);
    expect(deleteIngestionSession).not.toHaveBeenCalled();
    expect(result.current.completion).toEqual(completedAnalysis);
  });

  it("finalizes the same session when the frame socket fails after video end", async () => {
    const { video, getCallback } = videoFixture();
    const base = {
      videoElement: video,
      sourceMode: "VIDEO_FILE" as const,
      mediaOrigin: "USER_VIDEO_FILE" as const,
      detectorMode: "AERIAL_HIGH_RECALL" as const,
      sourceReady: true,
      sourceGeneration: 10,
    };
    const { result, rerender } = renderHook(
      ({ active, complete }) => useFrameIngestion({
        ...base,
        captureActive: active,
        sourceComplete: complete,
      }),
      { initialProps: { active: true, complete: false } },
    );
    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledOnce());
    act(() => handlers?.onOpen());
    await waitFor(() => expect(getCallback()).not.toBeNull());
    act(() => getCallback()?.(300, {} as VideoFrameCallbackMetadata));
    await waitFor(() => expect(sendFrame).toHaveBeenCalledOnce());

    rerender({ active: false, complete: true });
    expect(result.current.completionState).toBe("FINALIZING");
    expect(completeIngestionSession).not.toHaveBeenCalled();

    act(() => handlers?.onError());

    await waitFor(() => expect(result.current.completionState).toBe("COMPLETE"));
    expect(completeIngestionSession).toHaveBeenCalledOnce();
    expect(completeIngestionSession).toHaveBeenCalledWith(session.session_id);
    expect(createIngestionSession).toHaveBeenCalledOnce();
    expect(deleteIngestionSession).not.toHaveBeenCalled();
  });

  it("finalizes the same session after an end-of-video acknowledgement timeout", async () => {
    const { video, getCallback } = videoFixture();
    const base = {
      videoElement: video,
      sourceMode: "VIDEO_FILE" as const,
      mediaOrigin: "USER_VIDEO_FILE" as const,
      detectorMode: "AERIAL_HIGH_RECALL" as const,
      sourceReady: true,
      sourceGeneration: 9,
    };
    const { result, rerender } = renderHook(
      ({ active, complete }) => useFrameIngestion({
        ...base,
        captureActive: active,
        sourceComplete: complete,
      }),
      { initialProps: { active: true, complete: false } },
    );
    await waitFor(() => expect(openIngestionSocket).toHaveBeenCalledOnce());
    act(() => handlers?.onOpen());
    await waitFor(() => expect(getCallback()).not.toBeNull());
    vi.useFakeTimers();
    await act(async () => {
      getCallback()?.(300, {} as VideoFrameCallbackMetadata);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(sendFrame).toHaveBeenCalledOnce();
    expect(result.current.metrics.capturedFrames).toBe(1);
    expect(result.current.metrics.acknowledgedFrames).toBe(0);

    rerender({ active: false, complete: true });
    expect(result.current.completionState).toBe("FINALIZING");
    expect(completeIngestionSession).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(6_000));
    vi.useRealTimers();

    await waitFor(() => expect(completeIngestionSession).toHaveBeenCalledOnce());
    expect(completeIngestionSession).toHaveBeenCalledWith(session.session_id);
    expect(createIngestionSession).toHaveBeenCalledOnce();
    expect(deleteIngestionSession).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.completionState).toBe("COMPLETE"));

    act(() => handlers?.onResult({
      type: "frame_result",
      session_id: session.session_id,
      frame_id: 0,
      accepted: true,
      code: "accepted",
      message: "Late acknowledgement.",
      received_at_ms: 2,
      processing_ms: 2,
      byte_length: 8,
      decoded_frame: { width: 640, height: 360, channels: 3 },
      quality: null,
      data_origin: "DERIVED_ANALYTIC",
    }));
    expect(result.current.completionState).toBe("COMPLETE");
    expect(result.current.metrics.connectionState).toBe("stopped");
    expect(result.current.metrics.lastError).toBeNull();
  });
});
