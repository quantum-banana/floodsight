import { useCallback, useEffect, useRef, useState } from "react";

import {
  INGEST_ACK_TIMEOUT_MS,
  INGEST_CAPTURE_FPS,
  INGEST_JPEG_QUALITY,
  INGEST_MAX_HEIGHT,
  INGEST_MAX_WIDTH,
} from "../config/environment";
import { captureJpeg } from "../services/frameCapture";
import { saveIngestionDiagnostics } from "../services/ingestionDiagnostics";
import { createIngestionSession, deleteIngestionSession } from "../services/ingestionApi";
import {
  openIngestionSocket,
  type IngestionSocketConnection,
} from "../services/ingestionSocket";
import type {
  ActualSourceMode,
  FrameMetadata,
  FrameResult,
  IngestionMetrics,
  MediaOrigin,
} from "../types/ingestion";

interface UseFrameIngestionOptions {
  videoElement: HTMLVideoElement | null;
  sourceMode: ActualSourceMode | null;
  mediaOrigin: MediaOrigin | null;
  sourceReady: boolean;
  captureActive: boolean;
  sourceGeneration: number;
}

const initialMetrics = (): IngestionMetrics => ({
  sessionId: null,
  sessionState: null,
  sourceMode: null,
  mediaOrigin: null,
  connectionState: "idle",
  requestedFps: INGEST_CAPTURE_FPS,
  measuredFps: 0,
  capturedFrames: 0,
  acknowledgedFrames: 0,
  rejectedFrames: 0,
  clientDroppedFrames: 0,
  latestFrameId: null,
  latestDimensions: null,
  latestBlurScore: null,
  latestLuminance: null,
  latestProcessingMs: null,
  latestQualityState: null,
  lastError: null,
  modelStatus: "NOT_CONFIGURED",
  analysisStatus: "DEMO_SIMULATED",
});

export interface FrameIngestionController {
  metrics: IngestionMetrics;
  retry: () => void;
}

export function useFrameIngestion({
  videoElement,
  sourceMode,
  mediaOrigin,
  sourceReady,
  captureActive,
  sourceGeneration,
}: UseFrameIngestionOptions): FrameIngestionController {
  const [metrics, setMetrics] = useState<IngestionMetrics>(initialMetrics);
  const [retryGeneration, setRetryGeneration] = useState(0);
  const socketRef = useRef<IngestionSocketConnection | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIdRef = useRef(0);
  const inFlightRef = useRef(false);
  const expectedFrameIdRef = useRef<number | null>(null);
  const ackTimerRef = useRef<number | null>(null);
  const captureStartedAtRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);
  const captureActiveRef = useRef(captureActive);

  useEffect(() => {
    captureActiveRef.current = captureActive;
  }, [captureActive]);

  const clearAckTimer = useCallback(() => {
    if (ackTimerRef.current !== null) {
      window.clearTimeout(ackTimerRef.current);
      ackTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    saveIngestionDiagnostics(metrics);
  }, [metrics]);

  useEffect(() => {
    if (!sourceReady || !sourceMode || !mediaOrigin) {
      setMetrics(initialMetrics());
      return;
    }

    let cancelled = false;
    let sessionId: string | null = null;
    intentionalCloseRef.current = false;
    frameIdRef.current = 0;
    inFlightRef.current = false;
    expectedFrameIdRef.current = null;
    captureStartedAtRef.current = null;
    setMetrics({
      ...initialMetrics(),
      sourceMode,
      mediaOrigin,
      connectionState: "creating_session",
    });

    void createIngestionSession(sourceMode, mediaOrigin)
      .then((session) => {
        if (cancelled) {
          void deleteIngestionSession(session.session_id).catch(() => undefined);
          return;
        }
        sessionId = session.session_id;
        setMetrics((current) => ({
          ...current,
          sessionId,
          sessionState: session.state,
          requestedFps: Math.min(INGEST_CAPTURE_FPS, session.limits.recommended_capture_fps),
          connectionState: "connecting",
        }));
        socketRef.current = openIngestionSocket(session.session_id, {
          onOpen: () => {
            if (cancelled) return;
            setMetrics((current) => ({
              ...current,
              connectionState: captureActiveRef.current ? "streaming" : "paused",
              sessionState: "ACTIVE",
              lastError: null,
            }));
          },
          onResult: (result: FrameResult) => {
            if (cancelled) return;
            if (
              result.session_id !== session.session_id ||
              result.frame_id !== expectedFrameIdRef.current
            ) {
              clearAckTimer();
              inFlightRef.current = false;
              setMetrics((current) => ({
                ...current,
                connectionState: "malformed",
                lastError: "The ingestion server acknowledged an unexpected frame.",
              }));
              return;
            }
            clearAckTimer();
            inFlightRef.current = false;
            expectedFrameIdRef.current = null;
            setMetrics((current) => ({
              ...current,
              acknowledgedFrames: current.acknowledgedFrames + 1,
              rejectedFrames: current.rejectedFrames + (result.accepted ? 0 : 1),
              latestFrameId: result.frame_id,
              latestDimensions: result.decoded_frame
                ? `${result.decoded_frame.width}×${result.decoded_frame.height}`
                : current.latestDimensions,
              latestBlurScore: result.quality?.laplacian_variance ?? current.latestBlurScore,
              latestLuminance: result.quality?.mean_luminance ?? current.latestLuminance,
              latestProcessingMs: result.processing_ms,
              latestQualityState: result.quality
                ? `${result.quality.brightness_status} / ${result.quality.sharpness_status}`
                : result.code,
              lastError: result.accepted ? null : result.message,
            }));
          },
          onMalformedMessage: () => {
            clearAckTimer();
            inFlightRef.current = false;
            setMetrics((current) => ({
              ...current,
              connectionState: "malformed",
              lastError: "The ingestion server returned a malformed acknowledgement.",
            }));
          },
          onError: () => {
            if (cancelled) return;
            clearAckTimer();
            inFlightRef.current = false;
            setMetrics((current) => ({
              ...current,
              connectionState: "offline",
              lastError: "The frame WebSocket encountered a connection error.",
            }));
          },
          onClose: () => {
            if (cancelled || intentionalCloseRef.current) return;
            clearAckTimer();
            inFlightRef.current = false;
            setMetrics((current) => ({
              ...current,
              connectionState: "offline",
              lastError: "The frame WebSocket disconnected.",
            }));
          },
        });
      })
      .catch((sessionError: unknown) => {
        if (cancelled) return;
        setMetrics((current) => ({
          ...current,
          connectionState: "offline",
          lastError:
            sessionError instanceof Error
              ? sessionError.message
              : "Unable to create an ingestion session.",
        }));
      });

    return () => {
      cancelled = true;
      intentionalCloseRef.current = true;
      clearAckTimer();
      inFlightRef.current = false;
      socketRef.current?.close();
      socketRef.current = null;
      if (sessionId) void deleteIngestionSession(sessionId).catch(() => undefined);
    };
  }, [clearAckTimer, mediaOrigin, retryGeneration, sourceGeneration, sourceMode, sourceReady]);

  useEffect(() => {
    if (!sourceReady || !sourceMode || !mediaOrigin || !captureActive) {
      if (sourceReady) {
        setMetrics((current) => ({
          ...current,
          connectionState:
            current.sessionId && current.connectionState !== "offline"
              ? "paused"
              : current.connectionState,
        }));
      }
      return;
    }
    if (metrics.connectionState !== "streaming" && metrics.connectionState !== "paused") return;
    if (!socketRef.current?.isOpen()) return;

    setMetrics((current) => ({ ...current, connectionState: "streaming" }));
    const video = videoElement;
    if (!video) return;
    canvasRef.current ??= document.createElement("canvas");
    const canvas = canvasRef.current;
    let cancelled = false;
    let frameCallbackId: number | null = null;
    let intervalId: number | null = null;
    let lastCaptureAt = 0;
    const intervalMs = 1_000 / metrics.requestedFps;

    const attemptCapture = async () => {
      if (cancelled || video.paused || video.ended) return;
      if (inFlightRef.current) {
        setMetrics((current) => ({
          ...current,
          clientDroppedFrames: current.clientDroppedFrames + 1,
        }));
        return;
      }
      inFlightRef.current = true;
      const capturedAt = Date.now();
      try {
        const captured = await captureJpeg(video, canvas, {
          maxWidth: INGEST_MAX_WIDTH,
          maxHeight: INGEST_MAX_HEIGHT,
          quality: INGEST_JPEG_QUALITY,
        });
        if (cancelled) {
          inFlightRef.current = false;
          return;
        }
        const frameId = frameIdRef.current++;
        const buffer = await captured.blob.arrayBuffer();
        const metadata: FrameMetadata = {
          type: "frame_metadata",
          frame_id: frameId,
          captured_at_ms: capturedAt,
          media_time_ms: Math.max(0, Math.round(video.currentTime * 1_000)),
          source_mode: sourceMode,
          media_origin: mediaOrigin,
          mime_type: "image/jpeg",
          byte_length: buffer.byteLength,
          width: captured.width,
          height: captured.height,
        };
        expectedFrameIdRef.current = frameId;
        if (!socketRef.current?.sendFrame(metadata, buffer)) {
          throw new Error("The frame WebSocket is not open.");
        }
        const now = performance.now();
        const firstCaptureAt = captureStartedAtRef.current;
        captureStartedAtRef.current ??= now;
        setMetrics((current) => {
          const nextCaptured = current.capturedFrames + 1;
          const elapsedSeconds = firstCaptureAt === null
            ? 0
            : Math.max(0.001, (now - firstCaptureAt) / 1_000);
          return {
            ...current,
            capturedFrames: nextCaptured,
            measuredFps: elapsedSeconds === 0
              ? 0
              : Number(((nextCaptured - 1) / elapsedSeconds).toFixed(2)),
          };
        });
        ackTimerRef.current = window.setTimeout(() => {
          inFlightRef.current = false;
          expectedFrameIdRef.current = null;
          setMetrics((current) => ({
            ...current,
            connectionState: "offline",
            lastError: "Frame acknowledgement timed out.",
          }));
        }, INGEST_ACK_TIMEOUT_MS);
      } catch (captureError) {
        inFlightRef.current = false;
        expectedFrameIdRef.current = null;
        setMetrics((current) => ({
          ...current,
          lastError:
            captureError instanceof Error ? captureError.message : "Frame capture failed.",
        }));
      }
    };

    if (typeof video.requestVideoFrameCallback === "function") {
      const onVideoFrame: VideoFrameRequestCallback = (now) => {
        if (cancelled) return;
        if (now - lastCaptureAt >= intervalMs) {
          lastCaptureAt = now;
          void attemptCapture();
        }
        frameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
      };
      frameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
    } else {
      intervalId = window.setInterval(() => void attemptCapture(), intervalMs);
    }

    return () => {
      cancelled = true;
      if (frameCallbackId !== null && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(frameCallbackId);
      }
      if (intervalId !== null) window.clearInterval(intervalId);
    };
  }, [captureActive, mediaOrigin, metrics.connectionState, metrics.requestedFps, sourceMode, sourceReady, videoElement]);

  return { metrics, retry: () => setRetryGeneration((value) => value + 1) };
}
