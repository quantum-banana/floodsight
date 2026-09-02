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
import {
  completeIngestionSession,
  createIngestionSession,
  deleteIngestionSession,
} from "../services/ingestionApi";
import {
  openIngestionSocket,
  type IngestionSocketConnection,
} from "../services/ingestionSocket";
import type {
  ActualSourceMode,
  DetectorInferenceMode,
  FrameIntelligence,
  FrameMetadata,
  FrameResult,
  IngestionMetrics,
  MediaOrigin,
  VideoAnalysisComplete,
} from "../types/ingestion";
import type { LiveResult, ModelStatus } from "../types/liveResult";

interface UseFrameIngestionOptions {
  videoElement: HTMLVideoElement | null;
  sourceMode: ActualSourceMode | null;
  mediaOrigin: MediaOrigin | null;
  detectorMode: DetectorInferenceMode;
  sourceReady: boolean;
  captureActive: boolean;
  sourceComplete: boolean;
  sourceGeneration: number;
}

const initialMetrics = (): IngestionMetrics => ({
  sessionId: null,
  sessionState: null,
  sourceMode: null,
  mediaOrigin: null,
  detectorMode: null,
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
  modelStatus: "UNAVAILABLE",
  analysisStatus: "AWAITING_FRAME",
});

const describeModels = (
  segmentation?: ModelStatus | null,
  detection?: ModelStatus | null,
) => `SEG ${segmentation?.mode ?? "UNAVAILABLE"} · DET ${detection?.mode ?? "UNAVAILABLE"}`;

export interface FrameIngestionController {
  metrics: IngestionMetrics;
  intelligence: LiveResult | null;
  completionState: IngestionCompletionState;
  completion: VideoAnalysisComplete | null;
  retry: () => void;
}

export type IngestionCompletionState = "IDLE" | "FINALIZING" | "COMPLETE" | "ERROR";

export function useFrameIngestion({
  videoElement,
  sourceMode,
  mediaOrigin,
  detectorMode,
  sourceReady,
  captureActive,
  sourceComplete,
  sourceGeneration,
}: UseFrameIngestionOptions): FrameIngestionController {
  const [metrics, setMetrics] = useState<IngestionMetrics>(initialMetrics);
  const [intelligence, setIntelligence] = useState<LiveResult | null>(null);
  const [completionState, setCompletionState] = useState<IngestionCompletionState>("IDLE");
  const [completion, setCompletion] = useState<VideoAnalysisComplete | null>(null);
  const [retryGeneration, setRetryGeneration] = useState(0);
  const [completionRetryGeneration, setCompletionRetryGeneration] = useState(0);
  const socketRef = useRef<IngestionSocketConnection | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIdRef = useRef(0);
  const inFlightRef = useRef(false);
  const expectedFrameIdRef = useRef<number | null>(null);
  const ackTimerRef = useRef<number | null>(null);
  const captureStartedAtRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);
  const captureActiveRef = useRef(captureActive);
  const sourceCompleteRef = useRef(sourceComplete);
  const latestSequenceRef = useRef(-1);
  const completionRequestedRef = useRef(false);
  const forceCompletionRef = useRef(false);
  const activeSessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    captureActiveRef.current = captureActive;
  }, [captureActive]);

  useEffect(() => {
    sourceCompleteRef.current = sourceComplete;
  }, [sourceComplete]);

  const clearAckTimer = useCallback(() => {
    if (ackTimerRef.current !== null) {
      window.clearTimeout(ackTimerRef.current);
      ackTimerRef.current = null;
    }
  }, []);

  const requestSameSessionCompletion = useCallback(() => {
    if (!sourceCompleteRef.current || completionRequestedRef.current) return false;
    if (forceCompletionRef.current) return true;
    clearAckTimer();
    inFlightRef.current = false;
    expectedFrameIdRef.current = null;
    forceCompletionRef.current = true;
    setCompletionState("FINALIZING");
    setCompletionRetryGeneration((value) => value + 1);
    return true;
  }, [clearAckTimer]);

  useEffect(() => {
    saveIngestionDiagnostics(metrics);
  }, [metrics]);

  useEffect(() => {
    if (!sourceReady || !sourceMode || !mediaOrigin) {
      activeSessionIdRef.current = null;
      completionRequestedRef.current = false;
      forceCompletionRef.current = false;
      setMetrics(initialMetrics());
      setIntelligence(null);
      setCompletionState("IDLE");
      setCompletion(null);
      return;
    }

    let cancelled = false;
    let sessionId: string | null = null;
    intentionalCloseRef.current = false;
    frameIdRef.current = 0;
    inFlightRef.current = false;
    expectedFrameIdRef.current = null;
    captureStartedAtRef.current = null;
    latestSequenceRef.current = -1;
    completionRequestedRef.current = false;
    forceCompletionRef.current = false;
    activeSessionIdRef.current = null;
    setIntelligence(null);
    setCompletionState("IDLE");
    setCompletion(null);
    setMetrics({
      ...initialMetrics(),
      sourceMode,
      mediaOrigin,
      detectorMode,
      connectionState: "creating_session",
    });

    void createIngestionSession(sourceMode, mediaOrigin, detectorMode)
      .then((session) => {
        if (cancelled) {
          void deleteIngestionSession(session.session_id).catch(() => undefined);
          return;
        }
        sessionId = session.session_id;
        activeSessionIdRef.current = session.session_id;
        setMetrics((current) => ({
          ...current,
          sessionId,
          sessionState: session.state,
          detectorMode: session.detector_mode,
          requestedFps: Math.min(INGEST_CAPTURE_FPS, session.limits.recommended_capture_fps),
          connectionState: "connecting",
        }));
        socketRef.current = openIngestionSocket(session.session_id, {
          onOpen: () => {
            if (cancelled) return;
            setMetrics((current) => current.sessionState === "FINALIZING" || current.sessionState === "COMPLETE"
              ? current
              : {
                  ...current,
                  connectionState: captureActiveRef.current ? "streaming" : "paused",
                  sessionState: "ACTIVE",
                  lastError: null,
                });
          },
          onResult: (result: FrameResult) => {
            if (cancelled) return;
            if (
              result.session_id === session.session_id
              && sourceCompleteRef.current
              && (completionRequestedRef.current || forceCompletionRef.current)
            ) {
              return;
            }
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
              modelStatus: describeModels(
                result.segmentation_status,
                result.detection_status,
              ),
              analysisStatus: result.inference_state ?? current.analysisStatus,
              lastError: result.accepted ? null : result.message,
            }));
          },
          onIntelligence: (message: FrameIntelligence) => {
            if (cancelled) return;
            if (
              message.session_id !== session.session_id ||
              message.sequence <= latestSequenceRef.current
            ) {
              return;
            }
            latestSequenceRef.current = message.sequence;
            setIntelligence(message.result);
            setMetrics((current) => ({
              ...current,
              latestFrameId: Math.max(current.latestFrameId ?? -1, message.frame_id),
              modelStatus: describeModels(
                message.result.system_status.segmentation_details,
                message.result.system_status.detection_details,
              ),
              analysisStatus: message.result.system_status.inference_state ?? "LIVE",
            }));
          },
          onMalformedMessage: () => {
            if (sourceCompleteRef.current && (completionRequestedRef.current || forceCompletionRef.current)) return;
            clearAckTimer();
            inFlightRef.current = false;
            const finalizingSameSession = requestSameSessionCompletion();
            setMetrics((current) => ({
              ...current,
              connectionState: "malformed",
              sessionState: finalizingSameSession ? "FINALIZING" : current.sessionState,
              lastError: finalizingSameSession
                ? "The final frame acknowledgement was malformed. Finalizing server-accepted work on the same session."
                : "The ingestion server returned a malformed acknowledgement.",
            }));
          },
          onError: () => {
            if (cancelled) return;
            if (sourceCompleteRef.current && (completionRequestedRef.current || forceCompletionRef.current)) return;
            clearAckTimer();
            inFlightRef.current = false;
            const finalizingSameSession = requestSameSessionCompletion();
            setMetrics((current) => ({
              ...current,
              connectionState: "offline",
              sessionState: finalizingSameSession ? "FINALIZING" : current.sessionState,
              lastError: finalizingSameSession
                ? "The frame connection failed after the video ended. Finalizing server-accepted work on the same session."
                : "The frame WebSocket encountered a connection error.",
            }));
          },
          onClose: () => {
            if (cancelled || intentionalCloseRef.current) return;
            if (sourceCompleteRef.current && (completionRequestedRef.current || forceCompletionRef.current)) return;
            clearAckTimer();
            inFlightRef.current = false;
            const finalizingSameSession = requestSameSessionCompletion();
            setMetrics((current) => ({
              ...current,
              connectionState: "offline",
              sessionState: finalizingSameSession ? "FINALIZING" : current.sessionState,
              lastError: finalizingSameSession
                ? "The frame connection closed after the video ended. Finalizing server-accepted work on the same session."
                : "The frame WebSocket disconnected.",
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
      if (activeSessionIdRef.current === sessionId) activeSessionIdRef.current = null;
      if (sessionId) void deleteIngestionSession(sessionId).catch(() => undefined);
    };
  }, [clearAckTimer, detectorMode, mediaOrigin, requestSameSessionCompletion, retryGeneration, sourceGeneration, sourceMode, sourceReady]);

  useEffect(() => {
    if (!sourceComplete) return;
    setCompletionState((current) => current === "IDLE" ? "FINALIZING" : current);
    const sessionId = metrics.sessionId;
    if (
      !sessionId ||
      (metrics.capturedFrames !== metrics.acknowledgedFrames && !forceCompletionRef.current) ||
      completionRequestedRef.current
    ) {
      return;
    }

    completionRequestedRef.current = true;
    forceCompletionRef.current = false;
    setMetrics((current) => ({
      ...current,
      sessionState: "FINALIZING",
      connectionState: current.connectionState === "offline" ? "offline" : "paused",
    }));
    void completeIngestionSession(sessionId)
      .then((completed) => {
        if (activeSessionIdRef.current !== sessionId) return;
        setCompletion(completed);
        setCompletionState("COMPLETE");
        setIntelligence(completed.latest_result);
        setMetrics((current) => ({
          ...current,
          sessionState: "COMPLETE",
          connectionState: "stopped",
          latestFrameId: completed.summary.last_analyzed_frame_id ?? current.latestFrameId,
          modelStatus: describeModels(
            completed.summary.segmentation_status,
            completed.summary.detection_status,
          ),
          analysisStatus: completed.summary.inference_state,
          lastError: null,
        }));
      })
      .catch((completionError: unknown) => {
        if (activeSessionIdRef.current !== sessionId) return;
        setCompletionState("ERROR");
        setMetrics((current) => ({
          ...current,
          sessionState: "FINALIZING",
          lastError: completionError instanceof Error
            ? completionError.message
            : "Unable to finalize whole-video findings.",
        }));
      });
  }, [completionRetryGeneration, metrics.acknowledgedFrames, metrics.capturedFrames, metrics.sessionId, sourceComplete]);

  useEffect(() => {
    if (!sourceReady || !sourceMode || !mediaOrigin || !captureActive) {
      if (sourceReady) {
        setMetrics((current) => ({
          ...current,
          connectionState:
            sourceComplete || current.sessionState === "FINALIZING" || current.sessionState === "COMPLETE"
              ? current.connectionState
              : current.sessionId && current.connectionState !== "offline"
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
        if (cancelled || !captureActiveRef.current) {
          inFlightRef.current = false;
          return;
        }
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
          const finalizingSameSession = requestSameSessionCompletion();
          setMetrics((current) => ({
            ...current,
            connectionState: "offline",
            sessionState: finalizingSameSession ? "FINALIZING" : current.sessionState,
            lastError: finalizingSameSession
              ? "The final frame acknowledgement timed out. Finalizing server-accepted work on the same session."
              : "Frame acknowledgement timed out.",
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
  }, [captureActive, mediaOrigin, metrics.connectionState, metrics.requestedFps, requestSameSessionCompletion, sourceComplete, sourceMode, sourceReady, videoElement]);

  return {
    metrics,
    intelligence,
    completionState,
    completion,
    retry: () => {
      if (sourceComplete && completionState !== "COMPLETE" && metrics.sessionId) {
        if (completionState === "FINALIZING" && completionRequestedRef.current) return;
        completionRequestedRef.current = false;
        requestSameSessionCompletion();
        setMetrics((current) => ({
          ...current,
          sessionState: "FINALIZING",
          lastError: null,
        }));
        return;
      }
      setRetryGeneration((value) => value + 1);
    },
  };
}
