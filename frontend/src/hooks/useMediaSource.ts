import { useCallback, useEffect, useRef, useState } from "react";

import { VIDEO_FILE_MAX_BYTES } from "../config/environment";
import type { ActualSourceMode, MediaOrigin } from "../types/ingestion";

export type MediaSourceMode = "SIMULATION" | ActualSourceMode;
export type MediaSourceState =
  | "IDLE"
  | "PREPARING"
  | "READY"
  | "PLAYING"
  | "PAUSED"
  | "STOPPED"
  | "ERROR";

export interface VideoFileInfo {
  name: string;
  size: number;
  type: string;
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
}

export interface MediaSourceController {
  mode: MediaSourceMode;
  state: MediaSourceState;
  mediaOrigin: MediaOrigin | null;
  videoElement: HTMLVideoElement | null;
  bindVideoElement: (element: HTMLVideoElement | null) => void;
  videoSrc: string | null;
  stream: MediaStream | null;
  fileInfo: VideoFileInfo | null;
  error: string | null;
  generation: number;
  readyForIngestion: boolean;
  isPlaying: boolean;
  selectMode: (mode: MediaSourceMode) => void;
  selectFile: (file: File) => boolean;
  start: () => Promise<void>;
  pause: () => void;
  resume: () => Promise<void>;
  stop: () => void;
  onLoadedMetadata: () => void;
  onEnded: () => void;
}

const cameraErrorMessage = (error: unknown) => {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Camera permission was denied. Allow camera access in the browser and retry.";
    }
    if (error.name === "NotFoundError") return "No camera device is available.";
    if (error.name === "NotReadableError") return "The camera is already in use or unavailable.";
    if (error.name === "OverconstrainedError") return "The camera cannot satisfy the requested video settings.";
  }
  return "Unable to start the camera. Check browser permissions and device availability.";
};

export function useMediaSource(initialMode: MediaSourceMode = "VIDEO_FILE"): MediaSourceController {
  const [mode, setMode] = useState<MediaSourceMode>(initialMode);
  const [state, setState] = useState<MediaSourceState>("IDLE");
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [fileInfo, setFileInfo] = useState<VideoFileInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [dimensionsReady, setDimensionsReady] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef(true);

  const bindVideoElement = useCallback((element: HTMLVideoElement | null) => {
    videoRef.current = element;
    setVideoElement(element);
    if (element && streamRef.current) element.srcObject = streamRef.current;
  }, []);

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const revokeObjectUrl = useCallback(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
  }, []);

  const releaseCurrentSource = useCallback(() => {
    videoRef.current?.pause();
    stopTracks();
    revokeObjectUrl();
  }, [revokeObjectUrl, stopTracks]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      releaseCurrentSource();
    };
  }, [releaseCurrentSource]);

  const selectMode = useCallback(
    (nextMode: MediaSourceMode) => {
      if (nextMode === mode) return;
      releaseCurrentSource();
      setMode(nextMode);
      setState("IDLE");
      setVideoSrc(null);
      setStream(null);
      setFileInfo(null);
      setDimensionsReady(false);
      setError(null);
      setGeneration((value) => value + 1);
    },
    [mode, releaseCurrentSource],
  );

  const selectFile = useCallback(
    (file: File) => {
      if (!file.type.toLowerCase().startsWith("video/")) {
        setError("Choose a browser-playable video file (for example MP4 or WebM).");
        setState("ERROR");
        return false;
      }
      if (file.size > VIDEO_FILE_MAX_BYTES) {
        setError(`Video files must be ${Math.round(VIDEO_FILE_MAX_BYTES / 1024 / 1024)} MB or smaller.`);
        setState("ERROR");
        return false;
      }

      videoRef.current?.pause();
      stopTracks();
      revokeObjectUrl();
      const nextUrl = URL.createObjectURL(file);
      objectUrlRef.current = nextUrl;
      setMode("VIDEO_FILE");
      setVideoSrc(nextUrl);
      setStream(null);
      setFileInfo({
        name: file.name,
        size: file.size,
        type: file.type,
        durationSeconds: null,
        width: null,
        height: null,
      });
      setError(null);
      setDimensionsReady(false);
      setState("PREPARING");
      setGeneration((value) => value + 1);
      return true;
    },
    [revokeObjectUrl, stopTracks],
  );

  const onLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (mode === "VIDEO_FILE") {
      setFileInfo((current) =>
        current
          ? {
              ...current,
              durationSeconds: Number.isFinite(video.duration) ? video.duration : null,
              width: video.videoWidth,
              height: video.videoHeight,
            }
          : current,
      );
    }
    setDimensionsReady(Boolean(video.videoWidth && video.videoHeight));
    setState((current) => (current === "PLAYING" ? current : "READY"));
  }, [mode]);

  const start = useCallback(async () => {
    setError(null);
    if (mode === "SIMULATION") return;
    const video = videoRef.current;
    if (!video) {
      setError("The video preview is not ready.");
      setState("ERROR");
      return;
    }

    if (mode === "WEBCAM" && !streamRef.current) {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("This browser does not provide camera access.");
        setState("ERROR");
        return;
      }
      setState("PREPARING");
      try {
        const nextStream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: { ideal: "environment" },
          },
        });
        if (!mountedRef.current) {
          nextStream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = nextStream;
        setStream(nextStream);
        video.srcObject = nextStream;
        setGeneration((value) => value + 1);
      } catch (cameraError) {
        setError(cameraErrorMessage(cameraError));
        setState("ERROR");
        return;
      }
    } else if (mode === "VIDEO_FILE" && !objectUrlRef.current) {
      setError("Select a local video file before starting playback.");
      setState("ERROR");
      return;
    }

    try {
      await video.play();
      if (mountedRef.current) setState("PLAYING");
    } catch {
      if (mountedRef.current) {
        setError("The browser could not play this media. The codec may be unsupported.");
        setState("ERROR");
      }
    }
  }, [mode]);

  const pause = useCallback(() => {
    videoRef.current?.pause();
    setState("PAUSED");
  }, []);

  const resume = useCallback(async () => {
    await start();
  }, [start]);

  const stop = useCallback(() => {
    const video = videoRef.current;
    video?.pause();
    if (mode === "VIDEO_FILE" && video) video.currentTime = 0;
    if (mode === "WEBCAM") {
      stopTracks();
      setStream(null);
      setDimensionsReady(false);
      setGeneration((value) => value + 1);
    }
    setState("STOPPED");
  }, [mode, stopTracks]);

  const onEnded = useCallback(() => setState("STOPPED"), []);
  const mediaOrigin =
    mode === "VIDEO_FILE"
      ? "USER_VIDEO_FILE"
      : mode === "WEBCAM"
        ? "USER_WEBCAM"
        : null;
  const readyForIngestion =
    state !== "STOPPED" && state !== "ERROR" && mode === "VIDEO_FILE"
      ? Boolean(videoSrc && fileInfo?.width && fileInfo.height)
      : state !== "STOPPED" && state !== "ERROR" && mode === "WEBCAM"
        ? Boolean(stream && dimensionsReady)
        : false;

  return {
    mode,
    state,
    mediaOrigin,
    videoElement,
    bindVideoElement,
    videoSrc,
    stream,
    fileInfo,
    error,
    generation,
    readyForIngestion,
    isPlaying: state === "PLAYING",
    selectMode,
    selectFile,
    start,
    pause,
    resume,
    stop,
    onLoadedMetadata,
    onEnded,
  };
}
