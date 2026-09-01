import { useCallback, useEffect, useRef, useState } from "react";

import { getDemoIncident } from "../services/api";
import { openDemoStream, type DemoStreamConnection } from "../services/demoStream";
import type { IncidentDetailResponse } from "../types/api";
import type { LiveResult } from "../types/liveResult";

export type ConnectionState =
  | "loading"
  | "connecting"
  | "connected"
  | "paused"
  | "reconnecting"
  | "complete"
  | "offline"
  | "malformed"
  | "disconnected";

interface DemoIncidentState {
  detail: IncidentDetailResponse | null;
  snapshot: LiveResult | null;
  connectionState: ConnectionState;
  error: string | null;
  start: () => void;
  pause: () => void;
  resume: () => void;
  reset: () => void;
  retry: () => void;
}

const INCIDENT_ID = "FS-001";
const MAX_RECONNECT_ATTEMPTS = 3;

export function useDemoIncident(enabled = true): DemoIncidentState {
  const [detail, setDetail] = useState<IncidentDetailResponse | null>(null);
  const [snapshot, setSnapshot] = useState<LiveResult | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("loading");
  const [error, setError] = useState<string | null>(null);
  const connectionRef = useRef<DemoStreamConnection | null>(null);
  const intentionalCloseRef = useRef(false);
  const currentIndexRef = useRef(0);
  const snapshotCountRef = useRef(1);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const connectionGenerationRef = useRef(0);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const closeCurrent = useCallback(() => {
    intentionalCloseRef.current = true;
    connectionGenerationRef.current += 1;
    const connection = connectionRef.current;
    connectionRef.current = null;
    connection?.close();
  }, []);

  const connect = useCallback(
    function connectStream(startIndex: number, reconnecting = false) {
      clearReconnectTimer();
      closeCurrent();
      intentionalCloseRef.current = false;
      const connectionGeneration = ++connectionGenerationRef.current;
      setConnectionState(reconnecting ? "reconnecting" : "connecting");
      setError(null);

      const connection = openDemoStream(INCIDENT_ID, startIndex, {
        onOpen: () => {
          if (!mountedRef.current) return;
          reconnectAttemptsRef.current = 0;
          setConnectionState("connected");
        },
        onMessage: (nextSnapshot) => {
          if (!mountedRef.current) return;
          currentIndexRef.current = nextSnapshot.snapshot_index;
          snapshotCountRef.current = nextSnapshot.snapshot_count;
          setSnapshot(nextSnapshot);
        },
        onMalformedMessage: () => {
          if (!mountedRef.current) return;
          intentionalCloseRef.current = true;
          connectionGenerationRef.current += 1;
          const activeConnection = connectionRef.current;
          connectionRef.current = null;
          activeConnection?.close();
          setError("The demo backend sent a message that did not match the live-result contract.");
          setConnectionState("malformed");
        },
        onError: () => {
          if (!mountedRef.current || intentionalCloseRef.current) return;
          setConnectionState("reconnecting");
        },
        onClose: (event) => {
          if (
            !mountedRef.current ||
            intentionalCloseRef.current ||
            connectionGenerationRef.current !== connectionGeneration
          ) return;
          connectionRef.current = null;
          const isComplete =
            event.code === 1_000 && currentIndexRef.current >= snapshotCountRef.current - 1;
          if (isComplete) {
            setConnectionState("complete");
            return;
          }

          reconnectAttemptsRef.current += 1;
          if (reconnectAttemptsRef.current > MAX_RECONNECT_ATTEMPTS) {
            setError("The deterministic demo stream disconnected after repeated retries.");
            setConnectionState("disconnected");
            return;
          }
          setConnectionState("reconnecting");
          reconnectTimerRef.current = window.setTimeout(() => {
            connectStream(currentIndexRef.current + 1, true);
          }, 900);
        },
      });
      connectionRef.current = connection;
    },
    [clearReconnectTimer, closeCurrent],
  );

  const loadIncident = useCallback(async () => {
    clearReconnectTimer();
    closeCurrent();
    setConnectionState("loading");
    setError(null);
    try {
      const incident = await getDemoIncident(INCIDENT_ID);
      if (!mountedRef.current) return;
      setDetail(incident);
      setSnapshot(incident.initial_snapshot);
      currentIndexRef.current = 0;
      snapshotCountRef.current = incident.snapshot_count;
      reconnectAttemptsRef.current = 0;
      connect(0);
    } catch (loadError) {
      if (!mountedRef.current) return;
      const message =
        loadError instanceof Error ? loadError.message : "Unable to reach the FloodSight API.";
      setError(message);
      setConnectionState("offline");
    }
  }, [clearReconnectTimer, closeCurrent, connect]);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled) {
      return () => {
        mountedRef.current = false;
        clearReconnectTimer();
        closeCurrent();
      };
    }
    const loadTimer = window.setTimeout(() => void loadIncident(), 0);
    return () => {
      mountedRef.current = false;
      window.clearTimeout(loadTimer);
      clearReconnectTimer();
      closeCurrent();
    };
  }, [clearReconnectTimer, closeCurrent, enabled, loadIncident]);

  const start = useCallback(() => {
    currentIndexRef.current = 0;
    connect(0);
  }, [connect]);

  const pause = useCallback(() => {
    clearReconnectTimer();
    closeCurrent();
    setConnectionState("paused");
  }, [clearReconnectTimer, closeCurrent]);

  const resume = useCallback(() => {
    const nextIndex = Math.min(currentIndexRef.current + 1, snapshotCountRef.current - 1);
    connect(nextIndex);
  }, [connect]);

  const reset = useCallback(() => {
    if (detail) {
      setSnapshot(detail.initial_snapshot);
      currentIndexRef.current = 0;
    }
    connect(0);
  }, [connect, detail]);

  return {
    detail,
    snapshot,
    connectionState,
    error,
    start,
    pause,
    resume,
    reset,
    retry: loadIncident,
  };
}
