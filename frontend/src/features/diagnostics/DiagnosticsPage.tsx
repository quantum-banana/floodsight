import { useEffect, useState } from "react";

import { LoadingScreen } from "../system-status/LoadingScreen";
import { OfflineScreen } from "../system-status/OfflineScreen";
import { SystemStatusScreen } from "../system-status/SystemStatusScreen";
import { getSystemSnapshot } from "../../services/api";
import type { SystemSnapshot } from "../../types/api";

type DiagnosticsState =
  | { status: "loading" }
  | { status: "ready"; snapshot: SystemSnapshot }
  | { status: "offline"; message: string };

export function DiagnosticsPage() {
  const [view, setView] = useState<DiagnosticsState>({ status: "loading" });

  const load = async (showLoading: boolean) => {
    if (showLoading) setView({ status: "loading" });
    try {
      const snapshot = await getSystemSnapshot();
      setView({ status: "ready", snapshot });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to reach the FloodSight API.";
      setView({ status: "offline", message });
    }
  };

  useEffect(() => {
    let isActive = true;
    void getSystemSnapshot()
      .then((snapshot) => {
        if (isActive) setView({ status: "ready", snapshot });
      })
      .catch((error: unknown) => {
        if (!isActive) return;
        const message = error instanceof Error ? error.message : "Unable to reach the FloodSight API.";
        setView({ status: "offline", message });
      });
    return () => {
      isActive = false;
    };
  }, []);

  if (view.status === "loading") return <LoadingScreen />;
  if (view.status === "offline") return <OfflineScreen message={view.message} onRetry={() => void load(true)} />;
  return <SystemStatusScreen snapshot={view.snapshot} />;
}
