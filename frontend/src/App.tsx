import { useCallback, useEffect, useState } from "react";

import { LoadingScreen } from "./features/system-status/LoadingScreen";
import { OfflineScreen } from "./features/system-status/OfflineScreen";
import { SystemStatusScreen } from "./features/system-status/SystemStatusScreen";
import { getSystemSnapshot } from "./services/api";
import type { SystemSnapshot } from "./types/api";

type ViewState =
  | { status: "loading" }
  | { status: "ready"; snapshot: SystemSnapshot }
  | { status: "offline"; message: string };

function App() {
  const [view, setView] = useState<ViewState>({ status: "loading" });

  const retry = useCallback(async () => {
    setView({ status: "loading" });
    try {
      const snapshot = await getSystemSnapshot();
      setView({ status: "ready", snapshot });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to reach the FloodSight API.";
      setView({ status: "offline", message });
    }
  }, []);

  useEffect(() => {
    let isActive = true;

    void getSystemSnapshot()
      .then((snapshot) => {
        if (isActive) {
          setView({ status: "ready", snapshot });
        }
      })
      .catch((error: unknown) => {
        if (isActive) {
          const message = error instanceof Error ? error.message : "Unable to reach the FloodSight API.";
          setView({ status: "offline", message });
        }
      });

    return () => {
      isActive = false;
    };
  }, []);

  if (view.status === "loading") {
    return <LoadingScreen />;
  }

  if (view.status === "offline") {
    return <OfflineScreen message={view.message} onRetry={() => void retry()} />;
  }

  return <SystemStatusScreen snapshot={view.snapshot} />;
}

export default App;
