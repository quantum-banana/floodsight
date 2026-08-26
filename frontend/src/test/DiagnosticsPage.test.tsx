import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { getSystemSnapshot } from "../services/api";
import { diagnosticsSnapshot } from "./fixtures";

vi.mock("../hooks/useDemoIncident", () => ({ useDemoIncident: vi.fn() }));
vi.mock("../services/api", () => ({
  getDemoIncident: vi.fn(),
  getSystemSnapshot: vi.fn(),
}));

describe("FloodSight system diagnostics", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/system");
    window.sessionStorage.setItem(
      "floodsight.ingestion.diagnostics.v1",
      JSON.stringify({
        sessionId: "session-1234567890",
        sourceMode: "VIDEO_FILE",
        mediaOrigin: "USER_VIDEO_FILE",
        connectionState: "streaming",
        requestedFps: 4,
        measuredFps: 3.8,
        capturedFrames: 12,
        acknowledgedFrames: 11,
        rejectedFrames: 1,
        clientDroppedFrames: 2,
        latestFrameId: 11,
        latestDimensions: "1280×720",
        latestBlurScore: 87.3,
        latestLuminance: 112.4,
        latestProcessingMs: 3.2,
        latestQualityState: "NORMAL / NORMAL",
        lastError: null,
        modelStatus: "NOT_CONFIGURED",
        analysisStatus: "DEMO_SIMULATED",
      }),
    );
    vi.mocked(getSystemSnapshot).mockResolvedValue(diagnosticsSnapshot);
  });

  it("preserves the Phase 0 API and model readiness diagnostics", async () => {
    render(<App />);

    expect(screen.getByLabelText("Connecting to FloodSight backend")).toBeInTheDocument();
    expect(await screen.findByText("Incident FS-001")).toBeInTheDocument();
    expect(screen.getByLabelText("Data origin: DEMO_SIMULATED")).toHaveTextContent(
      "DEMO_SIMULATED",
    );
    expect(screen.getByLabelText("Backend connection: Connected")).toBeInTheDocument();
    expect(screen.getByLabelText("Segmentation model: Not configured")).toBeInTheDocument();
    expect(screen.getByLabelText("Detection model: Not configured")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Command center" })).toHaveAttribute("href", "/");
    expect(screen.getByRole("heading", { name: "Frame ingestion diagnostics" })).toBeInTheDocument();
    expect(screen.getByText("1280×720")).toBeInTheDocument();
    expect(screen.getByText("USER_VIDEO_FILE")).toBeInTheDocument();
    expect(screen.getAllByText("NOT_CONFIGURED").length).toBeGreaterThan(0);
  });
});
