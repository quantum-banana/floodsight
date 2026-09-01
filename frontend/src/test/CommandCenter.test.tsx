import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CommandCenter } from "../features/command-center/CommandCenter";
import { useDemoIncident } from "../hooks/useDemoIncident";
import { useFrameIngestion } from "../hooks/useFrameIngestion";
import { getDemoIncidentReport } from "../services/api";
import { getLiveIncidentReport } from "../services/ingestionApi";
import type { IncidentReport } from "../types/api";
import { commandSnapshot, incidentDetail, liveSnapshot, reorderedSnapshot } from "./fixtures";

vi.mock("../hooks/useDemoIncident", () => ({
  useDemoIncident: vi.fn(),
}));
vi.mock("../hooks/useFrameIngestion", () => ({
  useFrameIngestion: vi.fn(),
}));
vi.mock("../services/api", () => ({
  getDemoIncidentReport: vi.fn(),
}));
vi.mock("../services/ingestionApi", () => ({
  getLiveIncidentReport: vi.fn(),
}));

const controls = {
  start: vi.fn(),
  pause: vi.fn(),
  resume: vi.fn(),
  reset: vi.fn(),
  retry: vi.fn(),
};

const readyState = {
  detail: incidentDetail,
  snapshot: commandSnapshot,
  connectionState: "complete" as const,
  error: null,
  ...controls,
};

const ingestionMetrics = {
  sessionId: null,
  sessionState: null,
  sourceMode: null,
  mediaOrigin: null,
  detectorMode: null,
  connectionState: "idle" as const,
  requestedFps: 4,
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
  analysisStatus: "AWAITING_FRAME" as const,
};

const incidentReport: IncidentReport = {
  incident_id: commandSnapshot.incident_id,
  title: commandSnapshot.incident.title,
  generated_at_ms: commandSnapshot.timestamp_ms,
  severity: commandSnapshot.incident_severity,
  statistics: commandSnapshot.statistics,
  critical_zone_count: 1,
  highest_priority_zone_id: "ZONE-2",
  highest_priority_zone_name: "Zone 2",
  explanation: "Backend-supplied priority explanation.",
  access_summary: "Backend-supplied relative route.",
  responsible_ai_statement: "FloodSight is decision support and requires human review.",
  data_origin: "DEMO_SIMULATED",
  generated_from_frame_id: 5,
  priority_order: ["ZONE-2", "ZONE-4", "ZONE-1"],
  reason_codes: ["FLOOD_EXPOSURE"],
  route: commandSnapshot.route,
  model_provenance: { segmentation: "SIMULATED", detection: "SIMULATED" },
};

describe("FloodSight command center", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useDemoIncident).mockReturnValue(readyState);
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: ingestionMetrics,
      intelligence: null,
      retry: vi.fn(),
    });
    vi.mocked(getDemoIncidentReport).mockResolvedValue(incidentReport);
    vi.mocked(getLiveIncidentReport).mockResolvedValue({
      ...incidentReport,
      incident_id: liveSnapshot.incident_id,
      title: liveSnapshot.incident.title,
      severity: liveSnapshot.incident_severity,
      statistics: liveSnapshot.statistics,
      data_origin: "DERIVED_ANALYTIC",
      model_provenance: { segmentation: "REAL_MODEL", detection: "PRETRAINED_FALLBACK" },
    });
  });

  it("shows a contract-loading state before incident data is available", () => {
    vi.mocked(useDemoIncident).mockReturnValue({
      ...readyState,
      detail: null,
      snapshot: null,
      connectionState: "loading",
    });

    render(<CommandCenter demoMode />);

    expect(screen.getByLabelText("Loading FloodSight command centre")).toBeInTheDocument();
    expect(screen.getByText("Preparing intelligence…")).toBeInTheDocument();
  });

  it("renders simulated provenance, headline statistics, and backend-ranked zones", () => {
    render(<CommandCenter demoMode />);

    expect(screen.getByText("Riverside Ward Flood Response")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Data origin: DEMO_SIMULATED").length).toBeGreaterThan(0);
    expect(screen.getByText("42%")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Stream connection: Replay complete").length).toBeGreaterThan(0);
    expect(screen.getByText("No geographic scale, distance, or travel time")).toBeInTheDocument();

    const orderedIds = [...document.querySelectorAll<HTMLElement>("[data-zone-id]")].map(
      (element) => element.dataset.zoneId,
    );
    expect(orderedIds).toEqual(["ZONE-2", "ZONE-4", "ZONE-1"]);
  });

  it("makes actual media the professional default without simulation or replay language", () => {
    render(<CommandCenter />);

    const selector = screen.getByRole("group", { name: "Media source selector" });
    expect(within(selector).getByRole("button", { name: "Video" })).toHaveAttribute("aria-pressed", "true");
    expect(within(selector).getByRole("button", { name: "Live camera" })).toBeEnabled();
    expect(screen.getAllByLabelText("Choose video").length).toBeGreaterThan(0);
    expect(screen.getByLabelText("Detection profile: High recall")).toBeInTheDocument();
    expect(vi.mocked(useFrameIngestion).mock.calls.at(-1)?.[0].detectorMode).toBe("AERIAL_HIGH_RECALL");
    expect(screen.queryByText(/simulation|replay|demo scenario/i)).not.toBeInTheDocument();
  });

  it("allows an operator to change the detector profile explicitly", () => {
    render(<CommandCenter />);

    fireEvent.click(screen.getByLabelText("Detection profile: High recall"));
    const detectorSelector = screen.getByRole("group", { name: "Detector inference mode" });
    expect(within(detectorSelector).getByRole("button", { name: /High recall/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(within(detectorSelector).getByRole("button", { name: /Standard/i }));

    expect(vi.mocked(useFrameIngestion).mock.calls.at(-1)?.[0].detectorMode).toBe("STANDARD");
    expect(screen.getByLabelText("Detection profile: Standard")).toBeInTheDocument();
  });

  it("labels actual local media honestly without substituting simulated analytics", () => {
    const createObjectURL = vi.fn(() => "blob:local-flood-video");
    const revokeObjectURL = vi.fn();
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: createObjectURL },
      revokeObjectURL: { configurable: true, value: revokeObjectURL },
    });
    render(<CommandCenter />);
    const input = document.querySelector<HTMLInputElement>('#video-file-input');
    expect(input).not.toBeNull();
    const file = new File(["local"], "incident.webm", { type: "video/webm" });
    fireEvent.change(input as HTMLInputElement, { target: { files: [file] } });

    expect(createObjectURL).toHaveBeenCalledWith(file);
    expect(screen.getByLabelText("Media origin: USER_VIDEO_FILE")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Normalized simulated flood observation/)).not.toBeInTheDocument();
    expect(screen.queryByText("42%")).not.toBeInTheDocument();
    expect(screen.getByText("Awaiting intelligence")).toBeInTheDocument();
    expect(screen.queryByText(/demo|simulation|replay/i)).not.toBeInTheDocument();
  });

  it("renders backend intelligence and class-aware provenance over actual media", async () => {
    const reroutedSnapshot = {
      ...liveSnapshot,
      route: liveSnapshot.route && {
        ...liveSnapshot.route,
        changed_reason: "Primary access became unsafe; a new relative route is recommended.",
        changed_reason_code: "ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE",
        previous_edge_ids: ["EDGE-C1-D1"],
      },
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        sessionId: "frame-session-1234567890",
        connectionState: "streaming",
        analysisStatus: "LIVE",
        modelStatus: "SEG REAL · DET FALLBACK",
      },
      intelligence: reroutedSnapshot,
      retry: vi.fn(),
    });
    render(<CommandCenter />);

    expect(screen.getByLabelText(/Normalized model-derived flood observation/)).toBeInTheDocument();
    expect(screen.getByLabelText("Actual media state: LIVE")).toBeInTheDocument();
    expect(screen.getByLabelText("Segmentation class legend")).toHaveTextContent("water");
    expect(screen.getByLabelText("Segmentation class legend")).toHaveTextContent("pool");
    expect(screen.getByLabelText("Inference model status")).toHaveTextContent("REAL");
    expect(screen.getByLabelText("Inference model status")).toHaveTextContent("FALLBACK");
    expect(screen.getByLabelText("Evidence frame freshness")).toHaveTextContent("current inference");
    expect(screen.getByText("Previous route no longer preferred")).toBeInTheDocument();
    expect(screen.getByText("ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE")).toBeInTheDocument();
    expect(screen.getByText("38%")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));
    expect(await screen.findByText("Backend-supplied priority explanation.")).toBeInTheDocument();
    expect(getLiveIncidentReport).toHaveBeenCalledWith("frame-session-1234567890");
  });

  it("reorders rescue cards when a later backend snapshot changes rank", () => {
    const { rerender } = render(<CommandCenter demoMode />);
    vi.mocked(useDemoIncident).mockReturnValue({
      ...readyState,
      snapshot: reorderedSnapshot,
      connectionState: "connected",
    });

    rerender(<CommandCenter demoMode />);

    const orderedIds = [...document.querySelectorAll<HTMLElement>("[data-zone-id]")].map(
      (element) => element.dataset.zoneId,
    );
    expect(orderedIds).toEqual(["ZONE-4", "ZONE-1", "ZONE-2"]);
  });

  it("opens an explainable zone drawer from the priority list", () => {
    render(<CommandCenter demoMode />);
    const zoneCard = document.querySelector<HTMLElement>('[data-zone-id="ZONE-2"]');
    expect(zoneCard).not.toBeNull();

    fireEvent.click(within(zoneCard as HTMLElement).getByRole("button", { name: "View zone" }));

    const drawer = screen.getByRole("dialog", { name: "Zone 2" });
    expect(drawer).toHaveTextContent("92/100");
    expect(drawer).toHaveTextContent("ISOLATED");
    expect(drawer).toHaveTextContent("Flood exposure");
    expect(drawer).toHaveTextContent("Potential stranded person");
    expect(drawer).toHaveTextContent("PERSON_IN_HIGH_FLOOD_ZONE");
    expect(drawer).toHaveTextContent("Supplied contribution total: 92");
  });

  it("uses one canvas layer toolbar to control observation and tactical overlays", () => {
    render(<CommandCenter demoMode />);
    const floodButtons = screen.getAllByRole("button", { name: "Flood" });
    expect(floodButtons).toHaveLength(1);
    expect(floodButtons[0]).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(floodButtons[0]);

    expect(screen.getAllByRole("button", { name: "Flood" })[0]).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("loads a responsible, copyable incident report from the backend", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<CommandCenter demoMode />);

    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));

    const report = screen.getByRole("dialog", { name: "Incident report" });
    expect(await within(report).findByText("42%")).toBeInTheDocument();
    expect(within(report).getByText("Flood coverage")).toBeInTheDocument();
    expect(within(report).getByText("Blocked roads")).toBeInTheDocument();
    expect(report).toHaveTextContent("requires human review");
    fireEvent.click(within(report).getByRole("button", { name: "Copy report text" }));
    expect(writeText).toHaveBeenCalledOnce();
    expect(getDemoIncidentReport).toHaveBeenCalledWith("FS-001");
  });

  it("shows an honest offline state and exposes retry without local fallback data", () => {
    vi.mocked(useDemoIncident).mockReturnValue({
      ...readyState,
      detail: null,
      snapshot: null,
      connectionState: "offline",
      error: "Unable to reach the FloodSight API.",
    });
    render(<CommandCenter demoMode />);

    expect(screen.getByRole("alert")).toHaveTextContent("No local incident values");
    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }));
    expect(controls.retry).toHaveBeenCalledOnce();
  });

  it("retains the last valid backend snapshot when a stream message is malformed", () => {
    vi.mocked(useDemoIncident).mockReturnValue({
      ...readyState,
      connectionState: "malformed",
      error: "The demo backend sent an invalid message.",
    });
    render(<CommandCenter demoMode />);

    expect(screen.getByText(/Last valid intelligence retained/)).toBeInTheDocument();
    expect(screen.getByText("42%")).toBeInTheDocument();
  });
});
