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

    render(<CommandCenter />);

    expect(screen.getByLabelText("Loading FloodSight command centre")).toBeInTheDocument();
    expect(screen.getByText("Loading deterministic incident")).toBeInTheDocument();
  });

  it("renders simulated provenance, headline statistics, and backend-ranked zones", () => {
    render(<CommandCenter />);

    expect(screen.getByText("Riverside Ward Flood Response")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Data origin: DEMO_SIMULATED").length).toBeGreaterThan(2);
    expect(screen.getByText("42%")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Stream connection: Replay complete").length).toBeGreaterThan(0);
    expect(screen.getByText("No geographic scale, distance, or travel time")).toBeInTheDocument();

    const orderedIds = [...document.querySelectorAll<HTMLElement>("[data-zone-id]")].map(
      (element) => element.dataset.zoneId,
    );
    expect(orderedIds).toEqual(["ZONE-2", "ZONE-4", "ZONE-1"]);
  });

  it("offers all media sources while keeping simulation as the functional default", () => {
    render(<CommandCenter />);

    const selector = screen.getByRole("group", { name: "Media source selector" });
    expect(within(selector).getByRole("button", { name: /Simulation/ })).toHaveAttribute("aria-pressed", "true");
    expect(within(selector).getByRole("button", { name: /Video file/ })).toBeEnabled();
    expect(within(selector).getByRole("button", { name: /Webcam/ })).toBeEnabled();
    expect(screen.getByLabelText("Simulation controls")).toBeInTheDocument();
  });

  it("labels actual local media honestly without substituting simulated analytics", () => {
    const createObjectURL = vi.fn(() => "blob:local-flood-video");
    const revokeObjectURL = vi.fn();
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: createObjectURL },
      revokeObjectURL: { configurable: true, value: revokeObjectURL },
    });
    render(<CommandCenter />);
    fireEvent.click(screen.getByRole("button", { name: /Video file/ }));
    const input = screen.getByLabelText("Choose video");
    const file = new File(["local"], "incident.webm", { type: "video/webm" });
    fireEvent.change(input, { target: { files: [file] } });

    expect(createObjectURL).toHaveBeenCalledWith(file);
    expect(screen.getByLabelText("Media origin: USER_VIDEO_FILE")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Normalized simulated flood observation/)).not.toBeInTheDocument();
    expect(screen.queryByText("42%")).not.toBeInTheDocument();
    expect(screen.getByText(/Awaiting the first backend-computed intelligence update/)).toBeInTheDocument();
    expect(screen.getByText(/Demo values are not substituted/)).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: /Video file/ }));

    expect(screen.getByLabelText(/Normalized model-derived flood observation/)).toBeInTheDocument();
    expect(screen.getByText(/BACKEND INTELLIGENCE/)).toBeInTheDocument();
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
    const { rerender } = render(<CommandCenter />);
    vi.mocked(useDemoIncident).mockReturnValue({
      ...readyState,
      snapshot: reorderedSnapshot,
      connectionState: "connected",
    });

    rerender(<CommandCenter />);

    const orderedIds = [...document.querySelectorAll<HTMLElement>("[data-zone-id]")].map(
      (element) => element.dataset.zoneId,
    );
    expect(orderedIds).toEqual(["ZONE-4", "ZONE-1", "ZONE-2"]);
  });

  it("opens an explainable zone drawer from the priority list", () => {
    render(<CommandCenter />);
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

  it("keeps observation and tactical layer controls synchronized", () => {
    render(<CommandCenter />);
    const floodButtons = screen.getAllByRole("button", { name: "Flood" });
    expect(floodButtons).toHaveLength(2);
    expect(floodButtons[0]).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(floodButtons[0]);

    expect(screen.getAllByRole("button", { name: "Flood" })[0]).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getAllByRole("button", { name: "Flood" })[1]).toHaveAttribute(
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
    render(<CommandCenter />);

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
    render(<CommandCenter />);

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
    render(<CommandCenter />);

    expect(screen.getByText(/Current values are retained from the last valid backend message/)).toBeInTheDocument();
    expect(screen.getByText("42%")).toBeInTheDocument();
  });
});
