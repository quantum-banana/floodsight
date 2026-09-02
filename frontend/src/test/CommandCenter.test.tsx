import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CommandCenter } from "../features/command-center/CommandCenter";
import { useDemoIncident } from "../hooks/useDemoIncident";
import { useFrameIngestion } from "../hooks/useFrameIngestion";
import { getDemoIncidentReport } from "../services/api";
import { getLiveIncidentReport } from "../services/ingestionApi";
import type { IncidentReport } from "../types/api";
import type { AggregateMetric, VideoAnalysisComplete } from "../types/ingestion";
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

const idleCompletion = {
  completionState: "IDLE" as const,
  completion: null,
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
  supporting_frame_count: 3,
  confidence: 0.9,
  data_origin: "DERIVED_ANALYTIC",
});

const unavailableMetric = (unit: AggregateMetric["unit"]): AggregateMetric => ({
  value: null,
  unit,
  availability: "MODEL_UNAVAILABLE",
  aggregation: unit === "percent" ? "PEAK_FRESH_SEGMENTATION" : "NOT_APPLICABLE",
  supporting_frame_count: 0,
  confidence: null,
  data_origin: "DERIVED_ANALYTIC",
});

const finalPriorityZone = liveSnapshot.zones.find((zone) => zone.zone_id === "ZONE-2")!;
const videoCompletion: VideoAnalysisComplete = {
  type: "video_analysis_complete",
  session_id: "frame-session-1234567890",
  state: "COMPLETE",
  summary: {
    session_id: "frame-session-1234567890",
    generated_at_ms: 1_725_000_020_000,
    frames_accepted: 18,
    frames_analyzed: 7,
    frames_dropped: 2,
    first_analyzed_frame_id: 0,
    last_analyzed_frame_id: 17,
    first_media_time_ms: 0,
    last_media_time_ms: 17_000,
    statistics: {
      flooded_area_percent: aggregateMetric(54.3, "percent", "PEAK_FRESH_SEGMENTATION"),
      people_detected: aggregateMetric(4, "count", "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"),
      vehicles_detected: aggregateMetric(3, "count", "PEAK_SIMULTANEOUS_DIRECT_DETECTIONS"),
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
      building_damage_coverage_percent: aggregateMetric(12.5, "percent", "PEAK_FRESH_SEGMENTATION"),
    },
    detected_classes: [
      {
        label: "person",
        category: "PERSON",
        peak_simultaneous_count: 4,
        max_confidence: 0.94,
        supporting_frame_count: 3,
        data_origin: "DERIVED_ANALYTIC",
      },
      {
        label: "car",
        category: "VEHICLE",
        peak_simultaneous_count: 3,
        max_confidence: 0.88,
        supporting_frame_count: 2,
        data_origin: "DERIVED_ANALYTIC",
      },
    ],
    detected_classes_truncated: false,
    priorities: [{
      zone: finalPriorityZone,
      source_frame_id: 5,
      media_time_ms: 5_000,
      supporting_update_count: 3,
      segmentation_evidence_available: true,
      detection_evidence_available: true,
      building_damage_count_availability: "NOT_SUPPORTED",
      associated_route: liveSnapshot.route,
      data_origin: "DERIVED_ANALYTIC",
    }],
    priorities_truncated: false,
    highest_priority_zone_id: finalPriorityZone.zone_id,
    incident_severity: finalPriorityZone.severity,
    segmentation_status: liveSnapshot.system_status.segmentation_details!,
    detection_status: liveSnapshot.system_status.detection_details!,
    inference_state: "LIVE",
    responsible_ai_statement: "Counts are peak sampled observations; human verification is required.",
    data_origin: "DERIVED_ANALYTIC",
  },
  latest_result: { ...liveSnapshot, frame_id: 17, zones: [], route: null },
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
      ...idleCompletion,
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
    expect(screen.getByLabelText("Detection profile: Standard")).toBeInTheDocument();
    expect(vi.mocked(useFrameIngestion).mock.calls.at(-1)?.[0].detectorMode).toBe("STANDARD");
    expect(screen.queryByText(/simulation|replay|demo scenario/i)).not.toBeInTheDocument();
  });

  it("allows an operator to change the detector profile explicitly", () => {
    render(<CommandCenter />);

    fireEvent.click(screen.getByLabelText("Detection profile: Standard"));
    const detectorSelector = screen.getByRole("group", { name: "Detector inference mode" });
    expect(within(detectorSelector).getByRole("button", { name: /Standard/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(within(detectorSelector).getByRole("button", { name: /^Aerial\b/i }));

    expect(vi.mocked(useFrameIngestion).mock.calls.at(-1)?.[0].detectorMode).toBe("AERIAL");
    expect(screen.getByLabelText("Detection profile: Aerial")).toBeInTheDocument();
  });

  it("locks the detector profile after analysis starts and through completion", async () => {
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: vi.fn(() => "blob:locked-profile-video") },
      revokeObjectURL: { configurable: true, value: vi.fn() },
    });
    render(<CommandCenter />);
    const input = document.querySelector<HTMLInputElement>("#video-file-input")!;
    fireEvent.change(input, { target: { files: [new File(["video"], "lock.webm", { type: "video/webm" })] } });
    const video = screen.getByLabelText("Selected local video preview");
    Object.defineProperties(video, {
      videoWidth: { configurable: true, value: 1280 },
      videoHeight: { configurable: true, value: 720 },
      duration: { configurable: true, value: 20 },
    });
    fireEvent.loadedMetadata(video);
    fireEvent.click(await screen.findByRole("button", { name: "Analyse" }));
    await screen.findByRole("button", { name: "Pause media" });

    const profile = screen.getByLabelText("Detection profile: Standard");
    expect(profile).toHaveAttribute("aria-disabled", "true");
    expect(profile).toHaveAttribute("title", expect.stringMatching(/locked for this analysis/i));
    const standardButton = document.querySelector<HTMLButtonElement>('.detector-profile-menu button[aria-pressed="false"]')!;
    expect(standardButton).toBeDisabled();
    fireEvent.click(standardButton);
    expect(vi.mocked(useFrameIngestion).mock.calls.at(-1)?.[0].detectorMode).toBe("STANDARD");

    fireEvent.ended(video);
    expect(screen.getByLabelText("Detection profile: Standard")).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(screen.getByRole("button", { name: "Stop media" }));
    expect(screen.getByLabelText("Detection profile: Standard")).toHaveAttribute("aria-disabled", "false");
  });

  it("locks source controls only while completion is finalizing", () => {
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: vi.fn(() => "blob:finalizing-video") },
      revokeObjectURL: { configurable: true, value: vi.fn() },
    });
    const { rerender } = render(<CommandCenter />);
    const input = document.querySelector<HTMLInputElement>("#video-file-input")!;
    fireEvent.change(input, { target: { files: [new File(["video"], "finalizing.webm", { type: "video/webm" })] } });
    fireEvent.loadedMetadata(screen.getByLabelText("Selected local video preview"));
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: videoCompletion.session_id, sessionState: "FINALIZING", connectionState: "paused" },
      intelligence: videoCompletion.latest_result,
      completionState: "FINALIZING",
      completion: null,
      retry: vi.fn(),
    });
    rerender(<CommandCenter />);

    expect(within(screen.getByRole("group", { name: "Media source selector" })).getByRole("button", { name: "Live camera" })).toBeDisabled();
    expect(screen.getByLabelText("Change video")).toHaveAttribute("aria-disabled", "true");
    expect(document.querySelector<HTMLInputElement>("#video-file-input")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Stop media" })).toBeDisabled();

    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: videoCompletion.session_id, sessionState: "FINALIZING", connectionState: "paused", lastError: "Finalization failed" },
      intelligence: videoCompletion.latest_result,
      completionState: "ERROR",
      completion: null,
      retry: vi.fn(),
    });
    rerender(<CommandCenter />);
    expect(within(screen.getByRole("group", { name: "Media source selector" })).getByRole("button", { name: "Live camera" })).toBeEnabled();
    expect(screen.getByLabelText("Change video")).toHaveAttribute("aria-disabled", "false");
    expect(screen.getByRole("button", { name: "Stop media" })).toBeEnabled();
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

  it("shows configured-model requirements and honest recovery paths when inference is unavailable", () => {
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        connectionState: "paused",
        acknowledgedFrames: 1,
        analysisStatus: "MODEL_UNAVAILABLE",
        modelStatus: "SEG UNAVAILABLE · DET UNAVAILABLE",
      },
      intelligence: null,
      ...idleCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const pendingRail = screen.getByLabelText("Intelligence pending");
    expect(within(pendingRail).getByText("MODEL UNAVAILABLE")).toBeInTheDocument();
    expect(within(pendingRail).queryByText("Ready")).not.toBeInTheDocument();
    expect(within(pendingRail).getByText(/Bounding boxes require a configured detection model/)).toBeInTheDocument();
    expect(within(pendingRail).getByText(/has not substituted simulated values/)).toBeInTheDocument();
    expect(within(pendingRail).getByRole("link", { name: "Open system status" })).toHaveAttribute("href", "/system");
    expect(within(pendingRail).getByRole("link", { name: "Open DEMO_SIMULATED replay" })).toHaveAttribute("href", "/demo");

    const observationStatus = screen.getByText("MODEL_UNAVAILABLE").closest("[role='status']");
    expect(observationStatus).toHaveTextContent("No simulated output is substituted for this media");
    expect(observationStatus).toHaveTextContent("SEG UNAVAILABLE · DET UNAVAILABLE");
  });

  it("explains that a paused source must resume before its first intelligence update", async () => {
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: vi.fn(() => "blob:paused-flood-video") },
      revokeObjectURL: { configurable: true, value: vi.fn() },
    });
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        connectionState: "paused",
      },
      intelligence: null,
      ...idleCompletion,
      retry: vi.fn(),
    });
    render(<CommandCenter />);
    const input = document.querySelector<HTMLInputElement>("#video-file-input");
    const file = new File(["local"], "paused.webm", { type: "video/webm" });
    fireEvent.change(input as HTMLInputElement, { target: { files: [file] } });
    const video = screen.getByLabelText("Selected local video preview");
    Object.defineProperties(video, {
      videoWidth: { configurable: true, value: 1280 },
      videoHeight: { configurable: true, value: 720 },
      duration: { configurable: true, value: 12 },
    });
    fireEvent.loadedMetadata(video);
    fireEvent.click(await screen.findByRole("button", { name: "Analyse" }));
    fireEvent.click(await screen.findByRole("button", { name: "Pause media" }));

    expect(await screen.findByText("ANALYSIS PAUSED")).toBeInTheDocument();
    expect(screen.getByText(/Analysis is paused before the first intelligence update/)).toBeInTheDocument();
    expect(screen.getByText("Analysis paused before first intelligence")).toBeInTheDocument();
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
      ...idleCompletion,
      retry: vi.fn(),
    });
    render(<CommandCenter />);

    const overlay = screen.getByLabelText(/Normalized model-derived flood observation/);
    expect(overlay).toBeInTheDocument();
    expect(screen.getByText(`BACKEND INTELLIGENCE · FRAME ${reroutedSnapshot.frame_id}`)).toBeInTheDocument();
    expect(within(overlay).getByText("R-1")).toBeInTheDocument();
    const zoneOverlayLabel = within(overlay).getByText("ZONE 2 · 92");
    expect(zoneOverlayLabel.closest("g")?.querySelector("polygon")).toHaveAttribute("stroke-dasharray", "2 1");
    const personLabel = within(overlay).getByText("PERSON 91");
    const personBox = personLabel.closest("g")?.querySelector('rect[fill="none"]');
    expect(personBox).toHaveAttribute("x", "48");
    expect(personBox).toHaveAttribute("y", "42");
    expect(personBox).toHaveAttribute("stroke", "#f8fafc");
    expect(screen.getByLabelText("Actual media state: LIVE")).toBeInTheDocument();
    expect(screen.getByLabelText("Segmentation class legend")).toHaveTextContent("water");
    expect(screen.getByLabelText("Segmentation class legend")).toHaveTextContent("pool");
    expect(screen.getByLabelText("Inference model status")).toHaveTextContent("REAL");
    expect(screen.getByLabelText("Inference model status")).toHaveTextContent("FALLBACK");
    expect(screen.getByLabelText("Evidence frame freshness")).toHaveTextContent("current inference");
    expect(screen.getByText("Previous route no longer preferred")).toBeInTheDocument();
    expect(screen.getByText("ROUTE_CHANGED_PRIMARY_ACCESS_UNSAFE")).toBeInTheDocument();
    expect(screen.getByText("38%")).toBeInTheDocument();
    const topPriority = document.querySelector<HTMLElement>('.priority-decision[data-zone-id="ZONE-2"]');
    expect(topPriority).not.toBeNull();
    expect(within(topPriority as HTMLElement).getByText("Zone 2")).toBeInTheDocument();
    expect(within(topPriority as HTMLElement).getByText("92")).toBeInTheDocument();
    expect(within(topPriority as HTMLElement).getByText("4")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));
    expect(await screen.findByText("Backend-supplied priority explanation.")).toBeInTheDocument();
    expect(getLiveIncidentReport).toHaveBeenCalledWith("frame-session-1234567890");
  });

  it("renders final whole-video findings while keeping the latest frame separate", async () => {
    vi.mocked(getLiveIncidentReport).mockResolvedValue({
      ...incidentReport,
      incident_id: liveSnapshot.incident_id,
      analysis_scope: "WHOLE_VIDEO",
      severity_established: false,
      priorities_truncated: true,
      aggregate_availability: {
        flooded_area_percent: "MODEL_UNAVAILABLE",
        people_detected: "AVAILABLE",
        vehicles_detected: "AVAILABLE",
        blocked_road_cells: "MODEL_UNAVAILABLE",
        damaged_buildings: "NOT_SUPPORTED",
      },
      statistics: {
        flooded_area_percent: { ...liveSnapshot.statistics.flooded_area_percent, value: 0 },
        people_detected: { ...liveSnapshot.statistics.people_detected, value: 4 },
        vehicles_detected: { ...liveSnapshot.statistics.vehicles_detected, value: 3 },
        blocked_roads: { ...liveSnapshot.statistics.blocked_roads, value: 0 },
        damaged_buildings: { ...liveSnapshot.statistics.damaged_buildings, value: 0 },
      },
      data_origin: "DERIVED_ANALYTIC",
    });
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        sessionId: videoCompletion.session_id,
        sessionState: "COMPLETE",
        connectionState: "stopped",
        analysisStatus: "LIVE",
      },
      intelligence: videoCompletion.latest_result,
      completionState: "COMPLETE",
      completion: videoCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const rail = screen.getByLabelText("Incident command intelligence");
    expect(within(rail).getByText("FINAL VIDEO FINDINGS")).toBeInTheDocument();
    expect(within(rail).getByText("54.3%")).toBeInTheDocument();
    expect(within(rail).getByText("Peak people").closest("div")).toHaveTextContent("4");
    expect(within(rail).getByText("Peak vehicles").closest("div")).toHaveTextContent("3");
    expect(within(rail).getByText("Zone 2 · 92")).toBeInTheDocument();
    expect(within(rail).getAllByText("person")[0].closest("li")).toHaveTextContent("94% confidence");
    expect(within(rail).getByText("1 priorities · 7 frames")).toBeInTheDocument();
    expect(within(rail).getByText("Relative route observed at 00:05 · source frame 5")).toBeInTheDocument();
    expect(within(rail).getByText(/Historical image-relative evidence; verify current access/)).toBeInTheDocument();
    expect(document.querySelector('.priority-decision[data-zone-id="ZONE-2"]')).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));
    const report = screen.getByRole("dialog", { name: "Incident report" });
    expect(await within(report).findByText("Whole video findings")).toBeInTheDocument();
    expect(within(report).getByText("Severity").parentElement).toHaveTextContent("Not established");
    expect(within(report).getByText("Peak flood coverage").parentElement).toHaveTextContent("Unavailable");
    expect(within(report).getByText("Peak blocked grid cells").parentElement).toHaveTextContent("Unavailable");
    expect(within(report).getByText("Damaged buildings").parentElement).toHaveTextContent("Unavailable");
    expect(within(report).getByText("Historical relative access observation")).toBeInTheDocument();
    expect(within(report).getByText(/verify current conditions/)).toBeInTheDocument();
    expect(within(report).getByText(/additional observations were omitted/)).toBeInTheDocument();
  });

  it("aligns the completed overlay to the last analyzed video timestamp", () => {
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: vi.fn(() => "blob:aligned-final-video") },
      revokeObjectURL: { configurable: true, value: vi.fn() },
    });
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: videoCompletion.session_id, sessionState: "COMPLETE", connectionState: "stopped" },
      intelligence: videoCompletion.latest_result,
      completionState: "COMPLETE",
      completion: videoCompletion,
      retry: vi.fn(),
    });
    render(<CommandCenter />);
    const input = document.querySelector<HTMLInputElement>("#video-file-input")!;
    fireEvent.change(input, { target: { files: [new File(["video"], "aligned.webm", { type: "video/webm" })] } });
    const video = screen.getByLabelText("Selected local video preview") as HTMLVideoElement;
    Object.defineProperties(video, {
      videoWidth: { configurable: true, value: 1280 },
      videoHeight: { configurable: true, value: 720 },
      duration: { configurable: true, value: 30 },
    });
    fireEvent.loadedMetadata(video);
    video.currentTime = 30;
    fireEvent.ended(video);

    expect(video.currentTime).toBe(17);
    expect(screen.getByText("LAST ANALYZED FRAME 17 · VIDEO 00:17")).toBeInTheDocument();
  });

  it("refetches an open report when finalization completes", async () => {
    const latestFrameReport: IncidentReport = {
      ...incidentReport,
      incident_id: liveSnapshot.incident_id,
      data_origin: "DERIVED_ANALYTIC",
      analysis_scope: "LATEST_FRAME",
    };
    const wholeVideoReport: IncidentReport = {
      ...latestFrameReport,
      analysis_scope: "WHOLE_VIDEO",
      aggregate_availability: {
        flooded_area_percent: "AVAILABLE",
        people_detected: "AVAILABLE",
        vehicles_detected: "AVAILABLE",
        blocked_road_cells: "AVAILABLE",
        damaged_buildings: "NOT_SUPPORTED",
      },
    };
    vi.mocked(getLiveIncidentReport)
      .mockResolvedValueOnce(latestFrameReport)
      .mockResolvedValueOnce(wholeVideoReport);
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: videoCompletion.session_id, sessionState: "FINALIZING", connectionState: "paused" },
      intelligence: liveSnapshot,
      completionState: "FINALIZING",
      completion: null,
      retry: vi.fn(),
    });
    const { rerender } = render(<CommandCenter />);
    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));
    expect(await screen.findByText("Backend generated")).toBeInTheDocument();

    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: videoCompletion.session_id, sessionState: "COMPLETE", connectionState: "stopped" },
      intelligence: videoCompletion.latest_result,
      completionState: "COMPLETE",
      completion: videoCompletion,
      retry: vi.fn(),
    });
    rerender(<CommandCenter />);

    expect(await screen.findByText("Whole video findings")).toBeInTheDocument();
    expect(getLiveIncidentReport).toHaveBeenCalledTimes(2);
    expect(getLiveIncidentReport).toHaveBeenNthCalledWith(1, videoCompletion.session_id);
    expect(getLiveIncidentReport).toHaveBeenNthCalledWith(2, videoCompletion.session_id);
  });

  it("discloses bounded final priority and object findings", () => {
    const truncatedCompletion: VideoAnalysisComplete = {
      ...videoCompletion,
      summary: {
        ...videoCompletion.summary,
        priorities_truncated: true,
        detected_classes_truncated: true,
      },
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: truncatedCompletion.session_id, sessionState: "COMPLETE", connectionState: "stopped" },
      intelligence: truncatedCompletion.latest_result,
      completionState: "COMPLETE",
      completion: truncatedCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const rail = screen.getByLabelText("Incident command intelligence");
    expect(within(rail).getByText(/additional observations were omitted/)).toBeInTheDocument();
    expect(within(rail).getByText(/additional class findings were omitted/)).toBeInTheDocument();
  });

  it("does not present missing segmentation evidence as zero in a retained priority or its details", () => {
    const detectionOnlyCompletion: VideoAnalysisComplete = {
      ...videoCompletion,
      summary: {
        ...videoCompletion.summary,
        priorities: videoCompletion.summary.priorities.map((observation) => ({
          ...observation,
          segmentation_evidence_available: false,
          detection_evidence_available: true,
        })),
        statistics: {
          ...videoCompletion.summary.statistics,
          flooded_area_percent: unavailableMetric("percent"),
          blocked_road_cells: unavailableMetric("count"),
          building_damage_coverage_percent: unavailableMetric("percent"),
        },
      },
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        sessionId: detectionOnlyCompletion.session_id,
        sessionState: "COMPLETE",
        connectionState: "stopped",
        analysisStatus: "DEGRADED",
      },
      intelligence: detectionOnlyCompletion.latest_result,
      completionState: "COMPLETE",
      completion: detectionOnlyCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const priority = document.querySelector<HTMLElement>('.priority-decision[data-zone-id="ZONE-2"]');
    expect(priority).not.toBeNull();
    expect(within(priority as HTMLElement).getByText("Flooded").closest("div")).toHaveTextContent("Unavailable");
    expect(within(priority as HTMLElement).getByText("Damage").closest("div")).toHaveTextContent("Unavailable");
    expect(within(priority as HTMLElement).getByText("Access").closest("div")).toHaveTextContent("Unavailable");
    expect(within(priority as HTMLElement).getByText("People").closest("div")).toHaveTextContent("4");
    fireEvent.click(within(priority as HTMLElement).getByRole("button", { name: "View zone" }));
    const drawer = screen.getByRole("dialog", { name: "Zone 2" });
    expect(within(drawer).getByText("Observed at 00:05 · source frame 5")).toBeInTheDocument();
    expect(within(drawer).getByText("Flood coverage").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Damage coverage").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Pool coverage").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Access status").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Building instances").parentElement).toHaveTextContent("Not supported");
    expect(within(drawer).getByText("People").parentElement).toHaveTextContent("4");
    expect(within(drawer).getByText(/Historical video evidence/)).toHaveTextContent("map remains on the latest analyzed frame");
    expect(within(drawer).queryByRole("button", { name: "Focus on map" })).not.toBeInTheDocument();
  });

  it("masks people and vehicles when a retained priority lacks detection evidence", () => {
    const segmentationOnlyCompletion: VideoAnalysisComplete = {
      ...videoCompletion,
      summary: {
        ...videoCompletion.summary,
        priorities: videoCompletion.summary.priorities.map((observation) => ({
          ...observation,
          segmentation_evidence_available: true,
          detection_evidence_available: false,
        })),
      },
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: segmentationOnlyCompletion.session_id, sessionState: "COMPLETE", connectionState: "stopped" },
      intelligence: segmentationOnlyCompletion.latest_result,
      completionState: "COMPLETE",
      completion: segmentationOnlyCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const priority = document.querySelector<HTMLElement>('.priority-decision[data-zone-id="ZONE-2"]')!;
    expect(within(priority).getByText("People").parentElement).toHaveTextContent("Unavailable");
    fireEvent.click(within(priority).getByRole("button", { name: "View zone" }));
    const drawer = screen.getByRole("dialog", { name: "Zone 2" });
    expect(within(drawer).getByText("People").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Vehicles").parentElement).toHaveTextContent("Unavailable");
    expect(within(drawer).getByText("Flood coverage").parentElement).not.toHaveTextContent("Unavailable");
  });

  it("shows a completed zero-result table with unavailable evidence instead of pending dashes", () => {
    const zeroCompletion: VideoAnalysisComplete = {
      ...videoCompletion,
      summary: {
        ...videoCompletion.summary,
        frames_accepted: 0,
        frames_analyzed: 0,
        frames_dropped: 0,
        first_analyzed_frame_id: null,
        last_analyzed_frame_id: null,
        first_media_time_ms: null,
        last_media_time_ms: null,
        statistics: {
          flooded_area_percent: unavailableMetric("percent"),
          people_detected: unavailableMetric("count"),
          vehicles_detected: unavailableMetric("count"),
          blocked_road_cells: unavailableMetric("count"),
          damaged_buildings: unavailableMetric("count"),
          building_damage_coverage_percent: unavailableMetric("percent"),
        },
        detected_classes: [],
        priorities: [],
        highest_priority_zone_id: null,
        incident_severity: null,
      },
      latest_result: null,
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        sessionId: zeroCompletion.session_id,
        sessionState: "COMPLETE",
        connectionState: "stopped",
        analysisStatus: "MODEL_UNAVAILABLE",
      },
      intelligence: null,
      completionState: "COMPLETE",
      completion: zeroCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const rail = screen.getByLabelText("Incident command intelligence");
    expect(within(rail).queryByText("Intelligence pending")).not.toBeInTheDocument();
    expect(within(rail).getAllByText("Unavailable").length).toBeGreaterThanOrEqual(5);
    expect(within(rail).getByText(/No frames were analyzed, so rescue priorities could not be established/)).toBeInTheDocument();
    expect(within(rail).getByText(/No frames were analyzed, so object findings could not be established/)).toBeInTheDocument();
    expect(within(rail).getByText("Incident severity").closest("div")).toHaveTextContent("Not established");
    expect(within(rail).getByText("Frames analyzed").closest("div")).toHaveTextContent("0");
  });

  it("does not claim negative findings when object detection was unavailable", () => {
    const unavailableDetectionCompletion: VideoAnalysisComplete = {
      ...videoCompletion,
      summary: {
        ...videoCompletion.summary,
        frames_analyzed: 3,
        detected_classes: [],
        priorities: [],
        highest_priority_zone_id: null,
        incident_severity: null,
        statistics: {
          ...videoCompletion.summary.statistics,
          people_detected: unavailableMetric("count"),
          vehicles_detected: unavailableMetric("count"),
        },
        detection_status: {
          ...videoCompletion.summary.detection_status,
          status: "unavailable",
          mode: "UNAVAILABLE",
        },
      },
      latest_result: null,
    };
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: { ...ingestionMetrics, sessionId: unavailableDetectionCompletion.session_id, sessionState: "COMPLETE", connectionState: "stopped" },
      intelligence: null,
      completionState: "COMPLETE",
      completion: unavailableDetectionCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    const rail = screen.getByLabelText("Incident command intelligence");
    expect(within(rail).getByText(/Object detection was unavailable, so object findings could not be established/)).toBeInTheDocument();
    expect(within(rail).getByText(/Rescue priorities could not be established conclusively/)).toBeInTheDocument();
    expect(within(rail).getByText("Highest priority").parentElement).toHaveTextContent("Not established");
    expect(within(rail).queryByText(/No supported object classes were detected/)).not.toBeInTheDocument();
  });

  it("explains missing evidence when live detection has no supported rescue zone", () => {
    vi.mocked(useFrameIngestion).mockReturnValue({
      metrics: {
        ...ingestionMetrics,
        sessionId: "vehicle-only-session",
        connectionState: "streaming",
        analysisStatus: "DEGRADED",
        modelStatus: "SEG UNAVAILABLE · DET FALLBACK",
      },
      intelligence: {
        ...liveSnapshot,
        highest_priority_zone_id: null,
        zones: [],
        route: null,
        segmentation: { status: "not_configured", classes: [], regions: [], mask: null },
        system_status: {
          ...liveSnapshot.system_status,
          segmentation_model: "unavailable",
          inference_state: "DEGRADED",
        },
      },
      ...idleCompletion,
      retry: vi.fn(),
    });

    render(<CommandCenter />);

    expect(screen.getByText(/No evidence-supported rescue zones/)).toHaveTextContent(
      "flood/road/damage segmentation unavailable",
    );
    expect(screen.getByText(/No evidence-supported rescue zones/)).toHaveTextContent(
      "missing evidence was not simulated",
    );
    expect(screen.queryByText(/No rescue zones in this simulated scenario/)).not.toBeInTheDocument();
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
