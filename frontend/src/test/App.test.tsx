import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { getSystemSnapshot } from "../services/api";
import type { SystemSnapshot } from "../types/api";

vi.mock("../services/api", () => ({
  getSystemSnapshot: vi.fn(),
}));

const snapshot: SystemSnapshot = {
  health: { status: "ok", service: "floodsight-api", version: "0.1.0" },
  models: {
    segmentation: { status: "not_configured", model: null },
    detection: { status: "not_configured", model: null },
  },
  sample: {
    incident_id: "FS-DEMO-001",
    frame_id: 0,
    timestamp_ms: 0,
    source_mode: "DEMO_REPLAY",
    data_origin: "DEMO_SIMULATED",
    system_status: {
      api: "operational",
      segmentation_model: "not_configured",
      detection_model: "not_configured",
    },
    statistics: {
      flooded_area_percent: { value: 43.2, unit: "percent", confidence: null, data_origin: "DEMO_SIMULATED" },
      people_detected: { value: 12, unit: "count", confidence: null, data_origin: "DEMO_SIMULATED" },
      vehicles_detected: { value: 6, unit: "count", confidence: null, data_origin: "DEMO_SIMULATED" },
      blocked_roads: { value: 3, unit: "count", confidence: null, data_origin: "DEMO_SIMULATED" },
      damaged_buildings: { value: 8, unit: "count", confidence: null, data_origin: "DEMO_SIMULATED" },
    },
    detections: [],
    segmentation: { status: "not_configured", classes: [] },
    roads: [],
    zones: [],
    events: [],
    route: null,
  },
};

describe("FloodSight system status", () => {
  beforeEach(() => {
    vi.mocked(getSystemSnapshot).mockResolvedValue(snapshot);
  });

  it("shows API/model readiness and clearly labels simulated incident data", async () => {
    render(<App />);

    expect(screen.getByLabelText("Connecting to FloodSight backend")).toBeInTheDocument();
    expect(await screen.findByText("Incident FS-DEMO-001")).toBeInTheDocument();
    expect(screen.getByLabelText("Data origin: DEMO_SIMULATED")).toHaveTextContent("Demo / Simulated");
    expect(screen.getByLabelText("Backend connection: Connected")).toBeInTheDocument();
    expect(screen.getByLabelText("Segmentation model: Not configured")).toBeInTheDocument();
    expect(screen.getByLabelText("Detection model: Not configured")).toBeInTheDocument();
    expect(screen.getByText("floodsight-api · v0.1.0")).toBeInTheDocument();
  });
});

