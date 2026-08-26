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
  });
});
