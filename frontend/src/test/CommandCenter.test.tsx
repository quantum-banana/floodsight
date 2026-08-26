import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CommandCenter } from "../features/command-center/CommandCenter";
import { useDemoIncident } from "../hooks/useDemoIncident";
import { commandSnapshot, incidentDetail, reorderedSnapshot } from "./fixtures";

vi.mock("../hooks/useDemoIncident", () => ({
  useDemoIncident: vi.fn(),
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

describe("FloodSight command center", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useDemoIncident).mockReturnValue(readyState);
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

  it("builds a responsible, copyable incident report from the current snapshot", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<CommandCenter />);

    fireEvent.click(screen.getByRole("button", { name: "Open incident report" }));

    const report = screen.getByRole("dialog", { name: "Incident report" });
    expect(within(report).getByText("42%")).toBeInTheDocument();
    expect(within(report).getByText("Flood coverage")).toBeInTheDocument();
    expect(within(report).getByText("Blocked roads")).toBeInTheDocument();
    expect(report).toHaveTextContent("requires human review");
    fireEvent.click(within(report).getByRole("button", { name: "Copy report text" }));
    expect(writeText).toHaveBeenCalledOnce();
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
