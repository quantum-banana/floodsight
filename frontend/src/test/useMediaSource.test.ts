import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useMediaSource } from "../hooks/useMediaSource";

const originalMediaDevices = navigator.mediaDevices;

function fakeVideo() {
  const video = document.createElement("video");
  Object.defineProperties(video, {
    videoWidth: { configurable: true, value: 1280 },
    videoHeight: { configurable: true, value: 720 },
    duration: { configurable: true, value: 12.5 },
  });
  video.play = vi.fn().mockResolvedValue(undefined);
  video.pause = vi.fn();
  return video;
}

describe("media source ownership", () => {
  const createObjectURL = vi.fn(() => "blob:floodsight-video");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL,
      revokeObjectURL,
    });
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
  });

  afterEach(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: originalMediaDevices,
    });
    vi.unstubAllGlobals();
  });

  it("accepts a local video, exposes metadata, and revokes its object URL", () => {
    const { result, unmount } = renderHook(() => useMediaSource());
    const video = fakeVideo();
    const file = new File(["browser-local"], "flood.webm", { type: "video/webm" });

    act(() => result.current.bindVideoElement(video));
    act(() => expect(result.current.selectFile(file)).toBe(true));
    act(() => result.current.onLoadedMetadata());

    expect(createObjectURL).toHaveBeenCalledWith(file);
    expect(result.current.mode).toBe("VIDEO_FILE");
    expect(result.current.mediaOrigin).toBe("USER_VIDEO_FILE");
    expect(result.current.fileInfo).toMatchObject({
      name: "flood.webm",
      durationSeconds: 12.5,
      width: 1280,
      height: 720,
    });
    expect(result.current.readyForIngestion).toBe(true);
    video.currentTime = 5;
    act(() => result.current.stop());
    expect(video.currentTime).toBe(0);
    expect(result.current.state).toBe("STOPPED");

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:floodsight-video");
  });

  it("rejects invalid file types without creating an object URL", () => {
    const { result } = renderHook(() => useMediaSource());

    act(() => expect(result.current.selectFile(new File(["x"], "notes.txt", { type: "text/plain" }))).toBe(false));

    expect(result.current.state).toBe("ERROR");
    expect(result.current.error).toMatch(/browser-playable video/i);
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("requests video-only webcam permission and supports pause, resume, and stop", async () => {
    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] } as unknown as MediaStream;
    const getUserMedia = vi.fn().mockResolvedValue(stream);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const { result } = renderHook(() => useMediaSource());
    const video = fakeVideo();

    act(() => {
      result.current.bindVideoElement(video);
      result.current.selectMode("WEBCAM");
    });
    await act(async () => result.current.start());

    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
    expect(result.current.state).toBe("PLAYING");
    expect(result.current.mediaOrigin).toBe("USER_WEBCAM");
    act(() => result.current.pause());
    expect(result.current.state).toBe("PAUSED");
    await act(async () => result.current.resume());
    expect(result.current.state).toBe("PLAYING");
    act(() => result.current.stop());
    expect(result.current.state).toBe("STOPPED");
    expect(track.stop).toHaveBeenCalledOnce();
  });

  it("surfaces webcam permission denial without a fake fallback", async () => {
    const getUserMedia = vi.fn().mockRejectedValue(
      new DOMException("Permission denied", "NotAllowedError"),
    );
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const { result } = renderHook(() => useMediaSource());

    act(() => {
      result.current.bindVideoElement(fakeVideo());
      result.current.selectMode("WEBCAM");
    });
    await act(async () => result.current.start());

    expect(result.current.state).toBe("ERROR");
    expect(result.current.error).toMatch(/permission was denied/i);
  });

  it("stops every camera track on source change and unmount", async () => {
    const firstTrack = { stop: vi.fn() };
    const secondTrack = { stop: vi.fn() };
    const getUserMedia = vi
      .fn()
      .mockResolvedValueOnce({ getTracks: () => [firstTrack] })
      .mockResolvedValueOnce({ getTracks: () => [secondTrack] });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const { result, unmount } = renderHook(() => useMediaSource());
    act(() => {
      result.current.bindVideoElement(fakeVideo());
      result.current.selectMode("WEBCAM");
    });
    await act(async () => result.current.start());
    act(() => result.current.selectMode("SIMULATION"));
    expect(firstTrack.stop).toHaveBeenCalledOnce();

    act(() => result.current.selectMode("WEBCAM"));
    await act(async () => result.current.start());
    unmount();
    expect(secondTrack.stop).toHaveBeenCalledOnce();
  });
});
