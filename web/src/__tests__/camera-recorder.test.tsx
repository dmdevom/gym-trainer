import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CameraRecorder } from "@/components/camera-recorder";

class RecorderMock {
  static isTypeSupported = vi.fn(() => true);
  state: RecordingState = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: (() => void) | null = null;

  start() { this.state = "recording"; }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["video"], { type: "video/webm" }) });
    this.onstop?.();
  }
}

describe("CameraRecorder", () => {
  const stopTrack = vi.fn();
  const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream;

  beforeEach(() => {
    vi.stubGlobal("MediaRecorder", RecorderMock);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    });
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:recording") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    Object.defineProperty(HTMLMediaElement.prototype, "play", { configurable: true, value: vi.fn().mockResolvedValue(undefined) });
    Object.defineProperty(HTMLMediaElement.prototype, "pause", { configurable: true, value: vi.fn() });
    Object.defineProperty(HTMLMediaElement.prototype, "load", { configurable: true, value: vi.fn() });
  });

  afterEach(() => vi.restoreAllMocks());

  it("requests the rear camera, records, and returns a file", async () => {
    const onCapture = vi.fn();
    const view = render(<StrictMode><CameraRecorder onCapture={onCapture} onError={vi.fn()} /></StrictMode>);

    const start = await screen.findByRole("button", { name: "Start recording" });
    await waitFor(() => expect(start).toBeEnabled());
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith(expect.objectContaining({
      audio: false,
      video: expect.objectContaining({ facingMode: { ideal: "environment" } }),
    }));

    fireEvent.click(start);
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    await waitFor(() => expect(onCapture).toHaveBeenCalled());
    expect(onCapture.mock.calls[0][0]).toBeInstanceOf(File);
    expect(onCapture.mock.calls[0][1]).toBe("blob:recording");
    const playbackVideo = view.container.querySelector("video");
    expect(playbackVideo?.srcObject).toBeNull();
    expect(playbackVideo?.src).toBe("blob:recording");
    expect(HTMLMediaElement.prototype.load).toHaveBeenCalled();

    view.unmount();
    expect(stopTrack).toHaveBeenCalled();
  });
});
