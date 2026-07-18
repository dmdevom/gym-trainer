import { describe, expect, it } from "vitest";
import { deriveStats, MAX_VIDEO_BYTES, resultVideoUrl, validateVideo } from "@/lib/api";
import { samplesForExercise } from "@/lib/samples";
import type { AnalysisResult } from "@/lib/types";

describe("video validation", () => {
  it("accepts supported video files", () => {
    expect(validateVideo(new File(["video"], "set.mp4", { type: "video/mp4" }))).toBeNull();
  });

  it("rejects empty and oversized files", () => {
    expect(validateVideo(new File([], "empty.mp4", { type: "video/mp4" }))).toContain("empty");
    const oversized = new File(["video"], "large.mp4", { type: "video/mp4" });
    Object.defineProperty(oversized, "size", { value: MAX_VIDEO_BYTES + 1 });
    expect(validateVideo(oversized)).toContain("50 MB");
  });

  it("accepts a known extension when a phone supplies a generic MIME type", () => {
    expect(validateVideo(new File(["video"], "phone.mov", { type: "application/octet-stream" }))).toBeNull();
  });
});

describe("result helpers", () => {
  const result = {
    per_rep: [
      { depth_pct: 100, duration_s: 2 },
      { depth_pct: 80, duration_s: 3 },
    ],
    video_url: "/results/example",
  } as AnalysisResult;

  it("derives average depth and tempo", () => {
    expect(deriveStats(result)).toEqual({ averageDepth: 90, averageTempo: 2.5 });
  });

  it("normalizes relative result URLs", () => {
    expect(resultVideoUrl(result)).toBe("/backend-api/results/example");
  });

  it("keeps a pass and form-check sample for every exercise", () => {
    for (const key of ["bicep_curl", "barbell_curl", "squat"] as const) {
      expect(samplesForExercise(key).map((sample) => sample.grade).sort()).toEqual(["fail", "pass"]);
    }
  });
});
