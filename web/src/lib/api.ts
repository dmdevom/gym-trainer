import type { AnalysisResult, Exercise, ExerciseKey, ProgressResponse } from "./types";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "/backend-api"
).replace(/\/$/, "");

// Keep in sync with the backend's MAX_UPLOAD_MB (default 100) and the Next proxy's
// proxyClientMaxBodySize. Override at build time with NEXT_PUBLIC_MAX_UPLOAD_MB;
// the backend is the real gatekeeper (413).
export const MAX_VIDEO_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB) || 100;
export const MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024;
export const ACCEPTED_VIDEO_EXTENSIONS = ["mp4", "mov", "webm", "avi", "mkv", "m4v", "3gp"];

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function readError(response: Response, fallback: string) {
  try {
    const body = await response.json();
    return body.detail || body.error || fallback;
  } catch {
    return fallback;
  }
}

export async function fetchExercises(signal?: AbortSignal): Promise<Exercise[]> {
  const response = await fetch(`${API_BASE_URL}/exercises`, { signal });
  if (!response.ok) throw new ApiError(await readError(response, "Could not load exercises."), response.status);
  const data = await response.json();
  return data.exercises || [];
}

export async function submitVideo(file: File, exercise: ExerciseKey, signal?: AbortSignal): Promise<string> {
  const body = new FormData();
  body.append("file", file);
  body.append("exercise", exercise);
  const response = await fetch(`${API_BASE_URL}/analyze/video`, { method: "POST", body, signal });
  if (response.status !== 202) {
    throw new ApiError(await readError(response, `Upload failed (${response.status}).`), response.status);
  }
  const data = await response.json();
  if (!data.token) throw new ApiError("The server did not return an analysis token.");
  return data.token;
}

export async function fetchProgress(token: string, signal?: AbortSignal): Promise<ProgressResponse> {
  const response = await fetch(`${API_BASE_URL}/progress/${encodeURIComponent(token)}`, { signal });
  if (!response.ok) {
    throw new ApiError(await readError(response, "The analysis result is no longer available."), response.status);
  }
  return response.json();
}

export function resultVideoUrl(result: AnalysisResult): string {
  const path = result.video_url.startsWith("/") ? result.video_url : `/${result.video_url}`;
  if (/^https?:\/\//.test(API_BASE_URL)) return new URL(path, `${API_BASE_URL}/`).toString();
  return `${API_BASE_URL}${path}`;
}

export function validateVideo(file: File): string | null {
  if (file.size > MAX_VIDEO_BYTES) return `Video must be smaller than ${MAX_VIDEO_MB} MB.`;
  if (file.size === 0) return "This video is empty. Choose or record another clip.";
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!file.type.startsWith("video/") && !ACCEPTED_VIDEO_EXTENSIONS.includes(extension)) {
    return "Choose an MP4, MOV, WebM, AVI, MKV, M4V, or 3GP video.";
  }
  return null;
}

export function deriveStats(result: AnalysisResult) {
  const reps = result.per_rep;
  return {
    averageDepth: reps.length ? Math.round(reps.reduce((sum, rep) => sum + rep.depth_pct, 0) / reps.length) : null,
    averageTempo: reps.length ? reps.reduce((sum, rep) => sum + rep.duration_s, 0) / reps.length : null,
  };
}
