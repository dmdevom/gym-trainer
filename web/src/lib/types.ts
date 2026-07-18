export type ExerciseKey = "bicep_curl" | "barbell_curl" | "squat";

export interface Exercise {
  key: ExerciseKey;
  name: string;
  vertex_name: string;
  film_tip: string;
  tips: string[];
}

export interface RepGrade {
  number: number;
  min_angle: number;
  duration_s: number;
  full: boolean;
  depth_pct: number;
  issues: string[];
  tags: string[];
  start_t: number;
  end_t: number;
}

export interface AnalysisResult {
  meta: {
    file: string;
    fps: number;
    rotation_applied: number;
    stride: number;
    sample_hz: number;
    frames_sampled: number;
    exercise: Pick<Exercise, "key" | "name" | "vertex_name">;
    side: string;
    side_visibility: Record<string, number>;
    coaching_source?: "llm" | "rules" | "llm+rules";
  };
  reps: number;
  full_reps: number;
  per_rep: RepGrade[];
  verdict: string;
  form_checks?: Array<{
    key: string;
    label: string;
    fault: string;
    cue: string;
    assessed: number;
    flagged: number;
    status: "ok" | "flag" | "not_assessed";
  }>;
  coaching: {
    focus: string;
    next_session: string[];
    keep_in_mind: string[];
    muscle: string;
    session_story?: string;
    mental_cue?: string;
  };
  thresholds: {
    up_enter: number;
    down_enter: number;
    full_rom: number;
    gauge_deep: number;
    tempo_min_s: number;
  };
  series: {
    t: number[];
    angle: Array<number | null>;
  };
  video_url: string;
}

export interface ProgressResponse {
  stage: string;
  pct: number | null;
  done: boolean;
  error: string | null;
  result?: AnalysisResult;
}

export type InputMode = "sample" | "upload" | "record";
export type AnalysisState = "idle" | "ready" | "uploading" | "analyzing" | "complete" | "error";

export interface DemoSample {
  id: string;
  exercise: ExerciseKey;
  grade: "pass" | "fail";
  label: string;
  description: string;
  src: string;
  sourceUrl: string;
  creator: string;
}
