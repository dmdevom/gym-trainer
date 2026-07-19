import type { DemoSample, ExerciseKey } from "./types";

export const DEMO_SAMPLES: DemoSample[] = [
  {
    id: "bicep-pass",
    exercise: "bicep_curl",
    grade: "pass",
    label: "Clean curl",
    description: "2/2 full reps with controlled tempo",
    src: "/samples/bicep-curl-pass.mp4",
    sourceUrl: "local-capcut-export",
    creator: "trAIner hackathon team",
  },
  {
    id: "bicep-fail",
    exercise: "bicep_curl",
    grade: "fail",
    label: "Needs work",
    description: "Stops short of full extension",
    src: "/samples/bicep-curl-fail.mp4",
    sourceUrl: "local-capcut-export",
    creator: "trAIner hackathon team",
  },
  {
    id: "barbell-pass",
    exercise: "barbell_curl",
    grade: "pass",
    label: "Clean barbell curl",
    description: "Stable, full-range reps",
    src: "/samples/barbell-curl-pass.mp4",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Video_of_EZ_Bar_Curl_and_Straight_Bar_Curl.webm",
    creator: "Colossus Fitness",
  },
  {
    id: "barbell-fail",
    exercise: "barbell_curl",
    grade: "fail",
    label: "Needs work",
    description: "Real-time reps with rushed tempo",
    src: "/samples/barbell-curl-fail.mp4",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Video_of_EZ_Bar_Curl_and_Straight_Bar_Curl.webm",
    creator: "Colossus Fitness",
  },
  {
    id: "squat-pass",
    exercise: "squat",
    grade: "pass",
    label: "Clean squat",
    description: "Full depth with control",
    src: "/samples/squat-pass.mp4",
    sourceUrl: "user-provided-footage",
    creator: "trAIner team",
  },
  {
    id: "squat-fail",
    exercise: "squat",
    grade: "fail",
    label: "Needs work",
    description: "See how rushed reps are graded",
    src: "/samples/squat-fail.mp4",
    sourceUrl: "user-provided-footage",
    creator: "trAIner team",
  },
];

export const samplesForExercise = (exercise: ExerciseKey) =>
  DEMO_SAMPLES.filter((sample) => sample.exercise === exercise);
