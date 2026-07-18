import type { DemoSample, ExerciseKey } from "./types";

export const DEMO_SAMPLES: DemoSample[] = [
  {
    id: "bicep-pass",
    exercise: "bicep_curl",
    grade: "pass",
    label: "Clean curl",
    description: "Full range, controlled tempo",
    src: "/samples/bicep-curl-pass.mp4",
    sourceUrl: "https://www.youtube.com/watch?v=2k9co4UIlEw",
    creator: "FITTR",
  },
  {
    id: "bicep-fail",
    exercise: "bicep_curl",
    grade: "fail",
    label: "Needs work",
    description: "Real-time curls with a rushed negative",
    src: "/samples/bicep-curl-fail.mp4",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Joe_Biden_video_for_Gimme_Five_challenge_2015.ogv",
    creator: "David Lienemann / Executive Office of the President",
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
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm",
    creator: "FitnessScape",
  },
  {
    id: "squat-fail",
    exercise: "squat",
    grade: "fail",
    label: "Needs work",
    description: "Real-time half squats that stop short",
    src: "/samples/squat-fail.mp4",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Muscle_Strengthening_at_Home_-_Half_squat.webm",
    creator: "U.S. Centers for Disease Control and Prevention",
  },
];

export const samplesForExercise = (exercise: ExerciseKey) =>
  DEMO_SAMPLES.filter((sample) => sample.exercise === exercise);
