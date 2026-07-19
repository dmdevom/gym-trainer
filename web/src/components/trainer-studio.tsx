"use client";

import {
  Activity, ArrowRight, Camera, Check, CircleAlert, Dumbbell,
  FileVideo, Gauge, LoaderCircle, Play, RotateCcw, Sparkles, Timer, Upload, Video,
} from "lucide-react";
import Image from "next/image";
import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { fetchExercises, fetchProgress, MAX_VIDEO_MB, submitVideo, validateVideo } from "@/lib/api";
import { DEMO_SAMPLES, samplesForExercise } from "@/lib/samples";
import type { AnalysisResult, AnalysisState, DemoSample, Exercise, ExerciseKey, InputMode, ProgressResponse } from "@/lib/types";
import { CameraRecorder } from "./camera-recorder";
import { ResultsDashboard } from "./results-dashboard";

const FALLBACK_EXERCISES: Exercise[] = [
  { key: "bicep_curl", name: "Bicep Curl", vertex_name: "elbow", film_tip: "Film side-on, whole arm in frame.", tips: [] },
  { key: "barbell_curl", name: "Barbell Curl", vertex_name: "elbow", film_tip: "Film side-on, whole torso and arms in frame.", tips: [] },
  { key: "squat", name: "Squat", vertex_name: "knee", film_tip: "Film side-on, whole body in frame — step back so your feet show.", tips: [] },
];

const MODE_ITEMS: Array<{ id: InputMode; label: string; icon: React.ReactNode }> = [
  { id: "sample", label: "Try a sample", icon: <Play size={17} /> },
  { id: "upload", label: "Upload video", icon: <Upload size={17} /> },
  { id: "record", label: "Use camera", icon: <Camera size={17} /> },
];

const HERO_INSIGHTS = [
  { id: "reps", label: "Rep count", value: "4/5", status: "clean reps", title: "Rep 4 counted", detail: "Full extension confirmed" },
  { id: "rom", label: "Range", value: "63°", status: "full ROM", title: "Depth target reached", detail: "100% range of motion" },
  { id: "tempo", label: "Tempo", value: "2.3s", status: "controlled", title: "Tempo on target", detail: "Smooth lowering phase" },
  { id: "coach", label: "Coaching", value: "1 cue", status: "next rep", title: "Keep your elbow pinned", detail: "Avoid shoulder drift" },
] as const;

export function TrainerStudio() {
  const [exercises, setExercises] = useState<Exercise[]>(FALLBACK_EXERCISES);
  const [selectedExercise, setSelectedExercise] = useState<ExerciseKey>("bicep_curl");
  const [mode, setMode] = useState<InputMode>("sample");
  const [selectedSample, setSelectedSample] = useState<DemoSample>(DEMO_SAMPLES[0]);
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState(DEMO_SAMPLES[0].src);
  const ownedPreviewRef = useRef<string | null>(null);
  const [state, setState] = useState<AnalysisState>("ready");
  const [progress, setProgress] = useState({ stage: "Ready", pct: 0 });
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [heroInsightIndex, setHeroInsightIndex] = useState(0);
  const activeRequestRef = useRef<AbortController | null>(null);

  const selectedExerciseInfo = exercises.find((exercise) => exercise.key === selectedExercise) || FALLBACK_EXERCISES[0];
  const busy = state === "uploading" || state === "analyzing";

  const releaseOwnedPreview = useCallback(() => {
    if (ownedPreviewRef.current) URL.revokeObjectURL(ownedPreviewRef.current);
    ownedPreviewRef.current = null;
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchExercises(controller.signal).then((items) => {
      if (items.length) setExercises(items.map((exercise) =>
        exercise.key === "bicep_curl" ? { ...exercise, name: "Bicep Curl" } : exercise
      ));
    }).catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => () => {
    activeRequestRef.current?.abort();
    releaseOwnedPreview();
  }, [releaseOwnedPreview]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setInterval(() => {
      setHeroInsightIndex((current) => (current + 1) % HERO_INSIGHTS.length);
    }, 3600);
    return () => window.clearInterval(timer);
  }, []);

  function changeExercise(key: ExerciseKey) {
    if (busy) return;
    setSelectedExercise(key);
    setError(null);
    setResult(null);
    if (mode === "sample") {
      const next = samplesForExercise(key)[0];
      if (next) chooseSample(next);
    } else if (state === "complete") {
      setState((mode === "upload" ? customFile : recordedFile) ? "ready" : "idle");
    }
  }

  function changeMode(nextMode: InputMode) {
    if (busy || nextMode === mode) return;
    releaseOwnedPreview();
    setMode(nextMode);
    setError(null);
    setResult(null);
    setCustomFile(null);
    setRecordedFile(null);
    if (nextMode === "sample") {
      const sample = samplesForExercise(selectedExercise)[0] || DEMO_SAMPLES[0];
      setSelectedSample(sample);
      setPreviewUrl(sample.src);
      setState("ready");
    } else {
      setPreviewUrl("");
      setState("idle");
    }
  }

  function chooseSample(sample: DemoSample) {
    releaseOwnedPreview();
    setSelectedSample(sample);
    setSelectedExercise(sample.exercise);
    setPreviewUrl(sample.src);
    setError(null);
    setState("ready");
  }

  function setFile(file: File | null) {
    if (!file) return;
    const issue = validateVideo(file);
    if (issue) {
      setError(issue);
      setState("error");
      return;
    }
    releaseOwnedPreview();
    const url = URL.createObjectURL(file);
    ownedPreviewRef.current = url;
    setCustomFile(file);
    setPreviewUrl(url);
    setError(null);
    setState("ready");
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] || null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    event.currentTarget.classList.remove("dragging");
    setFile(event.dataTransfer.files?.[0] || null);
  }

  function handleRecording(file: File | null, url?: string) {
    setRecordedFile(file);
    if (!file || !url) {
      setPreviewUrl("");
      setState("idle");
      return;
    }
    releaseOwnedPreview();
    ownedPreviewRef.current = url;
    setPreviewUrl(url);
    setError(null);
    setState("ready");
  }

  async function resolveInputFile(): Promise<File> {
    if (mode === "upload" && customFile) return customFile;
    if (mode === "record" && recordedFile) return recordedFile;
    if (mode === "sample") {
      const response = await fetch(selectedSample.src);
      if (!response.ok) throw new Error("This demo clip is unavailable. Choose another sample or upload your own.");
      const blob = await response.blob();
      return new File([blob], selectedSample.src.split("/").pop() || "sample.mp4", { type: blob.type || "video/mp4" });
    }
    throw new Error(mode === "record" ? "Record a clip first." : "Choose a video first.");
  }

  async function analyze() {
    activeRequestRef.current?.abort();
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setError(null);
    setResult(null);
    setState("uploading");
    setProgress({ stage: "Preparing video", pct: 2 });
    try {
      const file = await resolveInputFile();
      const issue = validateVideo(file);
      if (issue) throw new Error(issue);
      setProgress({ stage: "Uploading securely", pct: 6 });
      const token = await submitVideo(file, selectedExercise, controller.signal);
      setState("analyzing");
      await pollAnalysis(token, controller);
    } catch (caught) {
      if (controller.signal.aborted) return;
      setError(caught instanceof Error ? caught.message : "Something went wrong. Please try again.");
      setState("error");
    }
  }

  async function pollAnalysis(token: string, controller: AbortController) {
    const deadline = Date.now() + 10 * 60 * 1000;
    let failedPolls = 0;
    while (!controller.signal.aborted && Date.now() < deadline) {
      let update: ProgressResponse;
      try {
        update = await fetchProgress(token, controller.signal);
        failedPolls = 0;
      } catch (caught) {
        // One dropped poll must not kill a 30s analysis - only give up after
        // several failures in a row. A server-reported job error still throws below.
        if (controller.signal.aborted || ++failedPolls >= 3) throw caught;
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        continue;
      }
      setProgress({ stage: update.stage || "Analyzing movement", pct: update.pct ?? 12 });
      if (update.error) throw new Error(update.error);
      if (update.done && update.result) {
        setResult(update.result);
        setState("complete");
        setProgress({ stage: "Complete", pct: 100 });
        window.setTimeout(() => document.querySelector("#results .results-heading")?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 750));
    }
    if (!controller.signal.aborted) throw new Error("Analysis is taking longer than expected. Please try again.");
  }

  function reset() {
    activeRequestRef.current?.abort();
    setResult(null);
    setError(null);
    setProgress({ stage: "Ready", pct: 0 });
    if (mode === "sample") {
      setState("ready");
      setPreviewUrl(selectedSample.src);
    } else {
      releaseOwnedPreview();
      setPreviewUrl("");
      setCustomFile(null);
      setRecordedFile(null);
      setState("idle");
    }
    // Bring the studio section (hidden while results were up) back into view. It remounts now
    // that state is no longer "complete", so scroll on the next tick, once it's in the DOM.
    window.setTimeout(() => document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  function goToStudio() {
    // Results unmount the studio (see the `state !== "complete"` gate below); clear the
    // completed state so it remounts, then scroll to it on the next tick, once it's in the DOM.
    if (state === "complete") {
      setResult(null);
      setState(mode === "sample" || customFile || recordedFile ? "ready" : "idle");
    }
    window.setTimeout(() => document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  function openSampleStudio() {
    if (mode !== "sample") changeMode("sample");
    goToStudio();
  }

  return (
    <main>
      <nav className="site-nav">
        <a className="brand" href="#top" aria-label="trAIner home"><span>tr</span><strong>AI</strong><span>ner</span><i /></a>
        <a className="nav-cta" href="#analyze" onClick={(event) => { event.preventDefault(); goToStudio(); }}>Analyze a set <ArrowRight size={15} /></a>
      </nav>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="eyebrow"><Sparkles size={14} /> AI movement analysis</span>
          <h1>Train smarter.<br /><em>Move better.</em></h1>
          <p>Upload or record one set. See every rep counted, graded, and explained — with a landmarked replay that shows exactly what to fix.</p>
          <div className="hero-actions">
            <button className="primary-button hero-button" type="button" onClick={openSampleStudio}><Play size={17} fill="currentColor" /> Try a sample</button>
            <button className="secondary-button hero-button" type="button" onClick={goToStudio}><Gauge size={17} /> Analyze a set</button>
          </div>
          <div className="hero-signals"><span><i /> 3 exercises</span><span><i /> Rep-by-rep feedback</span><span><i /> Results in seconds</span></div>
        </div>
        <div className="hero-visual">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" />
          <div className="hero-panel">
            <Image className="hero-photo" src="/hero-dumbbell-curl.jpg" alt="Athlete performing a dumbbell curl while trAIner analyzes the movement" fill sizes="(max-width: 700px) 330px, 390px" priority />
            <div className="hero-analysis-status"><i /> Analyzing · Bicep curl</div>
            <div className="hero-insight-stack" aria-label="Example analysis feedback">
              {HERO_INSIGHTS.map((insight, index) => {
                const position = (index - heroInsightIndex + HERO_INSIGHTS.length) % HERO_INSIGHTS.length;
                return (
                  <button key={insight.id} type="button" className={`hero-insight-card stack-${position}`} aria-hidden={position !== 0} tabIndex={position === 0 ? 0 : -1} onClick={() => position === 0 && setHeroInsightIndex((current) => (current + 1) % HERO_INSIGHTS.length)}>
                    <span>{insight.id === "tempo" ? <Timer size={18} /> : insight.id === "rom" ? <Gauge size={18} /> : insight.id === "coach" ? <Sparkles size={18} /> : <Activity size={18} />}</span>
                    <div><small>{insight.label}</small><strong>{insight.title}</strong><p>{insight.detail}</p></div>
                    <div className="hero-insight-value"><strong>{insight.value}</strong><small>{insight.status}</small></div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {state !== "complete" && (
      <section className="studio-section">
        <div className="section-heading" id="analyze"><span className="eyebrow">01 · Pick your movement</span><h2>What are we analyzing?</h2><p>Choose an exercise, then try a sample or use your own video.</p></div>
        <div className="exercise-grid">
          {exercises.map((exercise) => (
            <button key={exercise.key} type="button" className={`exercise-card ${selectedExercise === exercise.key ? "selected" : ""}`} onClick={() => changeExercise(exercise.key)} disabled={busy}>
              <span className="exercise-icon">{exercise.key === "squat" ? <Activity size={25} /> : <Dumbbell size={25} />}</span>
              <span><strong>{exercise.name}</strong><small>Tracks {exercise.vertex_name} angle</small></span>
              <i className="select-check">{selectedExercise === exercise.key && <Check size={14} />}</i>
            </button>
          ))}
        </div>

        <div className="analyzer-card">
          <div className="mode-tabs" role="tablist" aria-label="Video source">
            {MODE_ITEMS.map((item) => <button key={item.id} type="button" role="tab" aria-selected={mode === item.id} className={mode === item.id ? "active" : ""} onClick={() => changeMode(item.id)} disabled={busy}>{item.icon}{item.label}</button>)}
          </div>

          <div className="analyzer-body">
            <div className="input-pane">
              {mode === "sample" && (
                <div className="sample-picker">
                  <div className="sample-options">{samplesForExercise(selectedExercise).map((sample) => (
                    <button type="button" key={sample.id} className={selectedSample.id === sample.id ? "selected" : ""} onClick={() => chooseSample(sample)} disabled={busy}>
                      <span className={`grade-dot ${sample.grade}`}><Check size={14} /></span><span><strong>{sample.label}</strong><small>{sample.description}</small></span>
                    </button>
                  ))}</div>
                  <div className="preview-shell"><video src={selectedSample.src} controls muted loop playsInline preload="metadata" /><span className={`preview-grade ${selectedSample.grade}`}>{selectedSample.grade === "pass" ? "PASS SAMPLE" : "FORM CHECK"}</span></div>
                </div>
              )}

              {mode === "upload" && (
                customFile && previewUrl ? <div className="uploaded-preview"><video src={previewUrl} controls playsInline /><button type="button" className="secondary-button" onClick={() => { releaseOwnedPreview(); setCustomFile(null); setPreviewUrl(""); setState("idle"); }} disabled={busy}><RotateCcw size={16} /> Choose another</button></div> :
                <div className="drop-zone" onDragOver={(event) => { event.preventDefault(); event.currentTarget.classList.add("dragging"); }} onDragLeave={(event) => event.currentTarget.classList.remove("dragging")} onDrop={handleDrop}>
                  <span className="upload-icon"><FileVideo size={28} /></span><h3>Drop your workout video here</h3><p>MP4, MOV, WebM and more · up to {MAX_VIDEO_MB} MB</p><label className="secondary-button">Browse files<input type="file" accept="video/*,.mp4,.mov,.webm,.avi,.mkv,.m4v,.3gp" onChange={handleFileInput} /></label>
                </div>
              )}

              {mode === "record" && <CameraRecorder disabled={busy} onCapture={handleRecording} onError={(message) => { setError(message); setState("error"); }} />}
            </div>

            <aside className="analyze-sidebar">
              <div className="film-tip"><span><Video size={18} /></span><div><small>How to film</small><p>{selectedExerciseInfo.film_tip}</p></div></div>
              <div className="analysis-includes"><small>Your analysis includes</small><ul><li><Check size={14} /> Automatic rep counting</li><li><Check size={14} /> Range-of-motion grading</li><li><Check size={14} /> Tempo and form feedback</li><li><Check size={14} /> Landmark overlay video</li></ul></div>
              {busy ? (
                <div className="progress-box" role="status"><div className="progress-label"><span><LoaderCircle className="spin" size={16} />{progress.stage}</span><strong>{Math.round(progress.pct)}%</strong></div><div className="progress-track"><i style={{ width: `${Math.max(3, progress.pct)}%` }} /></div><small>Keep this tab open while we map your movement.</small></div>
              ) : (
                <button className="primary-button analyze-button" type="button" onClick={analyze} disabled={state === "idle" || (mode === "record" && !recordedFile)}><Gauge size={18} /> Analyze this set <ArrowRight size={17} /></button>
              )}
              {error && <div className="error-box" role="alert"><CircleAlert size={18} /><div><strong>Couldn&apos;t analyze that</strong><span>{error}</span></div></div>}
            </aside>
          </div>
        </div>
      </section>
      )}

      {state === "complete" && result && previewUrl && <ResultsDashboard result={result} originalUrl={previewUrl} onReset={reset} />}

      <footer><a className="brand" href="#top"><span>tr</span><strong>AI</strong><span>ner</span><i /></a><p>Train every rep.</p></footer>
    </main>
  );
}
