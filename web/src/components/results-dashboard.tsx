"use client";

import { Activity, BadgeCheck, Clock3, Dumbbell, RotateCcw, Sparkles, Target } from "lucide-react";
import { useState } from "react";
import { deriveStats, resolveCoachingTokens, resultVideoUrl } from "@/lib/api";
import type { AnalysisResult } from "@/lib/types";
import { AngleChart } from "./angle-chart";

interface ResultsDashboardProps {
  result: AnalysisResult;
  originalUrl: string;
  onReset: () => void;
}

export function ResultsDashboard({ result, originalUrl, onReset }: ResultsDashboardProps) {
  const [mobileVideo, setMobileVideo] = useState<"original" | "analyzed">("analyzed");
  const stats = deriveStats(result);
  const coachingText = (text: string) => resolveCoachingTokens(text, result.thresholds.tempo_min_s);

  return (
    <section className="results-section" id="results" aria-labelledby="results-title">
      <div className="results-heading">
        <div>
          <span className="eyebrow"><Sparkles size={14} /> Analysis complete</span>
          <h2 id="results-title">Your set, decoded.</h2>
          <p>{result.meta.exercise.name} · {result.meta.side} side tracked</p>
        </div>
        <button className="secondary-button" type="button" onClick={onReset}><RotateCcw size={17} /> Analyze another</button>
      </div>

      <div className="stat-grid">
        <Stat icon={<Dumbbell />} value={String(result.reps)} label="Total reps" />
        <Stat icon={<BadgeCheck />} value={`${result.full_reps}/${result.reps}`} label="Full ROM" />
        <Stat icon={<Target />} value={stats.averageDepth === null ? "—" : `${stats.averageDepth}%`} label="Avg. depth" />
        <Stat icon={<Clock3 />} value={stats.averageTempo === null ? "—" : `${stats.averageTempo.toFixed(1)}s`} label="Avg. tempo" />
      </div>

      <div className="result-card verdict-card">
        <div className="verdict-mark"><Activity size={23} /></div>
        <div>
          <span className="card-kicker">Set verdict</span>
          <h3>{result.reps ? result.verdict : "No complete reps found"}</h3>
        </div>
      </div>

      <div className="mobile-video-tabs" role="tablist" aria-label="Result video">
        <button role="tab" aria-selected={mobileVideo === "original"} className={mobileVideo === "original" ? "active" : ""} onClick={() => setMobileVideo("original")}>Original</button>
        <button role="tab" aria-selected={mobileVideo === "analyzed"} className={mobileVideo === "analyzed" ? "active" : ""} onClick={() => setMobileVideo("analyzed")}>AI landmarked</button>
      </div>

      <div className="video-comparison">
        <VideoPanel className={mobileVideo === "original" ? "mobile-active" : ""} label="Original clip" src={originalUrl} />
        <VideoPanel className={mobileVideo === "analyzed" ? "mobile-active" : ""} label="AI landmarked" src={resultVideoUrl(result)} accent />
      </div>

      <div className="results-columns">
        <article className="result-card coaching-card">
          <span className="card-kicker">Improve &amp; next session</span>
          {result.coaching.session_story && <p className="coach-story">{coachingText(result.coaching.session_story)}</p>}
          <div className="coach-callouts">
            <div><span>Focus next</span><strong>{coachingText(result.coaching.focus)}</strong></div>
            {result.coaching.mental_cue && <div><span>Cue</span><strong className="mental-cue">{coachingText(result.coaching.mental_cue)}</strong></div>}
          </div>
          <div className="coach-list">
            {result.coaching.next_session.map((item) => <p key={item}>{coachingText(item)}</p>)}
          </div>
          <div className="cue-block">
            <span>Keep in mind</span>
            <ul>{result.coaching.keep_in_mind.map((tip) => <li key={tip}>{coachingText(tip)}</li>)}</ul>
          </div>
          <p className="muscle-note">{coachingText(result.coaching.muscle)}</p>
        </article>

        <article className="result-card chart-card">
          <div className="card-title-row">
            <div><span className="card-kicker">Movement signal</span><h3>{capitalize(result.meta.exercise.vertex_name)} angle</h3></div>
            <span className="chart-legend"><i /> Live angle</span>
          </div>
          <AngleChart result={result} />
        </article>
      </div>

      <article className="result-card reps-card">
        <div className="card-title-row"><div><span className="card-kicker">Rep by rep</span><h3>Form breakdown</h3></div></div>
        {result.per_rep.length ? (
          <>
            <div className="rep-table-wrap">
              <table className="rep-table">
                <thead><tr><th>Rep</th><th>Deepest</th><th>Depth</th><th>Tempo</th><th>Status</th><th>Coach note</th></tr></thead>
                <tbody>{result.per_rep.map((rep) => (
                  <tr key={rep.number}>
                    <td>#{rep.number}</td><td>{rep.min_angle.toFixed(0)}°</td><td>{rep.depth_pct}%</td><td>{rep.duration_s.toFixed(1)}s</td>
                    <td><span className={`status-pill ${rep.full && !rep.tags.length ? "good" : "warn"}`}>{rep.full && !rep.tags.length ? "Clean" : "Improve"}</span></td>
                    <td>{rep.issues.length ? rep.issues.join(" ") : "Full and controlled."}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="rep-mobile-list">{result.per_rep.map((rep) => (
              <div className="rep-mobile-card" key={rep.number}>
                <div><strong>Rep {rep.number}</strong><span className={`status-pill ${rep.full && !rep.tags.length ? "good" : "warn"}`}>{rep.full && !rep.tags.length ? "Clean" : "Improve"}</span></div>
                <dl><div><dt>Depth</dt><dd>{rep.depth_pct}%</dd></div><div><dt>Deepest</dt><dd>{rep.min_angle.toFixed(0)}°</dd></div><div><dt>Tempo</dt><dd>{rep.duration_s.toFixed(1)}s</dd></div></dl>
                <p>{rep.issues.length ? rep.issues.join(" ") : "Full and controlled."}</p>
              </div>
            ))}</div>
          </>
        ) : <div className="empty-reps">Try filming side-on with your full working limb visible, then analyze again.</div>}
        {!!result.form_checks?.length && <FormChecks checks={result.form_checks} />}
      </article>
    </section>
  );
}

function FormChecks({ checks }: { checks: NonNullable<AnalysisResult["form_checks"]> }) {
  return (
    <div className="form-checks">
      <span className="card-kicker">What we checked</span>
      <div className="check-chips">
        {checks.map((check) => {
          const tone = check.status === "flag" ? "bad" : check.status === "not_assessed" ? "dim" : "ok";
          const text = check.status === "flag" ? `${check.label} · ${check.flagged}/${check.assessed}`
            : check.status === "not_assessed" ? `${check.label} · not in frame` : `${check.label} ✓`;
          return <span key={check.key} className={`check-chip ${tone}`} title={check.status === "flag" ? check.cue : undefined}>{text}</span>;
        })}
      </div>
    </div>
  );
}

function Stat({ icon, value, label }: { icon: React.ReactNode; value: string; label: string }) {
  return <div className="stat-card"><span>{icon}</span><div><strong>{value}</strong><small>{label}</small></div></div>;
}

function VideoPanel({ label, src, accent, className = "" }: { label: string; src: string; accent?: boolean; className?: string }) {
  return <article className={`video-panel ${accent ? "accent" : ""} ${className}`}><div className="video-label"><span>{accent && <i />} {label}</span>{accent && <small>Pose + live metrics</small>}</div><video src={src} controls playsInline preload="metadata" /></article>;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
