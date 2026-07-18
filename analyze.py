"""
Phase 2 checkpoint: a video in, a coached rep summary out.

    python analyze.py data/videos/curl_right.mp4

The product-logic layer - the thin orchestrator that turns the two hard-won
pieces into an answer a human cares about:

    video.py  ->  a smoothed per-frame elbow-angle series   (the signal)
    reps.py   ->  cycles, then a grade per cycle             (the meaning)
    here      ->  a summary you can read out loud            (the product)

Nothing is computed here that isn't computed upstream. If a rep count looks
wrong the fix is in video.py's angles or reps.py's thresholds, never in this
file - it only arranges what those two produced. That is the whole reason it
stays this short.

`summarize()` is the seam Phase 3 leans on: the /analyze/video endpoint and the
annotated-video renderer both build the same (arm, angle series, times) and call
it, so the CLI, the JSON the browser gets, and the overlay on the video can
never disagree about how many reps you did.

Exercise and rotation work exactly like video.py (it shares the same reader):
    EXERCISE=squat ROTATE_DEG=270 python analyze.py clip.mp4
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import llm_coach
from exercises import Exercise, get_exercise
from reps import evaluate_checks, find_reps, form_feedback
from video import VideoError, extract_series, median_smooth, pick_side


def summarize(
    meta: dict,
    exercise: Exercise,
    side: str,
    scores: dict,
    raw: Sequence[Optional[float]],
    times: Sequence[float],
    samples: Optional[Sequence["object"]] = None,
) -> dict:
    """
    The one place a rep summary is shaped. Give it the exercise, the tracked side,
    its raw per-sample angle series and the matching timestamps; get back the
    JSON-able dict that the CLI prints, the endpoint returns, and the renderer draws.

    `samples` are the full video.Sample objects (each carrying every landmark, not
    just the primary angle). When present, the secondary form checks run - torso
    lean, elbow drift, grounded legs - graded within each detected rep. When absent,
    or when the exercise defines no checks, the summary is exactly depth + tempo, as
    before. Either way the shape is identical; only the extra tags/issues appear.

    Everything below is derived, never measured here: smoothing and the reps.py
    layers do the work, and this only packs their output - now labelled with which
    exercise it was - into one shape. Keeping that shape in a single function is
    what lets three very different consumers stay honest with each other.
    """
    # The exact series video.py plots, so the reps counted here are the valleys you
    # see there - and the same series the browser chart and the overlay draw. The
    # thresholds come from the exercise now, not from module constants.
    smoothed = median_smooth(raw)
    reps = find_reps(smoothed, exercise.up_enter, exercise.down_enter)

    # The whole-body form checks, graded within each rep - but only if we were handed
    # the landmarks to judge them with. reps.evaluate_checks bows out (per rep, per
    # check) wherever a limb was out of frame, so this never invents a verdict.
    checks_per_rep = (
        [evaluate_checks(rep, samples, smoothed, side, exercise) for rep in reps]
        if samples is not None else []
    )
    grades = form_feedback(reps, times, exercise, checks_per_rep or None)
    form_checks = _form_summary(exercise, checks_per_rep)

    # start_t / end_t are the wire to the video overlay: the renderer bumps the
    # visible rep counter the instant the clock passes an end_t, so the number on
    # the video and the number in this dict are the same number by construction.
    per_rep = [
        {
            "number": g.number,
            "min_angle": round(g.min_angle, 1),
            "duration_s": round(g.duration_s, 2),
            "full": g.full,
            "depth_pct": g.depth_pct,
            "issues": g.issues,
            "tags": g.tags,
            "start_t": round(times[rep.start_idx], 3),
            "end_t": round(times[rep.end_idx], 3),
        }
        for rep, g in zip(reps, grades)
    ]

    full_reps = sum(g.full for g in grades)
    verdict = _verdict(len(grades), full_reps, per_rep)

    # The one soft field. Try the LLM (llm_coach.generate); on anything short of a
    # clean, valid answer it returns None and we use the offline rules below. Either
    # way the shape is identical, so every consumer is unaffected - meta just records
    # which path wrote it, for the demo and for debugging.
    coach = llm_coach.generate(exercise, per_rep, len(grades), full_reps, verdict, form_checks)
    coaching_source = "llm" if coach is not None else "rules"
    if coach is None:
        coach = coaching(grades, exercise, form_checks)
    elif grades and all(not r["tags"] for r in per_rep):
        # On a fully clean set, "what to fix next" has one correct answer - progressive
        # overload - so it's a deterministic call, not an interpretation to delegate. A
        # small model reliably eyeballs the raw per-rep durations and coaches natural
        # variation the grader already cleared ("your last rep was quicker"), which makes
        # the card contradict the video's own "3/3 clean". Take the actionable fields from
        # the rules and keep only the LLM's session_story for narrative colour.
        story = coach.get("session_story")
        coach = coaching(grades, exercise, form_checks)
        if story:
            coach["session_story"] = story
        coaching_source = "llm+rules"

    meta = {
        **meta,
        "exercise": {"key": exercise.key, "name": exercise.name, "vertex_name": exercise.vertex_name},
        "side": side,
        "side_visibility": {k: round(v, 3) for k, v in scores.items()},
        "coaching_source": coaching_source,
    }

    return {
        "meta": meta,
        "reps": len(grades),
        "full_reps": full_reps,
        "per_rep": per_rep,
        "verdict": verdict,
        # Which whole-body checks ran this set, and how they went (assessed / flagged /
        # not-in-frame). Drives the page's "what we checked" row and the coaching nudge.
        "form_checks": form_checks,
        # What to work on next - LLM-written when available, this session's rules otherwise.
        "coaching": coach,
        # The chart on the page is this signal plus these lines - the same
        # oscilloscope video.py draws to a PNG, redrawn live in the browser.
        "thresholds": {
            "up_enter": exercise.up_enter,
            "down_enter": exercise.down_enter,
            "full_rom": exercise.full_rom,
            "gauge_deep": exercise.gauge_deep,
            "tempo_min_s": exercise.tempo_min_s,
        },
        "series": {
            "t": [round(t, 3) for t in times],
            "angle": [None if v is None else round(v, 1) for v in smoothed],
        },
    }


def _verdict(reps: int, full_reps: int, per_rep: List[dict]) -> str:
    """The one line a coach would say out loud, built once and reused everywhere."""
    if not reps:
        return "no complete reps found"
    shallow = sum(1 for r in per_rep if "shallow" in r["tags"])
    rushed = sum(1 for r in per_rep if "rushed" in r["tags"])
    # A form fault is any tag that isn't the depth/tempo pair - kept as one count so
    # the line stays short whether it was an elbow drift, a swing, or a forward fold.
    form = sum(1 for r in per_rep if any(t not in ("shallow", "rushed") for t in r["tags"]))
    bits = [f"{full_reps}/{reps} full reps"]
    if shallow:
        bits.append(f"{shallow} shallow")
    if rushed:
        bits.append(f"{rushed} rushed")
    if form:
        bits.append(f"{form} form")
    return ", ".join(bits)


def _form_summary(exercise: Exercise, checks_per_rep: Sequence[Sequence["object"]]) -> List[dict]:
    """Roll the per-rep form checks up to a per-set view: for each check the exercise
    defines, how many reps it was assessable on, how many it flagged, and a single
    status. `not_assessed` means the limb was out of frame the whole set - the honest
    'we couldn't see this', which the page shows dimmed and the coaching turns into a
    gentle 'film your whole body' nudge rather than a fake pass."""
    out: List[dict] = []
    for chk in exercise.checks:
        rows = [cr for rep in checks_per_rep for cr in rep if cr.key == chk.key]
        assessed = [r for r in rows if r.status != "skip"]
        flagged = [r for r in assessed if r.status == "flag"]
        out.append({
            "key": chk.key,
            "label": chk.label,
            "fault": chk.fault,
            "cue": chk.cue,
            "assessed": len(assessed),
            "flagged": len(flagged),
            "status": ("flag" if flagged else "ok") if assessed else "not_assessed",
        })
    return out


def coaching(grades: Sequence["object"], exercise: Exercise,
             form_checks: Optional[Sequence[dict]] = None) -> dict:
    """
    The "what next" section - built from THIS session, not a canned tip sheet.

    Same philosophy as the on-video flash: don't just say what was wrong, say what
    to do about it. `focus` is the single biggest lever; `next_session` turns the
    session's own tag counts into actions; `keep_in_mind` and `muscle` are the
    exercise's standing cues and why the movement is worth doing well.

    `form_checks` is _form_summary()'s roll-up. Posture/safety outranks depth and
    tempo, so a flagged check takes the focus; a check that was never in frame
    becomes one gentle "film your whole body" line, never a complaint. This is the
    offline mirror of what the LLM path writes, so the app reads the same with or
    without a key.
    """
    reps = len(grades)
    shallow = sum(1 for g in grades if "shallow" in g.tags)
    rushed = sum(1 for g in grades if "rushed" in g.tags)
    flagged = [f for f in (form_checks or []) if f["status"] == "flag"]
    unseen = [f for f in (form_checks or []) if f["status"] == "not_assessed"]

    if not reps:
        limb = "body" if exercise.vertex_name == "knee" else "arm"
        return {
            "focus": "Get a clean side-on clip",
            "next_session": [
                f"No full reps were counted - that's usually the camera angle, not "
                f"you. Re-film side-on with your whole {limb} in frame and try again."
            ],
            "keep_in_mind": list(exercise.tips),
            "muscle": exercise.muscle,
            "mental_cue": "",   # no reps to cue off; the UI hides an empty cue row
        }

    # Pick the one thing worth fixing first. Posture/safety leads (a folded-forward
    # squat or a swung curl is the thing to fix before anything else), then depth
    # beats tempo (a shallow rep skips the muscle; a fast full rep at least did the
    # work), and a clean session earns the nicest problem to have - add load.
    # `mental_cue` mirrors the LLM path's extra field so the offline card reads the
    # same shape: one short physical cue for the next set. A flagged check's label is
    # already a crisp cue ("Elbow pinned", "Chest up"); the rest get a fixed phrase.
    if flagged:
        focus = flagged[0]["label"]
        mental_cue = flagged[0]["label"]
    elif shallow:
        focus = "Full range of motion"
        mental_cue = "Full depth every rep"
    elif rushed:
        focus = "Slower, controlled tempo"
        mental_cue = "Two-second lower each rep"
    else:
        focus = "Progressive overload"
        mental_cue = "Add a little, hold form"

    next_session: List[str] = []
    for f in flagged:
        next_session.append(
            f"{f['flagged']} of {f['assessed']} reps: {f['fault'].lower()}. {f['cue']}"
        )
    if shallow:
        next_session.append(
            f"{shallow} of {reps} reps stopped short of full range. {exercise.depth_cue} "
            "Drop the load a little if you need to so every rep reaches depth."
        )
    if rushed:
        next_session.append(
            f"{rushed} of {reps} reps were rushed. {exercise.tempo_cue} "
            "Count a two-second lower on each one."
        )
    if not flagged and not shallow and not rushed:
        next_session.append(
            "Clean session - every rep full and controlled. Add a rep or a little "
            "weight next time and hold this same form."
        )
    # Whatever we couldn't see, said once and gently - never a fault, just an invite
    # to frame the next clip so the app can check more of the lift.
    if unseen:
        what = ", ".join(f["label"].lower() for f in unseen)
        next_session.append(
            f"Film with your whole body in frame next time and I can also check {what}."
        )

    return {
        "focus": focus,
        "next_session": next_session,
        "keep_in_mind": list(exercise.tips),
        "muscle": exercise.muscle,
        "mental_cue": mental_cue,
    }


def analyze_video(path: str, exercise_key: str = "bicep_curl") -> dict:
    """
    Run the full pipeline and return the summary. The /analyze/video endpoint
    reaches this same result through the renderer (which needs the frames too);
    the CLI below is only a pretty-printer around it.
    """
    exercise = get_exercise(exercise_key)
    samples, meta = extract_series(path, exercise)
    side, scores = pick_side(samples)

    # The one side, chosen for the whole video, then handed to summarize - along with
    # the samples themselves, so the form checks have every landmark to read.
    raw = [getattr(s, side) for s in samples]
    times = [s.t for s in samples]
    return summarize(meta, exercise, side, scores, raw, times, samples)


def _print_summary(s: dict) -> None:
    m = s["meta"]
    ex = m["exercise"]["name"]
    print(f"\n{m['file']}  |  {ex}  |  {m['side']} side  |  {m['sample_hz']} Hz  |  rot {m['rotation_applied']}")
    print(f"reps counted   : {s['reps']}")
    print(f"full-ROM reps  : {s['full_reps']} / {s['reps']}")

    if not s["per_rep"]:
        print(
            "\nNo complete reps found. The usual causes are rotation (run:\n"
            "python scripts/probe_video.py <file>) or thresholds that don't match\n"
            "your plot (open out/angle_series.png)."
        )
    else:
        print()
        for r in s["per_rep"]:
            mark = "OK" if not r["issues"] else "!!"
            note = "clean" if not r["issues"] else "; ".join(r["issues"])
            print(
                f"  rep {r['number']:>2}  {mark}  "
                f"min {r['min_angle']:>5.1f}deg  {r['duration_s']:>4.1f}s  {r['depth_pct']:>3.0f}%   {note}"
            )
        print(f"\nverdict: {s['verdict']}.")

    fc = s.get("form_checks") or []
    if fc:
        marks = {"ok": "ok", "flag": "!!", "not_assessed": "--"}
        print("\nform checks:")
        for f in fc:
            detail = (f"{f['flagged']}/{f['assessed']} flagged" if f["status"] == "flag"
                      else "not in frame" if f["status"] == "not_assessed"
                      else "clean")
            print(f"  {marks.get(f['status'], '  ')}  {f['label']:<16} {detail}")

    c = s["coaching"]
    if c.get("session_story"):
        print(f"\n{c['session_story']}")
    print(f"\nfocus next : {c['focus']}")
    if c.get("mental_cue"):
        print(f"cue        : {c['mental_cue']}")
    for item in c["next_session"]:
        print(f"  - {item}")


def main(path: str) -> None:
    summary = analyze_video(path, os.environ.get("EXERCISE", "bicep_curl"))
    _print_summary(summary)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / "rep_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dest}  <- the exact shape the /analyze/video endpoint returns.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze.py path/to/video.mp4")
    try:
        main(sys.argv[1])
    except VideoError as e:
        sys.exit(f"VideoError: {e}")
