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

Rotation works exactly like video.py (it shares the same reader):
    ROTATE_DEG=270 python analyze.py data/videos/curl_right.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from reps import DOWN_ENTER, FULL_ROM, UP_ENTER, find_reps, form_feedback
from video import VideoError, extract_series, median_smooth, pick_arm


def summarize(
    meta: dict,
    arm: str,
    scores: dict,
    raw: Sequence[Optional[float]],
    times: Sequence[float],
) -> dict:
    """
    The one place a rep summary is shaped. Give it the arm, its raw per-sample
    angle series and the matching timestamps; get back the JSON-able dict that
    the CLI prints, the endpoint returns, and the renderer draws from.

    Everything below is derived, never measured here: smoothing and the two
    reps.py layers do the work, and this only packs their output into one shape.
    Keeping that shape in a single function is what lets three very different
    consumers stay honest with each other.
    """
    meta = {
        **meta,
        "arm": arm,
        "arm_visibility": {k: round(v, 3) for k, v in scores.items()},
    }

    # The exact series video.py plots, so the reps counted here are the valleys
    # you see there - and the same series the browser chart and the overlay draw.
    smoothed = median_smooth(raw)
    reps = find_reps(smoothed)
    grades = form_feedback(reps, times)

    # start_t / end_t are the wire to the video overlay: the renderer bumps the
    # visible rep counter the instant the clock passes an end_t, so the number on
    # the video and the number in this dict are the same number by construction.
    per_rep = [
        {
            "number": g.number,
            "min_angle": round(g.min_angle, 1),
            "duration_s": round(g.duration_s, 2),
            "full": g.full,
            "issues": g.issues,
            "start_t": round(times[rep.start_idx], 3),
            "end_t": round(times[rep.end_idx], 3),
        }
        for rep, g in zip(reps, grades)
    ]

    full_reps = sum(g.full for g in grades)

    return {
        "meta": meta,
        "reps": len(grades),
        "full_reps": full_reps,
        "per_rep": per_rep,
        "verdict": _verdict(len(grades), full_reps, per_rep),
        # The chart on the page is this signal plus these three lines - the same
        # oscilloscope video.py draws to a PNG, redrawn live in the browser.
        "thresholds": {
            "up_enter": UP_ENTER,
            "down_enter": DOWN_ENTER,
            "full_rom": FULL_ROM,
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
    partial = sum(1 for r in per_rep if any("partial" in i for i in r["issues"]))
    fast = sum(1 for r in per_rep if any("too fast" in i for i in r["issues"]))
    bits = [f"{full_reps}/{reps} full reps"]
    if partial:
        bits.append(f"{partial} partial")
    if fast:
        bits.append(f"{fast} rushed")
    return ", ".join(bits)


def analyze_video(path: str) -> dict:
    """
    Run the full pipeline and return the summary. The /analyze/video endpoint
    reaches this same result through the renderer (which needs the frames too);
    the CLI below is only a pretty-printer around it.
    """
    samples, meta = extract_series(path)
    arm, scores = pick_arm(samples)

    # The one arm, chosen for the whole video, then handed to summarize.
    raw = [getattr(s, arm) for s in samples]
    times = [s.t for s in samples]
    return summarize(meta, arm, scores, raw, times)


def _print_summary(s: dict) -> None:
    m = s["meta"]
    print(f"\n{m['file']}  |  {m['arm']} arm  |  {m['sample_hz']} Hz  |  rot {m['rotation_applied']}")
    print(f"reps counted   : {s['reps']}")
    print(f"full-ROM reps  : {s['full_reps']} / {s['reps']}")

    if not s["per_rep"]:
        print(
            "\nNo complete reps found. If you did curl, the usual causes are\n"
            "rotation (run: python scripts/probe_video.py <file>) or thresholds\n"
            "that don't match your plot (open out/angle_series.png)."
        )
        return

    print()
    for r in s["per_rep"]:
        mark = "OK" if not r["issues"] else "!!"
        note = "clean" if not r["issues"] else "; ".join(r["issues"])
        print(
            f"  rep {r['number']:>2}  {mark}  "
            f"min {r['min_angle']:>5.1f}deg  {r['duration_s']:>4.1f}s   {note}"
        )

    print(f"\nverdict: {s['verdict']}.")


def main(path: str) -> None:
    summary = analyze_video(path)
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
