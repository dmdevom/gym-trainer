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

Rotation works exactly like video.py (it shares the same reader):
    ROTATE_DEG=270 python analyze.py data/videos/curl_right.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reps import find_reps, form_feedback
from video import VideoError, extract_series, median_smooth, pick_arm


def analyze_video(path: str) -> dict:
    """
    Run the full pipeline and return a JSON-able summary. Phase 3's endpoint
    will call exactly this; the CLI below is only a pretty-printer around it.
    """
    samples, meta = extract_series(path)
    arm, scores = pick_arm(samples)
    meta["arm"] = arm
    meta["arm_visibility"] = {k: round(v, 3) for k, v in scores.items()}

    # The one arm, chosen for the whole video, then smoothed. This is the exact
    # series video.py plots - so the reps counted here are the valleys you see.
    raw = [getattr(s, arm) for s in samples]
    smoothed = median_smooth(raw)
    times = [s.t for s in samples]

    reps = find_reps(smoothed)
    grades = form_feedback(reps, times)

    return {
        "meta": meta,
        "reps": len(grades),
        "full_reps": sum(g.full for g in grades),
        "per_rep": [
            {
                "number": g.number,
                "min_angle": round(g.min_angle, 1),
                "duration_s": round(g.duration_s, 2),
                "full": g.full,
                "issues": g.issues,
            }
            for g in grades
        ],
    }


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

    # The one line a coach would actually say out loud.
    partial = sum(1 for r in s["per_rep"] if any("partial" in i for i in r["issues"]))
    fast = sum(1 for r in s["per_rep"] if any("too fast" in i for i in r["issues"]))
    bits = [f"{s['full_reps']}/{s['reps']} full reps"]
    if partial:
        bits.append(f"{partial} partial")
    if fast:
        bits.append(f"{fast} rushed")
    print(f"\nverdict: {', '.join(bits)}.")


def main(path: str) -> None:
    summary = analyze_video(path)
    _print_summary(summary)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / "rep_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dest}  <- the exact shape Phase 3's endpoint will return.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze.py path/to/video.mp4")
    try:
        main(sys.argv[1])
    except VideoError as e:
        sys.exit(f"VideoError: {e}")
