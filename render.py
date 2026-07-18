"""
Phase 3/4, the money shot: draw the whole analysis onto every frame, then write a
video a browser will actually play. Most people watching a demo won't read a JSON
table under the video - so the video itself has to say everything: which rep, how
deep (a traffic-light range-of-motion gauge), how fast (a tempo bar), what went
wrong on each rep (a detailed flash), a running tally, and an end card that sums
up the set and says what to work on next.

Two decisions here cost real debugging time, so they keep their comments:

  1. CODEC. This OpenCV cannot encode H.264 - there is no software x264 and the
     hardware v4l2m2m path isn't present - and a browser <video> refuses the mp4v
     it CAN make. VP8 inside a .webm is the way out: OpenCV writes it, every
     browser plays it, and it needs no extra dependency. See LEARNINGS #12.

  2. ONE detection pass, then held. We detect at the SAME stride analyze.py uses,
     so the rep count burned into the video equals the count in the JSON - always,
     by construction. Then we decode every frame and draw a skeleton interpolated
     between the two nearest samples. The numbers come from the sparse pass; only
     the picture is dense. See LEARNINGS #13.

The exercise (which joint, which thresholds, what to call it) comes from
exercises.py, so this file draws a squat's knee exactly the way it draws a curl's
elbow - it never hard-codes the arm.

Usage:
    python render.py data/videos/curl_right.mp4          # -> out/annotated.webm
    EXERCISE=squat ROTATE_DEG=270 python render.py clip.mp4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from analyze import _print_summary, summarize
from exercises import Exercise, get_exercise
from video import (
    CONF_MIN,
    MODEL_PATH,
    STRIDE,
    Sample,
    VideoError,
    _ROT,
    _open,
    _to_sample,
    pick_side,
)

# Cap the LONGER side. A smaller frame detects faster, encodes smaller, and still
# reads fine - the plan's "downscale before inference" made concrete. Capping the
# long side (not the width) shrinks portrait phone clips too. Landmarks are stored
# in THIS space, so drawing lines up for free.
MAX_SIDE = 960
REF = float(MAX_SIDE)

# BlazePose-33 skeleton. MediaPipe deleted POSE_CONNECTIONS along with the legacy
# API (LEARNINGS #6), so the edge list is ours to own. Face and fingers are left
# off deliberately - detail the overlay doesn't need.
POSE_EDGES = [
    (11, 12),                                          # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),            # arms
    (11, 23), (12, 24), (23, 24),                      # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # right leg
]

# BGR. The tracked limb is drawn in the same orange as video.py's smoothed line,
# so the wave on the chart and the limb on the video read as the same object.
C_BODY = (180, 140, 90)     # muted steel-blue for the rest of the skeleton
C_ARM = (0, 165, 255)       # orange - the limb being measured
C_JOINT = (235, 235, 235)
C_TEXT = (245, 245, 245)
C_DIM = (165, 165, 165)
C_PANEL = (16, 16, 16)

# The traffic light, in BGR. Same meaning wherever it appears (gauge fill, the live
# angle number, the tempo bar): red = barely into the rep, green = full range / on
# pace. Dark tints are the gauge's background bands.
C_GREEN = (95, 200, 95)
C_YELLOW = (60, 200, 240)
C_RED = (70, 80, 235)
BAND_GREEN = (40, 70, 40)
BAND_YELLOW = (40, 75, 95)
BAND_RED = (48, 38, 80)

FONT = cv2.FONT_HERSHEY_SIMPLEX

Landmarks = Optional[Dict[int, Tuple[float, float, float]]]


# --- frame prep (identical in both passes) --------------------------------

def _prep(frame: np.ndarray, rot: int) -> np.ndarray:
    """Rotate (ours, never OpenCV's auto) then downscale so the long side is
    MAX_SIDE. Applied identically in both passes, which is the whole reason stored
    landmark pixels still land on the right joint when we draw."""
    if rot in _ROT:
        frame = cv2.rotate(frame, _ROT[rot])
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _k(img: np.ndarray) -> float:
    """Overlay scale for this frame: 1.0 at MAX_SIDE, less on a smaller clip."""
    return max(img.shape[:2]) / REF


def _thick(k: float, base: float) -> int:
    return max(1, round(base * k))


def _landmarks_px(result, w: int, h: int) -> Landmarks:
    """All 33 landmarks as {index: (x_px, y_px, visibility)} in output space, or
    None when no pose was found. Same normalized->pixel multiply the angle depends
    on - forget it and the skeleton draws itself onto the top-left corner."""
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    return {i: (lm.x * w, lm.y * h, lm.visibility or 0.0) for i, lm in enumerate(lms)}


# --- detection pass: sparse, and the single source of the count -----------

def _detect_pass(path: str, exercise: Exercise, progress_cb=None):
    """
    Sample at STRIDE, exactly like analyze.py's signal, but keep the landmark
    pixels too - analyze.py throws them away and the overlay can't. Returns the
    samples (for counting), the aligned landmark maps (for drawing), the meta, the
    fps/rotation, the output frame size the writer must match, and the total frame
    count (for the progress bar).
    """
    cap, rot, fps = _open(Path(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if progress_cb:
        progress_cb("Analysing", 0)   # a stage to show while the model loads, so the
                                      # bar moves off "Starting" instead of sitting there
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=CONF_MIN,
        min_tracking_confidence=CONF_MIN,
    )

    samples: List[Sample] = []
    lms_px: List[Landmarks] = []
    out_w = out_h = 0
    idx = 0
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if not cap.grab():
                break
            if idx % STRIDE:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break

            frame = _prep(frame, rot)
            out_h, out_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(idx * 1000 / fps))

            samples.append(_to_sample(result, idx, idx / fps, out_w, out_h, exercise))
            lms_px.append(_landmarks_px(result, out_w, out_h))
            idx += 1

            # Detection is the first ~40% of the wait. Report it throttled - the
            # callback writes into a shared job dict, so don't spam it every frame.
            if progress_cb and idx % 15 == 0:
                progress_cb("Analysing", round(40 * min(idx, total) / total) if total else None)
    cap.release()

    if not samples:
        raise VideoError("No frames decoded. Run scripts/probe_video.py on this file.")

    meta = {
        "file": Path(path).name,
        "fps": round(fps, 2),
        "rotation_applied": rot,
        "stride": STRIDE,
        "sample_hz": round(fps / STRIDE, 2),
        "frames_sampled": len(samples),
    }
    return samples, lms_px, meta, fps, rot, (out_w, out_h), total


# --- interpolation: sparse samples, smooth picture ------------------------

def _lerp_landmarks(a: Landmarks, b: Landmarks, f: float) -> Landmarks:
    """Blend two landmark maps. A joint is blended only where BOTH samples saw it;
    if one sample lost the limb, holding is honest and teleporting across the gap is
    not - so we fall back to whichever end actually has it."""
    if a is None:
        return b
    if b is None or f <= 0:
        return a
    if f >= 1:
        return b
    out: Dict[int, Tuple[float, float, float]] = {}
    for i in a:
        xa, ya, va = a[i]
        xb, yb, vb = b[i]
        out[i] = (xa + (xb - xa) * f, ya + (yb - ya) * f, min(va, vb))
    return out


def _lerp_angle(angle: List[Optional[float]], j: int, f: float) -> Optional[float]:
    """The live HUD number, blended between the two smoothed samples that bracket
    this frame. Same series the chart draws, so the number on the video and the
    valley on the chart are the same measurement."""
    a = angle[j]
    b = angle[j + 1] if j + 1 < len(angle) else None
    if a is None:
        return b
    if b is None:
        return a
    return a + (b - a) * f


# --- text + colour helpers ------------------------------------------------

def _pt(p) -> Tuple[int, int]:
    return int(p[0]), int(p[1])


def _shadow_text(img, text, org, scale, color, k, weight=2) -> None:
    """A black outline under coloured text so it stays legible over any background -
    a bright gym floor or a dark wall, the overlay reads either way.

    Fill thickness is floored at 2: a 1px fill (thin small text) all but vanishes
    under the outline, and neighbouring glyphs' outlines merge into a muddy blur
    that reads as doubled text. Two pixels is the difference between smudge and
    crisp; the outline stays a step thicker so the halo survives."""
    fill = max(2, _thick(k, weight))
    outline = max(fill + 1, _thick(k, weight + 2))
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, fill, cv2.LINE_AA)


def _text_w(text, scale, k, weight=2) -> int:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, _thick(k, weight))
    return tw


def _center(img, text, cx, y, scale, color, k, weight=2) -> None:
    _shadow_text(img, text, (int(cx - _text_w(text, scale, k, weight) / 2), y), scale, color, k, weight)


def _wrap(text, scale, k, maxw, max_lines=3) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if _text_w(trial, scale, k) > maxw and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _zone_color(angle: Optional[float], thr: dict):
    """Traffic light on range of motion. A lower angle is always deeper (a curl and
    a squat both flex DOWN in angle), so the test is the same for every exercise:
    green once you pass full range, yellow while working, red barely into it."""
    if angle is None:
        return C_DIM
    if angle <= thr["full_rom"]:
        return C_GREEN
    if angle <= thr["up_enter"]:
        return C_YELLOW
    return C_RED


def _depth_pct(angle: Optional[float], thr: dict) -> Optional[float]:
    if angle is None:
        return None
    span = max(1.0, thr["down_enter"] - thr["full_rom"])
    return max(0.0, min(100.0, (thr["down_enter"] - angle) / span * 100.0))


# --- drawing the skeleton + the measured joint ----------------------------

def _draw_skeleton(img, lm: Landmarks, exercise: Exercise, side: str, k: float) -> None:
    if lm is None:
        return
    a, b, c = exercise.sides[side]
    hot = {(a, b), (b, c)}
    for i, j in POSE_EDGES:
        pi, pj = lm.get(i), lm.get(j)
        if not pi or not pj or pi[2] < CONF_MIN or pj[2] < CONF_MIN:
            continue
        highlight = (i, j) in hot or (j, i) in hot
        cv2.line(img, _pt(pi), _pt(pj), C_ARM if highlight else C_BODY,
                 _thick(k, 4 if highlight else 2), cv2.LINE_AA)
    for i in (a, b, c):
        p = lm.get(i)
        if p and p[2] >= CONF_MIN:
            cv2.circle(img, _pt(p), _thick(k, 5), C_JOINT, -1, cv2.LINE_AA)


def _draw_vertex_angle(img, lm: Landmarks, exercise: Exercise, side: str,
                       angle: Optional[float], thr: dict, k: float) -> None:
    """The live angle, printed at the joint it's measured at (elbow or knee) and
    coloured by the same traffic light as the gauge - so the number greens up at the
    exact moment the rep hits full range."""
    if lm is None or angle is None:
        return
    _, b, _ = exercise.sides[side]
    p = lm.get(b)
    if not p or p[2] < CONF_MIN:
        return
    x, y = _pt(p)
    _shadow_text(img, f"{angle:.0f}", (x + int(12 * k), y + int(4 * k)), 0.9 * k, _zone_color(angle, thr), k)


# --- the range-of-motion gauge (traffic light) ----------------------------

def _draw_rom_gauge(img, angle: Optional[float], thr: dict, k: float) -> None:
    """A slim vertical bar on the right: extended at the bottom, deepest at the top,
    with full-range marked. Its background is a red/yellow/green ladder and the fill
    is coloured by the zone the current angle sits in - the thresholds reps.py grades
    on, made visible while you move."""
    h, w = img.shape[:2]
    bw = int(16 * k)
    x = w - int(34 * k)
    top, bot = int(150 * k), h - int(150 * k)
    lo, hi = thr["gauge_deep"], thr["down_enter"]        # top = deepest, bottom = extended

    def y_of(a: float) -> int:
        a = max(lo, min(hi, a))
        return int(top + (a - lo) / (hi - lo) * (bot - top))

    yg, yy = y_of(thr["full_rom"]), y_of(thr["up_enter"])
    cv2.rectangle(img, (x, top), (x + bw, yg), BAND_GREEN, -1)     # full-range zone
    cv2.rectangle(img, (x, yg), (x + bw, yy), BAND_YELLOW, -1)     # working zone
    cv2.rectangle(img, (x, yy), (x + bw, bot), BAND_RED, -1)       # barely-in zone
    cv2.line(img, (x - int(6 * k), yg), (x + bw + int(6 * k), yg), C_GREEN, _thick(k, 2), cv2.LINE_AA)

    if angle is not None:
        yc = y_of(angle)
        col = _zone_color(angle, thr)
        cv2.rectangle(img, (x, yc), (x + bw, bot), col, -1)        # fill up from the bottom
        cv2.circle(img, (x + bw // 2, yc), _thick(k, 7), col, -1, cv2.LINE_AA)

    cx = x + bw // 2
    _center(img, "ROM", cx, top - int(12 * k), 0.5 * k, C_TEXT, k, 1)
    pct = _depth_pct(angle, thr)
    if pct is not None:
        _center(img, f"{pct:.0f}%", cx, bot + int(26 * k), 0.55 * k, _zone_color(angle, thr), k, 1)


# --- the tempo bar --------------------------------------------------------

def _draw_tempo(img, active: Optional[dict], target: float, k: float) -> None:
    """A horizontal bar, bottom-left, that fills with the current rep's elapsed time
    toward the controlled-tempo target (the tick at the far end). Green if that rep
    ends controlled, red if it was rushed - we already graded it, so we can colour it
    honestly as it happens. Idle between reps, it just names the target."""
    h, w = img.shape[:2]
    x0, bw, bh = int(16 * k), int(120 * k), int(12 * k)
    y0 = h - int(30 * k)
    _shadow_text(img, "TEMPO", (x0, y0 - int(8 * k)), 0.5 * k, C_TEXT, k, 1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (55, 55, 55), -1)
    cv2.line(img, (x0 + bw, y0 - int(3 * k)), (x0 + bw, y0 + bh + int(3 * k)), C_DIM, _thick(k, 1), cv2.LINE_AA)

    if active:
        frac = max(0.0, min(1.0, active["elapsed"] / active["target"]))
        col = C_RED if active["rushed"] else C_GREEN
        cv2.rectangle(img, (x0, y0), (x0 + int(bw * frac), y0 + bh), col, -1)
        _shadow_text(img, f"{active['elapsed']:.1f}s", (x0 + bw + int(9 * k), y0 + bh), 0.5 * k, col, k, 1)
    else:
        _shadow_text(img, f"aim {target:.1f}s", (x0 + bw + int(9 * k), y0 + bh), 0.5 * k, C_DIM, k, 1)


# --- the per-rep flash + the running tally --------------------------------

def _flash_for(per_rep: List[dict], t: float, hold: float = 1.6):
    """Just after a rep closes, name it - clean, or the full detailed reason it
    wasn't. This is form_feedback's verdict surfacing at the moment the rep happened,
    the whole message, not a two-word tag."""
    for r in per_rep:
        e = r["end_t"]
        if e <= t <= e + hold:
            if not r["issues"]:
                return f"Rep {r['number']}: clean - full range, controlled.", C_GREEN
            return f"Rep {r['number']}: {r['issues'][0]}", C_RED
    return None


def _tally(per_rep: List[dict], t: float) -> dict:
    done = [r for r in per_rep if r["end_t"] <= t + 1e-9]
    return {
        "reps": len(done),
        "full": sum(1 for r in done if r["full"]),
        "shallow": sum(1 for r in done if "shallow" in r["tags"]),
        "rushed": sum(1 for r in done if "rushed" in r["tags"]),
    }


def _active_tempo(per_rep: List[dict], t: float, tempo_min_s: float) -> Optional[dict]:
    for r in per_rep:
        if r["start_t"] <= t <= r["end_t"]:
            return {"elapsed": t - r["start_t"], "target": tempo_min_s, "rushed": "rushed" in r["tags"]}
    return None


def _draw_flash(img, flash, k: float) -> None:
    text, color = flash
    h, w = img.shape[:2]
    lines = _wrap(text, 0.6 * k, k, int(w * 0.88))
    lh = int(28 * k)
    box_h = lh * len(lines) + int(16 * k)
    y0 = h - int(64 * k) - box_h
    ov = img.copy()
    cv2.rectangle(ov, (int(w * 0.04), y0), (int(w * 0.96), y0 + box_h), C_PANEL, -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    y = y0 + int(26 * k)
    for ln in lines:
        _center(img, ln, w // 2, y, 0.6 * k, color, k)
        y += lh


def _draw_hud(img, exercise: Exercise, reps_done: int, total: int, tally: dict,
              angle: Optional[float], thr: dict, flash, active, k: float) -> None:
    h, w = img.shape[:2]
    bar_h = int(70 * k)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, bar_h), C_PANEL, -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

    r1, r2 = int(bar_h * 0.44), int(bar_h * 0.82)
    _shadow_text(img, f"REP {reps_done}/{total}", (int(14 * k), r1), 0.85 * k, C_TEXT, k)
    _shadow_text(img, exercise.name, (int(14 * k), r2), 0.5 * k, C_DIM, k, 1)

    if angle is not None:
        txt = f"{angle:.0f} {exercise.vertex_name}"
        _shadow_text(img, txt, (w - _text_w(txt, 0.8 * k, k) - int(14 * k), r1), 0.8 * k, _zone_color(angle, thr), k)
    tally_txt = f"{tally['full']} full  {tally['shallow']} short  {tally['rushed']} fast"
    _shadow_text(img, tally_txt, (w - _text_w(tally_txt, 0.5 * k, k, 1) - int(14 * k), r2), 0.5 * k, C_DIM, k, 1)

    _draw_rom_gauge(img, angle, thr, k)
    _draw_tempo(img, active, thr["tempo_min_s"], k)
    if flash:
        _draw_flash(img, flash, k)


# --- the end card: the video, alone, telling the whole story --------------

def _draw_end_card(base: np.ndarray, summary: dict, exercise: Exercise, k: float) -> np.ndarray:
    """A few seconds tacked on the end so someone who watched nothing but the video
    still leaves with the verdict and the one thing to fix. The last frame, dimmed,
    with the numbers on top."""
    img = base.copy()
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)

    cx, y = w // 2, int(h * 0.24)
    _center(img, exercise.name.upper(), cx, y, 0.75 * k, C_DIM, k, 1)
    y += int(58 * k)

    reps, full = summary["reps"], summary["full_reps"]
    if reps == 0:
        _center(img, "No full reps counted", cx, y, 0.95 * k, C_RED, k)
        y += int(54 * k)
    else:
        _center(img, f"{full} / {reps}", cx, y, 2.1 * k, C_ARM, k, 3)
        y += int(50 * k)
        _center(img, "full reps", cx, y, 0.7 * k, C_TEXT, k)
        y += int(46 * k)
        t = _tally(summary["per_rep"], 1e18)
        _center(img, f"{t['full']} full    {t['shallow']} shallow    {t['rushed']} rushed", cx, y, 0.6 * k, C_DIM, k, 1)
        y += int(54 * k)

    c = summary["coaching"]
    _center(img, "FOCUS NEXT", cx, y, 0.55 * k, C_DIM, k, 1)
    y += int(34 * k)
    _center(img, c["focus"], cx, y, 0.9 * k, C_GREEN, k)
    y += int(46 * k)
    if c["next_session"]:
        for ln in _wrap(c["next_session"][0], 0.55 * k, k, int(w * 0.82), max_lines=3):
            _center(img, ln, cx, y, 0.55 * k, C_TEXT, k, 1)
            y += int(28 * k)
    return img


def _open_writer(dst: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """VP8 in a .webm. The one writer this OpenCV opens AND a browser plays."""
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        raise VideoError(
            "Could not open a VP8/webm writer - no usable encoder. See LEARNINGS #12."
        )
    return writer


def annotate_video(src, dst=None, exercise_key: str = "bicep_curl", progress_cb=None) -> dict:
    """
    The whole product in one call: analyse the clip, then paint the analysis back
    onto it. Returns the exact summarize() dict the CLI and endpoint use, plus a
    "video" path to the rendered .webm. progress_cb(stage, pct) is called as it goes
    so the endpoint can show a real progress bar.
    """
    src = str(src)
    exercise = get_exercise(exercise_key)
    samples, lms_px, meta, fps, rot, size, total = _detect_pass(src, exercise, progress_cb)

    side, scores = pick_side(samples)
    raw = [getattr(s, side) for s in samples]
    times = [s.t for s in samples]
    summary = summarize(meta, exercise, side, scores, raw, times)

    angle = summary["series"]["angle"]                 # smoothed, aligned to samples
    per_rep = summary["per_rep"]
    ends = [r["end_t"] for r in per_rep]
    thr = summary["thresholds"]
    total_reps = summary["reps"]

    dst = (Path(dst) if dst else Path("out") / "annotated.webm").with_suffix(".webm")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(dst, fps, size)

    cap, _, _ = _open(Path(src))       # a second, dense decode - every frame this time
    idx = 0
    last_clean = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _prep(frame, rot)
            k = _k(frame)
            t = idx / fps
            last_clean = frame.copy()    # snapshot BEFORE the overlay, for a tidy end card

            j = min(idx // STRIDE, len(samples) - 1)     # the sample at or before this frame
            f = (idx % STRIDE) / STRIDE                  # ...and how far toward the next
            lm = _lerp_landmarks(lms_px[j], lms_px[j + 1] if j + 1 < len(lms_px) else None, f)
            live = _lerp_angle(angle, j, f)

            _draw_skeleton(frame, lm, exercise, side, k)
            _draw_vertex_angle(frame, lm, exercise, side, live, thr, k)
            reps_done = sum(1 for e in ends if e <= t + 1e-9)
            _draw_hud(frame, exercise, reps_done, total_reps, _tally(per_rep, t), live, thr,
                      _flash_for(per_rep, t), _active_tempo(per_rep, t, thr["tempo_min_s"]), k)

            writer.write(frame)
            idx += 1
            if progress_cb and idx % 10 == 0:
                progress_cb("Generating", round(40 + 58 * min(idx, total) / total) if total else None)
    finally:
        # The end card: hold a summary for a couple of seconds so the video stands
        # on its own. Drawn on a CLEAN last frame (no HUD bleeding through) and from
        # the same summary, so it can't disagree with the reps it just counted.
        if last_clean is not None:
            if progress_cb:
                progress_cb("Finishing up", 99)
            card = _draw_end_card(last_clean, summary, exercise, _k(last_clean))
            for _ in range(int(fps * 2.5)):
                writer.write(card)
        cap.release()
        writer.release()

    return {**summary, "video": str(dst)}


def main(path: str) -> None:
    summary = annotate_video(path, exercise_key=os.environ.get("EXERCISE", "bicep_curl"))
    _print_summary(summary)
    print(f"\nwrote {summary['video']}  <- OPEN IT. Skeleton, gauge, tempo, tally, end card.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python render.py path/to/video.mp4")
    try:
        main(sys.argv[1])
    except VideoError as e:
        sys.exit(f"VideoError: {e}")
