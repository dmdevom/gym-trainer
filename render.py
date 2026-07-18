"""
Phase 3, the money shot: draw the skeleton, the live elbow angle and a running
rep counter onto every frame, then write a video a browser will actually play.

Two decisions here cost real debugging time, so they get real comments:

  1. CODEC. This OpenCV cannot encode H.264 - there is no software x264 and the
     hardware v4l2m2m path isn't present - and a browser <video> refuses the
     mp4v it CAN make. VP8 inside a .webm is the way out: OpenCV writes it, every
     browser plays it, and it needs no extra dependency. See LEARNINGS #12.

  2. ONE detection pass, then held. We detect at the SAME stride analyze.py uses,
     so the rep count burned into the video equals the count in the JSON - always,
     by construction, not by luck. Then we decode every frame and draw a skeleton
     interpolated between the two nearest samples. The numbers come from the
     sparse pass; only the picture is dense. Detecting every frame would be a
     third more model calls to move a wrist a few pixels.

Usage:
    python render.py data/videos/curl_right.mp4          # -> out/annotated.webm
    ROTATE_DEG=270 python render.py data/videos/curl_right.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from analyze import _print_summary, summarize
from video import (
    ARMS,
    CONF_MIN,
    MODEL_PATH,
    STRIDE,
    Sample,
    VideoError,
    _ROT,
    _open,
    _to_sample,
    pick_arm,
)

# Cap the LONGER side. A smaller frame detects faster, encodes smaller, and still
# reads fine - the plan's "downscale before inference" made concrete. Capping the
# long side (not the width) shrinks portrait phone clips too, where the width was
# never the problem. Landmarks are stored in THIS space, so drawing lines up free.
MAX_SIDE = 960

# The overlay is authored at MAX_SIDE and scaled by k = long_side / MAX_SIDE, so a
# 540px clip and a 960px clip get text and lines in the same visual proportion.
REF = float(MAX_SIDE)

# BlazePose-33 skeleton. MediaPipe deleted POSE_CONNECTIONS along with the legacy
# API (LEARNINGS #6), so the edge list is ours to own now. Face and fingers are
# left off deliberately - detail the overlay doesn't need.
POSE_EDGES = [
    (11, 12),                                          # shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),            # arms
    (11, 23), (12, 24), (23, 24),                      # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # right leg
]

# BGR. The tracked arm is drawn in the same orange as video.py's smoothed line,
# so the wave on the chart and the limb on the video read as the same object.
C_BODY = (180, 140, 90)     # muted steel-blue for the rest of the skeleton
C_ARM = (0, 165, 255)       # orange - the arm actually being measured
C_JOINT = (235, 235, 235)
C_TEXT = (255, 255, 255)
C_OK = (90, 210, 90)
C_BAD = (70, 90, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

Landmarks = Optional[Dict[int, Tuple[float, float, float]]]


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
    None when no pose was found. Same normalized->pixel multiply the angle
    depends on - forget it and the skeleton draws itself onto the top-left corner."""
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    return {i: (lm.x * w, lm.y * h, lm.visibility or 0.0) for i, lm in enumerate(lms)}


def _detect_pass(path: str):
    """
    Sample at STRIDE, exactly like analyze.py's signal, but keep the landmark
    pixels too - analyze.py throws them away and the overlay can't. Returns the
    samples (for the counting), the aligned landmark maps (for the drawing), the
    meta, the fps/rotation, and the output frame size the writer must match.
    """
    cap, rot, fps = _open(Path(path))
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

            samples.append(_to_sample(result, idx, idx / fps, out_w, out_h))
            lms_px.append(_landmarks_px(result, out_w, out_h))
            idx += 1
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
    return samples, lms_px, meta, fps, rot, (out_w, out_h)


# --- interpolation: sparse samples, smooth picture ------------------------

def _lerp_landmarks(a: Landmarks, b: Landmarks, f: float) -> Landmarks:
    """Blend two landmark maps. A joint is blended only where BOTH samples saw it;
    if one sample lost the arm, holding is honest and teleporting across the gap
    is not - so we fall back to whichever end actually has it."""
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


# --- drawing --------------------------------------------------------------

def _pt(p) -> Tuple[int, int]:
    return int(p[0]), int(p[1])


def _draw_skeleton(img: np.ndarray, lm: Landmarks, arm: str, k: float) -> None:
    if lm is None:
        return
    s, e, w = ARMS[arm]
    hot = {(s, e), (e, w)}
    for i, j in POSE_EDGES:
        pi, pj = lm.get(i), lm.get(j)
        if not pi or not pj or pi[2] < CONF_MIN or pj[2] < CONF_MIN:
            continue
        highlight = (i, j) in hot or (j, i) in hot
        cv2.line(img, _pt(pi), _pt(pj), C_ARM if highlight else C_BODY,
                 _thick(k, 4 if highlight else 2), cv2.LINE_AA)
    for i in (s, e, w):
        p = lm.get(i)
        if p and p[2] >= CONF_MIN:
            cv2.circle(img, _pt(p), _thick(k, 5), C_JOINT, -1, cv2.LINE_AA)


def _draw_elbow_angle(img: np.ndarray, lm: Landmarks, arm: str, angle: Optional[float], k: float) -> None:
    if lm is None or angle is None:
        return
    _, e, _ = ARMS[arm]
    p = lm.get(e)
    if not p or p[2] < CONF_MIN:
        return
    x, y = _pt(p)
    label = f"{angle:.0f}"
    pos = (x + int(12 * k), y + int(4 * k))
    cv2.putText(img, label, pos, FONT, 0.9 * k, (0, 0, 0), _thick(k, 5), cv2.LINE_AA)
    cv2.putText(img, label, pos, FONT, 0.9 * k, C_ARM, _thick(k, 2), cv2.LINE_AA)


def _text_right(img: np.ndarray, text: str, right: int, y: int, scale: float, color, k: float) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, _thick(k, 2))
    cv2.putText(img, text, (right - tw, y), FONT, scale, color, _thick(k, 2), cv2.LINE_AA)


def _draw_depth_gauge(img: np.ndarray, angle: Optional[float], thr: dict, k: float, deep: float = 30.0) -> None:
    """A slim bar on the right: straight arm at the bottom, deep curl at the top,
    the FULL_ROM line marked. It fills orange once the rep passes full range - the
    thresholds reps.py judges on, made visible while you lift."""
    if angle is None:
        return
    h, w = img.shape[:2]
    bw = int(13 * k)
    x, top, bot = w - int(30 * k), int(88 * k), h - int(88 * k)
    lo, hi = deep, thr["down_enter"]

    def y_of(a: float) -> int:
        a = max(lo, min(hi, a))
        return int(top + (a - lo) / (hi - lo) * (bot - top))

    cv2.rectangle(img, (x, top), (x + bw, bot), (60, 60, 60), -1)
    yf = y_of(thr["full_rom"])
    cv2.line(img, (x - int(5 * k), yf), (x + bw + int(5 * k), yf), C_OK, _thick(k, 2), cv2.LINE_AA)
    yc = y_of(angle)
    good = angle <= thr["full_rom"]
    cv2.rectangle(img, (x, yc), (x + bw, bot), C_ARM if good else (120, 120, 120), -1)
    cv2.circle(img, (x + bw // 2, yc), _thick(k, 7), C_ARM if good else C_JOINT, -1, cv2.LINE_AA)


def _flash_for(ends: List[float], by_end: Dict[float, dict], t: float, hold: float = 0.9):
    """Just after a rep closes, name it: clean, or exactly what was wrong. This is
    form_feedback's verdict surfacing at the moment the rep happened."""
    for e in ends:
        if e <= t <= e + hold:
            r = by_end[e]
            if not r["issues"]:
                return f"rep {r['number']}: clean", C_OK
            partial = any("partial" in i for i in r["issues"])
            fast = any("too fast" in i for i in r["issues"])
            tag = " + ".join(t for t, on in (("partial", partial), ("fast", fast)) if on)
            return f"rep {r['number']}: {tag}", C_BAD
    return None


def _draw_hud(img, reps_done, total, angle, flash, thr, k) -> None:
    h, w = img.shape[:2]
    bar_h = int(56 * k)
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, bar_h), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)

    base = int(bar_h * 0.68)
    cv2.putText(img, f"REP {reps_done}/{total}", (int(14 * k), base), FONT, 1.0 * k, C_TEXT, _thick(k, 2), cv2.LINE_AA)
    if angle is not None:
        _text_right(img, f"{angle:.0f} deg", w - int(46 * k), base, 0.9 * k, C_TEXT, k)

    _draw_depth_gauge(img, angle, thr, k)

    if flash:
        text, color = flash
        (tw, _), _ = cv2.getTextSize(text, FONT, 1.0 * k, _thick(k, 3))
        x, y = (w - tw) // 2, h - int(32 * k)
        cv2.putText(img, text, (x, y), FONT, 1.0 * k, (0, 0, 0), _thick(k, 6), cv2.LINE_AA)
        cv2.putText(img, text, (x, y), FONT, 1.0 * k, color, _thick(k, 3), cv2.LINE_AA)


def _open_writer(dst: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """VP8 in a .webm. The one writer this OpenCV opens AND a browser plays."""
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        raise VideoError(
            "Could not open a VP8/webm writer - no usable encoder. See LEARNINGS #12."
        )
    return writer


def annotate_video(src, dst=None) -> dict:
    """
    The whole product in one call: analyse the clip, then paint the analysis back
    onto it. Returns the exact summarize() dict the CLI and endpoint use, plus a
    "video" path to the rendered .webm.
    """
    src = str(src)
    samples, lms_px, meta, fps, rot, size = _detect_pass(src)

    arm, scores = pick_arm(samples)
    raw = [getattr(s, arm) for s in samples]
    times = [s.t for s in samples]
    summary = summarize(meta, arm, scores, raw, times)

    angle = summary["series"]["angle"]                 # smoothed, aligned to samples
    ends = [r["end_t"] for r in summary["per_rep"]]
    by_end = {r["end_t"]: r for r in summary["per_rep"]}
    total = summary["reps"]

    dst = (Path(dst) if dst else Path("out") / "annotated.webm").with_suffix(".webm")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(dst, fps, size)

    cap, _, _ = _open(Path(src))       # a second, dense decode - every frame this time
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _prep(frame, rot)
            k = _k(frame)
            t = idx / fps

            j = min(idx // STRIDE, len(samples) - 1)     # the sample at or before this frame
            f = (idx % STRIDE) / STRIDE                  # ...and how far toward the next
            lm = _lerp_landmarks(lms_px[j], lms_px[j + 1] if j + 1 < len(lms_px) else None, f)
            live = _lerp_angle(angle, j, f)

            _draw_skeleton(frame, lm, arm, k)
            _draw_elbow_angle(frame, lm, arm, live, k)
            reps_done = sum(1 for e in ends if e <= t + 1e-9)
            _draw_hud(frame, reps_done, total, live, _flash_for(ends, by_end, t), summary["thresholds"], k)

            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()

    return {**summary, "video": str(dst)}


def main(path: str) -> None:
    summary = annotate_video(path)
    _print_summary(summary)
    print(f"\nwrote {summary['video']}  <- OPEN IT. Skeleton, live angle, rep count.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python render.py path/to/video.mp4")
    try:
        main(sys.argv[1])
    except VideoError as e:
        sys.exit(f"VideoError: {e}")
