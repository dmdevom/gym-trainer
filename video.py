"""
Phase 2 core: a video file -> a per-frame elbow angle series -> a plot.

This is plumbing. Read it, don't write it - reps.py is where your hour goes.
But there are four decisions baked in here that are worth understanding,
because each one is a bug you would otherwise have to find by hand:

  1. Rotation is applied by US, explicitly, never by OpenCV's auto-rotate.
  2. Angles are computed in PIXELS, never in normalized coordinates.
  3. MediaPipe runs in VIDEO mode, not IMAGE mode. Different object, real
     timestamps, carries tracking state between frames.
  4. The tracked arm is chosen ONCE per video, from the whole video.

Usage:
    ROTATE_DEG=270 python video.py data/videos/curls_good.mp4

Writes angle_series.png (look at it) and angle_series.json (feed it to reps.py).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from analyzer import calc_angle  # Phase 1's trig, reused unchanged. That was the point.

MODEL_PATH = Path(os.environ.get("POSE_MODEL", "models/pose_landmarker_lite.task"))
CONF_MIN = 0.5      # the threshold you validated in Phase 0: real joints 0.81-1.00, fake limb 0.12
STRIDE = 3          # every 3rd frame. 30 fps -> 10 Hz -> ~25 samples per rep. Plenty.
SMOOTH_K = 5        # median window: 5 samples at 10 Hz = 0.5 s. A curl is 2-3 s, so this
                    # kills spikes without flattening the peaks you need to threshold on.

# Set from probe_video.py's four pictures. None = trust the file's metadata.
ROTATE_DEG = int(os.environ["ROTATE_DEG"]) if "ROTATE_DEG" in os.environ else None

# BlazePose 33-landmark indices. "left" is the person's anatomical left.
ARMS = {
    "left":  (11, 13, 15),   # shoulder, elbow, wrist
    "right": (12, 14, 16),
}


class VideoError(ValueError):
    """Something wrong with the file itself. Becomes a 4xx, never a stack trace."""


@dataclass
class Sample:
    idx: int
    t: float                       # seconds from start
    left: Optional[float]          # elbow angle, degrees, None = not confidently seen
    right: Optional[float]
    left_vis: float
    right_vis: float


def _open(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not open this video.")

    meta_flag = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    auto_flag = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)

    rot = int(cap.get(meta_flag) or 0) % 360 if meta_flag is not None else 0
    if auto_flag is not None:
        # Rotate it ourselves. OpenCV's auto-rotate depends on how its FFmpeg was
        # built, so your Pop OS laptop and the HF Space container are allowed to
        # disagree - and the failure is silent: MediaPipe just stops finding a
        # person, and you go debugging calc_angle for an hour. Explicit is cheap.
        cap.set(auto_flag, 0)
    if ROTATE_DEG is not None:
        rot = ROTATE_DEG % 360

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # 0.0 or NaN
        raise VideoError("Could not read a frame rate from this video.")
    return cap, rot, fps


_ROT = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def extract_series(path, stride: int = STRIDE) -> tuple[List[Sample], dict]:
    path = Path(path)
    cap, rot, fps = _open(path)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        # VIDEO, not IMAGE. This is a different object from Phase 1: it keeps
        # tracking state across calls, which is why one landmarker must handle
        # one video start-to-finish, and why timestamps must strictly increase.
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=CONF_MIN,
        min_tracking_confidence=CONF_MIN,
    )

    samples: List[Sample] = []
    idx = 0
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok = cap.grab()          # grab() decodes without converting - skipping is free
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break

            if rot in _ROT:
                frame = cv2.rotate(frame, _ROT[rot])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Real timestamp, not a counter. Free to compute, and it makes the
            # x-axis of your plot the actual clock, which is what you want anyway.
            ts_ms = int(idx * 1000 / fps)
            result = landmarker.detect_for_video(image, ts_ms)

            samples.append(_to_sample(result, idx, idx / fps, w, h))
            idx += 1

    cap.release()
    if not samples:
        raise VideoError("No frames decoded. Run scripts/probe_video.py on this file.")

    meta = {
        "file": path.name,
        "fps": round(fps, 2),
        "rotation_applied": rot,
        "stride": stride,
        "sample_hz": round(fps / stride, 2),
        "frames_sampled": len(samples),
    }
    return samples, meta


def _to_sample(result, idx: int, t: float, w: int, h: int) -> Sample:
    if not result.pose_landmarks:
        return Sample(idx, t, None, None, 0.0, 0.0)
    lms = result.pose_landmarks[0]

    out = {}
    for side, (s, e, wr) in ARMS.items():
        vis = min(lms[s].visibility, lms[e].visibility, lms[wr].visibility)
        if vis < CONF_MIN:
            out[side] = (None, vis)
            continue
        # PIXELS, not normalized coords. MediaPipe divides x by width and y by
        # height SEPARATELY, so on a 1080x1920 frame normalized space is
        # squashed 1.78x on one axis and every angle in it is wrong. calc_angle
        # is scale-invariant; it is not aspect-ratio-invariant. Nothing errors -
        # you just get plausible garbage. Multiply back before you measure.
        pts = [np.array([lms[i].x * w, lms[i].y * h]) for i in (s, e, wr)]
        out[side] = (float(calc_angle(*pts)), vis)

    return Sample(idx, t, out["left"][0], out["right"][0], out["left"][1], out["right"][1])


def pick_arm(samples: List[Sample]) -> tuple[str, dict]:
    """
    Decide once, with the whole video in hand - not per frame.

    You shoot side-on, so one arm is occluded in every frame BY DESIGN. Choosing
    per frame means a noisy frame flips you to the hidden arm mid-rep, your angle
    series jumps 100 degrees, and the state machine counts a rep that never
    happened. One decision, held for the whole video, cannot do that.
    """
    scores = {
        side: float(np.mean([getattr(s, f"{side}_vis") for s in samples]))
        for side in ARMS
    }
    return max(scores, key=scores.__getitem__), scores


def median_smooth(values: List[Optional[float]], k: int = SMOOTH_K) -> List[Optional[float]]:
    """
    Median, not mean. One garbage frame drags a 5-mean by 20+ degrees; a 5-median
    ignores up to two outliers completely. Side effect worth knowing: a lone
    dropout gets filled in from its neighbours, while a long dropout stays None.
    That is the behaviour you want, and you get it for free.
    """
    half, out = k // 2, []
    for i in range(len(values)):
        window = [v for v in values[max(0, i - half): i + half + 1] if v is not None]
        out.append(float(np.median(window)) if window else None)
    return out


def _plot(samples, raw, smoothed, arm, meta, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reps import UP_ENTER, DOWN_ENTER, FULL_ROM

    nan = lambda xs: [np.nan if v is None else v for v in xs]
    t = [s.t for s in samples]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhspan(UP_ENTER, DOWN_ENTER, color="grey", alpha=0.10)
    ax.axhline(UP_ENTER, ls="--", lw=1, color="tab:blue", label=f"UP_ENTER {UP_ENTER:.0f}")
    ax.axhline(DOWN_ENTER, ls="--", lw=1, color="tab:red", label=f"DOWN_ENTER {DOWN_ENTER:.0f}")
    ax.axhline(FULL_ROM, ls=":", lw=1, color="tab:green", label=f"FULL_ROM {FULL_ROM:.0f}")
    ax.plot(t, nan(raw), lw=1, alpha=0.35, color="k", label="raw")
    ax.plot(t, nan(smoothed), lw=2, color="tab:orange", label=f"median({SMOOTH_K})")

    ax.set_xlabel("seconds")
    ax.set_ylabel("elbow angle (deg)")
    ax.set_title(f"{meta['file']}  |  {arm} arm  |  {meta['sample_hz']} Hz  |  rot {meta['rotation_applied']}")
    ax.invert_yaxis()   # curl = up on the plot. Matches what your body did.
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    print(f"\nwrote {dest}  <- OPEN IT. This is the gate.")


def main(path: str):
    samples, meta = extract_series(path)
    arm, scores = pick_arm(samples)
    meta["arm"] = arm
    meta["arm_visibility"] = {k: round(v, 3) for k, v in scores.items()}

    raw = [getattr(s, arm) for s in samples]
    smoothed = median_smooth(raw)
    seen = sum(v is not None for v in raw)

    print(json.dumps(meta, indent=2))
    print(f"\narm visibility   : left {scores['left']:.2f}   right {scores['right']:.2f}   -> tracking {arm}")
    print(f"usable frames    : {seen}/{len(raw)}  ({seen / len(raw):.0%})")
    if seen:
        vals = [v for v in raw if v is not None]
        print(f"angle range      : {min(vals):.1f} to {max(vals):.1f} deg")
        print("\n  ^ THAT RANGE IS YOUR THRESHOLDS. Not mine. Read them off the plot.")

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    (out_dir / "angle_series.json").write_text(json.dumps({
        "meta": meta,
        "t": [round(s.t, 3) for s in samples],
        "raw": raw,
        "smoothed": smoothed,
    }, indent=2))
    _plot(samples, raw, smoothed, arm, meta, out_dir / "angle_series.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python video.py path/to/video.mp4")
    try:
        main(sys.argv[1])
    except VideoError as e:
        sys.exit(f"VideoError: {e}")