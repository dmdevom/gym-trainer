"""
Phase 2 afternoon: the YOLO-vs-MediaPipe question, answered on VIDEO.

LEARNINGS.md #5 and #9 compared the two on a single photo and on import cost.
This runs them frame-for-frame on the same clip and asks the two questions that
actually pick the backend that ships:

  1. TIME  : ms/frame for each, here, on this machine, ON CPU - because the
             deploy target is a 2-vCPU box with no GPU. Multiply by ~300 frames
             per request and you have the number that decides whether the
             endpoint returns before the user gives up.
  2. AGREE : do the two elbow-angle series tell the same story? Where they
             diverge is where at least one model is guessing, and you cannot
             tell which from a single series - only from the pair.

Both models are pinned to CPU on purpose. YOLO would otherwise grab the GPU that
this laptop happens to have and the deploy box does not, and the timing would
flatter it into a choice that dies on Hugging Face.

Usage:
    ROTATE_DEG=270 python scripts/compare_backends.py data/videos/curl_right.mp4

Writes out/compare_backends.png (both series overlaid) and prints the timing +
agreement summary. Needs the dev deps (ultralytics/torch) - this never ships.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# scripts/ is not the repo root; put the root on the path so `analyzer`/`video`
# import the same way they do everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from analyzer import calc_angle
from video import CONF_MIN, MODEL_PATH, STRIDE, _ROT, _open

# COCO-17 (YOLO) indices per arm: shoulder, elbow, wrist.
YOLO_ARMS = {"left": (5, 7, 9), "right": (6, 8, 10)}
# BlazePose-33 (MediaPipe) indices for the same three joints.
MP_ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}


def _mp_angle(result, side, w, h):
    """MediaPipe -> elbow angle in PIXELS, or None below the confidence floor."""
    if not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    s, e, wr = MP_ARMS[side]
    if min(lms[s].visibility, lms[e].visibility, lms[wr].visibility) < CONF_MIN:
        return None
    pts = [np.array([lms[i].x * w, lms[i].y * h]) for i in (s, e, wr)]
    return float(calc_angle(*pts))


def _yolo_angle(result, side):
    """YOLO -> elbow angle. Its keypoints are already pixels; nothing to scale."""
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None
    xy = result.keypoints.xy[0].tolist()
    conf = result.keypoints.conf[0].tolist()
    s, e, wr = YOLO_ARMS[side]
    if min(conf[s], conf[e], conf[wr]) < CONF_MIN:
        return None
    pts = [np.array(xy[i]) for i in (s, e, wr)]
    return float(calc_angle(*pts))


def compare(path: str) -> None:
    from ultralytics import YOLO

    cap, rot, fps = _open(Path(path))
    yolo = YOLO("yolov8n-pose.pt")

    mp_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp_tasks.BaseOptions.Delegate.CPU,   # deploy has no GPU
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=CONF_MIN,
        min_tracking_confidence=CONF_MIN,
    )

    t_series, mp_ms, yolo_ms = [], [], []
    mp_series = {"left": [], "right": []}
    yolo_series = {"left": [], "right": []}

    idx = 0
    with mp_vision.PoseLandmarker.create_from_options(mp_opts) as landmarker:
        while True:
            if not cap.grab():
                break
            if idx % STRIDE:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if rot in _ROT:
                frame = cv2.rotate(frame, _ROT[rot])
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_series.append(idx / fps)

            # MediaPipe: VIDEO mode, RGB, needs a strictly increasing timestamp.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            mp_res = landmarker.detect_for_video(image, int(idx * 1000 / fps))
            mp_ms.append((time.perf_counter() - t0) * 1000)

            # YOLO: per-frame, BGR array (ultralytics assumes cv2 order), CPU.
            t0 = time.perf_counter()
            yolo_res = yolo(frame, verbose=False, device="cpu")[0]
            yolo_ms.append((time.perf_counter() - t0) * 1000)

            for side in ("left", "right"):
                mp_series[side].append(_mp_angle(mp_res, side, w, h))
                yolo_series[side].append(_yolo_angle(yolo_res, side))
            idx += 1
    cap.release()

    # Judge on the arm the product would track: the one MediaPipe saw best.
    arm = max(("left", "right"), key=lambda s: sum(v is not None for v in mp_series[s]))
    mp_a, yolo_a = mp_series[arm], yolo_series[arm]

    both = [(m, y) for m, y in zip(mp_a, yolo_a) if m is not None and y is not None]
    gaps = [abs(m - y) for m, y in both]
    n = len(t_series)

    print(f"\nframes compared  : {n}   (arm: {arm}; both pinned to CPU = the deploy target)")
    print(f"MediaPipe        : {np.median(mp_ms):5.0f} ms/frame median  ->  ~{np.median(mp_ms) * 300 / 1000:.1f}s per 300-frame request")
    print(f"YOLOv8n-pose     : {np.median(yolo_ms):5.0f} ms/frame median  ->  ~{np.median(yolo_ms) * 300 / 1000:.1f}s per 300-frame request")
    print(f"first frame      : MediaPipe {mp_ms[0]:.0f} ms   YOLO {yolo_ms[0]:.0f} ms   (both pay a one-off warmup)")
    print(f"arm seen (>={CONF_MIN}) : MediaPipe {sum(v is not None for v in mp_a)}/{n}   YOLO {sum(v is not None for v in yolo_a)}/{n}")
    if gaps:
        print(f"angle agreement  : median {np.median(gaps):.1f}deg   max {max(gaps):.1f}deg   over {len(gaps)} shared frames")

    _plot(t_series, mp_a, yolo_a, arm, Path(path).name, fps)


def _plot(t, mp_a, yolo_a, arm, fname, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nan = lambda xs: [np.nan if v is None else v for v in xs]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, nan(mp_a), lw=2, color="tab:orange", label="MediaPipe")
    ax.plot(t, nan(yolo_a), lw=2, color="tab:purple", label="YOLOv8n-pose")
    ax.set_xlabel("seconds")
    ax.set_ylabel("elbow angle (deg)")
    ax.set_title(f"{fname}  |  {arm} arm  |  MediaPipe vs YOLO  (CPU)")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = Path("out")
    out.mkdir(exist_ok=True)
    dest = out / "compare_backends.png"
    fig.savefig(dest, dpi=110)
    print(f"\nwrote {dest}  <- overlay both lines. Where they part is where one guessed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/compare_backends.py path/to/video.mp4")
    compare(sys.argv[1])
