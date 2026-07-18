"""
Phase 2, step 0. Run this on every video BEFORE you trust a single frame.

It answers three questions that will otherwise cost you an hour each:

  1. Can OpenCV decode this codec at all, or does read() just quietly say False?
  2. Is the person upright, or is OpenCV handing MediaPipe a person lying down?
  3. How many milliseconds does one MediaPipe inference actually cost here?

Usage:  python scripts/probe_video.py data/videos/curls_good.mp4
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path("models/pose_landmarker_lite.task")
TIMING_FRAMES = 40


def _prop(cap, name, default=0.0):
    """CAP_PROP_ORIENTATION_* only exist on newer OpenCV. Don't AttributeError."""
    flag = getattr(cv2, name, None)
    if flag is None:
        return None
    try:
        return cap.get(flag)
    except cv2.error:
        return None


def probe(path: str) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"No such file: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        sys.exit("isOpened() is False. OpenCV cannot even open the container.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dec_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    dec_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    meta = _prop(cap, "CAP_PROP_ORIENTATION_META")
    auto = _prop(cap, "CAP_PROP_ORIENTATION_AUTO")

    print(f"file             : {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"codec (FOURCC)   : {codec!r}")
    print(f"header says w x h: {dec_w:.0f} x {dec_h:.0f}")
    print(f"fps              : {fps:.2f}")
    if fps > 0:
        print(f"frames           : {n:.0f}   (~{n / fps:.1f} s)")
    print(f"ORIENTATION_META : {meta}   <- rotation the file asks for")
    print(f"ORIENTATION_AUTO : {auto}   <- 1.0 means OpenCV applies it silently")

    # ---- trap 1: the codec -------------------------------------------------
    ok, frame = cap.read()
    if not ok:
        print("\n*** read() returned False on frame 0, but isOpened() was True.")
        print("*** That combination means CODEC, not path. Almost certainly HEVC/H.265.")
        print("*** This is the silent one - a loop over a dead capture just does nothing.")
        print(f"***   ffmpeg -i {p} -c:v libx264 -crf 20 -an {p.with_name(p.stem + '_h264.mp4')}")
        cap.release()
        return
    print("\ncodec check      : OK, frame 0 decoded")

    h, w = frame.shape[:2]
    print(f"read() gave you  : {w} x {h}")
    if (round(dec_w), round(dec_h)) != (w, h):
        print("  ^ header and reality disagree. OpenCV rotated it for you.")

    # ---- trap 2: rotation --------------------------------------------------
    # I am not going to tell you which way to rotate. The sign convention on
    # MP4 display-matrix metadata is a swamp, ffmpeg and OpenCV have disagreed
    # about it across versions, and reasoning about it is how you lose an hour.
    # Look at the pictures instead. Phase 0's lesson was "look at the output" -
    # this is the same lesson wearing a different hat.
    out = Path("out")
    out.mkdir(exist_ok=True)
    variants = {
        0: frame,
        90: cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(frame, cv2.ROTATE_180),
        270: cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    for deg, img in variants.items():
        cv2.imwrite(str(out / f"frame0_rot{deg:03d}.jpg"), img)

    print(f"\nrotation check   : wrote 4 images to {out.resolve()}/")
    print("  Open them. Exactly one has a person standing up in it.")
    print("  Then run video.py with ROTATE_DEG=<that number>.")
    print("  Do this once per source device, not once per video.")

    # ---- trap 3: how slow is this actually ---------------------------------
    if not MODEL_PATH.exists():
        print(f"\ntiming           : skipped, no model at {MODEL_PATH}")
        cap.release()
        return

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    times, detected = [], 0
    with mp_vision.PoseLandmarker.create_from_options(options) as lm:
        for i in range(TIMING_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * 1000 / fps) if fps > 0 else i * 33
            t0 = time.perf_counter()
            res = lm.detect_for_video(image, ts)
            times.append((time.perf_counter() - t0) * 1000)
            if res.pose_landmarks:
                detected += 1

    cap.release()
    if not times:
        return

    med = float(np.median(times))
    print(f"\nMediaPipe timing : first frame {times[0]:.0f} ms, median {med:.0f} ms")
    print("  (first frame is slow: full-image detect. After that it tracks a crop.)")
    print(f"  300-frame request projects to ~{med * 300 / 1000:.1f} s of pure inference.")
    print(f"  YOLO measured 133 ms/frame on this machine -> ~40 s. That is your README table.")
    print(f"\nperson detected  : {detected}/{len(times)} frames")
    if detected < len(times) * 0.8:
        print("  ^ LOW. Almost always rotation. Fix ROTATE_DEG before you debug anything else.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/probe_video.py path/to/video.mp4")
    probe(sys.argv[1])