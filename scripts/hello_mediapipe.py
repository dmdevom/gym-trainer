"""
Phase 0, Script 2 — rewritten for the MediaPipe Tasks API.

WHY THIS LOOKS NOTHING LIKE THE TUTORIALS ONLINE:
MediaPipe deleted the legacy `mp.solutions.*` namespace in version 0.10.31.
`mp.solutions.pose`, `mp.solutions.drawing_utils`, `POSE_CONNECTIONS` — all gone,
not renamed. The supported replacement is the Tasks API, which needs an
explicit .task model file instead of a `model_complexity` argument.

ONE-TIME SETUP:
    mkdir -p models
    wget -O models/pose_landmarker_heavy.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

    (heavy = most accurate, slowest. Fine for one photo. When you hit video in
     Phase 2, swap BOTH "heavy" -> "lite" in that URL for a much faster model.
     Choosing the model file IS the new model_complexity knob.)

RUN:
    python hello_mediapipe.py path/to/photo.jpg [path/to/model.task]

GOAL — compare against YOLO. Three things to internalise:
  1. 33 landmarks here vs 17 for YOLO.
  2. Coordinates are NORMALISED (0.0-1.0), not pixels. You multiply by
     width/height yourself. Mixing this up with YOLO's pixel output silently
     produces plausible-but-wrong angles. No exception, no crash. Just wrong.
  3. MediaPipe reports visibility AND presence; YOLO reports one confidence.
     Related ideas, different scales. Don't reuse thresholds across them.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

DEFAULT_MODEL = "models/pose_landmarker_heavy.task"

# The landmarks a curl analyser actually cares about.
# Note the index shift vs YOLO: left_elbow is 13 here, but 7 in YOLO.
INTERESTING = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",  # <- YOLO calls this 7. Same joint, different index.
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

# mp.solutions.pose.POSE_CONNECTIONS is gone too, so here it is as plain data.
# Drawing the skeleton by hand is more instructive than calling draw_landmarks anyway.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),            # face: nose -> left eye -> left ear
    (0, 4), (4, 5), (5, 6), (6, 8),            # face: nose -> right eye -> right ear
    (9, 10),                                   # mouth
    (11, 12),                                  # shoulders
    (11, 13), (13, 15),                        # left arm  <- the curl lives here
    (15, 17), (15, 19), (15, 21), (17, 19),    # left hand
    (12, 14), (14, 16),                        # right arm
    (16, 18), (16, 20), (16, 22), (18, 20),    # right hand
    (11, 23), (12, 24), (23, 24),              # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # right leg
]


def fmt(value) -> str:
    """Tasks API may return None for visibility/presence. Don't crash on it."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "  --"


def main(image_path: str, model_path: str) -> None:
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}\n")
        print("Download it first:")
        print("  mkdir -p models")
        print("  wget -O models/pose_landmarker_heavy.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/pose_landmarker"
              "/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        sys.exit(1)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    h, w = image_bgr.shape[:2]
    print(f"\nImage size: {w} x {h} px")

    # OpenCV loads BGR. MediaPipe wants RGB. Skip this and you get quietly
    # worse results, not an error — the model just sees a blue-tinted human.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("No person detected.")
        print("Note: num_poses=1 here. YOLO found however many people were there,")
        print("with no configuration at all. That difference matters for a gym app.")
        return

    lms = result.pose_landmarks[0]
    print(f"Total landmarks: {len(lms)}   (YOLO gave you 17)\n")

    print(f"{'idx':<4} {'name':<16} {'x_norm':>7} {'y_norm':>7} "
          f"{'x_px':>8} {'y_px':>8} {'vis':>6} {'pres':>6}")
    for i, name in INTERESTING.items():
        lm = lms[i]
        print(f"{i:<4} {name:<16} {lm.x:>7.3f} {lm.y:>7.3f} "
              f"{lm.x * w:>8.1f} {lm.y * h:>8.1f} "
              f"{fmt(lm.visibility):>6} {fmt(lm.presence):>6}")

    # Draw it ourselves — this is the part that used to be draw_landmarks().
    annotated = image_bgr.copy()
    px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for a, b in POSE_CONNECTIONS:
        cv2.line(annotated, px[a], px[b], (200, 200, 200), 2)
    for i, point in enumerate(px):
        colour = (0, 0, 255) if i in INTERESTING else (0, 200, 0)
        cv2.circle(annotated, point, 4, colour, -1)

    out = Path("out/out_mediapipe.jpg")
    cv2.imwrite(str(out), annotated)

    print(f"\nAnnotated image saved to: {out.resolve()}")
    print("Put out_yolo.jpg and out_mediapipe.jpg side by side.")
    print("Which one placed your wrist more accurately? That answer is a README section.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hello_mediapipe.py path/to/photo.jpg [path/to/model.task]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL)