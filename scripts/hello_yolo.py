"""
Phase 0, Script 1: First contact with YOLOv8-pose.

Run:   python hello_yolo.py path/to/photo.jpg
Goal:  See the raw keypoints array with your own eyes, then open the annotated
       image and confirm the numbers actually match the picture.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO-17 keypoint order. YOLO-pose always returns these, always in this order.
# NOTE: "left" means the PERSON'S left, not the left side of the image.
KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7   <- the one that matters for curls
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]


def main(image_path: str) -> None:
    # First run downloads ~6.5MB of weights into the current directory, then caches.
    model = YOLO("yolov8n-pose.pt")

    results = model(image_path)
    r = results[0]

    if r.keypoints is None or len(r.keypoints.xy) == 0:
        print("\nNo person detected.")
        print("Not a bug — this is a real case your API will have to handle in Phase 1.")
        return

    n_people = len(r.keypoints.xy)
    print(f"\nPeople detected: {n_people}")

    for person_idx in range(n_people):
        xy = r.keypoints.xy[person_idx].tolist()  # pixel coordinates
        conf = (
            r.keypoints.conf[person_idx].tolist()
            if r.keypoints.conf is not None
            else [float("nan")] * len(xy)
        )

        print(f"\n--- Person {person_idx} ---")
        print(f"{'idx':<4} {'name':<16} {'x':>8} {'y':>8} {'conf':>6}")
        for i, name in enumerate(KEYPOINT_NAMES):
            x, y = xy[i]
            print(f"{i:<4} {name:<16} {x:>8.1f} {y:>8.1f} {conf[i]:>6.2f}")

    # The entire point of this script: look at it.
    annotated = r.plot()  # BGR numpy array with the skeleton drawn on
    out = Path("out/out_yolo.jpg")
    cv2.imwrite(str(out), annotated)

    print(f"\nAnnotated image saved to: {out.resolve()}")
    print("Open it. Find your left elbow in the picture. Check it against index 7 above.")
    print("Does the x-coordinate land where you expected, or on the other side?")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hello_yolo.py path/to/photo.jpg")
        sys.exit(1)
    main(sys.argv[1])
