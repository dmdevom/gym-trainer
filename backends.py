"""
Pose backends: one contract, two implementations.

This file is where Phase 0 gets paid back. Everything you learned the hard way
lives here and nowhere else:

  - MediaPipe returns 33 landmarks, NORMALISED coords (0-1), `visibility`.
  - YOLO returns 17 keypoints, PIXEL coords, one `conf` per point.
  - "left elbow" is index 13 in one and index 7 in the other.

Nothing downstream knows any of that. Both hand back exactly
{joint_name: (x_px, y_px, confidence)}.

MediaPipe is the default and the thing that ships. YOLO is opt-in, for the
benchmark and the disagreement check:  POSE_BACKEND=yolo python ...
"""

import os
import pathlib
from functools import lru_cache

JOINTS = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)


class PoseBackend:
    name = "base"

    def keypoints(self, image_path: str):
        """Return {joint: (x_px, y_px, conf)}, or None if no person found."""
        raise NotImplementedError

    def info(self) -> dict:
        """What's actually loaded. Printed at boot, and served from /health."""
        return {"backend": self.name}


class MediaPipeBackend(PoseBackend):
    """The one that ships. No torch. ~200MB resident, fast on CPU."""

    name = "mediapipe"
    IDX = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
    }

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.model_path = pathlib.Path(
            os.environ.get("POSE_MODEL", "models/pose_landmarker_lite.task")
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {self.model_path}\n"
                f"  mkdir -p models && wget -O {self.model_path} \\\n"
                f"    https://storage.googleapis.com/mediapipe-models/pose_landmarker"
                f"/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def info(self):
        return {
            "backend": self.name,
            "model": self.model_path.name,
            "model_path": str(self.model_path.resolve()),
            "model_mb": round(self.model_path.stat().st_size / 1e6, 1),
        }

    def keypoints(self, image_path):
        import cv2

        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        return self.keypoints_from_bgr(bgr)

    def keypoints_from_bgr(self, bgr):
        import cv2

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(img)
        if not res.pose_landmarks:
            return None

        lms = res.pose_landmarks[0]
        # Normalised -> pixels. Forget this line and every angle downstream is
        # quietly wrong with no error. This is the whole reason this file exists.
        return {
            j: (lms[i].x * w, lms[i].y * h, lms[i].visibility or 0.0)
            for j, i in self.IDX.items()
        }


class YoloBackend(PoseBackend):
    """Opt-in. Local only — needs torch, which we deliberately don't deploy."""

    name = "yolo"
    IDX = {
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
    }

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n-pose.pt")

    def info(self):
        return {"backend": self.name, "model": "yolov8n-pose.pt"}

    def keypoints(self, image_path):
        r = self.model(image_path, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return None
        xy = r.keypoints.xy[0].tolist()
        conf = r.keypoints.conf[0].tolist()
        # Already pixels. Nothing to convert.
        return {j: (xy[i][0], xy[i][1], conf[i]) for j, i in self.IDX.items()}


@lru_cache(maxsize=1)
def get_backend() -> PoseBackend:
    """Cached: the model loads once per process, not once per request."""
    choice = os.environ.get("POSE_BACKEND", "mediapipe").lower()
    if choice == "mediapipe":
        return MediaPipeBackend()
    if choice == "yolo":
        return YoloBackend()
    raise ValueError(f"Unknown POSE_BACKEND: {choice!r} (use 'mediapipe' or 'yolo')")
