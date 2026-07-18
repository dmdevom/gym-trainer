"""
Phase 2 - short videos to rep counts and coaching tips.

This is intentionally bounded for free hosting: sample a small number of frames,
reuse the photo pose backend, and turn angle changes into practical feedback.
"""

from dataclasses import dataclass

import cv2

from analyzer import CONF_MIN, calc_angle, classify, pick_arm
from backends import get_backend


MAX_VIDEO_FRAMES = 24
TOP_ANGLE = 70
BOTTOM_ANGLE = 145


@dataclass
class FrameMeasurement:
    frame: int
    time_s: float
    angle: float
    side: str
    confidence: float
    phase: str


def _measurement_from_keypoints(kp: dict, frame_index: int, time_s: float):
    side = pick_arm(kp)
    if side is None:
        return None

    shoulder = kp[f"{side}_shoulder"]
    elbow = kp[f"{side}_elbow"]
    wrist = kp[f"{side}_wrist"]
    angle = calc_angle(shoulder[:2], elbow[:2], wrist[:2])

    return FrameMeasurement(
        frame=frame_index,
        time_s=round(time_s, 2),
        angle=round(angle, 1),
        side=side,
        confidence=round(min(shoulder[2], elbow[2], wrist[2]), 2),
        phase=classify(angle),
    )


def count_reps(measurements: list[FrameMeasurement]) -> int:
    reps = 0
    saw_bottom = False
    saw_top_after_bottom = False

    for m in measurements:
        if m.angle >= BOTTOM_ANGLE:
            if saw_top_after_bottom:
                reps += 1
                saw_top_after_bottom = False
            saw_bottom = True
        elif saw_bottom and m.angle <= TOP_ANGLE:
            saw_top_after_bottom = True

    if saw_top_after_bottom:
        reps += 1
    return reps


def build_tips(measurements: list[FrameMeasurement], reps: int, duration_s: float) -> list[str]:
    if not measurements:
        return ["Keep your full arm visible and record from the side."]

    angles = [m.angle for m in measurements]
    min_angle = min(angles)
    max_angle = max(angles)
    avg_conf = sum(m.confidence for m in measurements) / len(measurements)
    tips = []

    if reps == 0:
        tips.append("Move through a full curl so the app can see bottom-to-top motion.")
    if max_angle < BOTTOM_ANGLE:
        tips.append("Extend your arm more at the bottom of each rep.")
    if min_angle > TOP_ANGLE:
        tips.append("Curl higher at the top so your biceps fully contract.")
    if avg_conf < CONF_MIN + 0.15:
        tips.append("Use a clearer side angle with your shoulder, elbow, and wrist in frame.")
    if duration_s and reps and duration_s / reps < 1.2:
        tips.append("Slow down slightly; controlled reps are easier to judge and usually safer.")
    if not tips:
        tips.append("Good range of motion. Keep the elbow steady and repeat that tempo.")

    return tips[:4]


def analyze_video(video_path: str, max_frames: int = MAX_VIDEO_FRAMES) -> dict:
    backend = get_backend()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "video_unreadable", "detail": "Couldn't read that video file."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if total_frames else 0.0
    step = max(1, total_frames // max_frames) if total_frames else 1

    measurements: list[FrameMeasurement] = []
    sampled = 0
    frame_index = 0

    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % step != 0:
                frame_index += 1
                continue

            sampled += 1
            if hasattr(backend, "keypoints_from_bgr"):
                kp = backend.keypoints_from_bgr(frame)
            else:
                kp = None
            if kp:
                measurement = _measurement_from_keypoints(kp, frame_index, frame_index / fps)
                if measurement:
                    measurements.append(measurement)

            frame_index += 1
    finally:
        cap.release()

    if not measurements:
        return {
            "error": "no_usable_pose",
            "detail": "Couldn't find a clear arm in the sampled video frames.",
            "frames_sampled": sampled,
            "duration_s": round(duration_s, 1),
        }

    reps = count_reps(measurements)
    angles = [m.angle for m in measurements]

    return {
        "reps": reps,
        "tips": build_tips(measurements, reps, duration_s),
        "summary": {
            "duration_s": round(duration_s, 1),
            "frames_sampled": sampled,
            "frames_used": len(measurements),
            "side_mostly_seen": max(
                ("left", "right"),
                key=lambda side: sum(1 for m in measurements if m.side == side),
            ),
            "min_elbow_angle": round(min(angles), 1),
            "max_elbow_angle": round(max(angles), 1),
            "avg_confidence": round(
                sum(m.confidence for m in measurements) / len(measurements), 2
            ),
            "backend": backend.name,
        },
        "timeline": [
            {
                "time_s": m.time_s,
                "angle": m.angle,
                "phase": m.phase,
                "confidence": m.confidence,
            }
            for m in measurements
        ],
    }


if __name__ == "__main__":
    fake = [
        FrameMeasurement(0, 0.0, 160, "right", 0.9, classify(160)),
        FrameMeasurement(1, 0.5, 100, "right", 0.9, classify(100)),
        FrameMeasurement(2, 1.0, 55, "right", 0.9, classify(55)),
        FrameMeasurement(3, 1.5, 155, "right", 0.9, classify(155)),
    ]
    assert count_reps(fake) == 1
    assert build_tips(fake, 1, 1.5)
    print("video analyzer tests passed")
