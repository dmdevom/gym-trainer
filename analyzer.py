"""
Phase 1 — keypoints to meaning.
"""

import math

CONF_MIN = 0.5  # Phase 0's lesson, in one constant. Never trust an unchecked point.


def calc_angle(a, b, c) -> float:
    """
    Angle at joint b, in degrees, formed by the points a-b-c.
    a, b, c are (x, y) pixel coords. b is the vertex — the elbow.
    Returns degrees in [0, 180].
    """
    # STEP 1 — translate so b sits at the origin.
    # The angle is a property of the two directions leaving the joint, not of
    # where the person is standing. This line is the whole conceptual move:
    # three dots on a photo become two vectors from a joint.
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    # STEP 2 — dot product, and the two magnitudes.
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    # Guard: two keypoints on the identical pixel means dividing by zero.
    # You watched MediaPipe collapse a whole forearm onto a hip. It happens.
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    # a·b = |a||b|cos(theta)   ->   cos(theta) = dot / (|a||b|)
    # Dividing by the magnitudes is what removes limb length, which is why this
    # same function works on a 3072px photo and a 640px video frame unchanged.
    cosine = dot / (mag_ba * mag_bc)

    # STEP 3 — THE CLIP. Non-negotiable.
    # A near-straight arm makes this evaluate to -1.0000000000000002.
    # math.acos of that raises ValueError: math domain error.
    # (numpy's arccos hands you a silent NaN instead — worse, because one NaN
    #  poisons every smoothed value after it in Phase 2.)
    # Your left arm measured 168.9 degrees. You are standing next to this cliff.
    cosine = max(-1.0, min(1.0, cosine))

    # STEP 4 — radians to degrees.
    # acos returns [0, pi] = [0, 180]. Exactly a joint's range.
    # Unsigned, though: this cannot tell "bent forward" from "bent backward".
    return math.degrees(math.acos(cosine))


def pick_arm(kp: dict):
    """
    Which arm did the camera actually see? Returns 'left', 'right', or None
    if neither arm clears CONF_MIN on all three joints.

    Returning None is not a failure — it's the honest answer for a photo of
    someone's back, and it's what stops you reporting an angle built from
    fabricated keypoints.
    """
    best, best_conf = None, 0.0
    for side in ("left", "right"):
        joints = [kp[f"{side}_{part}"] for part in ("shoulder", "elbow", "wrist")]
        if any(j[2] < CONF_MIN for j in joints):
            continue
        avg = sum(j[2] for j in joints) / 3
        if avg > best_conf:
            best, best_conf = side, avg
    return best


def classify(angle: float) -> str:
    if angle > 150:
        return "arm extended (bottom of curl)"
    if angle < 60:
        return "fully curled (top of rep)"
    return "mid-rep"


def analyze(image_path: str) -> dict:
    from backends import get_backend

    backend = get_backend()
    kp = backend.keypoints(image_path)

    if kp is None:
        return {
            "error": "no_person_detected",
            "detail": "Couldn't find a person in that image.",
        }

    side = pick_arm(kp)
    if side is None:
        return {
            "error": "arm_not_visible",
            "detail": "Found a person, but neither arm was clear enough to measure. "
                      "Try a side-on shot with the whole arm in frame.",
        }

    s = kp[f"{side}_shoulder"]
    e = kp[f"{side}_elbow"]
    w = kp[f"{side}_wrist"]
    angle = calc_angle(s[:2], e[:2], w[:2])

    return {
        "elbow_angle": round(angle, 1),
        "phase": classify(angle),
        "side_analyzed": side,
        "confidence": round(min(s[2], e[2], w[2]), 2),
        "backend": backend.name,
    }


if __name__ == "__main__":
    # Run me:  python analyzer.py
    # Synthetic first — you can verify these on paper without a model.
    assert abs(calc_angle((0, 0), (0, 1), (1, 1)) - 90.0) < 0.01, "right angle"
    assert abs(calc_angle((0, 0), (0, 1), (0, 2)) - 180.0) < 0.01, "straight line (the NaN trap)"
    assert abs(calc_angle((0, 2), (0, 1), (0, 2)) - 0.0) < 0.01, "folded flat"

    # Now your real curl1.jpg numbers, straight from YOLO.
    right = calc_angle((1205.9, 1497.5), (989.8, 2060.7), (538.8, 1760.7))
    left = calc_angle((2060.7, 1425.6), (2254.4, 2216.7), (2289.3, 2959.0))
    assert abs(right - 77.4) < 0.1, f"right arm: got {right}"
    assert abs(left - 168.9) < 0.1, f"left arm: got {left}"

    print("all passed")