"""
Phase 4: one exercise, generalised to three.

Everything upstream of here was built around a single movement - the bicep curl -
and a single joint - the elbow. But look at what a curl, a barbell curl and a
squat actually are as a *signal*: a joint angle that sits high (arm straight, legs
standing), dips low (arm curled, hips down), and comes back. Same shape, same
state machine, same summary. The only things that differ between them are:

  - which three landmarks make the angle (elbow vs knee),
  - where the thresholds sit (a curl bottoms out near 40 deg, a squat near 90),
  - what the coaching says when a rep falls short.

So an exercise is not code. It is a table of numbers and a few strings. This file
is that table. Adding a fourth movement (overhead press, lunge, ...) is one more
entry here - not a change to video.py, reps.py, analyze.py or render.py. That is
the same bet backends.py made: find the real seam and the rest stops needing to
know. Here the seam is "which joint, which thresholds"; there it was "17 vs 33
keypoints." Both hide behind one shape.

BlazePose-33 landmark indices, for reference:
    11/12 shoulder   13/14 elbow   15/16 wrist
    23/24 hip        25/26 knee    27/28 ankle
"left" is the person's anatomical left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Exercise:
    """One movement, described as data. See the module docstring for why this is a
    table and not a class hierarchy."""

    key: str                       # url-safe id: "bicep_curl", "squat", ...
    name: str                      # human label, for the UI and the video overlay
    film_tip: str                  # one line: how to point the camera
    tips: Tuple[str, ...]          # form cues, reused as the "keep in mind" list

    # The three landmarks that form the measured angle, per side. The MIDDLE index
    # is the vertex - the joint the angle is *at*: elbow for a curl, knee for a
    # squat. calc_angle takes exactly (a, vertex, c).
    sides: Dict[str, Tuple[int, int, int]]
    vertex_name: str               # "elbow" / "knee" - labels the number everywhere

    # The hysteresis band + grading thresholds. These WERE the module-level
    # constants in reps.py; they live here now because they are the per-exercise
    # part. Same meaning as before, and still the ones you calibrate off the plot:
    #   below up_enter    -> the joint is meaningfully flexed  (a rep opens)
    #   above down_enter  -> extended again                    (the rep closes)
    #   deepest <= full_rom -> it counted as a full-range rep
    up_enter: float
    down_enter: float
    full_rom: float
    tempo_min_s: float             # a cycle faster than this was thrown, not lifted

    # The top of the on-video ROM gauge - the deepest angle the scale is drawn to.
    # A curl can reach ~30 deg; a squat bottoms out much higher, so its gauge does too.
    gauge_deep: float

    # The actionable half of a bad-form line. form_feedback() prepends the measured
    # numbers ("reached 96 deg..."); these say what to DO about it.
    depth_cue: str
    tempo_cue: str

    # Primary movers + a one-line growth note, for the coaching card.
    muscle: str

    # Secondary form checks: the whole-body rules graded WITHIN each detected rep,
    # beside the one primary angle find_reps counts on. Data, not code (see
    # FormCheck). Empty is fine and is the default - an exercise with no checks grades
    # exactly on depth + tempo, the way everything did before this table existed.
    checks: Tuple["FormCheck", ...] = ()

    def brief(self) -> dict:
        """The public description the /exercises endpoint serves and the page's
        selector renders. Thresholds stay server-side; the client only needs to
        name the movement and tell the user where to point the camera."""
        return {
            "key": self.key,
            "name": self.name,
            "vertex_name": self.vertex_name,
            "film_tip": self.film_tip,
            "tips": list(self.tips),
        }


# =========================================================================
# FORM CHECKS - the "look at the whole body, not just the one angle" layer.
#
# find_reps stays exactly what it was: a state machine on ONE clean angle per
# exercise (elbow, knee). It is not asked to understand posture. Everything else a
# side-on camera can honestly see - a squat folding forward, a curl's elbow
# drifting, the torso swinging to cheat the weight up - is a FormCheck, graded
# WITHIN each rep the primary angle already found. Same bet as the rest of this
# file: a check is a row of numbers and two strings, not a new code path.
#
# A check reduces a rep's frames to one number and compares it to a limit:
#   measure  how a single frame becomes a scalar -
#              "angle"    -> calc_angle over 3 landmarks (a joint angle)
#              "vertical" -> a 2-landmark segment's tilt off vertical, 0..90 deg
#   reduce   how the per-frame scalars collapse to one -
#              "at_bottom"-> the value at the rep's deepest frame
#              "max"/"min"-> the worst / smallest across the rep
#              "range"    -> max - min across the rep (swing, drift)
#   compare  "over" flags when the reduced value exceeds limit, "under" when below
#
# Out of frame is not a fault: if too few of a rep's frames saw all the check's
# landmarks it is reported "not assessed", never penalised (see reps.evaluate_checks).
# =========================================================================

@dataclass(frozen=True)
class FormCheck:
    key: str                       # short id, and the tag that colours the overlay/table
    label: str                     # positive name for the "what we checked" row ("Elbow pinned")
    fault: str                     # what to say when it's violated ("Elbow drifting forward")
    measure: str                   # "angle" (3 landmarks) | "vertical" (2 landmarks)
    sides: Dict[str, Tuple[int, ...]]   # the landmark indices per side (3 for angle, 2 for vertical)
    reduce: str                    # "at_bottom" | "max" | "min" | "range"
    compare: str                   # "over" | "under" - which side of `limit` is the fault
    limit: float                   # the threshold, in degrees
    cue: str                       # the actionable fix, appended to the flagged issue line
    min_vis: float = 0.5           # a landmark below this visibility doesn't count as seen (CONF_MIN)


# --- the curl checks (shared by both curls; barbell tightens the swing below) ---

# A strict curl keeps the upper arm hanging ~vertical - only the forearm swings up.
# Let the elbow drift forward to help and the shoulder->elbow segment tilts away
# from vertical, so the WORST tilt across the rep is the tell. Read off the curl
# clips: a clean rep stays well under this; loosen it if your top-of-curl trips it.
_ELBOW_PINNED = FormCheck(
    key="elbow", label="Elbow pinned", fault="Elbow drifting forward",
    measure="vertical", sides={"left": (11, 13), "right": (12, 14)},
    reduce="max", compare="over", limit=40.0,
    cue="Pin your elbow to your side - let only the forearm move.",
)

# Body swing: rocking the torso (shoulder->hip) back and forth to throw the
# weight up. A clean rep barely moves it, so the RANGE of torso tilt over the rep -
# not its absolute lean - is what flags the cheat.
_TORSO_STILL = FormCheck(
    key="swing", label="No body swing", fault="Swinging the torso",
    measure="vertical", sides={"left": (11, 23), "right": (12, 24)},
    reduce="range", compare="over", limit=12.0,
    cue="Keep your torso still - no leaning back to heave the weight up.",
)

# Barbell curls are stricter about this - the tips already say "if your hips move,
# the weight's too heavy" - so the same check, tighter.
_TORSO_STILL_STRICT = FormCheck(
    key="swing", label="No body swing", fault="Swinging the torso",
    measure="vertical", sides={"left": (11, 23), "right": (12, 24)},
    reduce="range", compare="over", limit=9.0,
    cue="Strict reps only - if your torso swings, the bar is too heavy.",
)

# Heaving the weight with a head/neck bob - the head cranes or rocks to help drive the rep,
# distinct from a whole-torso swing. Measured as the RANGE of the nose->shoulder tilt off
# vertical across the rep. CALIBRATED off the curl clips: clean reps stay under ~13 deg (pass
# clips ran 6.5-12.7), a heaved rep swings 40+ (the barbell-fail cheat reps hit 42-54), so 20
# leaves clean reps clean with headroom. The nose tracks reliably even on the lite model, which
# is why this one holds up where the hand/foot landmarks are too noisy to grade.
_HEAD_STILL = FormCheck(
    key="head", label="Head still", fault="Head bobbing",
    measure="vertical", sides={"left": (0, 11), "right": (0, 12)},
    reduce="range", compare="over", limit=20.0,
    cue="Keep your head still and eyes forward - drive with the arm, not a neck bob.",
)

# Standing curl: the legs should stay planted and straight. Dip and drive with the
# knees to help and the hip-knee-ankle angle drops, so its SMALLEST value across the
# rep is the tell. Needs the legs in frame - the #1 cropped-out case, which degrades
# to "not assessed", not a wrong grade.
_LEGS_GROUNDED = FormCheck(
    key="legs", label="Legs grounded", fault="Legs bending to help",
    measure="angle", sides={"left": (23, 25, 27), "right": (24, 26, 28)},
    reduce="min", compare="under", limit=150.0,
    cue="Stand tall and still - drive with your arms, not your legs.",
)

# --- the squat check ---

# The upper-body check a knee-only squat grade misses entirely: at the bottom, is
# the lifter still "chest up", or folded forward over the bar? We measure the torso
# (shoulder->hip) tilt off vertical AT THE DEEPEST frame. A squat leans forward by
# nature, so the bar is set high on purpose - this flags a genuine fold, not a normal
# hip hinge. MediaPipe has no spine points, so this is torso *lean*, never back
# *rounding* - the cue says "chest up," it never claims to see your spine.
# UNCALIBRATED: there's no squat footage yet - read this number off a real plot
# before trusting it (the standing rule for every threshold in this file).
_SQUAT_TORSO = FormCheck(
    key="lean", label="Chest up", fault="Folding forward at the bottom",
    measure="vertical", sides={"left": (11, 23), "right": (12, 24)},
    reduce="at_bottom", compare="over", limit=60.0,
    cue="Keep your chest up and sit between your hips instead of folding over.",
)


# =========================================================================
# The three the demo ships. Numbers are starting points, not gospel - the same
# "read them off your own plot" rule from reps.py applies. bicep_curl deliberately
# reuses the exact constants reps.py shipped with, so the curl clips grade
# identically to before this refactor.
# =========================================================================

BICEP_CURL = Exercise(
    key="bicep_curl",
    name="Single-arm Bicep Curl",
    film_tip="Film side-on, whole arm in frame.",
    tips=(
        "Keep your elbow pinned to your side - it shouldn't drift forward.",
        "Squeeze at the top; don't just swing the weight up.",
        "Lower under control - the way down builds as much as the way up.",
    ),
    sides={"left": (11, 13, 15), "right": (12, 14, 16)},
    vertex_name="elbow",
    up_enter=100.0,
    down_enter=150.0,
    full_rom=70.0,
    tempo_min_s=1.2,
    gauge_deep=30.0,
    depth_cue="Curl higher - bring the wrist up toward the shoulder for a full contraction.",
    tempo_cue="Slow it down - control the lowering instead of dropping the weight.",
    muscle="Biceps brachii. It grows from full-range reps under control - the stretch "
           "at the bottom matters as much as the squeeze at the top.",
    checks=(_ELBOW_PINNED, _TORSO_STILL, _HEAD_STILL, _LEGS_GROUNDED),
)

BARBELL_CURL = Exercise(
    key="barbell_curl",
    name="Barbell Curl",
    film_tip="Film side-on, whole torso and arms in frame.",
    tips=(
        "Pin both elbows to your sides - they're a hinge, not a lever.",
        "No swinging or leaning back - if your hips move, the weight's too heavy.",
        "Control the negative; don't let the bar drop.",
    ),
    # Same joints as the single-arm curl. Side-on we track the near arm exactly as
    # before; what makes it "barbell" is the thresholds and the coaching, not the
    # geometry (a barbell's fixed grip usually stops a touch short of a dumbbell).
    sides={"left": (11, 13, 15), "right": (12, 14, 16)},
    vertex_name="elbow",
    up_enter=100.0,
    down_enter=145.0,
    full_rom=75.0,
    tempo_min_s=1.2,
    gauge_deep=35.0,
    depth_cue="Curl the bar all the way to your chest for a full contraction.",
    tempo_cue="Control the negative - a 2-3s lower beats dropping the bar.",
    muscle="Biceps brachii, both arms at once. Strict reps without swinging your body or "
           "jerking the weight up keep the tension on the muscle instead of the momentum.",
    checks=(_ELBOW_PINNED, _TORSO_STILL_STRICT, _HEAD_STILL, _LEGS_GROUNDED),
)

SQUAT = Exercise(
    key="squat",
    name="Squat",
    film_tip="Film side-on, whole body in frame - step back so your feet show.",
    tips=(
        "Sit back and down - break at the hips and knees together.",
        "Keep your chest up and your back flat.",
        "Push your knees out so they track over your toes.",
        "Drive up through your heels.",
    ),
    sides={"left": (23, 25, 27), "right": (24, 26, 28)},
    vertex_name="knee",
    # A squat is the same high->low->high signal as a curl, just at the knee and at
    # different angles: you stand near 175 deg, break past 130 into the descent,
    # and stand back up past 155 to close the rep. Parallel is ~100 deg.
    up_enter=130.0,
    down_enter=155.0,
    full_rom=100.0,
    tempo_min_s=1.3,
    gauge_deep=70.0,
    depth_cue="Sit deeper - drop your hips until your thighs are at least parallel.",
    tempo_cue="Control the descent - own the bottom instead of bouncing out of it.",
    muscle="Quads, glutes and hamstrings. Depth is what pays: hitting parallel or "
           "below trains the glutes and hams a half-squat skips.",
    checks=(_SQUAT_TORSO,),
)

EXERCISES: Dict[str, Exercise] = {e.key: e for e in (BICEP_CURL, BARBELL_CURL, SQUAT)}

DEFAULT = BICEP_CURL.key


def get_exercise(key: Optional[str]) -> Exercise:
    """Resolve a key to an Exercise, defaulting rather than raising. A stale or
    fat-fingered key from the client should still analyse *something* - the wrong
    thresholds are a bad grade, but a 500 is a broken app. main.py can be stricter
    if it wants; the pipeline stays forgiving."""
    return EXERCISES.get((key or "").lower().strip(), EXERCISES[DEFAULT])
