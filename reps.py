"""
Phase 2's actual work. Everything else today is plumbing; this file is the
part that turns a wobbly line into a number a human cares about.

Goal: make `python reps.py` print `all passed`.

Two layers live here, and keeping them separate is the whole design:

  1. DETECTION  - find the cycles. A cycle is "arm bent, then straightened".
                  Its only job is to notice motion. It does not grade it.
  2. GRADING    - once you have a cycle, look at how deep it went and judge it.

Conflating these is the tempting mistake. If you set the detector's threshold
at "a proper rep", then a sloppy 95-degree half-curl is invisible: the user did
ten reps, your app says seven, and it has nothing to say about the other three.
Detect permissively, grade strictly, and the feedback feature falls out for free.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

# --- Detection thresholds: the hysteresis band ---------------------------
# The gap between these two is the dead zone. Noise that rattles around inside
# it cannot change the state, which is the entire reason there are two numbers
# instead of one.
#
# CALIBRATE THESE OFF YOUR OWN PLOT. Do not inherit them from me. Your
# curl1.jpg measured 77.4 degrees on a bent arm - if that is near your real
# top-of-curl, a 70-degree threshold would never fire and you would spend an
# hour blaming this file.
UP_ENTER = 100.0     # below this -> the arm is meaningfully bent
DOWN_ENTER = 150.0   # above this -> the arm is straight again, cycle closes

# --- Grading thresholds --------------------------------------------------
FULL_ROM = 70.0      # a cycle whose deepest point beats this is a full rep
TEMPO_MIN_S = 1.2    # a cycle faster than this was swung, not lifted. Same rule
                     # as the thresholds above: calibrate off your own plot. A
                     # controlled curl-and-lower runs 2-3 s; under ~1 s the weight
                     # is riding momentum and the depth number flatters you.


@dataclass
class Rep:
    """One detected cycle: bent, then straightened."""
    start_idx: int      # frame index where the arm entered the bend
    end_idx: int        # frame index where it came back to straight
    min_angle: float    # deepest point reached during the cycle

    @property
    def full(self) -> bool:
        return self.min_angle <= FULL_ROM


def find_reps(
    angles: Sequence[Optional[float]],
    up_enter: float = UP_ENTER,
    down_enter: float = DOWN_ENTER,
) -> List[Rep]:
    """
    Walk the (smoothed) angle series and return one Rep per completed cycle.

    `angles` may contain None. A None means "this frame was dropped, I have no
    measurement" - it is NOT an angle of zero and it is NOT a reason to reset.
    Think about what your state should do while blind. The tests below have an
    opinion about it.

    A cycle only counts when it CLOSES. Bending the arm and holding it there
    until the video ends is not a rep; you never put the weight down.
    """
    # `bent` is the entire state machine: are we inside a cycle, or waiting at
    # the bottom for one to start? Hysteresis is the fact that the two edges use
    # DIFFERENT thresholds - you enter the bend at <up_enter but only leave it at
    # >down_enter, so a wobble trapped in the dead zone between them can't flip
    # anything. One threshold would count that wobble as reps; two cannot.
    reps: List[Rep] = []
    bent = False
    start_idx = 0
    min_angle = 0.0

    for i, a in enumerate(angles):
        if a is None:
            # Blind, not reset. A dropout is missing information, not a straight
            # arm - so hold state and keep the min we already had. This is what
            # lets a rep survive the tracker losing the arm for a few frames.
            continue

        if not bent:
            # Waiting at the bottom. Only a genuine bend past the near edge opens
            # a cycle; a shallow twitch to 110 never crosses it, so it's ignored.
            if a < up_enter:
                bent = True
                start_idx = i
                min_angle = a
        else:
            # Inside a cycle: remember the deepest point (grading needs it)...
            if a < min_angle:
                min_angle = a
            # ...and close only when the arm is straight again. Until then the
            # weight is still up, so the rep isn't done - and if the video ends
            # first this never fires and the half-rep correctly doesn't count.
            if a > down_enter:
                reps.append(Rep(start_idx, i, min_angle))
                bent = False

    return reps


# =========================================================================
# GRADING - layer 2. The cycle exists; now judge how it was performed.
# Kept deliberately downstream of find_reps (see the module docstring): the
# detector counts everything that moves, the grader is the strict one.
# =========================================================================

@dataclass
class RepGrade:
    """One rep, judged. An empty `issues` list means a clean rep."""
    number: int              # 1-based, the way a human counts them
    min_angle: float         # deepest point reached, straight off the Rep
    duration_s: float        # enter-the-bend to back-to-straight, in seconds
    full: bool               # did it reach full range of motion?
    issues: List[str]        # human-readable complaints; empty == nothing wrong

    @property
    def clean(self) -> bool:
        return not self.issues


def form_feedback(
    reps: Sequence[Rep],
    times: Sequence[float],
    full_rom: float = FULL_ROM,
    tempo_min_s: float = TEMPO_MIN_S,
) -> List[RepGrade]:
    """
    Turn detected reps into coaching.

    `times[i]` is the clock reading, in seconds, of the i-th angle sample - the
    SAME index find_reps walked, so a Rep's start/end indices drop straight into
    it. That shared index is the only wire between the two layers; all it carries
    is "which sample happened when."

    Two rules, because side-on video can honestly see exactly two ways a curl
    goes wrong:
      - depth : did the arm reach the top?  (min_angle vs full_rom)
      - tempo : lifted, or swung?           (duration vs tempo_min_s)
    """
    grades: List[RepGrade] = []
    for n, rep in enumerate(reps, start=1):
        duration = times[rep.end_idx] - times[rep.start_idx]
        is_full = rep.min_angle <= full_rom
        issues: List[str] = []

        if not is_full:
            issues.append(
                f"partial rep - curled only to {rep.min_angle:.0f}deg "
                f"(a full rep passes {full_rom:.0f}deg)"
            )
        if duration < tempo_min_s:
            issues.append(
                f"too fast - {duration:.1f}s per rep "
                f"(controlled is >={tempo_min_s:.1f}s; that's a swing)"
            )

        grades.append(RepGrade(n, rep.min_angle, duration, is_full, issues))
    return grades


# =========================================================================
# Tests. These are the spec - read them before you write the function.
# =========================================================================

def _clean_rep(bottom: float = 170.0, top: float = 40.0) -> List[float]:
    """One tidy rep: straight -> curled -> straight."""
    return [bottom, 140.0, 100.0, 60.0, top, 60.0, 100.0, 140.0, bottom]


def _check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got}")
        print(f"        want {want}")
    return ok


def run_tests() -> bool:
    results = []

    # 1. Three clean reps, all deep. The happy path.
    reps = find_reps(_clean_rep() * 3)
    results.append(_check("3 clean reps -> 3 cycles", len(reps), 3))
    results.append(_check("3 clean reps -> all full", [r.full for r in reps], [True] * 3))

    # 2. Hysteresis. This signal oscillates hard across 125 - the midpoint a
    #    naive single-threshold counter would sit on. It would count several
    #    reps here. Yours must count zero, because the signal never leaves the
    #    dead zone.
    noise = [120.0, 130.0, 122.0, 128.0, 121.0, 129.0, 123.0, 127.0] * 4
    results.append(_check("noise inside the dead zone -> 0", len(find_reps(noise)), 0))

    # 3. A shallow twitch that never really bends the arm.
    results.append(_check("110-degree twitch -> 0", len(find_reps([170.0, 140.0, 110.0, 140.0, 170.0])), 0))

    # 4. A real attempt that stops short. This is the case the two-layer design
    #    exists for: it IS a cycle, and it is NOT a full rep.
    partial = find_reps([170.0, 130.0, 95.0, 130.0, 170.0])
    results.append(_check("95-degree curl -> 1 cycle", len(partial), 1))
    results.append(_check("95-degree curl -> not full", partial[0].full if partial else None, False))
    results.append(_check("95-degree curl -> min_angle 95", partial[0].min_angle if partial else None, 95.0))

    # 5. Dropouts. The tracker lost the arm for two frames mid-curl. You were
    #    blind, not reset. The rep still happened.
    dropped = [170.0, 130.0, 60.0, None, None, 45.0, 90.0, 140.0, 170.0]
    reps = find_reps(dropped)
    results.append(_check("dropout mid-rep -> 1 cycle", len(reps), 1))
    results.append(_check("dropout mid-rep -> full", reps[0].full if reps else None, True))
    results.append(_check("dropout ignored by min_angle", reps[0].min_angle if reps else None, 45.0))

    # 6. Curled up and the video ended. Never came back down. Not a rep.
    results.append(_check("never returned to straight -> 0", len(find_reps([170.0, 130.0, 40.0])), 0))

    # 7. Mixed set: 2 good, 1 lazy, 1 good. Count is 4, quality is 3.
    mixed = _clean_rep() + [170.0, 130.0, 92.0, 130.0, 170.0] + _clean_rep() * 2
    reps = find_reps(mixed)
    results.append(_check("mixed set -> 4 cycles", len(reps), 4))
    results.append(_check("mixed set -> 3 full", sum(r.full for r in reps), 3))

    # 8. Bookkeeping. Indices must point at real frames, in order.
    reps = find_reps(_clean_rep() * 2)
    ordered = all(r.start_idx < r.end_idx for r in reps) and reps[0].end_idx <= reps[1].start_idx
    results.append(_check("indices ordered and non-overlapping", ordered, True))

    # 9. Degenerate input shouldn't explode.
    results.append(_check("empty series -> 0", len(find_reps([])), 0))
    results.append(_check("all dropped -> 0", len(find_reps([None] * 20)), 0))

    # --- GRADING layer (form_feedback). Detection above, judgement here. ------
    # Explicit tempo_min_s so these don't move when you calibrate the default.

    # A deep rep at a controlled pace has nothing to say.
    clean_t = [i * 0.25 for i in range(9)]            # the rep spans ~1.25 s
    g = form_feedback(find_reps(_clean_rep()), clean_t, tempo_min_s=1.0)
    results.append(_check("clean rep -> full, no issues", (g[0].full, g[0].issues), (True, [])))

    # A 92-degree rep IS a rep (detection is permissive) but grades partial.
    partial_t = [i * 0.7 for i in range(5)]
    g = form_feedback(find_reps([170.0, 130.0, 92.0, 130.0, 170.0]), partial_t, tempo_min_s=1.0)
    results.append(_check("92-degree rep -> not full", g[0].full, False))
    results.append(_check("92-degree rep -> says partial", any("partial" in s for s in g[0].issues), True))

    # The same deep rep, crammed into 0.4 s, is a swing.
    fast_t = [i * 0.05 for i in range(9)]
    g = form_feedback(find_reps(_clean_rep()), fast_t, tempo_min_s=1.0)
    results.append(_check("rushed rep -> says too fast", any("too fast" in s for s in g[0].issues), True))

    return all(results)


if __name__ == "__main__":
    print("reps.py")
    if run_tests():
        print("all passed")
    else:
        print("\nnot yet. Read the failing case - it is telling you something.")