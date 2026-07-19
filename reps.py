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

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from analyzer import calc_angle  # Phase 1's trig, reused for the secondary form angles too.

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

# --- Partial ("bad rep") detection ---------------------------------------
# The smallest peak-to-valley swing that counts as a real attempt (not sensor
# noise). Dead-zone jitter is ~10 deg (see the hysteresis noise test); a genuine
# curl, even a shallow one, moves 40 deg+. 30 deg sits cleanly between the two, so
# a real half-rep is recorded as a bad rep while noise stays ignored.
PARTIAL_MIN_AMPLITUDE = 30.0


@dataclass
class Rep:
    """One detected cycle: bent, then straightened.

    `reason` is None for a real, completed cycle (what find_reps returns). find_partials
    sets it to "under_extension" or "under_contraction" for an attempt that did NOT
    complete a valid cycle - the lifter moved, but the rep wasn't counted. Grading treats
    a reason'd rep as a "bad rep": counted in the total, never full."""
    start_idx: int      # frame index where the arm entered the bend
    end_idx: int        # frame index where it came back to straight
    min_angle: float    # deepest point reached during the cycle
    reason: Optional[str] = None   # None = completed; else why it wasn't a full rep

    @property
    def full(self) -> bool:
        return self.reason is None and self.min_angle <= FULL_ROM


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


# --- Partial detection: the reps find_reps deliberately doesn't count ------
# find_reps counts a cycle only when it CLOSES (bent past up_enter, straightened
# past down_enter). Two honest efforts slip through that gate and vanish:
#   - under-extension: curled fine, never straightened back -> the cycle never closes.
#   - under-contraction: bent a little, never past up_enter -> the cycle never opens.
# The lifter did something and got no credit and no reason why. find_partials finds
# exactly those attempts so they can be shown as "bad reps" - counted in the total,
# never full. find_reps is untouched; this reads the same series a second way.

def _extrema(angles: Sequence[Optional[float]], min_swing: float) -> List[tuple]:
    """Alternating turning points (idx, value, "max"|"min") where each swing is at
    least `min_swing`. A prominence/zigzag filter: track the running high and low since
    the last committed pivot; when the series retraces `min_swing` off one of them,
    commit that extreme and flip direction. None samples are skipped - blind, not a
    reset, the same rule find_reps uses - so a dropout can't fake a turning point."""
    pts = [(i, a) for i, a in enumerate(angles) if a is not None]
    if len(pts) < 2:
        return []
    out: List[tuple] = []
    hi_i, hi_v = pts[0]
    lo_i, lo_v = pts[0]
    trend = 0                      # +1 = rising leg, -1 = falling leg, 0 = not yet moving
    for i, v in pts[1:]:
        if v > hi_v:
            hi_i, hi_v = i, v
        if v < lo_v:
            lo_i, lo_v = i, v
        if trend >= 0 and hi_v - v >= min_swing:
            out.append((hi_i, hi_v, "max"))     # was rising; dropped off the high -> a peak
            trend, lo_i, lo_v = -1, i, v
        elif trend <= 0 and v - lo_v >= min_swing:
            out.append((lo_i, lo_v, "min"))      # was falling; rose off the low -> a valley
            trend, hi_i, hi_v = 1, i, v
    if trend > 0:                  # close the final leg on its running extreme
        out.append((hi_i, hi_v, "max"))
    elif trend < 0:
        out.append((lo_i, lo_v, "min"))
    return out


def find_partials(
    angles: Sequence[Optional[float]],
    up_enter: float = UP_ENTER,
    down_enter: float = DOWN_ENTER,
    min_amplitude: float = PARTIAL_MIN_AMPLITUDE,
) -> List[Rep]:
    """Attempts that moved but never became a counted rep, each tagged with why.

    Returns Reps with `reason` set. A valley (curl bottom) is classified by whether the
    arm came back up afterwards (a recovery peak) and how far:
      - crossed up_enter AND recovered past down_enter -> a real rep (find_reps has it): skip.
      - crossed up_enter, recovered but short of down_enter -> under_extension.
      - never crossed up_enter (but did move ≥ min_amplitude) -> under_contraction.
    A valley with NO recovery peak - the clip ended mid-dip - is left alone: we can't tell
    a cut-off rep from a failed one, so we don't presume a fault (the same call find_reps
    makes for an unclosed cycle). Anything overlapping a find_reps window is dropped, so a
    counted rep can never also be reported as a bad rep."""
    ext = _extrema(angles, min_amplitude)
    counted = find_reps(angles, up_enter, down_enter)
    partials: List[Rep] = []
    for k, (idx, val, kind) in enumerate(ext):
        if kind != "min":
            continue
        following = ext[k + 1] if k + 1 < len(ext) else None   # the recovery peak, if any
        if following is None:
            continue                                            # clip ended mid-dip: don't presume
        crossed_up = val < up_enter
        if crossed_up and following[1] > down_enter:
            continue                                            # a completed rep; find_reps owns it
        reason = "under_extension" if crossed_up else "under_contraction"
        preceding = ext[k - 1] if k - 1 >= 0 else None          # the top the descent began from
        start_idx = preceding[0] if preceding is not None else idx
        end_idx = following[0]
        # Strict interior overlap only: a bad rep may SHARE the peak that ends a counted
        # rep (that peak is the top it descends from) - it just can't sit inside one.
        if any(start_idx < r.end_idx and end_idx > r.start_idx for r in counted):
            continue                                            # inside a counted rep's window
        partials.append(Rep(start_idx, end_idx, val, reason=reason))
    return partials


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
    depth_pct: float = 0.0   # how much of the target range it covered, 0-100
    tags: List[str] = field(default_factory=list)   # short codes ("shallow", "rushed")
                             # the same verdict as `issues`, but for colouring the
                             # video overlay and table without re-parsing English
    flash: str = ""          # short one-line overlay flash: the terse lead cause, number-free
                             # and without the "Rep N:" prefix (render adds it). The
                             # deterministic fallback when the LLM path is off.

    @property
    def clean(self) -> bool:
        return not self.issues


def _join_faults(labels: List[str]) -> str:
    """Join up to the top 3 fault labels into one video-flash phrase: the first capitalised,
    the rest lower-cased, an Oxford 'and' before the last. 'Swinging the torso', 'elbow
    drifting forward', 'rushed' -> 'Swinging the torso, elbow drifting forward, and rushed'."""
    labels = labels[:3]
    parts = [labels[0][:1].upper() + labels[0][1:]] + [l[:1].lower() + l[1:] for l in labels[1:]]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def form_feedback(
    reps: Sequence[Rep],
    times: Sequence[float],
    exercise: "Exercise",   # exercises.Exercise; string-annotated to keep this file dep-free
    checks_per_rep: Optional[Sequence[Sequence["CheckResult"]]] = None,
) -> List[RepGrade]:
    """
    Turn detected reps into coaching, in this exercise's own words.

    `times[i]` is the clock reading, in seconds, of the i-th angle sample - the
    SAME index find_reps walked, so a Rep's start/end indices drop straight into
    it. That shared index is the only wire between the two layers; all it carries
    is "which sample happened when."

    The always-on rules are the two a side-on camera can read from the ONE primary
    angle, and both generalise across the movements:
      - depth : did the joint reach full range?  (min_angle vs full_rom)
      - tempo : lifted, or thrown?               (duration vs tempo_min_s)

    `checks_per_rep`, when supplied, is evaluate_checks()'s whole-body verdict for
    each rep in the same order - torso lean, elbow drift, and so on. Its flagged
    issues lead the list, because posture/safety outranks depth and tempo in what to
    fix next, and because the on-video flash leads with them. Leave it None (the
    default) and this behaves exactly as it did before form checks existed - which is
    what keeps the nine find_reps tests and the curl grades unchanged.

    The messages are detailed on purpose: a bare "partial rep" gives the user
    nothing to DO. Each names the number it missed and hands over a fix. `tags` and
    `depth_pct` carry the same verdict in a machine-readable form so the overlay and
    the table can colour and size things without re-reading the English.
    """
    full_rom = exercise.full_rom
    down = exercise.down_enter
    tempo_min_s = exercise.tempo_min_s
    vtx = exercise.vertex_name
    span = max(1.0, down - full_rom)   # the arc a "full" rep is expected to cover

    up = exercise.up_enter

    grades: List[RepGrade] = []
    for n, rep in enumerate(reps, start=1):
        duration = times[rep.end_idx] - times[rep.start_idx]
        # A bad rep (reason set) is never full - it didn't complete a valid cycle - even
        # if it went deep. It's counted in the total, but full_reps excludes it.
        is_full = rep.reason is None and rep.min_angle <= full_rom
        # 100% once the joint reaches full_rom; linear back down to 0 at the top of
        # the movement. Capped, so going deeper than full range still reads as 100.
        depth_pct = max(0.0, min(100.0, (down - rep.min_angle) / span * 100.0))

        issues: List[str] = []
        tags: List[str] = []
        posture_flash: List[tuple] = []   # (severity, terse fault) per flagged form check,
                                          # used only to order the video flash worst-first

        # A bad rep leads with WHY it wasn't counted - that's the one thing the lifter
        # needs. The redundant shallow/rushed depth/tempo tags are skipped for it (the
        # reason already says the range was incomplete), but posture checks still run.
        if rep.reason == "under_extension":
            issues.append(
                f"Not counted as a full rep - {vtx} curled to {rep.min_angle:.0f} deg but "
                f"didn't straighten back past {down:.0f} deg before the next rep. Extend all "
                f"the way between reps so it counts."
            )
            tags.append("under_extension")
        elif rep.reason == "under_contraction":
            issues.append(
                f"Not counted - {vtx} only bent to {rep.min_angle:.0f} deg; a rep has to pass "
                f"{up:.0f} deg to count. {exercise.depth_cue}"
            )
            tags.append("under_contraction")

        # Posture/safety: a flagged form check is added next (it leads for a clean rep and
        # rides just under the incompleteness line for a bad one). This is what the on-video
        # flash surfaces (issues[0]) and what coaching prioritises fixing.
        if checks_per_rep is not None:
            for cr in checks_per_rep[n - 1]:
                if cr.status == "flag":
                    issues.append(cr.issue)
                    tags.append(cr.key)
                    posture_flash.append((cr.severity, cr.fault))

        if rep.reason is None:
            if not is_full:
                issues.append(
                    f"Shallow - {vtx} stopped at {rep.min_angle:.0f} deg, short of the "
                    f"{full_rom:.0f} deg that counts as full range. {exercise.depth_cue}"
                )
                tags.append("shallow")
            if duration < tempo_min_s:
                issues.append(
                    f"Rushed - {duration:.1f}s for the rep; controlled is >= "
                    f"{tempo_min_s:.1f}s. {exercise.tempo_cue}"
                )
                tags.append("rushed")

        # The one-line flash: the rep's top faults, worst-first, read in a glance. Each fault
        # contributes (single_phrase, join_label) - the hand-tuned phrase used when it is the
        # ONLY fault, or the terse label used when several are joined. Order mirrors the issues
        # list (not-counted reason, posture worst-first by severity, depth, tempo); the flash
        # shows the top 3, and the full list still lands in the coach note.
        elements: List[tuple] = []
        if rep.reason == "under_extension":
            elements.append(("Not counted - didn't fully extend", "Not counted - didn't fully extend"))
        elif rep.reason == "under_contraction":
            elements.append(("Not counted - didn't bend far enough", "Not counted - didn't bend far enough"))
        for _sev, fault in sorted(posture_flash, key=lambda pf: pf[0], reverse=True):
            elements.append((fault, fault))
        if "shallow" in tags:
            elements.append(("Shallow - short of full depth", "short of full depth"))
        if "rushed" in tags:
            elements.append(("Rushed - control the lowering", "rushed"))

        if not elements:
            flash = "Clean - full range, controlled."
        elif len(elements) == 1:
            flash = elements[0][0]
        else:
            flash = _join_faults([e[1] for e in elements])

        grades.append(RepGrade(n, rep.min_angle, duration, is_full, issues, round(depth_pct), tags, flash))
    return grades


# =========================================================================
# FORM CHECKS - layer 2b. The rep exists and its depth/tempo are graded above;
# this looks at the REST of the body across the same rep window. It is deliberately
# separate from find_reps: the detector still counts on one clean angle, and a check
# that can't be seen (limb out of frame) is reported "not assessed", never a penalty.
# The per-exercise checks themselves are data in exercises.py (FormCheck).
# =========================================================================

ASSESS_MIN = 0.6   # a check needs all its landmarks visible in >= this fraction of a
                   # rep's frames to be judged at all; below it, the rep is "not assessed"


@dataclass
class CheckResult:
    """One FormCheck's verdict for one rep. `status` is the whole story:
      "ok"   - assessed and within bounds
      "flag" - assessed and out of bounds (issue/tag populated)
      "skip" - not assessable this rep (too much of it was out of frame)"""
    key: str
    label: str
    status: str                    # "ok" | "flag" | "skip"
    value: Optional[float] = None  # the reduced measurement, degrees (None when skipped)
    issue: str = ""                # human line; populated only when status == "flag"
    fault: str = ""                # terse phrase for the overlay flash ("Elbow drifting
                                   # forward"); populated only when status == "flag"
    severity: float = 0.0          # how far past the limit as a ratio (>= 1, bigger = worse);
                                   # populated only when status == "flag", orders the flash


def _vertical_dev(p0, p1) -> float:
    """How far the p0->p1 segment leans off vertical, in degrees: 0 = plumb,
    90 = flat. Sign-free - which end is up doesn't matter, only the tilt - so it
    reads the same whether we hand it shoulder->hip or hip->shoulder."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _reduce(check: "FormCheck", measured: List[tuple], bottom_i: int) -> float:
    """Collapse a rep's per-frame measurements to the one number the limit compares
    against. `measured` is [(sample_index, value), ...]; `bottom_i` is the rep's
    deepest frame, for the at_bottom kind."""
    vals = [v for _, v in measured]
    if check.reduce == "max":
        return max(vals)
    if check.reduce == "min":
        return min(vals)
    if check.reduce == "range":
        return max(vals) - min(vals)
    # at_bottom: the measurement on the frame nearest the rep's deepest point.
    return min(measured, key=lambda iv: abs(iv[0] - bottom_i))[1]


def evaluate_checks(
    rep: Rep,
    samples: Sequence["Sample"],        # video.Sample; each carries .landmarks (33-pt map)
    angles: Sequence[Optional[float]],  # the SAME smoothed primary series find_reps walked
    side: str,
    exercise: "Exercise",
) -> List[CheckResult]:
    """
    Grade every FormCheck this exercise carries, within one detected rep window.

    find_reps stays the one-angle state machine it has always been; this is the "and
    also look at the whole body" pass bolted on beside it. For each check we look only
    at the frames of THIS rep, measure it wherever its landmarks are visible, and
    reduce to one number to compare against the check's limit.

    Out of frame is not a fault. A check is judged only if enough of the rep's frames
    actually saw all its landmarks (ASSESS_MIN); below that it is "skip" - surfaced as
    "not assessed", never a penalty. A legs-cropped curl clip therefore degrades to
    exactly the behaviour this file had before form checks existed.
    """
    if not exercise.checks:
        return []

    lo = rep.start_idx
    hi = min(rep.end_idx, len(samples) - 1)
    window = range(lo, hi + 1)
    win_len = max(1, hi - lo + 1)

    # The rep's deepest frame, for at_bottom checks: the smallest primary angle we
    # actually measured inside the window (fall back to the start if all were blind).
    bottom_i, best = lo, None
    for i in window:
        a = angles[i] if i < len(angles) else None
        if a is not None and (best is None or a < best):
            best, bottom_i = a, i

    results: List[CheckResult] = []
    for chk in exercise.checks:
        idxs = chk.sides.get(side)
        if not idxs:                       # this check doesn't watch the tracked side
            results.append(CheckResult(chk.key, chk.label, "skip"))
            continue

        # Per-frame measurements, only where EVERY landmark the check needs is visible.
        measured: List[tuple] = []         # (sample_index, value)
        for i in window:
            lm = samples[i].landmarks if i < len(samples) else None
            if lm is None:
                continue
            pts = [lm.get(j) for j in idxs]
            if any(p is None or p[2] < chk.min_vis for p in pts):
                continue
            xy = [(p[0], p[1]) for p in pts]
            val = calc_angle(*xy) if chk.measure == "angle" else _vertical_dev(*xy)
            measured.append((i, val))

        if len(measured) / win_len < ASSESS_MIN:
            results.append(CheckResult(chk.key, chk.label, "skip"))
            continue

        value = _reduce(chk, measured, bottom_i)
        flagged = value > chk.limit if chk.compare == "over" else value < chk.limit
        if flagged:
            issue = f"{chk.fault} ({value:.0f} deg). {chk.cue}"
            # Severity as a >= 1 ratio, direction-aware, so the flash can lead with the worst
            # fault whether the check trips going over its limit (drift, swing) or under it
            # (legs bending, heels lifting).
            if chk.compare == "over":
                severity = value / chk.limit if chk.limit else 0.0
            else:
                severity = chk.limit / value if value else 0.0
            results.append(CheckResult(chk.key, chk.label, "flag", round(value, 1), issue, chk.fault, severity))
        else:
            results.append(CheckResult(chk.key, chk.label, "ok", round(value, 1)))
    return results


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

    # --- PARTIALS (find_partials). The attempts find_reps deliberately drops, now
    #     surfaced as "bad reps" so the lifter learns WHY a rep didn't count. Detection
    #     stays permissive-but-not-noisy: a real half-rep is caught, dead-zone jitter isn't.
    # Under-extension: curled deep (50), came back up but only to 120 - short of the 150
    # that closes a rep. find_reps counts 0, find_partials records one and says why.
    ue = find_partials([170.0, 50.0, 120.0])
    results.append(_check("under-extension -> 1 partial", len(ue), 1))
    results.append(_check("under-extension -> tagged", ue[0].reason if ue else None, "under_extension"))

    # A curl that the clip cuts off mid-lift (no recovery seen) is NOT flagged - it might
    # just be a truncated recording, exactly what find_reps assumes.
    results.append(_check("clip ends mid-dip -> 0 partials", len(find_partials([170.0, 130.0, 40.0])), 0))

    # Under-contraction: a real 55-degree dip that never reaches the bend gate. Invisible
    # to find_reps; recorded here.
    uc = find_partials([170.0, 140.0, 115.0, 140.0, 170.0])
    results.append(_check("under-contraction -> 1 partial", len(uc), 1))
    results.append(_check("under-contraction -> tagged", uc[0].reason if uc else None, "under_contraction"))

    # A clean rep is a COUNTED rep, never a partial - the demo sets must stay unflagged.
    results.append(_check("clean rep -> 0 partials", len(find_partials(_clean_rep())), 0))
    # Dead-zone noise (the same signal as test 2) is below the amplitude gate: not a rep,
    # not a bad rep, nothing.
    results.append(_check("dead-zone noise -> 0 partials", len(find_partials(noise)), 0))
    # Mixed: one clean rep, then a curl that comes back up short (to 120) instead of
    # straightening. 1 counted + 1 bad, and they don't step on each other.
    mixed_p = _clean_rep() + [130.0, 40.0, 120.0]
    results.append(_check("clean + short-return -> 1 counted", len(find_reps(mixed_p)), 1))
    results.append(_check("clean + short-return -> 1 partial", len(find_partials(mixed_p)), 1))

    # --- GRADING layer (form_feedback). Detection above, judgement here. ------
    # A concrete exercise supplies the thresholds now; clone the shipped curl with
    # an explicit tempo so these tests don't move if that default is ever calibrated.
    from dataclasses import replace
    from exercises import BICEP_CURL
    ex = replace(BICEP_CURL, tempo_min_s=1.0)

    # A deep rep at a controlled pace has nothing to say.
    clean_t = [i * 0.25 for i in range(9)]            # the rep spans ~1.25 s
    g = form_feedback(find_reps(_clean_rep()), clean_t, ex)
    results.append(_check("clean rep -> full, no issues", (g[0].full, g[0].tags), (True, [])))
    results.append(_check("clean rep -> clean flash", g[0].flash, "Clean - full range, controlled."))

    # A 92-degree rep IS a rep (detection is permissive) but grades shallow.
    partial_t = [i * 0.7 for i in range(5)]
    g = form_feedback(find_reps([170.0, 130.0, 92.0, 130.0, 170.0]), partial_t, ex)
    results.append(_check("92-degree rep -> not full", g[0].full, False))
    results.append(_check("92-degree rep -> tagged shallow", "shallow" in g[0].tags, True))
    results.append(_check("shallow rep -> shallow flash", g[0].flash, "Shallow - short of full depth"))

    # The same deep rep, crammed into 0.4 s, is a swing.
    fast_t = [i * 0.05 for i in range(9)]
    g = form_feedback(find_reps(_clean_rep()), fast_t, ex)
    results.append(_check("rushed rep -> tagged rushed", "rushed" in g[0].tags, True))
    results.append(_check("rushed rep -> rushed flash", g[0].flash, "Rushed - control the lowering"))

    # A bad rep (reason set) grades not-full and carries ONLY its reason tag - no shallow/
    # rushed noise on top - so the table and the flash say one clear thing.
    bad = form_feedback([Rep(0, 2, 120.0, reason="under_contraction")], [0.0, 0.5, 1.0], ex)
    results.append(_check("bad rep -> not full", bad[0].full, False))
    results.append(_check("bad rep -> reason tag only", bad[0].tags, ["under_contraction"]))
    results.append(_check("under-contraction -> not-counted flash", bad[0].flash,
                          "Not counted - didn't bend far enough"))
    bad_ue = form_feedback([Rep(0, 2, 60.0, reason="under_extension")], [0.0, 0.5, 1.0], ex)
    results.append(_check("under-extension -> not-counted flash", bad_ue[0].flash,
                          "Not counted - didn't fully extend"))

    # --- FORM CHECKS (evaluate_checks). Synthetic landmark maps, no video needed. ---
    # These are the spec for the "look at the whole body" layer the same way the
    # cases above are the spec for detection: build a rep out of hand-placed joints
    # and assert the check fires, stays quiet, or bows out when a limb is unseen.
    from types import SimpleNamespace
    from exercises import BICEP_CURL

    def S(pts):
        """A stand-in Sample - only .landmarks is read here. pts maps a BlazePose
        index to (x, y) or (x, y, visibility); visibility defaults to fully seen."""
        return SimpleNamespace(
            landmarks={i: (p[0], p[1], p[2] if len(p) > 2 else 1.0) for i, p in pts.items()}
        )

    rep5 = Rep(0, 4, 50.0)                       # a 5-frame rep, deepest in the middle
    ang5 = [160.0, 120.0, 50.0, 120.0, 160.0]
    keys = lambda res: {r.key: r for r in res}

    # A clean curl: upper arm hanging vertical, torso still, legs straight and planted.
    clean_lm = {11: (100, 100), 13: (100, 200), 23: (110, 300), 25: (110, 400), 27: (110, 500)}
    good = keys(evaluate_checks(rep5, [S(clean_lm)] * 5, ang5, "left", BICEP_CURL))
    results.append(_check("clean form -> elbow ok", good["elbow"].status, "ok"))
    results.append(_check("clean form -> swing ok", good["swing"].status, "ok"))
    results.append(_check("clean form -> legs ok", good["legs"].status, "ok"))

    # Elbow drifts forward through the middle of the rep: the upper arm tilts off
    # vertical, and the worst tilt crosses the limit.
    drift = [S({**clean_lm, 13: (215, 200)}) if i in (1, 2, 3) else S(clean_lm) for i in range(5)]
    results.append(_check("elbow drift -> flag",
                          keys(evaluate_checks(rep5, drift, ang5, "left", BICEP_CURL))["elbow"].status, "flag"))

    # Torso rocks back and forth (shoulder swings out and back): the swing RANGE, not
    # any single lean, is what trips it - and the arm stays clean underneath.
    sway = [S({**clean_lm, 11: (100 + 35 * abs(2 - i), 100)}) for i in range(5)]
    swres = keys(evaluate_checks(rep5, sway, ang5, "left", BICEP_CURL))
    results.append(_check("torso swing -> flag", swres["swing"].status, "flag"))
    results.append(_check("torso swing, elbow still ok", swres["elbow"].status, "ok"))

    # Legs cropped out of frame (knee/ankle below CONF_MIN): not assessed, never a
    # penalty - and the arm, still in frame, is judged as normal.
    cropped = {11: (100, 100), 13: (100, 200), 23: (110, 300),
               25: (110, 400, 0.1), 27: (110, 500, 0.1)}
    cres = keys(evaluate_checks(rep5, [S(cropped)] * 5, ang5, "left", BICEP_CURL))
    results.append(_check("legs cropped -> not assessed", cres["legs"].status, "skip"))
    results.append(_check("legs cropped, arm still assessed", cres["elbow"].status, "ok"))

    # The flag reaches the RepGrade: posture leads the issues, so it's what the video
    # flashes and what coaching fixes first.
    graded = form_feedback([rep5], [i * 0.4 for i in range(5)], BICEP_CURL,
                           [evaluate_checks(rep5, drift, ang5, "left", BICEP_CURL)])
    results.append(_check("form flag -> tagged on the rep", "elbow" in graded[0].tags, True))
    results.append(_check("form flag -> fault flash", graded[0].flash, "Elbow drifting forward"))

    # Multiple faults on ONE rep: the flash lists the top 3, worst posture first (by severity),
    # then tempo. The full detail still goes to issues/the coach note; the flash is the glance.
    multi = [[
        CheckResult("elbow", "Elbow pinned", "flag", 60.0, "e", "Elbow drifting forward", 1.5),
        CheckResult("swing", "No body swing", "flag", 28.0, "s", "Swinging the torso", 3.1),
    ]]
    gm = form_feedback([Rep(0, 4, 40.0)], [0.0, 0.1, 0.2, 0.3, 0.4], ex, multi)   # full but rushed
    results.append(_check("multi-fault -> top-3, worst posture first",
                          gm[0].flash, "Swinging the torso, elbow drifting forward, and rushed"))
    two = [[CheckResult("swing", "No body swing", "flag", 20.0, "s", "Swinging the torso", 2.2)]]
    g2 = form_feedback([Rep(0, 4, 40.0)], [0.0, 0.05, 0.1, 0.15, 0.2], ex, two)   # swing + rushed
    results.append(_check("two faults -> 'A and B' (no Oxford comma)",
                          g2[0].flash, "Swinging the torso and rushed"))
    gue = form_feedback([Rep(0, 2, 60.0, reason="under_extension")], [0.0, 0.5, 1.0], ex,
                        [[CheckResult("swing", "No body swing", "flag", 15.0, "s", "Swinging the torso", 1.7)]])
    results.append(_check("bad rep + posture -> reason leads the flash",
                          gue[0].flash, "Not counted - didn't fully extend and swinging the torso"))

    return all(results)


if __name__ == "__main__":
    print("reps.py")
    if run_tests():
        print("all passed")
    else:
        print("\nnot yet. Read the failing case - it is telling you something.")