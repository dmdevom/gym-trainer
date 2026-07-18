"""
The coaching layer, optionally written by an LLM - but never depending on one.

Everything numeric about a set is measured upstream and is deterministic: the rep
count, each rep's depth, its tempo, the faults reps.py tagged. That work does NOT
happen here and must never be second-guessed here. What DOES happen here is the
soft part - which fault to fix first, how to phrase the fix, a plain-language read
of how the set went. That is interpretation, and an LLM does it better than a fixed
rule table (analyze.coaching()).

So this module takes the measured summary and asks an LLM to write the coaching, in
the SAME shape analyze.coaching() returns, so render.py's end card, the browser
card, and the CLI all consume it without caring who wrote it.

The contract every caller leans on: generate() returns a valid coaching dict, or
None. None on ANY failure - no key, package missing, API error, timeout, refusal, a
reply that doesn't fit the schema, or simply the feature switched off. The caller
falls back to analyze.coaching(), the offline rules that shipped before this file
existed. The LLM can therefore only ever ADD quality; it cannot break the app,
because "broken" collapses to None and None means "use the rules we already had."

Two things the model is NOT trusted with, on purpose:
  - It never re-states a number. The counts and angles are burned into the video;
    a coach paragraph that disagrees with them makes the whole app look wrong. The
    system prompt forbids it and we hand back only prose fields.
  - It never writes the standing cues. `keep_in_mind` and `muscle` are the
    exercise's own reference text (exercises.py); we inject them after the call so
    they can't drift, and so we don't pay tokens to regenerate a constant.

We reach the model through OpenRouter (one key, the whole catalogue) using the
OpenAI-compatible wire format - so the SDK is `openai`, just pointed at OpenRouter's
base URL. A public Space runs on our key, so the default is a cheap, reliable model;
the numbers are already measured, the model only writes three short prose fields.

Config, all via env so the Space/Railway secret is the only switch:
  OPENROUTER_API_KEY      the key. Absent -> feature off, rules used.
  LLM_COACH               force on/off (1/0). Default: on iff a key is present.
  LLM_MODEL               override the model (vendor/model). Default
                          openai/gpt-4.1-nano (a fast, non-reasoning model - see
                          the DEFAULT_MODEL note on why reasoning models time out).
  OPENROUTER_BASE_URL     override the API base (rarely needed).
  LLM_MAX_CALLS_PER_HOUR  cheap abuse cap for a public endpoint on our key.
  LLM_LOG_FILE            local dev only: trace each call's full request+response
                          (pretty JSON). Set a file path to append to a file, or
                          "stderr"/"stdout"/"-" to print on the terminal - handy under
                          `uvicorn --reload`. Unset -> nothing written. The key is
                          never logged - only the messages, reply, and timing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import List, Optional, Sequence

from pydantic import BaseModel

# The API dep is optional. If it isn't installed the whole feature is simply off -
# the guarded import is the first rung of the fallback ladder, not an error.
try:
    import openai
except ImportError:  # pragma: no cover - exercised only where the dep is absent
    openai = None  # type: ignore[assignment]

# uvicorn owns logging in the deployed app; borrow its logger so a fallback is
# visible in the Space logs (the one window in). Under the CLI it just no-ops up to
# root at WARNING, which is fine - the fallback is silent-by-design either way.
log = logging.getLogger("uvicorn.error")

# OpenRouter model id (vendor/model). A public Space runs on our key, so the default
# is a cheap, fast model - plenty for a 3-field coaching blurb, and it must answer
# INSIDE the _TIMEOUT_S window this call sits behind. Do NOT use a reasoning model
# here: gpt-5-nano burns the entire token budget on hidden reasoning (1500+ tokens ->
# empty reply) and takes ~28s, so it always times out to the rules.
#
# Chosen by A/B over five set profiles (clean / depth-fade / rushed / posture / mixed),
# 3 runs each, scoring whether `focus` picks the right priority (safety > depth > tempo
# > load): gemini-2.5-flash-lite went 15/15 JSON, ~1.2s, and nailed the priority on all
# five - including "clean set -> progressive overload" with no invented tempo fault.
# mistral-nemo (~2s, cheaper) was solid but mis-prioritised the mixed set (led with depth,
# not the posture fault). gpt-4.1-nano fixates on tempo for a clean set AND returned a
# provider 400 mid-test, so it's out as the default. Pin any of them with LLM_MODEL.
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
_BASE_URL = "https://openrouter.ai/api/v1"
_TIMEOUT_S = 15.0
_MAX_TOKENS = 1500

# The client is built once and reused. Timeout and retries are deliberately tight:
# this call sits in the request path behind a progress bar, so it must fail FAST to
# the rules rather than hang. One retry rides out a transient blip; beyond that the
# rules answer (which is good) is better than a spinning bar.
_client = None  # type: ignore[var-annotated]
_client_lock = threading.Lock()

# Rolling-window rate cap. annotate_video runs in a threadpool worker, so guard the
# shared window with a lock. Over the cap, requests silently take the rules path -
# the fallback machinery already handles "no LLM answer" for free.
_calls: List[float] = []
_calls_lock = threading.Lock()


class _Coaching(BaseModel):
    """Exactly the prose we trust the model to write. Numbers stay out; the
    standing cues (keep_in_mind, muscle) are injected from the Exercise afterward.
    messages.parse() validates the reply against this, so a malformed answer raises
    and we fall back instead of shipping half a card."""

    focus: str
    next_session: List[str]
    session_story: str
    mental_cue: str = ""   # additive: one physical cue for the next set. Defaulted so a
                           # model that omits it still yields a usable card, not a fallback.


_SYSTEM = """\
You are a strength coach reviewing ONE set of a single exercise for one lifter.

You are given the MEASURED results of that set as JSON. A pose-estimation pipeline
has already counted the reps and computed each rep's depth, tempo, and faults. Your
job is only to turn those numbers into coaching the lifter can act on.

Hard rules:
- Interpret the numbers; never invent or restate different ones. Do NOT state any
  rep count, angle, percentage, or duration that is not already in the data. The
  lifter is watching a video with these exact numbers drawn on it - contradicting
  them makes the app look broken.
- Pick ONE focus: the single most valuable thing to fix next. Prioritise in this
  order: safety/posture, then range of motion (depth), then tempo, then adding
  load. If every rep was full and controlled, the focus is progressive overload.
- Each rep's "faults" list holds short tags. "shallow"/"rushed" are depth/tempo;
  any other tag is a form/posture fault, and "form_checks" (when present) names what
  each watched and how many reps it caught - a flagged one is a safety/posture issue
  and outranks depth and tempo. Only assessed checks are sent; don't coach on a check
  that isn't there (it just wasn't in frame).
- Do NOT invent faults. A rep with no "shallow"/"rushed"/form tag was clean - full
  range and controlled. Natural variation in the raw numbers is NOT a fault: a rep at
  or above min_controlled_tempo_s is controlled even if it's a little faster than
  another, and a rep marked full_range covered the range even if a touch shallower than
  its neighbour - never flag these or make them the focus. The set carries a boolean
  "all_reps_clean": when it is true, EVERY rep already passed depth, tempo AND form, so
  do not raise tempo or depth at all - the focus MUST be progressive overload (add load
  or a rep), and session_story should read as a clean set. Coach a fault only when a rep
  is actually tagged with it.
- A "trends" object is also given: the ARC of the set, pre-computed from the same
  per-rep numbers so you interpret a conclusion instead of aggregating rows yourself.
  depth_arc and tempo_arc are "up" / "steady" / "down" (depth "down" = getting shallower
  as they tire; tempo "down" = reps getting shorter, i.e. rushing later in the set).
  reps_by_fault maps each fault tag to the rep NUMBERS that had it; deepest_rep,
  shallowest_rep and clean_reps are rep numbers. You MAY name these specific rep numbers
  ("your last two reps were the rushed ones") - they are given to you - but you still
  must never state a count, angle, percentage, or duration.
- Be specific and grounded in THIS set. "Control the last two reps, they were the
  rushed ones" beats "go slower". Reuse the exercise's own depth_cue / tempo_cue
  wording where it fits.
- Talk to the lifter directly ("you"), plainly. No emoji, no hype, no preamble.

Return:
- focus: 3-6 words naming the one thing to work on next.
- next_session: 1 to 3 short, actionable sentences for their next session.
- session_story: 2-3 sentences describing how THIS set actually went - the arc
  across the reps (e.g. strong then fading), what stood out. Honest, conversational.
- mental_cue: 3-6 words, ONE physical thing to hold in mind during the next set - a
  cue they can feel, not a topic. "Pin the elbow, own the lower", not "form".

Respond with ONLY a JSON object (no markdown, no code fence) with exactly these keys:
  "focus" (string), "next_session" (array of strings), "session_story" (string),
  "mental_cue" (string).
"""


def _model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def is_enabled() -> bool:
    """On only when we could actually call the API. LLM_COACH forces the decision;
    otherwise the presence of a key is the switch, so the default deployment posture
    is offline until a secret is set."""
    if openai is None:
        return False
    flag = os.environ.get("LLM_COACH", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def info() -> dict:
    """What /health reports, so the Space can tell at a glance whether the LLM path
    is live and which model it would use."""
    on = is_enabled()
    return {"enabled": on, "model": _model() if on else None}


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                # OpenRouter speaks the OpenAI wire format, so the openai SDK IS the
                # client - just pointed at OpenRouter's base URL with our key.
                _client = openai.OpenAI(
                    base_url=os.environ.get("OPENROUTER_BASE_URL", _BASE_URL),
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    timeout=_TIMEOUT_S,
                    max_retries=1,
                )
    return _client


def _rate_ok() -> bool:
    """True if we're under the hourly cap (and records this call). A public Space on
    our key is the threat model; this keeps a bad afternoon from becoming a bill."""
    limit = int(os.environ.get("LLM_MAX_CALLS_PER_HOUR", "60"))
    if limit <= 0:
        return False
    now = time.time()
    with _calls_lock:
        cutoff = now - 3600
        _calls[:] = [t for t in _calls if t > cutoff]
        if len(_calls) >= limit:
            return False
        _calls.append(now)
        return True


def _strip_fence(text: str) -> str:
    """json_object mode should return bare JSON, but some cheap/free models still wrap
    it in a ```json fence. Peel one leading/trailing fence so model_validate_json sees
    clean JSON; a well-behaved reply passes straight through untouched."""
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _debug_log(record: dict) -> None:
    """Local-only call tracing. When LLM_LOG_FILE is set, write one pretty-printed JSON
    record - the full request and response of a single call - so the exact request and
    reply to/from OpenRouter can be eyeballed while developing. The value is either a
    file path (appended to) or a terminal sink: "stderr"/"stdout"/"-" prints straight
    to the console, which is what you want under `uvicorn --reload`. A direct stream
    write (not the logger) so it shows regardless of uvicorn's log config. Off unless
    the env var is set, and it NEVER writes the API key: only the messages, the reply,
    and timing. Best-effort - any write error is swallowed so tracing can never break
    the coaching path."""
    dest = os.environ.get("LLM_LOG_FILE", "").strip()
    if not dest:
        return
    text = json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n\n"
    try:
        if dest in ("stderr", "stdout", "-"):
            stream = sys.stdout if dest in ("stdout", "-") else sys.stderr
            stream.write(text)
            stream.flush()
        else:
            with open(dest, "a", encoding="utf-8") as fh:
                fh.write(text)
    except Exception:  # pragma: no cover - tracing must never break coaching
        log.debug("llm debug log write failed", exc_info=True)


def _derive_features(per_rep: Sequence[dict]) -> dict:
    """Cheap, deterministic trend features distilled from the per-rep numbers, so the
    model interprets a conclusion instead of re-aggregating rows in its head. Because
    these are computed here, they can't be hallucinated - the rep numbers named below
    are given to the model, never invented by it. Additive: purely more input to reason
    over, it changes nothing the pipeline measures."""
    n = len(per_rep)
    if n < 1:
        return {}
    depth = [r["depth_pct"] for r in per_rep]
    dur = [r["duration_s"] for r in per_rep]

    def arc(vals: Sequence[float], flat: float) -> str:
        # First third vs last third. Below `flat` the drift is noise, not a trend, and
        # a short set (< 4 reps) has no arc worth naming.
        if n < 4:
            return "too_few_to_tell"
        k = max(1, n // 3)
        delta = sum(vals[-k:]) / k - sum(vals[:k]) / k
        return "steady" if abs(delta) < flat else ("up" if delta > 0 else "down")

    fault_reps: dict = {}
    for r in per_rep:
        for tag in r["tags"]:
            fault_reps.setdefault(tag, []).append(r["number"])

    return {
        "depth_arc": arc(depth, 8.0),     # "down" = reps getting shallower as they tire
        "tempo_arc": arc(dur, 0.3),       # "down" = reps getting shorter = rushing later
        "deepest_rep": max(per_rep, key=lambda r: r["depth_pct"])["number"],
        "shallowest_rep": min(per_rep, key=lambda r: r["depth_pct"])["number"],
        "reps_by_fault": fault_reps,      # {"shallow": [6, 7, 8], "rushed": [8]}
        "clean_reps": [r["number"] for r in per_rep if not r["tags"]],
    }


def _payload(exercise, per_rep: Sequence[dict], reps: int, full_reps: int, verdict: str,
             form_checks: Optional[Sequence[dict]] = None) -> dict:
    """The few KB of NUMBERS the model reasons over. No frames, no landmarks, no
    video - just the graded set. Cheap to send, nothing sensitive leaves, and it's
    the same data the video already shows the user."""
    # A single categorical flag the model can't misread: no rep carries any fault tag.
    # A weak model will otherwise eyeball the raw per-rep durations/angles and "coach"
    # natural variation that the grader already cleared - the flag pins it to the truth.
    all_reps_clean = all(not r["tags"] for r in per_rep)
    payload = {
        "exercise": {
            "name": exercise.name,
            "measured_joint": exercise.vertex_name,
            "depth_cue": exercise.depth_cue,
            "tempo_cue": exercise.tempo_cue,
        },
        "set": {
            "reps": reps,
            "full_range_reps": full_reps,
            "all_reps_clean": all_reps_clean,
            "verdict": verdict,
        },
        "targets": {
            "full_range_angle_deg": exercise.full_rom,
            "min_controlled_tempo_s": exercise.tempo_min_s,
        },
        "per_rep": [
            {
                "number": r["number"],
                "depth_pct": r["depth_pct"],
                "deepest_angle_deg": r["min_angle"],
                "duration_s": r["duration_s"],
                "full_range": r["full"],
                "faults": r["tags"],
            }
            for r in per_rep
        ],
        # The ARC of the set, pre-digested so the model reads a conclusion instead of
        # re-aggregating the rows above - and can't hallucinate it, since we compute it.
        "trends": _derive_features(per_rep),
    }
    # The whole-body form checks, so the per-rep `faults` tags aren't opaque: each
    # names what was watched (elbow drift, torso swing, ...) and how many reps it
    # caught. Only checks we could actually see are sent - a not-assessed one is the
    # camera's fault, not the lifter's, and the model shouldn't coach on it.
    assessed = [f for f in (form_checks or []) if f.get("assessed")]
    if assessed:
        payload["form_checks"] = [
            {"check": f["label"], "flagged_reps": f["flagged"], "assessed_reps": f["assessed"]}
            for f in assessed
        ]
    return payload


def generate(
    exercise,
    per_rep: Sequence[dict],
    reps: int,
    full_reps: int,
    verdict: str,
    form_checks: Optional[Sequence[dict]] = None,
) -> Optional[dict]:
    """Ask the LLM for the coaching. Return a dict shaped exactly like
    analyze.coaching() (plus an additive `session_story`), or None to tell the
    caller to use the offline rules. None is a normal outcome, not an error - it's
    every rung of the fallback ladder collapsed to one return value.

    reps < 1 short-circuits to None on purpose: the no-reps case is almost always a
    camera-angle problem, and analyze.coaching() already has a good, specific
    message for it - no reason to spend a call second-guessing it."""
    if not is_enabled() or reps < 1:
        return None
    if not _rate_ok():
        log.info("llm coaching over hourly cap; using rules")
        return None

    # Build the request once, so the exact bytes we send can also be traced. The
    # system prompt is a constant; the user turn is the measured numbers, nothing else.
    payload = _payload(exercise, per_rep, reps, full_reps, verdict, form_checks)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": _model(),
        "request": {"system": _SYSTEM, "user": payload},
    }

    t0 = time.time()
    try:
        # JSON mode (not native structured outputs): near-universal across cheap/free
        # OpenRouter models, where strict json_schema often isn't. _Coaching still
        # validates the reply, and the except -> None -> rules ladder catches anything
        # malformed - so a wobbly small model degrades to the rules, never a broken card.
        resp = _get_client().chat.completions.create(
            model=_model(),
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ],
        )
        # Capture the whole reply verbatim for the trace: id, the model that actually
        # served it (OpenRouter may route), the content, finish_reason, token usage.
        try:
            record["response"] = resp.model_dump()
        except Exception:  # pragma: no cover - defensive; SDK objects normally dump
            record["response"] = str(resp)

        raw = resp.choices[0].message.content
        if not raw:                         # refusal / truncation -> no usable answer
            return None
        c = _Coaching.model_validate_json(_strip_fence(raw))   # bad shape -> raises -> caught -> rules

        focus = c.focus.strip()
        next_session = [s.strip() for s in c.next_session if s.strip()]
        if not focus or not next_session:   # an empty card is worse than the rules
            return None

        # Prose from the model; standing cues from the exercise (authoritative,
        # never regenerated). Same 4 keys the UI/CLI/end-card read, plus the extras
        # (session_story, mental_cue) - additive, so a consumer that ignores them is fine.
        return {
            "focus": focus,
            "next_session": next_session,
            "keep_in_mind": list(exercise.tips),
            "muscle": exercise.muscle,
            "session_story": c.session_story.strip(),
            "mental_cue": c.mental_cue.strip(),
        }
    except Exception as e:
        # ANY failure - bad key, network, rate limit, refusal, schema mismatch, an
        # SDK surprise - is the same decision: quietly use the rules. exc_info so the
        # reason is in the Space logs without ever reaching the user.
        record["error"] = repr(e)
        log.warning("llm coaching failed; using rule-based fallback", exc_info=True)
        return None
    finally:
        # Local-only: when LLM_LOG_FILE is set, persist THIS call's request+response
        # (success, empty reply, or error) for inspection. A no-op otherwise.
        record["latency_s"] = round(time.time() - t0, 2)
        _debug_log(record)
