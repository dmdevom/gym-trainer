"""
Minimal, fail-open usage logging - "did anyone use it, and what did the coach say?"

Why this exists: the app is stateless by design (no accounts, no history) and the prod
container runs on ephemeral disk, so nothing records whether the deployed app was ever
used or what the LLM coach actually wrote. This module keeps a compact, append-only
JSONL trail - ONE line per event - in a directory that survives restarts (a mounted
volume in prod). Text only: never a video, never landmarks, never the raw angle series.

The posture is lifted straight from llm_coach.py, on purpose:
  - OFF unless configured. No TELEMETRY_DIR -> record() is a no-op. Nothing to install,
    nothing to set up, for local dev or for anyone running the app without opting in.
  - Best-effort, never fatal. Every public call is wrapped so a telemetry failure can
    never raise into - or slow - the analysis path. A broken disk loses telemetry, not
    the app. Same spirit as the coaching fallback: the feature can only ADD, never break.
  - Off the request path. record() only appends to an in-memory buffer; a background
    daemon thread does the file IO on a timer. The worker that calls record() never
    touches the disk, so a slow/full volume can't stall a request.

Config, all via env, all optional:
  TELEMETRY_DIR            enable + where to write (e.g. /data, a mounted volume).
                           Unset -> feature off.
  TELEMETRY_RETENTION_DAYS delete event files older than this many days. Default 30.
  TELEMETRY_SALT           salt for the daily IP hash. Set it to keep the hash stable
                           across restarts; unset -> a per-process random salt (fine -
                           the hashes just won't line up across a redeploy).

Read the trail back with the ADMIN_TOKEN-guarded GET /admin/telemetry endpoint, or
straight off the volume (e.g. `railway run cat /data/events-*.jsonl`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

# uvicorn owns logging in the deployed app; borrow its logger so a telemetry hiccup is
# visible in the server logs (the one window in), exactly like llm_coach does.
log = logging.getLogger("uvicorn.error")

FLUSH_EVERY_S = 30.0      # the writer thread drains the buffer at least this often
MAX_BUF = 100             # ...or sooner, the moment the buffer reaches this many events
_MAX_FIELD_CHARS = 2000   # safety clip so one runaway string field can't bloat a record

# The buffer and the two locks. record() only ever touches _buf under _lock (fast); the
# file IO happens under _write_lock so a disk write never blocks a caller appending.
_buf: List[dict] = []
_lock = threading.Lock()
_write_lock = threading.Lock()
_flush_event = threading.Event()   # set by record() at MAX_BUF to wake the writer early
_stop = threading.Event()
_thread: Optional[threading.Thread] = None

# Salt for the daily IP hash. A configured secret keeps within-day hashes stable across
# restarts; absent, a per-process random salt is fine for a coarse "unique-ish" signal.
_SALT = os.environ.get("TELEMETRY_SALT") or os.urandom(16).hex()


def _dir() -> Optional[Path]:
    d = os.environ.get("TELEMETRY_DIR", "").strip()
    return Path(d) if d else None


def enabled() -> bool:
    """On only when a destination is set - the presence of TELEMETRY_DIR is the switch,
    so the default posture is off until someone opts in (mirrors llm_coach.is_enabled)."""
    return _dir() is not None


def ip_hash(ip: Optional[str]) -> str:
    """A short, salted, per-day digest of the client IP. The raw IP is NEVER stored; this
    is only enough to gauge 'roughly how many distinct visitors today' without holding PII.
    The day is folded in so the same IP hashes differently tomorrow (no cross-day tracking)."""
    if not ip:
        return ""
    day = time.strftime("%Y%m%d")
    return hashlib.sha256(f"{_SALT}:{day}:{ip}".encode()).hexdigest()[:12]


def _clip(v):
    """Bound a single value's size so one runaway field can't bloat a record. Strings are
    truncated; everything else passes through - rep_notes are already length-capped upstream
    by llm_coach's validators, and beacon props are size-bounded at the endpoint."""
    if isinstance(v, str) and len(v) > _MAX_FIELD_CHARS:
        return v[:_MAX_FIELD_CHARS] + "…"
    return v


def record(event: str, **fields) -> None:
    """Append one event to the in-memory buffer. Fast, non-blocking, and it NEVER raises -
    a telemetry problem must never touch the caller. No-op unless TELEMETRY_DIR is set.

    `None`-valued fields are dropped so records stay compact and columns stay meaningful."""
    if not enabled():
        return
    try:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event}
        for k, v in fields.items():
            if v is not None:
                rec[k] = _clip(v)
        with _lock:
            _buf.append(rec)
            full = len(_buf) >= MAX_BUF
        if full:
            _flush_event.set()   # wake the writer early; the writer does the IO, not us
    except Exception:  # pragma: no cover - telemetry must never break the caller
        log.debug("telemetry record failed", exc_info=True)


def _retention_days() -> int:
    try:
        return int(os.environ.get("TELEMETRY_RETENTION_DAYS", "30"))
    except ValueError:
        return 30


def _prune(dirpath: Path) -> None:
    """Delete event files past the retention window so the trail stays bounded. Best-effort;
    a retention of 0 (or less) disables pruning."""
    days = _retention_days()
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    for f in dirpath.glob("events-*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        except OSError:
            pass


def _flush() -> None:
    """Drain the buffer to today's JSONL file. Snapshot-and-clear under the buffer lock, then
    do the file IO under a separate write lock so record() is never blocked on disk. Best-effort:
    a write failure drops that batch (fail-open) rather than growing memory without bound."""
    with _lock:
        if not _buf:
            return
        batch = _buf[:]
        _buf.clear()
    dirpath = _dir()
    if dirpath is None:
        return
    try:
        with _write_lock:
            dirpath.mkdir(parents=True, exist_ok=True)
            path = dirpath / f"events-{time.strftime('%Y%m%d')}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for rec in batch:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            _prune(dirpath)
    except Exception:  # pragma: no cover - disk problems must not crash the writer thread
        log.warning("telemetry flush failed; dropped %d event(s)", len(batch), exc_info=True)


def read_recent(limit: int = 200) -> List[dict]:
    """For the admin endpoint: flush pending events, then return the most recent `limit`
    records in chronological order (oldest first). Reads newest files first and stops once
    it has enough, so it never slurps the whole trail. Best-effort -> [] if off/unreadable."""
    _flush()
    dirpath = _dir()
    if dirpath is None:
        return []
    try:
        files = sorted(dirpath.glob("events-*.jsonl"))   # oldest -> newest by name
    except OSError:
        return []
    collected: List[str] = []                            # newest-first while gathering
    for f in reversed(files):
        try:
            file_lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for ln in reversed(file_lines):
            if ln.strip():
                collected.append(ln)
        if len(collected) >= limit:
            break
    out: List[dict] = []
    for ln in reversed(collected[:limit]):               # back to chronological order
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _run() -> None:
    while not _stop.is_set():
        _flush_event.wait(FLUSH_EVERY_S)
        _flush_event.clear()
        _flush()


def start() -> None:
    """Begin the background flush thread. Called once from the FastAPI lifespan; a no-op
    when telemetry is off or already started."""
    global _thread
    if not enabled() or _thread is not None:
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="telemetry-flush", daemon=True)
    _thread.start()


def stop() -> None:
    """Stop the flush thread and write anything still buffered - so a clean shutdown doesn't
    drop the last few events."""
    global _thread
    if _thread is None:
        _flush()
        return
    _stop.set()
    _flush_event.set()
    _thread.join(timeout=5)
    _thread = None
    _flush()


if __name__ == "__main__":
    # Offline self-test, in the house style (python telemetry.py -> "all passed"). No
    # network, no server: point TELEMETRY_DIR at a temp dir, record a couple of events,
    # flush, and assert what landed - including that the raw IP never does.
    import tempfile

    _ok = True

    def _check(name, cond):
        global _ok
        _ok = _ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    # Disabled by default: no dir -> record() is a silent no-op and nothing is written.
    os.environ.pop("TELEMETRY_DIR", None)
    record("page_view", cid="x")
    _check("off by default -> not enabled", not enabled())

    with tempfile.TemporaryDirectory() as d:
        os.environ["TELEMETRY_DIR"] = d
        os.environ["TELEMETRY_SALT"] = "test-salt"
        _check("enabled once TELEMETRY_DIR set", enabled())

        iph = ip_hash("203.0.113.7")
        _check("ip_hash is short", len(iph) == 12)
        _check("ip_hash hides the raw ip", "203.0.113.7" not in iph)
        _check("ip_hash stable within a day", iph == ip_hash("203.0.113.7"))
        _check("ip_hash of None is empty", ip_hash(None) == "")

        record("page_view", src="web", cid="c1", ip_hash=iph, dropme=None)
        record("analysis_complete", src="api", exercise="squat", reps=5,
               focus="Reach full depth", session_story="Strong then fading.")
        _flush()

        lines = [json.loads(l) for l in
                 Path(d, f"events-{time.strftime('%Y%m%d')}.jsonl").read_text().splitlines()]
        _check("both events written", len(lines) == 2)
        _check("envelope has ts+event", all("ts" in r and "event" in r for r in lines))
        _check("None fields dropped", "dropme" not in lines[0])
        _check("payload preserved", lines[1]["focus"] == "Reach full depth")

        got = read_recent(10)
        _check("read_recent returns chronological", [r["event"] for r in got] ==
               ["page_view", "analysis_complete"])

    os.environ.pop("TELEMETRY_DIR", None)
    print("telemetry.py")
    print("all passed" if _ok else "\nnot yet - a telemetry decision is wrong.")
