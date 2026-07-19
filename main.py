import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

# Local dev convenience: load a gitignored .env (OPENROUTER_API_KEY, LLM_LOG_FILE, …)
# BEFORE anything reads the environment, so `uvicorn main:app` picks the key up with no
# exports, in any terminal, across restarts. Dev-only and optional: python-dotenv may be
# absent (a minimal prod image) and there may be no .env - both are fine. Real env vars
# always win: load_dotenv defaults to override=False, so a Railway secret is never
# clobbered by a stray .env.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # dep not installed -> just use the real environment
    pass

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import llm_coach
import telemetry
from analyzer import CONF_MIN
from backends import get_backend
from exercises import EXERCISES
from render import annotate_video
from video import VideoError

# uvicorn owns logging config, so borrow its logger. A bare getLogger(__name__)
# would silently go nowhere — root sits at WARNING and nobody added a handler.
log = logging.getLogger("uvicorn.error")

# Upload cap. The endpoint reads the whole file into RAM before writing it to
# disk, so this bound doubles as the per-upload memory ceiling: raise MAX_UPLOAD_MB
# for longer clips, but mind the container's memory - N concurrent uploads each
# buffer up to this much (the render CPU is separately capped by RENDER_SEMAPHORE).
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_VIDEO_BYTES = MAX_UPLOAD_MB * 1024 * 1024

TEMPLATES = Path(__file__).parent / "templates"

# Rendered videos land here and are served back by token. A temp dir, because the
# result is disposable: no accounts, no history, nothing to persist (the Phase 3
# guardrail). On a restart they're gone, which for an upload->analyse->watch tool
# is exactly right.
RESULTS_DIR = Path(tempfile.mkdtemp(prefix="gymtrainer-"))

# In-memory job table. The upload returns a token the instant the file lands, and
# the page polls /progress while the analysis runs in the background and reports
# here. A dict, not a database, for the same reason RESULTS_DIR is a temp dir: the
# result is disposable and there are no accounts. Pruned so a long-lived server
# process doesn't slowly leak jobs and their rendered webms.
JOBS: dict = {}

# At most this many clips render at once; later uploads wait in line as "Queued".
# MediaPipe + VP8 encoding saturate a core each and the deploy boxes have 2 vCPUs,
# so unbounded concurrency just makes every progress bar crawl - and is a free DoS
# on a public, anonymous endpoint. (The LLM call has its own hourly cap.)
RENDER_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_RENDERS", "2")))


def _prune_jobs(max_age_s: float = 3600, max_jobs: int = 128) -> None:
    now = time.time()
    for tok in list(JOBS):
        if now - JOBS[tok].get("created", now) > max_age_s:
            JOBS.pop(tok, None)
            (RESULTS_DIR / f"{tok}.webm").unlink(missing_ok=True)
    if len(JOBS) > max_jobs:                     # hard cap, oldest first
        for tok in sorted(JOBS, key=lambda t: JOBS[t].get("created", 0))[: len(JOBS) - max_jobs]:
            JOBS.pop(tok, None)
            (RESULTS_DIR / f"{tok}.webm").unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at boot, not on the first request. Also: this is where
    # MediaPipe dumps its absl/EGL noise, so logging afterwards puts our banner
    # last — the one thing you can actually see.
    info = get_backend().info()

    log.info("─" * 52)
    log.info("  pose backend : %s", info["backend"])
    if "model" in info:
        mb = f"  ({info['model_mb']} MB)" if "model_mb" in info else ""
        log.info("  model        : %s%s", info["model"], mb)
    if "model_path" in info:
        log.info("  path         : %s", info["model_path"])
    log.info("  conf floor   : %s", CONF_MIN)
    li = llm_coach.info()
    log.info("  llm coaching : %s", li["model"] if li["enabled"] else "off (rule-based)")
    log.info("  results dir  : %s", RESULTS_DIR)
    # Usage/LLM-response trail. Off unless TELEMETRY_DIR is set; the writer thread lives
    # for the app's lifetime and does a final flush on shutdown so no events are lost.
    telemetry.start()
    log.info("  telemetry    : %s", os.environ.get("TELEMETRY_DIR") or "off")
    log.info("─" * 52)
    yield
    telemetry.stop()


app = FastAPI(title="trAIner", version="0.2.0", lifespan=lifespan)

# The analysis API is intentionally anonymous and uses no cookies. Keep browser
# access origin-scoped while making local Next.js development work out of the box.
# Production deployments can add one or more comma-separated frontend origins.
default_origins = "http://localhost:3000,http://127.0.0.1:3000"
cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index():
    # One static page, served by hand — no Jinja, no StaticFiles mount, no new
    # dependency. The page does the dynamic part in the browser.
    return (TEMPLATES / "index.html").read_text()


@app.get("/health")
def health():
    # Same facts as the boot banner. On the hosted API this is your only window in.
    return {"status": "ok", "conf_min": CONF_MIN, "llm": llm_coach.info(), **get_backend().info()}


@app.get("/exercises")
def exercises():
    # Served to the page's selector so the client's list of movements can't drift
    # from the server's. Thresholds stay here; the client only gets names and tips.
    return {"exercises": [e.brief() for e in EXERCISES.values()]}


def _analysis_fields(summary: dict, ctx: dict) -> dict:
    """Flatten a finished summary into one compact telemetry record: the measured set plus
    the LLM coaching text, and NOTHING heavy - no `series`, no `thresholds`, no media. This
    is the "save the LLM response" record; a long set is still only a few KB."""
    meta = summary.get("meta") or {}
    ex = meta.get("exercise") or {}
    coaching = summary.get("coaching") or {}
    li = llm_coach.info()
    return {
        "src": "api",
        "exercise": ex.get("key"),
        "mode": ctx.get("mode"),
        "ip_hash": ctx.get("ip_hash"),
        "reps": summary.get("reps"),
        "full_reps": summary.get("full_reps"),
        "verdict": summary.get("verdict"),
        "coaching_source": meta.get("coaching_source"),      # llm / rules / llm+rules
        "rep_notes_source": meta.get("rep_notes_source"),
        "llm_model": li.get("model") if li.get("enabled") else None,
        "focus": coaching.get("focus"),
        "session_story": coaching.get("session_story"),
        "mental_cue": coaching.get("mental_cue"),
        "rep_notes": [
            {"number": r.get("number"), "flash": r.get("flash_note"), "coach": r.get("coach_note")}
            for r in (summary.get("per_rep") or [])
        ] or None,
    }


async def _run_job(token: str, src: str, dst: str, exercise_key: str) -> None:
    """Do the slow work off to the side and report progress into JOBS[token].

    The endpoint handed back a token the moment the upload landed; the page polls
    /progress while this runs. annotate_video is sync and CPU-bound (seconds of
    MediaPipe + encoding), so it goes to the threadpool - and its progress callback
    is invoked FROM that worker thread, writing a couple of ints into the job dict,
    which is safe enough under the GIL for a status line. RENDER_SEMAPHORE bounds
    the concurrent renders; a waiting job says "Queued" instead of showing a stuck
    bar. Finalizers go through JOBS.get because _prune_jobs may have evicted the
    job mid-run - losing that result is fine, crashing the task is not."""
    def progress_cb(stage: str, pct) -> None:
        job = JOBS.get(token)
        if job is not None and not job["done"]:
            job["stage"] = stage
            job["pct"] = pct

    def finish(**fields) -> None:
        job = JOBS.get(token)
        if job is not None:
            job.update(fields)

    # Snapshot the request context stashed by the endpoint (mode, hashed IP) up front, so
    # the telemetry record is complete even if _prune_jobs evicts the job mid-render.
    job0 = JOBS.get(token) or {}
    ctx = {"mode": job0.get("mode"), "ip_hash": job0.get("ip_hash")}

    try:
        if RENDER_SEMAPHORE.locked():
            progress_cb("Queued", 0)
        async with RENDER_SEMAPHORE:
            summary = await run_in_threadpool(annotate_video, src, dst, exercise_key, progress_cb)
        summary.pop("video", None)               # a local path; the browser gets a URL
        summary["video_url"] = f"/results/{token}"
        finish(stage="Done", pct=100, done=True, result=summary)
        telemetry.record("analysis_complete", **_analysis_fields(summary, ctx))
    except VideoError as e:
        finish(stage="Error", done=True, error=str(e))
        telemetry.record("analysis_error", src="api", exercise=exercise_key,
                         kind="video", detail=str(e), **ctx)
    except Exception:
        log.exception("analysis failed for token %s", token)
        finish(stage="Error", done=True, error="Analysis failed on this clip. Try another.")
        telemetry.record("analysis_error", src="api", exercise=exercise_key,
                         kind="internal", **ctx)
    finally:
        Path(src).unlink(missing_ok=True)        # keep only the rendered .webm


@app.post("/analyze/video", status_code=202)
async def analyze_video_endpoint(
    request: Request,
    file: UploadFile = File(...),
    exercise: str = Form("bicep_curl"),
    mode: str = Form("unknown"),      # "sample"/"upload"/"record" from the client; a label only
):
    # Phones love to send video as application/octet-stream, so accept on either
    # the content-type OR a known extension rather than rejecting a real clip.
    name = (file.filename or "").lower()
    looks_video = (file.content_type or "").startswith("video/") or name.endswith(
        (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".3gp")
    )
    if not looks_video:
        return JSONResponse(
            status_code=415,
            content={"error": "not_a_video", "detail": f"Got {file.content_type or 'unknown type'}"},
        )

    data = await file.read(MAX_VIDEO_BYTES + 1)
    if len(data) > MAX_VIDEO_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "too_large", "detail": f"Max {MAX_UPLOAD_MB}MB. Trim it or drop the resolution."},
        )

    token = uuid.uuid4().hex
    # src must NOT collide with dst. A recorded clip arrives as .webm, and dst is
    # ALWAYS .webm — so `{token}.webm` for both is the same file, and the writer
    # truncates the upload before the dense pass can re-read it ("Could not open
    # this video"). The "-src" suffix keeps them apart. (An uploaded .mp4 never
    # collided, which is exactly why only the Record path hit this.)
    src = RESULTS_DIR / f"{token}-src{Path(name).suffix or '.mp4'}"
    dst = RESULTS_DIR / f"{token}.webm"
    src.write_bytes(data)

    # Register the job, then return the token straight away (202). The work runs in
    # the background so the page can show a real progress bar instead of a spinner
    # that says nothing for 20 seconds. get_exercise (inside annotate_video) defaults
    # a bad key rather than erroring, so no validation is needed here.
    #
    # Stash the request context (input mode, a hashed IP - never the raw one) on the job
    # so _run_job can attach it to the analysis_complete/error telemetry record; the raw
    # user-agent is kept only for the start event's device signal.
    mode = (mode or "unknown")[:20]
    iph = telemetry.ip_hash(request.client.host if request.client else None)
    ua = request.headers.get("user-agent", "")[:200]
    JOBS[token] = {"stage": "Starting", "pct": 0, "done": False, "error": None,
                   "result": None, "created": time.time(),
                   "mode": mode, "ip_hash": iph, "ua": ua, "exercise": exercise}
    _prune_jobs()
    telemetry.record("analysis_started", src="api", exercise=exercise, mode=mode,
                     ip_hash=iph, ua=ua)
    asyncio.create_task(_run_job(token, str(src), str(dst), exercise))
    return JSONResponse(status_code=202, content={"token": token})


@app.get("/progress/{token}")
def progress(token: str):
    # token is our uuid4 hex; isalnum() rejects any '../' path-traversal attempt.
    if not token.isalnum():
        return JSONResponse(status_code=400, content={"error": "bad_token"})
    job = JOBS.get(token)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": "This job expired - analyse again."},
        )
    out = {"stage": job["stage"], "pct": job["pct"], "done": job["done"], "error": job["error"]}
    if job["done"] and job["result"] is not None:
        out["result"] = job["result"]           # the full summary + video_url, once
    return out


@app.get("/results/{token}")
def results(token: str):
    # token is a uuid4 hex; isalnum() rejects any '../' path-traversal attempt.
    if not token.isalnum():
        return JSONResponse(status_code=400, content={"error": "bad_token"})
    path = RESULTS_DIR / f"{token}.webm"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": "This result has expired — analyse again."},
        )
    return FileResponse(path, media_type="video/webm")


# --- Anonymous client-side usage beacon ------------------------------------------------
# The frontend posts tiny fire-and-forget events here (a page view, which input they chose,
# whether they saw a result). Allowlisted, size-capped, per-visitor rate-limited, and
# best-effort: anything off-shape is silently dropped with a 204, never an error the page
# has to handle. Only persists when TELEMETRY_DIR is set (else telemetry.record no-ops).
_BEACON_EVENTS = {"page_view", "exercise_selected", "mode_changed",
                  "input_ready", "result_viewed", "error_shown"}
_BEACON_MAX_PER_HOUR = int(os.environ.get("BEACON_MAX_PER_HOUR", "600"))
_beacon_calls: dict = {}
_beacon_lock = threading.Lock()


def _beacon_ok(key: str) -> bool:
    """Rolling-hour cap per hashed IP, the same idea as llm_coach._rate_ok - keeps a loop or
    a bad actor from flooding the trail. Fail-open on an empty key (allowed)."""
    if _BEACON_MAX_PER_HOUR <= 0:
        return True
    now = time.time()
    with _beacon_lock:
        if len(_beacon_calls) > 10000:        # pathological cardinality -> reset, don't leak
            _beacon_calls.clear()
        times = [t for t in _beacon_calls.get(key, ()) if t > now - 3600]
        if len(times) >= _BEACON_MAX_PER_HOUR:
            _beacon_calls[key] = times
            return False
        times.append(now)
        _beacon_calls[key] = times
        return True


@app.post("/e", status_code=204)
async def beacon(request: Request):
    # Read a small JSON body, keep only allowlisted events, stamp a hashed IP + truncated UA
    # server-side, and record. Every early return is a 204 so a dropped beacon never surfaces
    # as an error in the browser console.
    try:
        raw = await request.body()
        if len(raw) > 4096:
            return Response(status_code=204)
        data = json.loads(raw or b"{}")
    except Exception:
        return Response(status_code=204)
    if not isinstance(data, dict) or str(data.get("event") or "") not in _BEACON_EVENTS:
        return Response(status_code=204)
    iph = telemetry.ip_hash(request.client.host if request.client else None)
    if not _beacon_ok(iph):
        return Response(status_code=204)
    props = data.get("props")
    if not isinstance(props, dict) or len(json.dumps(props, default=str)) > 1024:
        props = None
    telemetry.record(str(data["event"]), src="web", cid=str(data.get("cid") or "")[:64],
                     ip_hash=iph, ua=request.headers.get("user-agent", "")[:200], props=props)
    return Response(status_code=204)


@app.get("/admin/telemetry")
def admin_telemetry(token: str = "", limit: int = 200):
    # Read the trail back without SSHing into the box. Guarded by ADMIN_TOKEN (constant-time
    # compare); unset or mismatched -> 403, so it is fail-closed by default.
    secret = os.environ.get("ADMIN_TOKEN", "")
    if not secret or not hmac.compare_digest(token, secret):
        return JSONResponse(status_code=403, content={"error": "forbidden"})
    return {"events": telemetry.read_recent(max(1, min(limit, 2000)))}


@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(token: str = "", limit: int = 2000):
    # A human view of the same trail: stat cards, a funnel, and a searchable table of every
    # coaching response. Served from the backend so it reads its own data same-origin (no CORS)
    # and dodges the strict CSP that stops a static claude.ai artifact from calling this API.
    # The events are embedded into the page; a full reload (the Refresh button) re-reads them.
    secret = os.environ.get("ADMIN_TOKEN", "")
    if not secret or not hmac.compare_digest(token, secret):
        return HTMLResponse(
            "<h1 style='font:600 18px sans-serif;color:#333;padding:24px'>403 — bad or missing ?token</h1>",
            status_code=403,
        )
    events = telemetry.read_recent(max(1, min(limit, 5000)))
    # JSON is valid JS; neutralize any literal </script> that could appear in coaching text.
    data = json.dumps(events).replace("<", "\\u003c")
    html = (TEMPLATES / "admin.html").read_text().replace("__EVENTS_JSON__", data)
    return HTMLResponse(html)
