import asyncio
from contextlib import asynccontextmanager
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

# Local dev convenience: load a gitignored .env (OPENROUTER_API_KEY, LLM_LOG_FILE, …)
# BEFORE anything reads the environment, so `uvicorn main:app` picks the key up with no
# exports, in any terminal, across restarts. Dev-only and optional: python-dotenv may be
# absent (a minimal prod image) and there may be no .env - both are fine. Real env vars
# always win: load_dotenv defaults to override=False, so a Space/Railway secret is never
# clobbered by a stray .env.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # dep not installed -> just use the real environment
    pass

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import llm_coach
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
# result is disposable and there are no accounts. Pruned so a long-lived process on
# Spaces doesn't slowly leak jobs and their rendered webms.
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
    log.info("─" * 52)
    yield


app = FastAPI(title="AI Gym Trainer", version="0.2.0", lifespan=lifespan)

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
    # Same facts as the boot banner. On Spaces this is your only window in.
    return {"status": "ok", "conf_min": CONF_MIN, "llm": llm_coach.info(), **get_backend().info()}


@app.get("/exercises")
def exercises():
    # Served to the page's selector so the client's list of movements can't drift
    # from the server's. Thresholds stay here; the client only gets names and tips.
    return {"exercises": [e.brief() for e in EXERCISES.values()]}


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

    try:
        if RENDER_SEMAPHORE.locked():
            progress_cb("Queued", 0)
        async with RENDER_SEMAPHORE:
            summary = await run_in_threadpool(annotate_video, src, dst, exercise_key, progress_cb)
        summary.pop("video", None)               # a local path; the browser gets a URL
        summary["video_url"] = f"/results/{token}"
        finish(stage="Done", pct=100, done=True, result=summary)
    except VideoError as e:
        finish(stage="Error", done=True, error=str(e))
    except Exception:
        log.exception("analysis failed for token %s", token)
        finish(stage="Error", done=True, error="Analysis failed on this clip. Try another.")
    finally:
        Path(src).unlink(missing_ok=True)        # keep only the rendered .webm


@app.post("/analyze/video", status_code=202)
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    exercise: str = Form("bicep_curl"),
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
    JOBS[token] = {"stage": "Starting", "pct": 0, "done": False, "error": None,
                   "result": None, "created": time.time()}
    _prune_jobs()
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
