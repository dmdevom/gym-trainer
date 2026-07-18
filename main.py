from contextlib import asynccontextmanager
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from analyzer import CONF_MIN, analyze
from backends import get_backend
from render import annotate_video
from video import VideoError

# uvicorn owns logging config, so borrow its logger. A bare getLogger(__name__)
# would silently go nowhere — root sits at WARNING and nobody added a handler.
log = logging.getLogger("uvicorn.error")

MAX_BYTES = 8 * 1024 * 1024        # a photo. 512MB of RAM total. Be rude about limits.
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # a phone clip is big; still cap it hard.

TEMPLATES = Path(__file__).parent / "templates"

# Rendered videos land here and are served back by token. A temp dir, because the
# result is disposable: no accounts, no history, nothing to persist (the Phase 3
# guardrail). On a restart they're gone, which for an upload->analyse->watch tool
# is exactly right.
RESULTS_DIR = Path(tempfile.mkdtemp(prefix="gymtrainer-"))


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
    log.info("  results dir  : %s", RESULTS_DIR)
    log.info("─" * 52)
    yield


app = FastAPI(title="AI Gym Trainer", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    # One static page, served by hand — no Jinja, no StaticFiles mount, no new
    # dependency. The page does the dynamic part in the browser.
    return (TEMPLATES / "index.html").read_text()


@app.get("/health")
def health():
    # Same facts as the boot banner. On Spaces this is your only window in.
    return {"status": "ok", "conf_min": CONF_MIN, **get_backend().info()}


@app.post("/analyze/photo")
async def analyze_photo(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        return JSONResponse(
            status_code=415,
            content={"error": "not_an_image", "detail": f"Got {file.content_type}"},
        )

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "too_large", "detail": "Max 8MB."},
        )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        result = analyze(path)
    finally:
        os.unlink(path)

    if "error" in result:
        return JSONResponse(status_code=422, content=result)
    return result


@app.post("/analyze/video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    # Phones love to send video as application/octet-stream, so accept on either
    # the content-type OR a known extension rather than rejecting a real clip.
    name = (file.filename or "").lower()
    looks_video = (file.content_type or "").startswith("video/") or name.endswith(
        (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
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
            content={"error": "too_large", "detail": "Max 50MB. Trim it or drop the resolution."},
        )

    token = uuid.uuid4().hex
    src = RESULTS_DIR / f"{token}{Path(name).suffix or '.mp4'}"
    dst = RESULTS_DIR / f"{token}.webm"
    src.write_bytes(data)

    # annotate_video is sync and CPU-bound (seconds of MediaPipe + encoding). Run
    # it off the event loop or a single upload freezes every other request. Each
    # call builds its own landmarker, so the threadpool is safe here.
    try:
        summary = await run_in_threadpool(annotate_video, str(src), str(dst))
    except VideoError as e:
        return JSONResponse(status_code=422, content={"error": "bad_video", "detail": str(e)})
    finally:
        src.unlink(missing_ok=True)  # keep only the rendered .webm, not the upload

    # The renderer returns a local path; the browser gets a URL instead.
    summary.pop("video", None)
    summary["video_url"] = f"/results/{token}"
    return summary


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
