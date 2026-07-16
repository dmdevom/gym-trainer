from contextlib import asynccontextmanager
import logging
import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from analyzer import CONF_MIN, analyze
from backends import get_backend

# uvicorn owns logging config, so borrow its logger. A bare getLogger(__name__)
# would silently go nowhere — root sits at WARNING and nobody added a handler.
log = logging.getLogger("uvicorn.error")

MAX_BYTES = 8 * 1024 * 1024  # 512MB of RAM total. Be rude about limits.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at boot, not on the first request. Also: this is where
    # MediaPipe dumps its absl/EGL noise, so logging afterwards puts our banner
    # last — the one thing you can actually see.
    info = get_backend().info()

    log.info("\u2500" * 52)
    log.info("  pose backend : %s", info["backend"])
    if "model" in info:
        mb = f"  ({info['model_mb']} MB)" if "model_mb" in info else ""
        log.info("  model        : %s%s", info["model"], mb)
    if "model_path" in info:
        log.info("  path         : %s", info["model_path"])
    log.info("  conf floor   : %s", CONF_MIN)
    log.info("\u2500" * 52)
    yield


app = FastAPI(title="AI Gym Trainer", version="0.1.0", lifespan=lifespan)


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