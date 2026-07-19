FROM python:3.11-slim

# opencv-python-headless and mediapipe link against these. mediapipe 0.10.35
# also dlopens the GLES/EGL stack (libGLESv2.so.2, libEGL.so.1) at startup -
# present on a desktop, absent on slim, so the import passes locally and the app
# dies on boot here. A few megabytes to delete a whole class of "cannot open
# shared object file" (libGLdispatch.so.0 comes in as a dependency of these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgles2 libegl1 gosu \
    && rm -rf /var/lib/apt/lists/*

# Managed container hosts commonly run containers as uid 1000. Stay root and the
# first thing that tries to write to the working directory kills the app at startup.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# requirements.txt, NOT requirements-dev.txt — no ultralytics, so no torch.
# ~400MB image instead of ~2.5GB, and a build that finishes in about a minute.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the weights into the image. The container disk is not persistent, so
# anything fetched at runtime gets fetched again on every single cold start.
RUN mkdir -p models && python -c "\
import urllib.request; \
urllib.request.urlretrieve( \
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', \
  'models/pose_landmarker_lite.task')"

ENV POSE_BACKEND=mediapipe \
    POSE_MODEL=models/pose_landmarker_lite.task

# Make the entrypoint executable (it was copied in with the source above).
RUN chmod +x docker-entrypoint.sh

# Start as root ONLY so the entrypoint can hand the root-owned volume mount (Railway attaches
# volumes root-owned) to the app user; it immediately drops to uid 1000 via gosu before
# exec'ing uvicorn, so the server itself still runs unprivileged. A host that forces a
# non-root uid (e.g. HF Spaces) runs uvicorn directly. Binds to $PORT when the platform
# injects one (Railway), else 7860 (image default).
USER root
EXPOSE 7860
ENTRYPOINT ["/home/user/app/docker-entrypoint.sh"]
