FROM python:3.11-slim

# opencv-python-headless and mediapipe still link against these. Ten megabytes
# to delete an entire class of "ImportError: libGL.so.1" at 23:00 on the 19th.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000. Stay root and the first thing that
# tries to write to the working directory kills the Space at startup.
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

# Bake the weights into the image. The Space's disk is not persistent, so
# anything fetched at runtime gets fetched again on every single cold start.
RUN mkdir -p models && python -c "\
import urllib.request; \
urllib.request.urlretrieve( \
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', \
  'models/pose_landmarker_lite.task')"

ENV POSE_BACKEND=mediapipe \
    POSE_MODEL=models/pose_landmarker_lite.task

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
