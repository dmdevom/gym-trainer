---
title: AI Gym Trainer
emoji: 💪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AI Gym Trainer

Upload a photo of a bicep curl. Get back the elbow angle and where you are in
the rep. Upload a short video to get rep counting and AI-style coaching tips.

Built for the namastedev.com hackathon. Pose estimation via MediaPipe Tasks,
served with FastAPI, deployed on Hugging Face Spaces.

## Try it

Open `/docs` for the interactive API. `POST /analyze/photo` with an image.
Use `POST /analyze/video` with a short video under 25MB.

```json
{
  "elbow_angle": 77.4,
  "phase": "mid-rep",
  "side_analyzed": "right",
  "confidence": 0.99,
  "backend": "mediapipe"
}
```

Video response:

```json
{
  "reps": 3,
  "tips": [
    "Extend your arm more at the bottom of each rep.",
    "Slow down slightly; controlled reps are easier to judge and usually safer."
  ],
  "summary": {
    "duration_s": 8.1,
    "frames_sampled": 90,
    "frames_used": 73,
    "min_elbow_angle": 48.2,
    "max_elbow_angle": 151.7,
    "backend": "mediapipe"
  }
}
```

## Why MediaPipe and not YOLO

<!-- TODO: fill this in after you run the benchmark. You have both backends
     behind one interface, so this is a real measured answer, not a vibe:
       - latency per frame, both models, same video
       - how far apart they place each joint on the same photo
       - why that gap matters for a 300-frame video request -->

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

mkdir -p models
wget -O models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

python analyzer.py                      # unit tests for the angle maths
uvicorn main:app --reload               # then open http://localhost:8000/docs
```

Swap models to compare:

```bash
POSE_BACKEND=yolo uvicorn main:app --reload
```

## Notes

See [LEARNINGS.md](LEARNINGS.md) for what broke and why.
