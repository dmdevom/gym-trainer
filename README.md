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
the rep. Video, rep counting and form feedback to follow.

Built for the namastedev.com hackathon. Pose estimation via MediaPipe Tasks,
served with FastAPI, deployed on Hugging Face Spaces.

## Try it

Open `/docs` for the interactive API. `POST /analyze/photo` with an image.

```json
{
  "elbow_angle": 77.4,
  "phase": "mid-rep",
  "side_analyzed": "right",
  "confidence": 0.99,
  "backend": "mediapipe"
}
```

## Why MediaPipe and not YOLO

Both backends sit behind one interface (`backends.py`), so this is measured, not
a vibe. Same clip, same arm, both pinned to CPU — the deploy box (HF Spaces,
2 vCPU) has no GPU — via `scripts/compare_backends.py`:

| backend      | CPU latency  | per 300-frame request | first-frame warmup |
| ------------ | ------------ | --------------------- | ------------------ |
| MediaPipe    | 41 ms/frame  | ~12 s                 | 61 ms              |
| YOLOv8n-pose | 129 ms/frame | ~39 s                 | 2.1 s              |

MediaPipe is ~3× faster on the hardware that actually serves the request, and
the two agree to a **median 5.2° (max 11°)** elbow angle across every shared
frame — so on clean side-on input they tell the same story, and the choice is
cost, not accuracy. YOLO also pulls in torch (~200–350 MB resident): fine on HF's
16 GB, but it's exactly what rules YOLO out on a 512 MB tier. MediaPipe ships;
YOLO stays opt-in behind `POSE_BACKEND=yolo`. Details in [LEARNINGS.md](LEARNINGS.md) #9 and #11.

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
