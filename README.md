---
title: AI Gym Trainer
emoji: 💪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AI Gym Trainer

Upload a side-on video of bicep curls. Get your reps counted, each one graded for
depth and tempo, and the whole analysis painted back onto the video — skeleton,
live elbow angle, and a running rep counter.

Built for the namastedev.com hackathon. Pose estimation via MediaPipe Tasks,
served with FastAPI, packaged for Hugging Face Spaces (Docker).

## Try it

Open `/` for the upload page: pick a clip, wait ~15 s, and watch the annotated
video next to a per-rep table and the elbow-angle chart. Or drive it directly:

- `POST /analyze/video` — a video in, JSON out (rep count, per-rep grade, angle
  series) plus a `/results/<id>` URL for the rendered `.webm`.
- `POST /analyze/photo` — a single frame in, elbow angle + phase out.
- `python analyze.py clip.mp4` — the graded summary in your terminal.
- `python render.py clip.mp4` — just the annotated video, to `out/annotated.webm`.

Example `/analyze/video` response:

```json
{
  "reps": 3,
  "full_reps": 3,
  "verdict": "3/3 full reps",
  "per_rep": [
    { "number": 1, "min_angle": 53.6, "duration_s": 2.7, "full": true, "issues": [] }
  ],
  "video_url": "/results/…"
}
```

## How it works

```
video.py    ->  a smoothed per-frame elbow-angle series      (the signal)
reps.py     ->  hysteresis state machine counts cycles,
                then grades each on depth + tempo             (the meaning)
analyze.py  ->  summarize(): one dict the CLI, API and video share
render.py   ->  paints that summary onto every frame -> VP8/webm
main.py     ->  FastAPI: the page, the endpoints, serving the result
```

Two decisions worth knowing:

- The rep count **on the video** and the rep count **in the JSON** are the same
  number by construction — one detection pass feeds both, and the on-screen
  counter is literally "reps whose end-time has passed." See LEARNINGS.md #13.
- The annotated output is `.webm`/VP8 because that is the one format this OpenCV
  can *encode* and a browser can *play* — H.264 wouldn't open, and the mp4 it does
  write, the browser won't decode. See LEARNINGS.md #12.

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

python reps.py                                 # the rep-counter's unit tests
python analyze.py data/videos/curl_right.mp4   # graded summary in the terminal
uvicorn main:app --reload                      # then open http://localhost:8000/
```

Compare the two backends on a real clip (needs the dev deps):

```bash
python scripts/compare_backends.py data/videos/curl_right.mp4
```

## Notes

See [LEARNINGS.md](LEARNINGS.md) for what broke and why.
