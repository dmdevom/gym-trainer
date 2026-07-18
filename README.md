---
title: AI Gym Trainer
emoji: 💪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AI Gym Trainer

Upload *or record* a side-on set — single-arm bicep curl, barbell curl, or squat.
Get your reps counted, each one graded for depth and tempo, and the whole analysis
painted back onto the video: skeleton, live joint angle, a traffic-light
range-of-motion gauge, a tempo bar, per-rep form notes, and an end card that sums
up the set and says what to work on next.

Built for the namastedev.com hackathon. Pose estimation via MediaPipe Tasks,
served with FastAPI, deployed on Hugging Face Spaces (Docker).

**▶ Live demo:** https://huggingface.co/spaces/dmdev261/ai-gym-trainer

## Try it

Open `/`: pick an exercise, **upload or record** a side-on clip, watch a real
progress bar, then get the annotated video next to a per-rep table, the angle
chart, and a coaching card ("what to work on next session"). Or drive it directly:

- `GET  /exercises` — the movements on offer (name + how to film each).
- `POST /analyze/video` — a video (plus an `exercise` field) in, a `{token}` back.
- `GET  /progress/{token}` — poll for `{stage, pct, done}`; the final payload
  carries the full summary and a `/results/<id>` URL for the rendered `.webm`.
- `POST /analyze/photo` — a single frame in, elbow angle + phase out.
- `python analyze.py clip.mp4` — the graded summary + coaching in your terminal
  (`EXERCISE=squat python analyze.py clip.mp4` for the others).
- `python render.py clip.mp4` — just the annotated video, to `out/annotated.webm`.

Example final `/progress` payload (the analysis summary):

```json
{
  "meta": { "exercise": { "name": "Single-arm Bicep Curl", "vertex_name": "elbow" }, "side": "right" },
  "reps": 3,
  "full_reps": 3,
  "verdict": "3/3 full reps",
  "per_rep": [
    { "number": 1, "min_angle": 53.6, "depth_pct": 100, "duration_s": 2.7, "full": true, "tags": [], "issues": [] }
  ],
  "coaching": { "focus": "Progressive overload", "next_session": ["…"], "keep_in_mind": ["…"], "muscle": "…" },
  "video_url": "/results/…"
}
```

## How it works

```
exercises.py ->  each movement as data: which joints, which thresholds, which cues
video.py     ->  a smoothed per-frame joint-angle series      (the signal)
reps.py      ->  hysteresis state machine counts cycles,
                 then grades each on depth + tempo             (the meaning)
analyze.py   ->  summarize() + coaching(): one dict the CLI, API and video share
render.py    ->  paints that summary onto every frame -> VP8/webm
main.py      ->  FastAPI: the page, a job/poll progress model, serving the result
```

Three decisions worth knowing:

- Three exercises, **one** pipeline: a curl and a squat are the same
  high→low→high angle signal, so only a table of joints + thresholds differs
  (`exercises.py`). The rep-counting state machine is untouched. See LEARNINGS.md #15.
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
