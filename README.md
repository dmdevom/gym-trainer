# trAIner

Upload *or record* a side-on set — single-arm bicep curl, barbell curl, or squat.
Get your reps counted, each one graded for depth and tempo, and the whole analysis
painted back onto the video: skeleton, live joint angle, a traffic-light
range-of-motion gauge, a tempo bar, per-rep form notes, and an end card that sums
up the set and says what to work on next.

Built for the namastedev.com hackathon. Pose estimation via MediaPipe Tasks,
served with FastAPI, a Next.js frontend in [`web/`](web/README.md) deployed on
Vercel, and a Dockerized API on Railway.

**▶ Live demo:** https://gym-trainer-web-ai.vercel.app

**API (+ built-in fallback UI):** https://gym-trainer-production-3c7f.up.railway.app

## Try it

Open the [live demo](https://gym-trainer-web-ai.vercel.app): pick an exercise,
try a bundled sample or **upload/record** a side-on clip, watch a real progress
bar, then get the annotated video next to a per-rep table, the angle chart,
form-check chips, and a coaching card ("what to work on next session"). The
API serves the same product through its built-in page. Or call it
directly:

### API

- `GET /exercises` — the movements on offer (name, filming tip, cues). The UI's
  selector is fed from this so client and server can't drift.
- `POST /analyze/video` — multipart `file` + `exercise` (`bicep_curl` ·
  `barbell_curl` · `squat`) in, `202 {token}` back the moment the upload lands.
  Accepts mp4/mov/webm/avi/mkv/m4v/3gp up to **100 MB** (`MAX_UPLOAD_MB`; `413`
  over the cap, `415` if it doesn't look like video).
- `GET /progress/{token}` — poll for `{stage, pct, done, error}`. `stage` walks
  Starting → (Queued) → Analysing → Coaching → Generating → Done; the final
  response also carries the full summary below. `404` once the job expires (~1 h).
- `GET /results/{token}` — the rendered, annotated `.webm`. Same ~1 h lifetime.
- `GET /health` — backend, model and LLM status. On the hosted API this is your
  only window in.

### CLI (same pipeline, no server)

- `python analyze.py clip.mp4` — the graded summary + coaching in your terminal
  (`EXERCISE=squat python analyze.py clip.mp4` for the others).
- `python render.py clip.mp4` — just the annotated video, to `out/annotated.webm`.

A real final `/progress` payload (squat sample, trimmed with `…`):

```json
{
  "meta": {
    "exercise": { "key": "squat", "name": "Squat", "vertex_name": "knee" },
    "side": "left", "side_visibility": { "left": 0.781, "right": 0.273 },
    "fps": 30.0, "stride": 3, "sample_hz": 10.0, "frames_sampled": 60,
    "coaching_source": "llm+rules", "rep_notes_source": "llm"
  },
  "reps": 2, "full_reps": 2, "verdict": "2/2 full reps",
  "per_rep": [
    { "number": 1, "min_angle": 45.8, "depth_pct": 100, "duration_s": 1.9,
      "full": true, "tags": [], "issues": [], "reason": null,
      "flash_note": "Clean - full range and controlled",
      "coach_note": ["You hit full range and controlled the descent well here.",
                     "Keep the chest up on the way down as assessed."],
      "start_t": 1.0, "end_t": 2.9 },
    { "number": 2, "min_angle": 48.7, "depth_pct": 100, "duration_s": 1.8,
      "full": true, "tags": [], "issues": [], "reason": null,
      "flash_note": "Clean - full range and controlled", "coach_note": ["…"],
      "start_t": 3.7, "end_t": 5.5 }
  ],
  "form_checks": [
    { "key": "lean", "label": "Chest up", "status": "ok", "assessed": 2, "flagged": 0,
      "fault": "Folding forward at the bottom",
      "cue": "Keep your chest up and sit between your hips instead of folding over." }
  ],
  "coaching": {
    "focus": "Progressive overload", "mental_cue": "Add a little, hold form",
    "session_story": "Both reps were performed with full range of motion and good control. …",
    "next_session": ["Clean session - every rep full and controlled. …"],
    "keep_in_mind": ["Sit back and down - break at the hips and knees together.", "…"],
    "muscle": "Quads, glutes and hamstrings. …"
  },
  "thresholds": { "up_enter": 130.0, "down_enter": 155.0, "full_rom": 100.0,
                  "gauge_deep": 70.0, "tempo_min_s": 1.3 },
  "series": { "t": [0.0, 0.1, 0.2, "…"], "angle": [170.5, 170.4, 170.2, "…"] },
  "video_url": "/results/e9872d12…"
}
```

`form_checks` are the whole-body checks (torso swing, elbow drift, head bob,
chest up) — each one is `ok`, `flag`, or honestly `not_assessed` when that body
part never made it into frame. Every rep also carries its two display notes:
`flash_note` (the one-line verdict burned onto the video) and `coach_note`
(detail bullets for the rep table), written by the deterministic grader or by
the LLM when its reply validates (`meta.rep_notes_source`). `reason` is `null`
for a counted rep, or says why one didn't count (`under_extension` /
`under_contraction`).

## How it works

```
exercises.py ->  each movement as data: joints, thresholds, cues, form checks
video.py     ->  a smoothed per-frame joint-angle series      (the signal)
reps.py      ->  hysteresis state machine counts cycles,
                 then grades each on depth + tempo + form      (the meaning)
analyze.py   ->  summarize() + coaching(): one dict the CLI, API and video share
llm_coach.py ->  optional LLM coaching card + per-rep notes; rules fallback on ANY failure
render.py    ->  paints that summary onto every frame -> VP8/webm
main.py      ->  FastAPI: job/poll progress model, render queue, results
backends.py  ->  pluggable MediaPipe/YOLO seam (boot banner, /health, benchmarks)
```

[PIPELINE.md](PIPELINE.md) has the full diagrams: system overview, the job/poll
request lifecycle, and the two-pass analysis pipeline.

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

### Coaching: an LLM with a rules floor

With an `OPENROUTER_API_KEY` set, an LLM (default `google/gemini-2.5-flash-lite`
via OpenRouter) writes the coaching card *and* each rep's display notes — the
one-line flash burned onto the video and the bulleted note in the rep table —
from this session's actual numbers, and the reply is validated hard. Wrong
shape, empty reply, timeout, rate cap, refusal: *any* failure silently falls
back to the offline rule-based coach, which produces the same shape. Rep notes
are all-or-nothing on top: unless the reply covers every rep within the length
caps, the deterministic grader text keeps the overlay. The LLM can add quality;
it can never break the app. `meta.coaching_source` (`llm` · `rules` ·
`llm+rules`) and `meta.rep_notes_source` (`llm` · `rules`) say who wrote what,
and calls are capped hourly because the endpoint is public and the key is yours.

## Why MediaPipe and not YOLO

Both backends sit behind one interface (`backends.py`), so this is measured, not
a vibe. Same clip, same arm, both pinned to CPU — the 2-vCPU deploy box has
no GPU — via `scripts/compare_backends.py`:

| backend      | CPU latency  | per 300-frame request | first-frame warmup |
| ------------ | ------------ | --------------------- | ------------------ |
| MediaPipe    | 41 ms/frame  | ~12 s                 | 61 ms              |
| YOLOv8n-pose | 129 ms/frame | ~39 s                 | 2.1 s              |

MediaPipe is ~3× faster on the hardware that actually serves the request, and
the two agree to a **median 5.2° (max 11°)** elbow angle across every shared
frame — so on clean side-on input they tell the same story, and the choice is
cost, not accuracy. YOLO also pulls in torch (~200–350 MB resident): fine on a
16 GB box, but it's exactly what rules YOLO out on a 512 MB tier. MediaPipe ships;
YOLO stays opt-in behind `POSE_BACKEND=yolo`. Details in [LEARNINGS.md](LEARNINGS.md) #9 and #11.

## The web frontend (`web/`)

A standalone Next.js 16 (React 19) app — the styled product UI: sample clips,
upload/record, live progress, original-vs-annotated video side by side, angle
chart, per-rep breakdown, form-check chips, coaching card. This is the live
demo, deployed on Vercel: https://gym-trainer-web-ai.vercel.app

```bash
cd web && npm install
BACKEND_API_URL=http://localhost:8000 npm run dev    # -> http://localhost:3000
```

The browser calls same-origin `/backend-api/*`, which Next rewrites server-side
to `BACKEND_API_URL` — no CORS involved. **Left unset it defaults to the
production Railway URL**, so a bare `npm run dev` exercises the live backend.
Alternative mode: set `NEXT_PUBLIC_API_BASE_URL` to call an API straight from
the browser, and add the frontend's origin to the backend's `CORS_ORIGINS`.
More in [web/README.md](web/README.md).

The API also serves a built-in, zero-build single-file page at `/`
(`templates/index.html`) — same features, no Node required; `web/` is the
primary UI.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pinned runtime deps; requirements-dev.txt
                                     # adds YOLO + the benchmark tooling

mkdir -p models
wget -O models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

cp .env.example .env                 # optional: your OpenRouter key for LLM coaching
uvicorn main:app --reload            # -> http://localhost:8000/
```

Compare the two pose backends on a real clip (needs the dev deps):

```bash
python scripts/compare_backends.py path/to/clip.mp4
```

## Tests

```bash
python reps.py       # rep counting, grading, form checks — the spec as tests
python analyzer.py   # single-frame angle math sanity
cd web && npm test && npm run typecheck && npm run lint
```

## Configuration

All optional — with nothing set, the app runs offline with rule-based coaching.

| Variable | Default | What it does |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | *(unset)* | Enables LLM coaching; absent → rule-based. |
| `LLM_MODEL` | `google/gemini-2.5-flash-lite` | Any non-reasoning model on openrouter.ai/models. |
| `LLM_COACH` | on iff key set | Force the LLM path on/off (`1`/`0`). |
| `LLM_MAX_CALLS_PER_HOUR` | `60` | Spend cap for a public endpoint on your key. |
| `LLM_LOG_FILE` | *(off)* | `stderr` or a file path — log each LLM request/response. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Alternate OpenAI-compatible endpoint. |
| `POSE_BACKEND` | `mediapipe` | `yolo` opt-in (needs `requirements-dev.txt`). |
| `POSE_MODEL` | `models/pose_landmarker_lite.task` | Which landmarker file to load. |
| `MAX_CONCURRENT_RENDERS` | `2` | Renders running at once; extra jobs wait as "Queued". |
| `MAX_UPLOAD_MB` | `100` | Upload size cap. Buffered in RAM per upload — mind container memory. |
| `CORS_ORIGINS` | localhost:3000 dev origins | Comma-separated frontend origins allowed to call the API. |
| `PORT` | `7860` | Injected by Railway; the image falls back to 7860. |

For local dev, `main.py` auto-loads a gitignored `.env` (template in
`.env.example`); real environment variables always win over it.

## Deploying

**Railway (Docker).** `railway up` from the repo root (`.railwayignore` keeps
the build context lean). The build bakes the lite model into the image —
container disks are ephemeral, so a runtime download would repeat on every
restart. The container binds `$PORT` when the platform injects one, else 7860.

**Secrets** (`OPENROUTER_API_KEY`) are platform env/secret variables on
Railway — never in the repo, never baked into the image.

**Frontend.** `web/` is deployed on Vercel at
https://gym-trainer-web-ai.vercel.app with `BACKEND_API_URL` pointed at the
Railway API; it builds and runs the same on any Node host (`npm run build &&
npm run start`). The built-in page served straight from the API stays as a
zero-deploy fallback.

## Limitations (honest ones)

- **One person, filmed side-on.** The tracker follows a single pose; mirrors or
  bystanders in frame can steal it, and depth grading assumes the sagittal plane.
- **Disposable by design.** No accounts, no history: jobs live in memory and
  rendered videos in a temp dir, both pruned after ~1 h and gone on restart.
- **Single worker.** The in-memory job table assumes one process (the shipped
  CMD). Run `uvicorn --workers 2` and progress polling breaks.
- **Three exercises**, thresholds tuned on a limited clip library — squat
  thresholds especially are first-pass.
- **100 MB upload cap** (`MAX_UPLOAD_MB`, buffered in RAM per upload); the Next
  proxy in front of the deployed UI also caps bodies (`proxyClientMaxBodySize`);
  renders queue two at a time on the 2-vCPU deploy boxes.

## Project layout

```
main.py          FastAPI app: endpoints, job/poll model, render queue
render.py        two-pass annotate_video: sparse detect, dense draw, VP8 writer
analyze.py       summarize() — the one summary the JSON, video and CLI share
reps.py          hysteresis rep counting + grading + form checks (and their tests)
video.py         decode -> rotate -> pose -> smoothed joint-angle series
exercises.py     the exercise table: joints, thresholds, cues, form checks
llm_coach.py     LLM coaching with validation + rules fallback
backends.py      pluggable pose backends (MediaPipe / opt-in YOLO)
analyzer.py      single-frame angle harness (CLI + tests)
templates/       built-in single-file UI served at /
web/             Next.js frontend (own README + tests)
scripts/         compare_backends.py benchmark
models/          pose_landmarker_lite.task (baked into the Docker image)
```

Personal test footage (`data/`) stays local: gitignored and excluded from every
deploy path.

## Notes

[LEARNINGS.md](LEARNINGS.md) is the honest engineering log — what broke and why,
from deploy-day GLES crashes to codec dead ends to threshold tuning.
[PIPELINE.md](PIPELINE.md) has the architecture diagrams.
