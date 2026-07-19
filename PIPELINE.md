# How the pipeline works

Upload *or* record a side-on clip of one of three exercises → the reps are counted
and graded, the analysis is painted back onto the video, and a coaching card says
what to work on next. The whole thing is a straight line:

> **exercise config → signal → meaning → one shared summary → two views of it (JSON + video)**

Three diagrams: the system at a glance, the request lifecycle over time, then the
analysis pipeline in detail.

---

## 1. System overview

Which module talks to which. `exercises.py` is the config seam — a table of numbers
and strings the whole pipeline reads (dotted lines), so adding a movement is a new
row, not a new code path.

```mermaid
flowchart LR
    HTML["Browser UI<br/>web/ (Next.js) · templates/index.html"]

    subgraph API["main.py — FastAPI"]
        VID["POST /analyze/video<br/>+ /progress + /results"]
        EXAPI["GET /exercises · /health"]
    end

    subgraph PIPE["Video analysis pipeline"]
        RENDER["render.py<br/>annotate_video (2-pass)"]
        VIDEO["video.py<br/>decode + pose → angle series"]
        ANALYZE["analyze.py<br/>summarize + coaching"]
        REPS["reps.py<br/>find_reps · grade · form checks"]
        LLM["llm_coach.py"]
    end

    MP["MediaPipe pose<br/>models/pose_landmarker_lite.task"]
    OR["OpenRouter API<br/>(optional, keyed)"]
    CFG["exercises.py<br/>Exercise table:<br/>joints · thresholds · cues"]

    HTML <--> API
    VID --> RENDER
    RENDER --> VIDEO
    RENDER --> ANALYZE
    ANALYZE --> REPS
    ANALYZE --> LLM
    VIDEO --> MP
    LLM --> OR

    CFG -.-> VIDEO
    CFG -.-> REPS
    CFG -.-> ANALYZE
    CFG -.-> RENDER
    EXAPI -.-> CFG
```

The pluggable MediaPipe/YOLO seam (`backends.py`, with `analyzer.py` as its
single-frame harness) survives from the Phase-0 POC for offline work — the boot
banner, `/health`, `scripts/compare_backends.py` — but no request path runs
through it anymore. The **video** path — the product — uses MediaPipe's Tasks
API in VIDEO mode directly (it needs tracking state across frames), which is why
it doesn't go through `backends.py`.

---

## 2. Request lifecycle — the job/poll model

The upload doesn't block for ~20 s. `POST /analyze/video` returns a **token**
immediately (202); the heavy work runs in a threadpool worker and reports progress
into an in-memory `JOBS` dict; the page **polls** `/progress/{token}` (~600 ms) to
drive a real bar. Polling, not SSE — SSE buffers behind the hosting proxy.

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser (index.html)
    participant A as FastAPI (main.py)
    participant J as JOBS dict
    participant W as Threadpool worker
    participant P as annotate_video (pipeline)

    B->>A: GET /exercises
    A-->>B: [{key, name, tips, ...}]
    Note over B: user picks exercise,<br/>uploads or records a clip

    B->>A: POST /analyze/video (file + exercise)
    A->>A: validate type + size, save {token}-src
    A->>J: JOBS[token] = {stage, pct:0, done:false}
    A-->>B: 202 {token}
    A->>W: asyncio.create_task(_run_job)
    W->>P: run_in_threadpool(annotate_video, progress_cb)

    loop every ~600 ms until done
        B->>A: GET /progress/{token}
        A->>J: read job
        A-->>B: {stage, pct, done:false}
        P->>J: progress_cb writes stage + pct
    end

    P-->>W: summary dict + rendered .webm
    W->>J: JOBS[token].update(done:true, result=summary)

    B->>A: GET /progress/{token}
    A-->>B: {done:true, result:{...}}
    B->>A: GET /results/{token}
    A-->>B: video/webm
    Note over B: render video + coaching card<br/>+ per-rep table + Chart.js
```

Both `JOBS` and the rendered `.webm`s are disposable — an in-memory dict and a temp
dir, pruned by age/count, gone on restart. No accounts, no DB, no history (a
deliberate guardrail).

---

## 3. The analysis pipeline — `annotate_video()`

The heart of it. **Two passes** over the clip with **one shared summary** in the
middle. The sparse detect pass produces the numbers; `summarize()` turns them into
meaning *once*; the dense pass paints that same summary onto every frame. Because
both the JSON and the on-screen counter read the *same* `per_rep` list, they can
never disagree about how many reps you did.

```mermaid
flowchart TD
    START(["annotate_video(src, exercise, progress_cb)"])

    subgraph D["① Detect pass — sparse, every STRIDE=3 frames"]
        direction TB
        D1["decode → rotate (ours) → downscale ≤ 960"]
        D2["MediaPipe pose, VIDEO mode"]
        D3["video._to_sample(exercise)"]
        D4["samples[]: primary angle (elbow/knee)<br/>+ all 33 landmarks, in draw space"]
        D1 --> D2 --> D3 --> D4
    end

    SIDE["pick_side(samples) → left | right<br/>chosen once for the whole clip"]

    subgraph S["③ summarize() — analyze.py — the shared summary"]
        direction TB
        S1["median_smooth → smoothed angle series"]
        S2["find_reps → Rep[] cycles<br/>hysteresis state machine · permissive"]
        S3["evaluate_checks → torso / elbow / legs<br/>form checks, per rep"]
        S4["form_feedback → RepGrade[]<br/>depth% + tempo + form · strict"]
        S5["_verdict — the one-line summary"]
        S6{"llm_coach.generate()?"}
        S7["LLM card + per-rep notes<br/>(OpenRouter, validated)"]
        S8["analyze.coaching() — offline rules<br/>+ the grader's own rep notes"]
        S9["summary dict:<br/>reps · per_rep · verdict · coaching<br/>· thresholds · series"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
        S6 -->|valid reply| S7 --> S9
        S6 -->|None on any failure| S8 --> S9
    end

    subgraph R["④ Render pass — dense, every frame"]
        direction TB
        R1["decode → same rotate + downscale"]
        R2["interpolate skeleton between the<br/>two nearest detect samples"]
        R3["draw HUD from the summary:<br/>skeleton · live angle · ROM gauge ·<br/>tempo bar · per-rep flash · tally ·<br/>rep counter = count(end_t ≤ now)"]
        R4["VP8 / .webm writer + 2.5s end card"]
        R1 --> R2 --> R3 --> R4
    end

    JSON["JSON → browser<br/>(progress result)"]
    WEBM[".webm → browser<br/>(/results)"]

    START --> D1
    D4 --> SIDE --> S1
    S9 --> R1
    S9 --> JSON
    R4 --> WEBM

    CFG["exercises.py — joints · thresholds · cues"]
    CFG -.-> D3
    CFG -.-> S2
    CFG -.-> S4
    CFG -.-> R3
```

---

## The seams that hold it together

- **`summarize()` = one number, two views.** The rep count in the JSON table and the
  counter burned onto the video come from the *same* `per_rep` list — the overlay
  counter is literally `count(end_t ≤ now)` over it. Computed once, drawn twice.
- **An exercise is data, not code.** A curl and a squat are the same
  high→low→high signal; only *which joint*, *which thresholds*, and *which cues*
  differ. That's a row in `exercises.py`. `video.py` / `reps.py` / `render.py`
  never hard-code "elbow" — a squat's knee is drawn exactly like a curl's elbow.
- **Detect permissive, grade strict.** `find_reps` counts every cycle that moves
  (so a sloppy half-rep is *seen*); `form_feedback` is the strict judge of depth,
  tempo and form. Two layers, deliberately kept apart.
- **The LLM can only add quality, never break the app.** `llm_coach.generate()`
  returns valid coaching *or* `None` on any failure (no key, timeout, bad JSON,
  refusal, rate cap, feature off). `None` falls back to the offline rule-based
  `analyze.coaching()`, so the card reads the same shape with or without a key.
  Per-rep notes ride the same contract: the LLM's flash/coach text overlays the
  grader's deterministic notes only if the reply covers every rep and passes the
  length caps (`meta.rep_notes_source` says which text shipped).
- **Sparse detect, dense draw.** Detection runs at `STRIDE=3`; the skeleton is
  interpolated between samples, never re-detected per frame — fast, and the count
  stays exact because it comes from the one detection pass.

---

## Where to start reading

| You want to… | Start in |
| --- | --- |
| follow a request end-to-end | `main.py` → `render.annotate_video` |
| understand rep counting | `reps.py` (`find_reps` + its tests) |
| change a threshold or add a movement | `exercises.py` |
| see how the overlay is drawn | `render.py` (`_draw_hud`, `_draw_rom_gauge`) |
| shape the JSON/summary | `analyze.py` (`summarize`) |
