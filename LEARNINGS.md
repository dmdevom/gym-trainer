# LEARNINGS

Things that broke, and what they taught me. Written while building, so the
numbers are the real ones off my machine.

---

## 1. The model never says "I don't know"

The single most useful thing I learned. A pose model returns **every** keypoint
on **every** call, whether or not it can see them.

My first test photo was cropped at the thigh. YOLO returned this:

```
13   left_knee     1494.6   2025.0   0.05
15   left_ankle    1538.8   2025.0   0.01
16   right_ankle   1291.5   2025.0   0.00
```

The image is 2025px tall. Those points are pinned to the bottom edge with
near-zero confidence, because the legs are not in the frame at all. No `None`,
no exception, no missing key. It guessed and lowered the number.

Then I reshot with my feet out of frame instead of my knees:

```
13   left_knee     1759.2   3882.6   0.82    <- real
15   left_ankle    1646.7   4096.0   0.09    <- invented
16   right_ankle   1285.3   4096.0   0.11    <- invented
```

Image height: 4096. Same fingerprint, and this time only the ankles, because
only my feet were cropped. **The confidence tracked the crop exactly, both
times.** That's not luck, it's the contract: you always get 17 points, and
confidence is the only thing telling you which ones are fiction.

`CONF_MIN = 0.5` in `analyzer.py` exists because of this.

## 2. Confidence means confident, not correct

I assumed a high confidence meant a reliable point. It doesn't.

On a stock gym photo where the subject's right arm was hidden behind his torso,
YOLO invented an entire right arm and rated it **0.84**. It put `left_wrist` at
**0.92** roughly 300px away from the actual hand — on his shirt.

MediaPipe hallucinated the same limb in the same place, but stamped
`visibility 0.12` on it. It told me. YOLO said 0.84 and kept a straight face.

So confidence is **necessary but not sufficient**. The real filter is
confidence _plus_ geometric plausibility — bone lengths that stay constant,
joints in a sane order, a forearm that isn't longer than an upper arm.

## 3. The visualisation lied by being helpful

`results[0].plot()` produced an annotated image with **no legs drawn at all**,
while the array it came from contained four fabricated keypoints. Ultralytics
filters keypoints below a confidence threshold before drawing.

My own MediaPipe script had no such filter, so _it_ showed the truth: a chaotic
web of landmarks dribbling down to the bottom edge. My sloppier code was the
honest one.

If I'd trusted the JPEG as ground truth I'd have built a rep counter on
invisible garbage. **The debug view and the data are different artifacts. My
code reads the array.**

## 4. The test image was the problem, not the model

`person 0.77` on the stock photo. `person 0.93` on a photo I took against a
plain wall. That bounding-box confidence was printed on the image the whole
time and I didn't know it was a photo-quality meter.

The stock photo had: a near-black background, hard rim lighting, a dumbbell
occluding the gripping hand, a hunched pose, one arm entirely hidden, and a
crop at mid-thigh. Lit for drama, not for keypoints.

Before blaming a model, check what you handed it.

## 5. Two models disagreeing is a free quality signal

Same joints, same photo, two unrelated architectures:

| joint       | YOLO             | MediaPipe        | gap      |
| ----------- | ---------------- | ---------------- | -------- |
| left hip    | (1803.0, 2909.6) | (1804.7, 2911.0) | **2 px** |
| left wrist  | (2289.3, 2959.0) | (2288.1, 2976.2) | 17 px    |
| right wrist | (538.8, 1760.7)  | (512.1, 1739.1)  | 34 px    |
| right elbow | (989.8, 2060.7)  | (997.6, 2125.9)  | 66 px    |

On a 3072px-wide image. On the bad stock photo the same joints were **175px
apart with opposite confidences**.

Neither model can tell you it's wrong. **The pair of them disagreeing can.**
Run both, compare, reject the frame past a threshold. That's the idea I'd build
out with more time.

## 6. MediaPipe deleted its entire legacy API

`mp.solutions.pose` → `AttributeError: module 'mediapipe' has no attribute
'solutions'`. Removed in **0.10.31** (Dec 2025). Not renamed — deleted.
`mp.solutions.drawing_utils` and `POSE_CONNECTIONS` went with it.

Which means essentially every MediaPipe pose tutorial online is now dead code.
The replacement is the Tasks API, where you pass an explicit `.task` model file
instead of a `model_complexity` int. The model stopped being a parameter and
became an asset you manage — which I felt again on deploy day, baking the file
into the image because the disk isn't persistent.

**The library moves faster than the content about it.** Read the changelog, not
the blog post.

## 7. The 180° floating-point cliff

`cos θ` for a near-straight arm evaluates to `-1.0000000000000002`.
`math.acos` of that raises `ValueError: math domain error`. numpy's `arccos`
quietly hands back `NaN` instead, which is worse — one NaN poisons every value
after it in a moving average.

My extended arm measured **168.9°**. The bottom of every single rep is
near-straight, so without the clip this fires at exactly the moment the rep
counter needs to see "extended," and I'd have spent an afternoon blaming the
state machine.

`max(-1.0, min(1.0, cosine))` before `acos`. One line.

## 8. Camera geometry decides whether the signal exists

A curl rotates the forearm in the sagittal plane. Shoot front-on and the
forearm swings toward and away from the lens — the wrist barely moves in (x, y)
while the true elbow angle sweeps 130°. The angle gets compressed into noise.

Side-on puts the whole arc in the image plane, so the 2D angle ≈ the real
angle. The cost is the far arm goes occluded, and lateral elbow flare becomes
invisible. Worth it: for curls, the drift that matters is forward, and forward
is what side-on shows.

**No amount of model quality fixes a camera angle that discards the signal.**

## 9. Deploy: RAM picks the model, CPU decides if it finishes

Render's free tier is 512MB — and so is the $7 Starter tier. `import torch`
alone is 200–350MB resident, so YOLO was never fitting in either. The first
Render tier that runs it is $25/mo.

The free tier we deployed on: **2 vCPU, 16GB**. Render's $85/mo Pro tier, for zero.

But the number that actually mattered wasn't RAM:

```
Speed: 4.4ms preprocess, 115.1ms inference, 13.9ms postprocess
```

~133ms/frame **on my laptop**, which has more than 2 cores. Video is ~300
frames per request. That's a 40-second floor on faster hardware than I'd be
deploying to. RAM decides _which model_; CPU decides _whether the endpoint
returns at all_. I picked MediaPipe on speed, not memory.

## 10. The abstraction paid for itself three times

I put both models behind one interface returning
`{joint: (x_px, y_px, conf)}` to solve exactly one problem: torch not fitting
in Render's 512MB.

It then also: cut my Docker image from ~2.5GB to ~400MB and my build from ten
minutes to one, and made the model benchmark a one-line env var instead of a
second script.

I'd have called it over-engineering if I'd planned all three. **An abstraction
drawn along a real seam keeps paying out for reasons you didn't think of.** The
seam here was real: 17 vs 33 keypoints, pixels vs normalised, different indices
for the same joint.

## 11. On clean input the two models agree, so speed is the whole argument

#5 compared YOLO and MediaPipe on one hard photo, where they landed 175px apart
with opposite confidences — the pathological case. Running both frame-for-frame
on a clean side-on curl (`scripts/compare_backends.py`, both pinned to CPU
because the deploy box has none of the GPU this laptop does) told the opposite,
more useful story:

```
                 CPU median     ~300-frame request    first frame
MediaPipe        41 ms/frame    ~12 s                 61 ms
YOLOv8n-pose     129 ms/frame   ~39 s                 2109 ms
```

The two elbow-angle series agreed to a **median 5.2° (max 11°)** across all 117
shared frames — visibly the same wave, YOLO just a hair deeper at every peak.

So #5 and this aren't in conflict; they're one lesson at two input qualities. On
a bad photo, disagreement is the quality signal. On good video, agreement means
the backend choice is pure cost — and MediaPipe is ~3× faster on the CPU that
serves the request, the one axis that was free to move. The 129 ms/frame here
also lands right on the 133 ms I measured back in #9 by a completely different
route, which is the kind of coincidence that makes me trust a number.

## 12. The browser won't play what OpenCV can write

The annotated video was meant to be the demo's money shot, and it nearly wasn't
playable at all. `cv2.VideoWriter` with an `avc1`/`H264` fourcc **fails to open**
on this box (OpenCV 5.0.0): it reaches for the `h264_v4l2m2m` hardware encoder,
finds no device, and there is no software x264 to fall back to. The one mp4 codec
that *does* open is `mp4v` — MPEG-4 Part 2 — which Chrome and Firefox refuse to
play in a `<video>` tag. So the happy path writes a file that plays fine in VLC
and shows a black rectangle in the browser, with no error at either end.

The escape hatch is **VP8 in a `.webm`**:

```
fourcc avc1  -> opened=False     <- no encoder
fourcc H264  -> opened=False
fourcc mp4v  -> opened=True       <- opens, but the browser won't decode it
fourcc VP80  -> opened=True       <- opens AND every browser plays it
```

`ffprobe` confirms the VP8 file is real (`codec_name=vp8`), it plays natively
everywhere, and it needs no extra dependency — no system `ffmpeg`, no
`imageio-ffmpeg` wheel baked into the image. One fourcc string, `VP80`.

The lesson that generalises: **"the encoder ran" and "the target can decode it"
are two different questions**, and the second is the one the demo lives or dies
on. Test playback on the actual target, not the exit code and the file size.

## 13. The rep count on the video and in the JSON must be the same number

The annotated video counts reps with an on-screen counter; the API counts them in
`find_reps`. If those two ever disagree the whole product looks broken, even when
each is internally correct. So I didn't let them be two computations. Both read
from **one** detection pass: `summarize()` produces the reps and their end times,
the JSON reports them, and the video's counter is literally `sum(end_t <= now)`
over that same list. There is nothing to drift because there is one source and two
views of it.

The skeleton *is* drawn on every frame, but interpolated between the sparse
samples, never re-detected. Detecting every frame would be a third more model
calls to move a wrist a few pixels — and worse, a second opinion the counter
could contradict. Sparse for the truth, dense only for the picture.

## 14. The library grew a system dependency and only the deploy noticed

The image built clean, then the container died on boot:

```
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

MediaPipe 0.10.35 `dlopen`s the GLES/EGL stack the moment it creates a landmarker.
My laptop has those libraries because it has a desktop GL stack; `python:3.11-slim`
does not. So `import mediapipe` succeeds in both places and the crash only happens
on the machine with no monitor — which is the only machine that serves the app.

The Dockerfile installed `libgl1` (opencv-headless needs it) and stopped there,
because an earlier MediaPipe was content with that. The fix was NOT to guess the
next missing `.so`, rebuild, guess again — that's a multi-minute round trip per
guess on a remote builder. I ran `ldd` on the actual `libmediapipe.so` locally and
read off every GL-family library it NEEDs in one shot:

```
libGLESv2.so.2      -> libgles2
libEGL.so.1         -> libegl1
libGLdispatch.so.0  -> (pulled in by the two above)
```

One rebuild, app up, `/health` green. This is #6 seen from the other side: the
library moves faster than the Dockerfile written against it. When a shared object
won't load, `ldd` the thing that fails — don't bisect the apt line.

## 15. Three exercises turned out to be one, plus a table

Adding barbell curl and squat next to the bicep curl looked like three rep
counters. It's one. Strip a curl and a squat down to the signal and they're the
same shape: a joint angle that sits high (arm straight, legs standing), dips low
(arm curled, hips down), and comes back. The hysteresis state machine that finds
the cycles never changed a line. What changed was a **table**: which three
landmarks make the angle (elbow vs knee), where the thresholds sit (a curl bottoms
near 40°, a squat near 90°), and what the coaching says.

So `exercises.py` is a frozen dataclass and three instances, and the pipeline
reads joints and thresholds off it. `video.py` measures whatever triplet it's
handed; `reps.py` counts whatever series it's given; the squat's knee flows
through the exact code the curl's elbow does.

The tell I'd built it right: on a squat clip the *elbow* never crosses the curl's
thresholds, and on a curl clip the *knee* never crosses the squat's — each
exercise counts zero on the other's footage, for free, because the only thing
that changed was the numbers. **When a new feature feels like N copies, look for
the axis they vary on; usually it's data, and the code is already written.** Same
bet as `backends.py` (#10), one layer up.

## 16. A spinner is a lie; a progress bar cost me a second request

Analysis takes 20–30s, and the old endpoint did the honest thing badly: it held
the POST open, ran MediaPipe and the encoder, and returned the finished result.
The browser could only show a spinner — no idea if it was 10% or 90% done, or
stuck.

You can't stream a percentage out of a request whose whole body is one blocking
call. So the one request became two: POST registers a job, kicks the work onto a
background task, and returns a token **immediately**; the page polls
`GET /progress/{token}` every 600ms and draws a real bar. The renderer takes a
`progress_cb(stage, pct)` and calls it through its two passes (detect 0–40%, draw
40–98%).

Two things that could have bitten and didn't: the callback fires from the
threadpool worker (annotate_video is sync, off the event loop) and just writes two
ints into a dict — safe enough under the GIL for a status line, no lock. And I
polled instead of Server-Sent Events on purpose: SSE buffers behind the proxies in
front of managed hosts, a poll is a plain GET that can't. **"Show progress" isn't a UI
task; it's a request-shape decision — you can't report on work you're blocking on.**

## 17. The preview lied by shrinking it

Checking the end card, the thin grey text looked doubled — a faint echo shifted
right on every line, while the bold orange "3/3" was crisp. I went hunting for a
double-draw in the text helper, then rendered the card straight to PNG to skip the
video codec: still there. Then I cropped one line at 1:1, no resize — **pixel
clean.**

The echo was never in the render. The image viewer downscales a 540px-wide frame
to show it, and downsampling thin anti-aliased strokes aliases them into a ghost;
the bold text has enough stroke to survive the shrink. I'd been debugging the
preview, not the pixels.

This is #3 again from a new angle: there I trusted a *helpful* visualisation
(Ultralytics hiding low-confidence points) over the array; here I trusted a
*downscaled* one over the buffer. **The thing on your screen is an artifact of how
it was displayed, not the ground truth. When a render looks wrong, check it at
native resolution before you touch the code.**

## 18. The beacon that couldn't preflight

Client-side usage events go out through `navigator.sendBeacon` — the one send that
survives the page unloading, which a normal `fetch` won't (the tab is gone before it
flushes). But a beacon has a constraint the tutorials skip: **it can't make a
preflighted request.** There is no OPTIONS round-trip on the way out — it fires once,
as the page dies, or not at all.

The page (Vercel) talks to the API (Railway) cross-origin whenever the client points
straight at it — which is why the app runs CORS in the first place — and a
cross-origin POST with an `application/json` body is exactly what a browser
preflights. My instinct was `application/json`, to match every other endpoint. That
body turns the beacon into a preflighted request, which `sendBeacon` quietly refuses
to send: it returns `false`, nothing reaches the network, and there is no error
anywhere to notice.

Keeping it CORS-**"simple"** fixes it. The body is a `text/plain` Blob — a safelisted
content type that needs no preflight — so the beacon flies. The server reads the raw
body and `json.loads` it, ignoring the content-type header entirely: the payload is
still JSON, it just doesn't announce itself as such. `track()` also checks the
boolean `sendBeacon` hands back and falls back to a `keepalive: true` fetch, because
a beacon whose return value you don't read is one you're only assuming fired.

This is #1 again at the network layer: **a fire-and-forget beacon has no error path
by design, so a broken one is indistinguishable from one nobody triggered.** When the
transport can only fail silently, pick the variant that can't fail — the safelisted
content-type — over the one that reads cleanest beside the endpoints around it.

## 19. Stateless by design meant blind by default

No accounts, no database, no history — upload, analyse, forget. I liked that rule.
Then the app went live and I couldn't answer the first question anyone asks: has
anyone used it, and what did the coach tell them when they did? The prod box has an
ephemeral disk, so results vanish on restart *by design* — and that same property
means there is nothing left to look at afterward. The guardrail I was proud of had
made me blind.

The wrong fix is to reverse it — add a DB, start keeping clips. The right one is a
single append-only JSONL trail (`telemetry.py`) that buys back the one missing signal
and nothing else, held to the same discipline that made the original choice good:

- **On a persistent volume, or it's pointless.** It writes only when `TELEMETRY_DIR`
  is set, and in prod that's a mounted `/data`, not the app dir. Point it at the
  ephemeral disk and the trail evaporates on the next deploy, right next to the
  results it was meant to outlive — the same lesson as baking the model into the
  image (#14): on this class of host, what you want to keep lives on a volume or it
  doesn't live.
- **The IP is hashed before it is ever written** — a salted SHA-256 with the *day*
  folded in and truncated to 12 chars, so the same visitor hashes differently
  tomorrow. Text only: never a clip, never landmarks, never a raw address. I wanted a
  headcount, not a dossier.
- **Off the request path or not at all.** `record()` drops the event into an
  in-memory buffer and returns; a daemon thread does the file IO on a timer and a
  final flush on shutdown. A slow or full volume loses telemetry, never a request —
  the same fail-open posture as the LLM coach, one layer over: a feature that
  measures the product must never be able to break it.

The through-line: **one design choice can be a strength and a blind spot at the same
time.** Statelessness was both. You don't resolve that by undoing the choice — you
add back the single signal it cost you, under the rules that made the choice worth
making: off by default, privacy-first, and incapable of harming the thing it watches.

---

## Smaller things that cost me time

- **Person's left is on the image's right.** `left_shoulder x=1610` vs
  `right_shoulder x=1223` — facing the camera, your left is on the viewer's
  right. Invisible bug, wrong-arm results.
- **MediaPipe returns normalised (0–1), YOLO returns pixels.** Mix them and you
  get plausible, wrong angles. No error. The adapter exists partly for this.
- **`uvicorn` owns logging config.** `logging.getLogger(__name__).info()`
  silently goes nowhere — root sits at WARNING with no handler. Use
  `getLogger("uvicorn.error")`.
- **`PYTHONUNBUFFERED=1` in Docker.** Python buffers stdout when it isn't a
  TTY, so your startup log is invisible exactly when you need it.
- **`opencv-python` vs `opencv-python-headless`.** The former wants libGL,
  which isn't on a slim image. Textbook deploy-day `ImportError`.
- **`.gitignore` only ignores untracked files.** Adding a rule after you've
  staged something does nothing.
- **`git init` doesn't create a branch — the first commit does.** And deploy
  platforms often build only `main`. Push `master` and it's accepted and never
  builds, with no error at all.
- **The browser records the codec we already write.** The Record button uses
  `MediaRecorder` with `video/webm;codecs=vp8` — the same VP8 (#12) the renderer
  encodes and the server serves. Record and render meet in the middle: the clip the
  user films is already the one format everything downstream speaks, no transcode.
- **Snapshot the request context before the slow work, not after.** `_run_job` copies
  `mode` and the hashed IP out of the job dict at the very top, because `_prune_jobs`
  can evict the job while the 20–30 s render runs; read them at the end — when the
  telemetry record is finally written — and they may already be gone. The work
  outlives the bookkeeping that launched it.
- **The freshest events aren't on disk yet.** `read_recent()` (the admin endpoint)
  calls `_flush()` before it reads, because the last few seconds of events are still
  in the in-memory buffer, not the file. Query the JSONL directly and you'd miss
  exactly the events you just triggered while testing — and conclude, wrongly, that
  logging is broken.
