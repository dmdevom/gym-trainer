Since your goal is learning, not just shipping, I'll structure this plan differently from a pure hackathon sprint: each phase pairs **concepts you're learning** with **things you're building**, so the hackathon becomes a forcing function for the learning rather than the other way around. The deadline (19 July, 23:59 IST) still shapes the schedule.

## The skills you're actually acquiring

Before the plan, name the learning targets explicitly — this project happens to be a near-perfect vehicle for five foundational skills that transfer to almost any AI/vision project:

1. **Working with pretrained models** — loading models, understanding inputs/outputs, reading model docs. (You'll never train from scratch in most real projects; using pretrained models well is the actual job.)
2. **Image/video fundamentals** — frames, color spaces (BGR vs RGB), resolution, fps, codecs. The unglamorous knowledge every CV project needs.
3. **Turning model output into product logic** — the keypoints→angles→state-machine→feedback pipeline. This "post-processing" layer is where 80% of the engineering happens in real AI products, and it's the least taught.
4. **Serving ML behind an API** — FastAPI, file uploads, handling slow processing, structuring responses.
5. **Deploying ML** — model weights, memory limits, cold starts. Where beginner projects usually die.

Keep a `LEARNINGS.md` file in your repo and jot down every "aha" and every bug that cost you an hour. It sounds silly; it doubles retention, and it makes great README/pitch material ("what we learned").

## Phase 0 — Setup + first contact with the models (today, ~2-3 hrs)

**Build:** Environment and two "hello world" scripts.

- Create the repo, a virtualenv, install `ultralytics`, `mediapipe`, `opencv-python`, `fastapi`, `uvicorn`, `numpy`, `matplotlib`.
- Script 1: load YOLOv8n-pose, run it on a single photo of a person, print the raw keypoints array. Then use `results[0].plot()` to save an annotated image and *look at it*.
- Script 2: same photo through MediaPipe, draw its landmarks.
- Record 3 test videos of yourself: good-form curls, sloppy half-reps, fast swinging reps. This is your ground-truth dataset.

**Learn (the point of this phase):** Don't rush past the raw output. Open the keypoints array and manually find your left elbow's coordinates. Understand: what does confidence/visibility mean? Why 17 keypoints for YOLO vs 33 for MediaPipe? What happens when you run it on a photo with no person? Two people? This 30 minutes of poking is where the mental model forms.

**Checkpoint:** You can look at a keypoints array and say "index 7 is the left elbow, and it's at pixel (312, 405) with 0.91 confidence."

## Phase 1 — Photo POC: keypoints → meaning (today, ~2-3 hrs)

**Build:** The `calc_angle` function, the photo analyzer from my earlier message, and a minimal FastAPI app with one `/analyze/photo` endpoint. Test through the auto-generated `/docs` UI.

**Learn:**
- The trigonometry: work out on paper why the dot-product formula gives the angle at the elbow. Don't copy it blindly — this pattern (vectors between keypoints → geometry → semantics) is the core trick of all pose-based apps.
- FastAPI basics: what `UploadFile` does, why the endpoint is `async`, what the auto docs are.
- Failure handling: upload a photo of a dog, a landscape, a person from behind. Watch what breaks. Making the code fail *gracefully* here teaches defensive ML coding.

**Checkpoint:** Upload a curl photo → get back `{"elbow_angle": 47.2, "phase": "fully curled"}`. Upload a dog photo → get a clean error, not a crash.

## Phase 2 — Video pipeline: the real learning day (17 July, full day)

This is the heart of both the project and the learning. Build it in this order, testing after each step:

**Morning — frames and angles (~3 hrs):**
- Write the `extract_angles` loop from my last message. First make it just *count frames* and print fps — understand what `cv2.VideoCapture` gives you.
- Add the backend call, collect the angle series from your good-form video.
- **Crucial learning step:** plot the angle series with matplotlib *before* any smoothing. Stare at the wave. You should literally see your reps as valleys. Then plot the sloppy video and see how the shape differs. This plot is how you'll debug everything later — angle-vs-time is your oscilloscope.

**Midday — smoothing and rep counting (~2-3 hrs):**
- Add smoothing, re-plot, compare. Seeing jitter disappear teaches you signal noise viscerally.
- Implement the hysteresis state machine. Run it on all 3 test videos. Does the count match reality? If not, adjust thresholds *by looking at your plot* — this threshold-tuning-against-ground-truth loop is exactly how real ML product work feels.

**Afternoon — form rules + comparison (~2-3 hrs):**
- Implement `form_feedback`. Verify the sloppy video triggers "partial curl" and the fast video triggers "too fast."
- Run your YOLO vs MediaPipe comparison: time both on the same video, overlay both angle plots. Write your conclusion in `LEARNINGS.md` and pick one backend for the product (keep the other in the repo — judges love seeing evaluated alternatives).

**Evening stretch goal — LLM coach feedback (~1-2 hrs):** Send the rep stats JSON to an LLM API and get natural-language coaching back. Learning: prompt design for structured-data-in, friendly-text-out — a hugely reusable pattern, and it directly scores on the hackathon's "AI fluency" criterion.

**Checkpoint:** `python analyze.py sloppy_video.mp4` prints the right rep count and the right complaints.

## Phase 3 — Product wrapper (18 July)

**Morning — API + annotated video (~3 hrs):**
- `/analyze/video` endpoint: accept upload, run pipeline, return JSON (rep count, per-rep stats, feedback, angle series for charting).
- Generate the annotated output video: draw the skeleton + live angle + rep counter on each frame with `cv2.VideoWriter`. Learning: how video *writing* works (codecs, fps matching). This is your demo's money shot.

**Afternoon — minimal frontend (~2 hrs):** One HTML page served by FastAPI: upload → spinner → results with the annotated video, rep table, angle chart (Chart.js), and coach feedback. Resist React; resist prettiness beyond clean.

**Evening — deploy (~2-3 hrs, budget more than you think):**
- Hugging Face Spaces (Docker or Gradio SDK) is the most forgiving free host for this. Learning: writing a Dockerfile, model weights caching, memory limits.
- Deploy early, test with a real phone-recorded video upload. The classic failure discovered only in production: large phone videos timing out — cap upload size and downscale frames (`cv2.resize` to 640px wide) before inference.

**Checkpoint:** A friend can open your public URL on their phone and analyze a video with zero instructions from you.

## Phase 4 — Submission package (19 July, morning — not night)

- **Demo video (3 min):** 30s problem ("form feedback is expensive/unavailable"), 90s live product walkthrough with the annotated video output, 30s how it works (your pipeline diagram!), 30s what's next. Record 2-3 takes.
- **README:** what it does, architecture diagram, how to run locally, the YOLO-vs-MediaPipe comparison, learnings. A README with a "what we learned" section stands out.
- **Optional deck (5-7 slides):** problem, demo screenshots, pipeline diagram, comparison results, roadmap (more exercises, real-time mode, mobile).
- Submit by afternoon. Never 23:50.

## Scope guardrails (read when tempted)

- **One exercise, done deeply.** Multi-exercise support is a roadmap slide, not a build task.
- **No user accounts, no database, no history.** Upload → analyze → results. Done.
- **If the state machine misbehaves,** the fix is almost always in the plot: look at the angle series before touching code.
- **If deployment fights you past 2 hours,** fall back: run locally + ngrok for a public URL, and note the tradeoff. A working demo beats a broken deploy.

The honest summary of what you'll walk away with: you'll have touched the full lifecycle — pretrained model → signal processing → product logic → API → deployment — on a project small enough to fully understand and real enough to demo. That's the actual curriculum of AI product engineering, compressed into four days.

Ready to start Phase 0? I can walk you through the environment setup and the first YOLO hello-world script whenever you are.