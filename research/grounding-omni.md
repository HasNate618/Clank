# Grounding: PLAN vs GROUND, one call or two (LFM3-Omni-2B)

**STATUS: NOT YET RUN.** This document is a pre-registration of the experiment
plus the harness that runs it. There are **zero MEASURED numbers below.** Every
prediction is labelled ASSUMED. Nothing here may be quoted as a result until the
harness output is pasted back into the Results section.

Harness: [`research/grounding/ground.py`](grounding/ground.py) — client only,
POSTs to `http://127.0.0.1:11434/v1/chat/completions`, never binds a port.

---

## Question

JobBot says *"Take my card."* The robot must emit one action:
`skill + side + optional named target`, e.g. `grab / right / "the JobBot ID card"`.

Should **PLAN** (which object) and **GROUND** (where is it) be one model call or two?

## What is already known (inherited, MEASURED elsewhere — not re-derived here)

- Same model, same object class, hand-drawn **crop**: 44/44. Same prompt on the
  **full frame**: 9/12. The crop is the whole difference.
- At `temperature: 0` the model is byte-identical across repeats.
- The model **always answers**; it does not reliably say "absent".

## Ground truth (MEASURED by inspection, this session)

`research/evidence/grounding-seed-view.png`, 960x540. The JobBot ID card is the
white badge leaning at the lower-left foot of the green cartoon-face monitor,
reading `1011.111.0101 / JobBot / MUSEUM CURATOR`.

- Full-frame pixel: **(705, 420)**, normalised (0.734, 0.778), origin top-left.
- Truth quadrant in the right-half crop (480,0,960,540): **bottom-left**.
- Truth quadrant in the container crop (600,240,960,500): **bottom-left**.

## Decision thresholds, fixed before seeing any output

| Field | Definition |
|---|---|
| **hit** | mapped full-frame error ≤ 48 px (5% of frame width) |
| **tight hit** | ≤ 24 px |
| **miss** | > 48 px, or unparseable, or `found:false` on a present object |
| **quadrant hit** | returned quadrant == truth quadrant |
| **hallucination** | `found:true` with a location for an absent object |

Accuracy is reported as `hits / attempts`, in the existing style (e.g. `9/12`).

---

## Runbook

```bash
cd /home/nate/Projects/Clank
P=spike/e3/.venv/bin/python
G=research/grounding/ground.py
O=research/grounding
mkdir -p "$O"

# 0. Sanity: print every prompt, send nothing. Read these before spending calls.
$P $G --mode all --dry-run

# A1. PLAN, text only (no image)
$P $G --mode plan               --jsonl "$O/raw.jsonl" | tee "$O/A1-plan.txt"

# A2. PLAN, with the full image
$P $G --mode plan-vision        --jsonl "$O/raw.jsonl" | tee "$O/A2-plan-vision.txt"

# A3-A7. GROUND, five input forms (gold description)
$P $G --mode ground-full             --jsonl "$O/raw.jsonl" | tee "$O/A3-ground-full.txt"
$P $G --mode ground-grid3            --jsonl "$O/raw.jsonl" | tee "$O/A4-ground-grid3.txt"
$P $G --mode ground-crop-right       --jsonl "$O/raw.jsonl" | tee "$O/A5-ground-crop-right.txt"
$P $G --mode ground-crop-right-grid3 --jsonl "$O/raw.jsonl" | tee "$O/A6-ground-crop-grid3.txt"
$P $G --mode ground-quadrant         --jsonl "$O/raw.jsonl" | tee "$O/A7-ground-quadrant.txt"

# B. ONE CALL (image + context + line -> skill + description + location)
$P $G --mode one-call           --jsonl "$O/raw.jsonl" | tee "$O/B1-one-call.txt"

# C. Hallucination controls (absent object: "the red fire extinguisher")
$P $G --mode absent-one-call    --jsonl "$O/raw.jsonl" | tee "$O/C1-absent-one-call.txt"
$P $G --mode absent-two-call    --jsonl "$O/raw.jsonl" | tee "$O/C2-absent-two-call.txt"

# D. Variance at temperature 0.7 (method repeated, seed free)
$P $G --mode ground-crop-right --temperature 0.7 --repeat 3 --jsonl "$O/raw.jsonl" \
    | tee "$O/D1-crop-t07.txt"
$P $G --mode one-call          --temperature 0.7 --repeat 3 --jsonl "$O/raw.jsonl" \
    | tee "$O/D2-one-call-t07.txt"
```

### Two things that will otherwise corrupt the numbers

1. **The first request of a session pays model-load time.** Run `--mode plan`
   first as a warm-up and treat its `wall_seconds` as an outlier, or discard it.
   Report latency from the second call onward.
2. **`absent-two-call` grounds a literal description**, not the PLAN output,
   because at temperature 0 PLAN is deterministic. To get the faithful two-call
   hallucination test, copy the `target` string out of `C2`'s `absent-plan`
   RECORD line and re-run:
   `$P $G --mode ground-full --description "<that target>"`.
   The same applies to the real object: feed the PLAN output back with
   `--description` to measure the true two-call pipeline.

---

## Exact prompts (verbatim, as the harness sends them)

### PLAN (no image) — `--mode plan`

```
You are the planner for a humanoid robot working a shift at the Job Simulator
museum counter.

Scene: you are facing a museum counter. In the middle is a white retro terminal.
To your right sits a green desktop computer with a cartoon face; this machine is
the NPC "JobBot", your co-worker. JobBot hands out tasks by speaking to you.

JobBot says: "Take my card."

Work out the single best low-level action. The action vocabulary is skill + side + target.
- skill: one of grab, press, push, turn, place
- side: which hand to use, left or right
- target: the object to act on

Reply with JSON only and nothing else:
{"skill": "<skill>", "side": "<left|right>", "target": "<the object, described precisely enough that someone who never heard JobBot speak can find it in a photo of this scene>"}
```

`--mode plan-vision` sends the identical text **plus** the 960x540 image. The
prompt hash is therefore the same and the only difference is the image part —
that isolates "does seeing the scene change the plan".

### GROUND on the full frame — `--mode ground-full`

```
This is a photo of the robot's view of a Job Simulator museum counter.

Find this object: the JobBot ID card: a small white badge with a barcode and small printed text reading JobBot, leaning against the lower left of the green monitor that has a cartoon face

Reply with JSON only and nothing else:
{"found": true, "x": <integer pixel x>, "y": <integer pixel y>}

x and y are pixel coordinates in the image you were given, origin at the TOP-LEFT corner, x increasing to the right. The image is 960 pixels wide and 540 pixels high. If the object is not in the image, reply {"found": false}.
```

### GROUND, 3x3 grid — `--mode ground-grid3` / `ground-crop-right-grid3`

```
This is a photo of the robot's view of a Job Simulator museum counter.

Find this object: <description>

The image is divided into a 3 by 3 grid. Reply with JSON only and nothing else:
{"found": true, "cell": "<row>,<col>"}

row is 0 for the top third, 1 for the middle third, 2 for the bottom third. col is 0 for the left third, 1 for the middle third, 2 for the right third. The image is <W> pixels wide and <H> pixels high. If the object is not in the image, reply {"found": false}.
```

Grid cells are converted to a full-frame pixel via the **cell centre**, then
`full = box_origin + (shown_px / crop_scale)`. Record the raw `row,col` as well
as the mapped pixel: on this model a 3x3 grid is expected to collapse toward the
image centre (ASSUMED — the inherited note), and a collapse to `1,1` across all
crops is itself the finding.

### GROUND, quadrant on a container crop — `--mode ground-quadrant`

```
This is a photo of the robot's view of a Job Simulator museum counter.

Find this object: <description>

Reply with JSON only and nothing else:
{"found": true, "quadrant": "<top-left|top-right|bottom-left|bottom-right>"}

The quadrant refers to the image you were given: top-left is the upper left corner. The image is <W> pixels wide and <H> pixels high. If the object is not in the image, reply {"found": false}.
```

Crop is (600,240)-(960,500) upscaled x2 → 720x520. Truth = **bottom-left**.

### Crops

| Name | Box (full-frame) | Shown size @ scale 2 |
|---|---|---|
| right half | (480, 0, 960, 540) | 960x1080 |
| container / monitor | (600, 240, 960, 500) | 720x520 |

Crop files are written next to the harness as `crop_<box>_x<scale>.png` so the
exact pixels the model saw are auditable.

### ONE CALL — `--mode one-call`

```
You are the planner for a humanoid robot working a shift at the Job Simulator
museum counter.

Scene: you are facing a museum counter. In the middle is a white retro terminal.
To your right sits a green desktop computer with a cartoon face; this machine is
the NPC "JobBot", your co-worker. JobBot hands out tasks by speaking to you.

JobBot says: "Take my card."

Decide the single best low-level action AND locate its target in the image you were given.
- skill: one of grab, press, push, turn, place
- side: which hand to use, left or right
- target: the object to act on

Reply with JSON only and nothing else:
{"skill": "<skill>", "side": "<left|right>", "target": "<the object>", "found": true, "x": <integer pixel x>, "y": <integer pixel y>}

x and y are pixel coordinates in the image you were given, origin at the TOP-LEFT corner, x increasing to the right. If the target is not in the image, reply {"found": false}.
```

### Hallucination control — `--mode absent-one-call`

Identical to ONE CALL except the line is `JobBot says: "Fetch the red fire
extinguisher."` No fire extinguisher exists in the frame (MEASURED by
inspection: the right side of the counter holds a mug, a red button on the
monitor bezel and a bowl; no extinguisher). A correct answer is
`{"found": false}`. Any pixel answer is a hallucination, and its distance from
the true card position tells you whether the model falls back to the salient
object or invents a plausible corner.

---

## Results

**UNMEASURED. Fill from harness output. Do not populate from expectation.**

| Method | Input form | Accuracy | Median latency (ms) | Median err (px) | Notes |
|---|---|---|---|---|---|
| A1 PLAN | text only | — | — | n/a | description quality only |
| A2 PLAN | + full frame | — | — | n/a | does vision change the plan? |
| A3 GROUND | full frame, pixel | — | — | — | baseline; expect≈9/12 per inherited note |
| A4 GROUND | full frame, 3x3 grid | — | — | — | check for centre collapse (`1,1`) |
| A5 GROUND | right-half crop, pixel | — | — | — | expect high per inherited 44/44 |
| A6 GROUND | right-half crop, 3x3 grid | — | — | — | |
| A7 GROUND | container crop, quadrant | — | — | n/a | truth = bottom-left |
| B ONE CALL | full frame | — | — | — | accuracy **and** latency vs A3+A5 |
| C1 absent | one call | — | — | n/a | hallucination count |
| C2 absent | two call | — | — | n/a | hallucination count |
| D1 crop @0.7 | right-half crop | — | — | — | variance |
| D2 one-call @0.7 | full frame | — | — | — | variance |

Latency to be reported as **median of N≥3** excluding the warm-up call, plus the
`usage` / `timings` fields the server returns in each RECORD line.

---

## Predictions (ASSUMED — to be overwritten by measurement)

| # | Prediction | Basis |
|---|---|---|
| P1 | GROUND on the right-half crop beats GROUND on the full frame, in pixels | inherited crop-vs-full-frame gap (44/44 vs 9/12) |
| P2 | 3x3 grid returns `1,1` on most crops — centre collapse | inherited note; untested here |
| P3 | ONE CALL is fast (one round trip) but its location is worse than two calls where GROUND gets a crop, because one call has no crop advantage | ASSUMED |
| P4 | The PLAN description is semi-discriminative: names "ID card"/"badge" but omits "on the green monitor" | ASSUMED |
| P5 | The model invents a location for the fire extinguisher (`found:true`) in at least 1 of 2 forms | inherited "always answers" |

P2 and P5 are the ones that would change the recommendation. If the model says
`found:false` cleanly for the extinguisher, the absent-object cost of one call
largely disappears and the latency argument wins.

---

## Recommendation

**Pending measurement.** The decision rule, fixed now so it cannot be
rationalised after the fact:

- If **ONE CALL** matches two-call accuracy (both ≥ tight hits on the same
  input form) **and** is at least ~40% cheaper in wall clock → **one call**, and
  keep a crop in the pipeline.
- If **ONE CALL** is accurate but **hallucinates on absent objects** while the
  two-call GROUND path returns `found:false` → **two calls**, because the
  separate GROUND step is where an absence check can be inserted and audited.
- If **ONE CALL** is worse in pixels than A5/A6 → **two calls with a crop**;
  the crop, not the call count, is the dominant variable.
- If the crop alone fixes accuracy regardless of call count → the real finding
  is "crop, then use one call", and report it as such rather than as a win for
  the two-call design.

---

## Failures worth stating bluntly

Nothing has been run, so there are no failures to state yet. The failure modes
to watch for and report without softening:

- 3x3 grid returning a constant cell (the interface is worthless, not the model).
- `found:true` on an absent object (no reliable absence signal; a safety problem
  for a robot that will reach for it).
- Unparseable JSON (the reply is not a usable action).
- Long-tail latency (p95 much worse than median; a robot cannot budget on median).

---

# MEASURED - harness executed 2026-11-04

Everything above was pre-registration. This section is what the harness actually returned
against `research/evidence/grounding-seed-view.png` (960x540), truth pixel **(705, 420)**.

| mode | input | error | verdict | wall |
|---|---|---|---|---|
| `ground-full` | full frame, pixel | **360 px** | off-frame, useless | 0.46 s |
| `ground-grid3` | full frame, 3x3 cell | **565 px**, cell 1,0 vs truth 2,2 | useless | 0.33 s |
| `ground-crop-right` | right half, pixel | **65 px** | usable | 0.49 s |
| `ground-crop-right-grid3` | right half, 3x3 cell | **230 px**, cell 1,2 vs truth 2,1 | useless | 0.48 s |
| `ground-quadrant` | tight crop, quadrant | **correct** (bottom-left) | categorical pass | 0.28 s |
| `one-call` | full frame, everything | **340 px** | useless | 0.60 s |
| **`one-call-crop-right`** | right half, everything | **39 px** | **best** | 0.72 s |
| `absent-one-call` | full frame, fire extinguisher | invented (680, 620) | **hallucinated** | 0.60 s |
| `absent-ground-full` | full frame, fire extinguisher | invented (720, 720) | **hallucinated** | 0.36 s |
| `absent-ground-crop-right` | right half, fire extinguisher | `found: false` | **correct refusal** | 0.35 s |

Latency is dominated by prompt processing (~150-260 ms) plus ~10 ms/token of decode, so
these are fast enough for a 2 Hz loop.

## The crop is still the whole difference

Full frame 340-360 px, the same call on the right half 39-65 px. This reproduces the earlier
finding (crop-to-tray 44/44 against 9/12 on the full frame) on a different object and a
different question. A 340 px error is a third of the frame: not a seed, a guess.

## One call beats two, on a crop

`one-call-crop-right` is **39 px** against **65 px** for the two-call path, and the two-call
path also has to pay for PLAN - measured 0.78 s for PLAN with the image, so 1.27 s total
against 0.72 s. There is no accuracy case for splitting them.

PLAN on its own is the weakest link either way: **without an image it invents the target**
("the red card sitting on the left middle grid"), and with one it described the card as *blue*
on the white console when it is white on the green monitor. The description is not worth a
separate round trip.

## Grids are worse than pixels on this model

230 px and 565 px from the two grid modes, both landing in the wrong cell. This is the
documented "image-global grid collapses toward the centre" behaviour, reproduced. Do not use
a 3x3 cell to ground; ask for a pixel.

## Refusal only happens when the crop is tight

For a fire extinguisher that is not in the frame: the full-frame modes **invent a location**
and report `found: true`. Only the cropped mode answers `found: false`. So presence detection
is not a property of the model, it is a property of how much of the frame you show it - which
means the crop must come from somewhere other than the model, and that is the open item
(container detection is currently hand-drawn).

## What this means for the pipeline

`one call on a crop` -> pixel (39 px, ~5-7 cm at the card's depth) -> depth sample -> 3D
point -> `track.py` follows it by projection, and the reach's own damped correction closes
the rest. The seed does not need to be exact; it needs to be on the right surface.

The quadrant answer is the only categorically correct one, so keep it as the confirmation
while the pixel is the seed: if quadrant and pixel disagree, the seed is wrong.

---

# Model capability note (not tested): which of these can reason

Recorded 2026-11-04 from the model cards, not from measurement. It matters because the whole
grounding design above treats the model as a classifier, and the failure modes that hurt
(inventing a location for an absent object, describing a white card as blue) are judgement
failures rather than resolution failures.

| model | sees | hears | thinks |
| --- | --- | --- | --- |
| **LFM3-Omni-2B** (in use) | image | audio | **no** - non-reasoning by design; `enable_thinking=True` unsupported, every answer opens with an empty think block |
| LFM3-VL-2B | image | no | **yes** - reasons step by step; server needs `--reasoning-parser qwen3` |
| LFM3-1.7B (text) | no | no | yes, by default; `enable_thinking=False` disables |
| LFM3-Audio-2B | no | audio | opt-in, by prefilling the think tag |

**Consequence for this project.** The model that hears the game is the one that cannot
reason, and the models that can reason cannot hear it. As long as the brain is Omni, no
prompt change turns on thinking - so "ask it to reason harder" is not available here.

**Local state.** `LFM3-VL-2B-Q8_0.gguf` is on disk (1.57 GB) but **its `mmproj` projector is
not**, so it cannot see anything yet; llama.cpp needs the pair. The running server exposes
only Omni.

**What to test when it is available**, in the order that the failures justify:

1. **Absence.** Does reasoning stop the invented location for an object that is not there?
   Measured failure today: both full-frame modes returned a pixel and `found: true` for a
   fire extinguisher that is not in the frame. This is the one failure that costs a real
   action, so it is the first thing worth a reasoning model.
2. **Target description.** Omni called a white card blue. Does a reasoning pass produce
   "the badge reading JobBot leaning on the green monitor" instead of a confident wrong
   colour?
3. **Multi-step lines.** "Take my card and put it in the slot" needs two actions; Omni emits
   one. This is where reasoning should help most, and it is untested.

**Not worth testing:** pixel accuracy. 39-65 px on a crop is a resolution limit, not a
thinking limit, and a bigger think block does not put more pixels on the object.

**The audio cost.** VL has no audio input, and `LFM3-Audio-2B` is not downloaded, so
`VL + reasoning` means a second stage: transcribe first (Omni, or a separate ASR), then reason
over text plus image. That is two calls and two failure modes where there is now one. Worth
paying only if test 1 or 3 above shows reasoning changes the answer.
