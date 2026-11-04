# VL vs Omni: does reasoning buy pixel grounding?

**STATUS: MEASURED, 2026-11-04.** Two local models, same image, same prompts, same
harness, one call per cell unless stated otherwise. Raw transcripts and JSONL
records are in [`grounding/runs/`](grounding/runs/).

## Answer, up front

**The reasoning model buys accuracy and costs latency and refusals.** LFM3-VL-2B
is 9-338 px more accurate than LFM3-Omni-2B on every grounding fixture, spends
6-130x the tokens and 5-36x the wall time doing it, and is **worse at saying
"not there"**: it hallucinated the absent fire extinguisher in 4/4 fixtures where
Omni hallucinated in 3/4. The reasoning is not wasted - it visibly locates the
card on the full frame, where Omni misses by 360 px - but the one place it should
help (deciding an object is absent) is exactly where it fails hardest.

Recommendation at the bottom. Short version: **Omni stays the live planner; keep
VL as an offline / one-shot locator when a single correct pixel is worth 5.7 s and
you already have a crop.** Neither model can be trusted to refuse, so absence must
be decided by the caller, not the model.

## Setup

| | LFM3-VL-2B (key `vl3`) | LFM3-Omni-2B (key `omni`) |
| --- | --- | --- |
| Model id on `/v1/models` | `/home/nate/AI/models/LFM3-VL-2B-Q8_0.gguf` | `/home/nate/AI/models/LFM3-Omni-2B-Q8_0.gguf` |
| Context | 32768 | 65536 |
| Reasoning flags | `--reasoning-format deepseek` (plus `--jinja`) | none |
| Modalities | text, image, video | text, image, video, **audio** |
| Server | llama.cpp b1-8ca4060, `~/AI/llama.sh` | same binary, same host |

All **MEASURED** except the audio column, which is **ASSUMED** from the wrapper's
own help text (`llama help`: "omni … text+vision+audio … vl3 … vision+video"). I
did not feed either model audio.

Note on the brief: the task said vl3 runs "`--reasoning-parser qwen3`". It does
not. The running command line is `--reasoning-format deepseek`; llama.cpp strips
the think block into `reasoning_content` and the server log says *"chat template
supports preserving reasoning, it is enabled by default"*. **MEASURED** from
`ps` and `/home/nate/AI/logs/vl3_20261104_085458.log`. The behaviour the brief
asked about - a separate thinking block - is present either way.

Method: **MEASURED** unless noted.

- Image: `research/evidence/grounding-seed-view.png`, 960x540. Truth for the
  JobBot ID card: **(705, 420)**, origin top-left, "hit" = error <= 48 px (the
  threshold pre-registered in `research/grounding-omni.md`).
- Crop: right half `(480,0,960,540)`, upscaled 2x to 960x1080, mapped back
  through the crop box+scale. Identical bytes for both models.
- `temperature: 0`, `max_tokens: 4096` (see the budget trap below; the harness
  default of 200 returns **nothing** from vl3).
- One warm-up request per model before any timed call, excluded. vl3 was warm
  before I started; omni was warmed with the same arithmetic question.
- `wall_seconds` is client-side round trip, so it includes ~0.15-0.25 s of
  prefill and HTTP overhead. Decode throughput is ~92-96 tok/s for both, so
  wall time tracks token count almost exactly (**MEASURED**, `timings` in every
  record).

## Results

Error is the returned pixel mapped back to full-frame pixels, versus (705, 420).
Every cell is **MEASURED**, `temperature: 0`.

| model | mode | error px | wall s | completion tokens | found | refusal correct? |
| --- | --- | --- | --- | --- | --- | --- |
| vl3 | `ground-full` | **22.4** | 15.51 | 1404 | true | n/a |
| vl3 | `ground-crop-right` | **35.1** | 5.75 | 505 | true | n/a |
| vl3 | `one-call` | 181.7 | 20.31 | 1863 | true | n/a |
| vl3 | `one-call-crop-right` | **30.0** | 19.33 | 1762 | true | n/a |
| vl3 | `multistep` | 204.1 | 10.33 | 946 | true | n/a |
| omni | `ground-full` | 360.3 | 0.43 | 19 | true | n/a |
| omni | `ground-crop-right` | 65.0 | 0.51 | 19 | true | n/a |
| omni | `one-call` | 340.3 | 0.60 | 41 | true | n/a |
| omni | `one-call-crop-right` | **39.1** | 0.73 | 41 | true | n/a |
| omni | `multistep` | 301.0 | 0.58 | 39 | true | n/a |
| vl3 | `absent-one-call` | (inv. 305.2) | 13.11 | 1209 | **true** | **no**, hallucinated |
| vl3 | `absent-ground-full` | (inv. 39.1) | 28.80 | 2579 | **true** | **no**, hallucinated |
| vl3 | `absent-ground-crop-right` | (inv. 74.9) | 4.07 | 340 | **true** | **no**, hallucinated |
| vl3 | `absent-plan` (text only) | - | 50.52 | 4598 | n/a | **no**, invented a place |
| omni | `absent-one-call` | (inv. 201.6) | 0.61 | 43 | **true** | **no**, hallucinated |
| omni | `absent-ground-full` | (inv. 300.4) | 0.36 | 19 | **true** | **no**, hallucinated |
| omni | `absent-ground-crop-right` | - | 0.37 | 7 | **false** | **yes** |
| omni | `absent-plan` (text only) | - | 0.30 | 26 | n/a | **no**, invented a target |

Bold = within the 48 px hit threshold. "(inv." = the object is not in the frame,
so the number is only shown to locate the hallucination: it is the distance from
the returned pixel to the card, i.e. what the model was actually pointing at.

Four readings that matter:

1. **On the full frame VL is 16x more accurate** (22 px vs 360 px) - that is the
   single largest difference in the table, and the only one that survives a wrong
   crop box.
2. **On the hand-drawn crop VL is 1.9x more accurate** (35 px vs 65 px) for
   `ground`, but just 1.3x for one call (30 px vs 39 px) - both are inside the
   48 px hit threshold, at 19.3 s vs 0.73 s.
3. **Both models put the answer outside the image on one-call/multistep**: vl3
   `(730, 600)` and omni `(720, 760)` in a 540-tall frame. Neither notices. The
   harness records `in_frame: false`; nothing in either reply flags it.
   **MEASURED.**
4. **Refusal inverts the intuition.** The model that thinks is the one that
   talks itself into a fire extinguisher. vl3's own reasoning for
   `absent-one-call` reads: *"Below it there's a white paper with \"JobBot\" text
   and a red object to its left/top - that red thing could be the fire
   extinguisher? Hmm."* - there is no red cylinder in the frame; it is reasoning
   about the red stripe printed on the card. **MEASURED** by inspection of the
   image and the reasoning text.

The Omni numbers reproduce the figures already recorded in `AGENTS.md`
(full-frame pixel 360 px, crop pixel 65 px, one call including the crop 39 px /
0.72 s). This run matched them to 0.1 px and 0.01 s (**MEASURED**), which is
strong though indirect evidence (**ASSUMED**) that those earlier numbers were
Omni.

## Exact prompts

**MEASURED** - copied verbatim from `ground.py --dry-run`. `${W}`/`${H}` are the
dimensions of the image the model is shown (960x540 full frame, 960x1080 for the
2x crop); `${LINE}` is what JobBot says.

`ground-full`, `ground-crop-right`, `absent-ground-full`, `absent-ground-crop-right`:

```
This is a photo of the robot's view of a Job Simulator museum counter.

Find this object: the JobBot ID card: a small white badge with a barcode and small printed text reading JobBot, leaning against the lower left of the green monitor that has a cartoon face

Reply with JSON only and nothing else:
{"found": true, "x": <integer pixel x>, "y": <integer pixel y>}

x and y are pixel coordinates in the image you were given, origin at the TOP-LEFT corner, x increasing to the right. The image is ${W} pixels wide and ${H} pixels high. If the object is not in the image, reply {"found": false}.
```

`one-call`, `one-call-crop-right`, `multistep`, `absent-one-call`:

```
You are the planner for a humanoid robot working a shift at the Job Simulator
museum counter.

Scene: you are facing a museum counter. In the middle is a white retro terminal.
To your right sits a green desktop computer with a cartoon face; this machine is
the NPC "JobBot", your co-worker. JobBot hands out tasks by speaking to you.

JobBot says: "${LINE}"

Decide the single best low-level action AND locate its target in the image you were given.
- skill: one of grab, press, push, turn, place
- side: which hand to use, left or right
- target: the object to act on

Reply with JSON only and nothing else:
{"skill": "<skill>", "side": "<left|right>", "target": "<the object>", "found": true, "x": <integer pixel x>, "y": <integer pixel y>}

x and y are pixel coordinates in the image you were given, origin at the TOP-LEFT corner, x increasing to the right. If the target is not in the image, reply {"found": false}.
```

`plan`, `plan-vision`, `absent-plan` (the same block, image attached only for
`plan-vision`):

```
You are the planner for a humanoid robot working a shift at the Job Simulator
museum counter.

Scene: you are facing a museum counter. In the middle is a white retro terminal.
To your right sits a green desktop computer with a cartoon face; this machine is
the NPC "JobBot", your co-worker. JobBot hands out tasks by speaking to you.

JobBot says: "${LINE}"

Work out the single best low-level action. The action vocabulary is skill + side + target.
- skill: one of grab, press, push, turn, place
- side: which hand to use, left or right
- target: the object to act on

Reply with JSON only and nothing else:
{"skill": "<skill>", "side": "<left|right>", "target": "<the object, described precisely enough that someone who never heard JobBot speak can find it in a photo of this scene>"}
```

`${LINE}`: `"Take my card."` for `plan`, `plan-vision`, `one-call`,
`one-call-crop-right`; `"Take my card and put it in the slot."` for `multistep`;
`"Fetch the red fire extinguisher."` for the `absent-*` modes.

## Target description (`plan-vision`)

The card is white, reads `JOB SIMULATOR INC. / JobBot / MUSEUM CURATOR / 101 JOB
STREET / JOB CITY`, and is clipped to the lower front of the beige-and-green
JobBot monitor. There is no hand in the frame holding it. **MEASURED** by
inspection of the image and a 6x zoom of the card.

- **vl3**, 376 tokens, 4.23 s:
  `{"skill": "grab", "side": "right", "target": "the white card/paper held by JobBot in its hand"}`
- **omni**, 59 tokens, 0.78 s:
  `{"skill": "grab", "side": "right", "target": "JobBot's card, the small rectangular card with the 'JobBot' logo and 'Museum Simulator' text, held in the green desktop computer on the right side of the counter"}`

Judgement (**ASSUMED**, my reading):

- vl3 names the object and its colour correctly ("white") and is discriminative,
  but "held by JobBot in its hand" is false - nothing holds it.
- omni is longer, also discriminative, and **fabricates "Museum Simulator"** as
  card text. The card says "JOB SIMULATOR INC." on one line and "MUSEUM CURATOR"
  on the next; the model merged them. It does *not* get the colour wrong, so the
  specific earlier failure the brief warned about ("called the white card blue")
  did not recur in either model **MEASURED for this run, one sample each**.
- Both pick `grab`/`right` correctly.

## Reasoning: present, separate, and mostly self-correction

**MEASURED.** The arithmetic row's vl3 time is the server's `predicted_ms`
(generation only); the rest are client wall clock.

| same question | vl3 completion tokens | omni completion tokens | vl3 wall | omni wall |
| --- | --- | --- | --- | --- |
| "What is 17 times 23? Think it through." | 265 | 70 | 2.84 s | 0.73 s |
| `plan` (text only, no image) | 3295 | 25 | 35.16 s | 0.31 s |
| `plan-vision` | 376 | 59 | 4.23 s | 0.78 s |
| `ground-crop-right` | 505 | 19 | 5.75 s | 0.51 s |

vl3 spends 3.8x to 132x the tokens. The reply body is unchanged in size - for the
pixel fixtures it is 20 to 40 characters of JSON either way, even where Omni's
`plan-vision` target sentence is the longest output in the run - so **the entire
difference is thinking**.

**The reasoning is not in the reply.** llama.cpp returns it in a separate
`reasoning_content` field; `content` holds only the JSON. A client that reads
`content` sees no reasoning at all. It is on the wire, not in the answer.

Representative block (vl3, `ground-full`, 4873 chars, first and last third; the
answer was 22.4 px from truth):

```
The white badge with JobBot text and barcode appears around the lower area of the
green monitor, roughly at [655, 706, 795, 881] region. Let me estimate pixel
coordinates. The image is 960x540. ... y around 706 to 881? But height is only
540. Wait, the image height is 540. Let me re-examine.
...
Let's settle: x from about 655 to 795 (in 0-1000) => pixels 655? Wait if width
960, then 655/960=0.68 -> pixel 655? No! 0.68*960 = 652.
...
Thus box approx [655, 380, 795, 480]. ... Center approx: x = (655+795)/2 = 725,
y = (380+480)/2 = 430.
```

Is it wasted? **Partly.** On `ground-full` it self-corrects a units error and
lands 22 px out where Omni (19 tokens, no reasoning) is 360 px out - that is the
reasoning earning its keep. On `absent-one-call` the same mechanism reasons from
"there might be a red object" to a confident `found: true` with coordinates for an
object that does not exist, and burns 13 s doing it. On `absent-plan` it does not
terminate: with `max_tokens: 4096` it spent **4096 tokens / 44.8 s and emitted an
empty reply**; at 8192 it finished in 4598 tokens / 50.5 s by inventing *"the red
fire extinguisher located on the museum counter to the left of the white retro
terminal"*. Nothing is to the left of the terminal.

Budget trap, **MEASURED**: at the harness default `--max-tokens 200` the one
fixture I re-ran (`ground-crop-right`) came back with `content: ""` after 2.5 s -
vl3 spent the whole budget thinking. Earlier, at 1200 tokens, five of the twelve
cells did the same, all of them mid-thinking. The shortest *complete* vl3 reply in the whole run was 340
tokens. Any caller of vl3 needs a budget in the thousands; Omni's replies are
7-59 tokens.

## Multi-step: "Take my card and put it in the slot."

One call, image attached, `max_tokens: 4096`. Both models emit exactly one action;
**neither indicates a second step, a plan, or an ordering**. **MEASURED.**

- vl3 (946 tokens, 10.33 s):
  `{"skill": "grab", "side": "right", "target": "card", "found": true, "x": 600, "y": 595}`
- omni (39 tokens, 0.58 s):
  `{"skill": "grab", "side": "right", "target": "card", "found": true, "x": 680, "y": 720}`

vl3's reasoning does discuss the second step - *"first I need to grab the card,
then place it … the immediate next step … is to grab the card"* - and then
discards it, because the schema it was given has room for one action. So the
information is there and the prompt format throws it away. Neither model is doing
task decomposition; do not read "grab" as a plan. Both point at nothing: vl3's
`y=595` and omni's `y=720` are below a 540-pixel-tall frame, and the card is at
(705, 420).

## Variance and determinism

**MEASURED.**

- `temperature: 0`, identical prompt, repeated: the **returned pixel is stable**
  (vl3 `445,770` / 35.1 px on three separate calls; omni `400,720` / 65.0 px on
  two), but vl3's **thinking is not**: 505, 505 and 430 completion tokens, wall
  5.75, 5.88 and 4.96 s. So "temperature 0 is byte-identical" (the harness's own
  note) is true of the answer and false of the reasoning. Omni's 19 tokens were
  identical on both calls. **ASSUMED** cause, not investigated: the server runs a
  quantised KV cache (`-ctk q8_0 -ctv q8_0`) with prefix-cache reuse, so an
  identical prompt can see a slightly different numeric path and a near-tie in
  the thinking flips. The answer never moved.
- `temperature: 0.7`, `ground-crop-right`, 3 samples:
  - vl3 error **35.1, 35.2, 35.1 px**, but tokens **808 / 485 / 365** and wall
    **9.09 / 5.56 / 4.24 s**. The answer is stable; the cost is not, and that is
    already true at temperature 0.
  - omni error **65.0, 65.0, 65.0 px**, tokens identical, 0.47 s each. Peakier
    than the temperature suggests.

## Recommendation

**Do not make the reasoning model the default. It buys accuracy you already have,
and it costs the two things you cannot spend: latency and the audio input.**

Reasoning (**MEASURED**):

- +338 px better on the full frame (22 vs 360). Real, large, and the only result
  here that is not fixable by cropping.
- +30 px better on the crop for `ground` (35 vs 65), +9 px for one-call (30 vs 39).
  Both Omni numbers are already inside the 48 px hit threshold.
- -1 on refusals: vl3 hallucinated 4/4 absent fixtures, Omni 3/4 (the crop case,
  the one that matters, is the one Omni gets right).
- 5-36x wall time: 5.7 s vs 0.5 s on a crop, 15.5 s vs 0.4 s on a full frame,
  19.3 s vs 0.7 s for one call.
- 6-130x tokens, and a 200-token budget returns nothing at all.

Cost of dropping Omni (**ASSUMED**, from the wrapper's help text): Omni is the
only one of the two that can hear the game. JobBot's instruction arrives as
speech. Choosing vl3 means transcribing the audio first (another model, another
latency) or losing the direction that drives the whole task.

Where each belongs:

- **Live loop, per task: Omni, one call, on the crop.** 39 px in 0.73 s is the
  best accuracy-per-second in the table and it keeps the audio path.
- **One-shot, offline: vl3 `ground-crop-right`.** 35 px in 5.75 s, or 22 px on the
  full frame in 15.5 s if the crop box is untrustworthy. Use it to audit the crop,
  or once per task if a 6-second pause is acceptable. Do not call it per tick -
  and per `AGENTS.md` you should not be re-grounding per tick anyway, because the
  pixel is re-derived from the projected 3D point every tick.
- **Absence, either model: do not ask.** Both invent. A `found: false` from Omni
  on the crop is the only correct refusal observed, and it is 1 for 4, not a
  capability. Gate on something mechanical - the glow cliff, a non-empty crop, the
  depth patch - as the existing notes already do.

If a single model must do it all, **Omni**: 39 px, 0.73 s, hears JobBot, and its
one correct refusal is the only one in the run.

## What this does not show

- One image, one object, one truth pixel. n=1 per cell except the three temp-0.7
  samples. **MEASURED but thin.**
- No audio was passed to Omni; the audio advantage is **ASSUMED** from the wrapper
  help text and `research/d1-omni-capabilities.md`, not measured here.
- The 2x crop and its box are hand-chosen. **ASSUMED** correct for the card; a
  wrong box invalidates every crop row and favours vl3's full-frame number.
- Both models were quantised Q8_0, so this says nothing about a smaller quant.
- `error_px` measures the single point returned. It says nothing about how
  confident either model is; neither emits a confidence.
- The transient: the first `python3 -m unittest discover -s tests -t .` run had 1
  error, two subsequent runs passed 748/748. Unrelated to this change (nothing
  imports `ground.py`), not diagnosed.

## Reproduce

```bash
cd /home/nate/Projects/Clank
P=spike/e3/.venv/bin/python ; G=research/grounding/ground.py

bash /home/nate/AI/llama.sh vl3     # or: omni
until curl -fsS http://127.0.0.1:11434/v1/models >/dev/null; do sleep 1; done

# warm-up (not timed), then the fixtures
$P $G --mode plan-vision --max-tokens 4096            # discard
for m in plan ground-full ground-crop-right one-call multistep plan-vision \
         absent-one-call absent-two-call; do
  $P $G --mode "$m" --max-tokens 4096 --jsonl research/grounding/runs/<key>/raw.jsonl
done
```

`absent-plan` is a fixture inside `absent-two-call`; it needed
`--max-tokens 8192` for vl3 and was re-sent as
`--mode plan --line "Fetch the red fire extinguisher."`, which is why the
`absent-plan_mt8192.txt` record in `runs/vl3/raw.jsonl` carries `mode: "plan"` -
same prompt hash as the `absent-two-call` fixture, larger budget.

Harness changes this session (`research/grounding/ground.py`, no existing mode
altered): the model id now comes from `--model` or from `/v1/models` instead of a
hardcoded Omni path; `reasoning_content` is captured and printed as a separate
`REASONING` block and stored in the JSONL; `--line` overrides JobBot's line; and
`--mode multistep` was added for the two-step question. `--mode all --dry-run`
still emits the same 11 configurations it did before.
